from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import imageio_ffmpeg
import librosa
import numpy as np
import torch
import torchvision.transforms as T

from .data_utils import EMOTIONS
from .models import AudioCNN, build_face_model, build_video_model


class MultiModalEmotionEngine:
    def __init__(self, artifact_dir: str):
        self.artifact_dir = Path(artifact_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = json.loads((self.artifact_dir / "fusion_config.json").read_text(encoding="utf-8"))
        self.labels = config.get("labels", EMOTIONS)
        self.weights = config.get("fusion_weights", {"audio": 0.4, "face": 0.3, "video": 0.3})
        n = len(self.labels)

        self.audio_model = AudioCNN(n).to(self.device).eval()
        self.face_model = build_face_model(n).to(self.device).eval()
        self.video_model = build_video_model(n).to(self.device).eval()

        self.audio_model.load_state_dict(torch.load(self.artifact_dir / "audio_model.pt", map_location=self.device))
        self.face_model.load_state_dict(torch.load(self.artifact_dir / "face_model.pt", map_location=self.device))
        self.video_model.load_state_dict(torch.load(self.artifact_dir / "video_model.pt", map_location=self.device))

        self.face_tf = T.Compose(
            [T.ToPILImage(), T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)]
        )

    def _audio_logits(self, audio_path: str):
        y, sr = librosa.load(audio_path, sr=16000)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        mel = librosa.power_to_db(mel, ref=np.max)
        mel = cv2.resize(mel, (128, 64))
        x = torch.from_numpy(mel).float().unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.audio_model(x)[0]

    def _face_logits(self, image_path: str):
        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError("Could not read image")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = self.face_tf(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.face_model(x)[0]

    def _video_logits(self, video_path: str, num_frames: int = 16):
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = np.linspace(0, max(total - 1, 0), num_frames, dtype=int)
        frames = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                frame = np.zeros((112, 112, 3), dtype=np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (112, 112))
            frames.append(frame)
        cap.release()
        x = torch.from_numpy(np.stack(frames)).float() / 255.0
        x = x.permute(3, 0, 1, 2).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.video_model(x)[0]

    def _extract_middle_frame(self, video_path: str) -> str:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        mid = max(total // 2, 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError("Could not decode frame from video")
        out_path = Path(tempfile.mkstemp(suffix=".jpg")[1])
        cv2.imwrite(str(out_path), frame)
        return str(out_path)

    def _extract_audio_from_video(self, video_path: str) -> str:
        out_path = Path(tempfile.mkstemp(suffix=".wav")[1])
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            video_path,
            "-ac",
            "1",
            "-ar",
            "16000",
            str(out_path),
        ]
        cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if cp.returncode != 0:
            raise RuntimeError("Audio extraction from video failed")
        return str(out_path)

    def predict(self, audio_path: str | None, image_path: str | None, video_path: str | None):
        probs = []
        branch = {}

        if audio_path:
            ap = torch.softmax(self._audio_logits(audio_path), dim=0).detach().cpu().numpy()
            branch["audio"] = {self.labels[i]: float(ap[i]) for i in range(len(self.labels))}
            probs.append(self.weights.get("audio", 0.0) * ap)
        if image_path:
            ip = torch.softmax(self._face_logits(image_path), dim=0).detach().cpu().numpy()
            branch["face"] = {self.labels[i]: float(ip[i]) for i in range(len(self.labels))}
            probs.append(self.weights.get("face", 0.0) * ip)
        if video_path:
            vp = torch.softmax(self._video_logits(video_path), dim=0).detach().cpu().numpy()
            branch["video"] = {self.labels[i]: float(vp[i]) for i in range(len(self.labels))}
            probs.append(self.weights.get("video", 0.0) * vp)

        if not probs:
            raise ValueError("Provide at least one modality")

        fused = np.sum(probs, axis=0)
        if fused.sum() > 0:
            fused = fused / fused.sum()
        pred_idx = int(np.argmax(fused))
        return {
            "predicted_emotion": self.labels[pred_idx],
            "confidence": float(fused[pred_idx]),
            "fusion_probs": {self.labels[i]: float(fused[i]) for i in range(len(self.labels))},
            "branch_probs": branch,
        }

    def predict_from_video_hybrid(self, video_path: str):
        audio_path = self._extract_audio_from_video(video_path)
        image_path = self._extract_middle_frame(video_path)
        out = self.predict(audio_path=audio_path, image_path=image_path, video_path=video_path)
        out["mode"] = "video_hybrid_unified"
        out["used_modalities"] = ["audio_from_video", "face_from_video_frame", "video_temporal"]
        return out
