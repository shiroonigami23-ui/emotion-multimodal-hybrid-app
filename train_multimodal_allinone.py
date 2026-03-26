from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as tvm
import torchvision.transforms as T
from huggingface_hub import HfApi, create_repo
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset


# 8-class target
EMOTIONS = ["angry", "calm", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
CODE_MAP = {
    "ANG": "angry",
    "CAL": "calm",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}
FACE_CLASS_MAP = {
    "0": "angry",
    "1": "disgust",
    "2": "fear",
    "3": "happy",
    "4": "sad",
    "5": "surprise",
    "6": "neutral",
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "sad": "sad",
    "surprise": "surprise",
    "neutral": "neutral",
}


def emotion_to_idx(e: str) -> int:
    return EMOTIONS.index(e)


def infer_emotion_from_name(path: str) -> str | None:
    stem = Path(path).stem.upper()
    parts = stem.split("-")
    if len(parts) >= 3 and parts[2] in CODE_MAP:
        return CODE_MAP[parts[2]]
    for code, emo in CODE_MAP.items():
        if f"_{code}_" in f"_{stem}_" or stem.endswith(f"_{code}"):
            return emo
    return None


def scan_files(root: str, suffixes: Tuple[str, ...]) -> List[str]:
    out = []
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(suffixes):
                out.append(str(Path(r) / f))
    return out


class AudioCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def build_face_model(num_classes: int) -> nn.Module:
    m = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    return m


def build_video_model(num_classes: int) -> nn.Module:
    m = tvm.video.r3d_18(weights=tvm.video.R3D_18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


class AudioDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        path, label = self.pairs[i]
        y, sr = librosa.load(path, sr=16000)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        mel = librosa.power_to_db(mel, ref=np.max)
        mel = cv2.resize(mel, (128, 64))
        x = torch.from_numpy(mel).float().unsqueeze(0)
        return x, torch.tensor(label, dtype=torch.long)


class FaceDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
        self.tf = T.Compose(
            [T.ToPILImage(), T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)]
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        path, label = self.pairs[i]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.tf(img), torch.tensor(label, dtype=torch.long)


class VideoDataset(Dataset):
    def __init__(self, pairs, num_frames=16):
        self.pairs = pairs
        self.num_frames = num_frames

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        path, label = self.pairs[i]
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = np.linspace(0, max(total - 1, 0), self.num_frames, dtype=int)
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
        x = x.permute(3, 0, 1, 2)
        return x, torch.tensor(label, dtype=torch.long)


def split_train_val(items: list, val_ratio=0.2):
    random.shuffle(items)
    c = int(len(items) * (1 - val_ratio))
    return items[:c], items[c:]


def train_branch(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    epochs: int,
    lr: float,
    resume: bool,
):
    model.to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=lr)
    ckpt = out_dir / f"{name}_ckpt.pt"
    best = {"acc": 0.0, "f1": 0.0, "epoch": -1}
    start_epoch = 0

    if resume and ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        best = state["best"]
        start_epoch = int(state["epoch"]) + 1

    for ep in range(start_epoch, epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()

        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for x, y in val_loader:
                p = torch.argmax(model(x.to(device)), dim=1).cpu().numpy().tolist()
                ys.extend(y.numpy().tolist())
                ps.extend(p)
        acc = accuracy_score(ys, ps) if ys else 0.0
        f1w = f1_score(ys, ps, average="weighted") if ys else 0.0
        if acc >= best["acc"]:
            best = {"acc": float(acc), "f1": float(f1w), "epoch": ep}
            torch.save(model.state_dict(), out_dir / f"{name}_model.pt")
        torch.save(
            {"model": model.state_dict(), "opt": opt.state_dict(), "best": best, "epoch": ep},
            ckpt,
        )

    return model, best


def gather_audio_pairs(root: str, max_items: int):
    pairs = []
    for p in scan_files(root, (".wav",)):
        emo = infer_emotion_from_name(p)
        if emo in EMOTIONS:
            pairs.append((p, emotion_to_idx(emo)))
    return pairs[:max_items]


def gather_video_pairs(root: str, max_items: int):
    pairs = []
    for p in scan_files(root, (".mp4", ".avi", ".mov")):
        emo = infer_emotion_from_name(p)
        if emo in EMOTIONS:
            pairs.append((p, emotion_to_idx(emo)))
    return pairs[:max_items]


def gather_face_pairs(root: str, max_items: int):
    pairs = []
    for p in scan_files(root, (".jpg", ".jpeg", ".png")):
        mapped = None
        for token in reversed([x.lower() for x in Path(p).parts]):
            if token in FACE_CLASS_MAP:
                mapped = FACE_CLASS_MAP[token]
                break
        if mapped in EMOTIONS:
            pairs.append((p, emotion_to_idx(mapped)))
            if mapped == "neutral":
                pairs.append((p, emotion_to_idx("calm")))
    return pairs[:max_items]


def choose_device(require_gpu: bool) -> tuple[torch.device, str, tuple[int, int] | None]:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        # Kaggle default torch may fail on P100 (sm60). avoid crash.
        if cap[0] >= 7:
            return torch.device("cuda"), name, cap
        if require_gpu:
            raise RuntimeError(
                f"Incompatible GPU '{name}' capability {cap}. Need sm_70+ (T4/L4/A10 etc) for this torch build."
            )
        return torch.device("cpu"), f"{name} (fallback-cpu)", cap
    if require_gpu:
        raise RuntimeError("GPU required but CUDA unavailable.")
    return torch.device("cpu"), "cpu", None


def upload_to_hf(out_dir: Path, repo_id: str, token: str):
    create_repo(repo_id, token=token, repo_type="model", exist_ok=True)
    api = HfApi(token=token)
    for f in ["audio_model.pt", "face_model.pt", "video_model.pt", "fusion_config.json", "metrics.json"]:
        api.upload_file(
            path_or_fileobj=str(out_dir / f),
            path_in_repo=f,
            repo_id=repo_id,
            repo_type="model",
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/kaggle/input")
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-audio", type=int, default=3500)
    ap.add_argument("--max-face", type=int, default=6000)
    ap.add_argument("--max-video", type=int, default=1200)
    ap.add_argument("--audio-epochs", type=int, default=2)
    ap.add_argument("--face-epochs", type=int, default=2)
    ap.add_argument("--video-epochs", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--require-gpu", action="store_true")
    ap.add_argument("--hf-repo-id", default="")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device, gpu_name, cap = choose_device(args.require_gpu)
    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("device:", device)
    print("gpu_name:", gpu_name)
    print("capability:", cap)

    audio_pairs = gather_audio_pairs(args.root, args.max_audio)
    face_pairs = gather_face_pairs(args.root, args.max_face)
    video_pairs = gather_video_pairs(args.root, args.max_video)
    print(f"audio_pairs: {len(audio_pairs)} face_pairs: {len(face_pairs)} video_pairs: {len(video_pairs)}")

    if len(audio_pairs) < 100 or len(face_pairs) < 100 or len(video_pairs) < 50:
        raise RuntimeError("Insufficient modality samples discovered. Check dataset mounts/paths.")

    a_tr, a_va = split_train_val(audio_pairs)
    f_tr, f_va = split_train_val(face_pairs)
    v_tr, v_va = split_train_val(video_pairs)

    audio_model = AudioCNN(len(EMOTIONS))
    face_model = build_face_model(len(EMOTIONS))
    video_model = build_video_model(len(EMOTIONS))

    audio_model, audio_m = train_branch(
        "audio",
        audio_model,
        DataLoader(AudioDataset(a_tr), batch_size=32, shuffle=True, num_workers=2),
        DataLoader(AudioDataset(a_va), batch_size=32, shuffle=False, num_workers=2),
        device,
        out_dir,
        args.audio_epochs,
        3e-4,
        args.resume,
    )
    face_model, face_m = train_branch(
        "face",
        face_model,
        DataLoader(FaceDataset(f_tr), batch_size=32, shuffle=True, num_workers=2),
        DataLoader(FaceDataset(f_va), batch_size=32, shuffle=False, num_workers=2),
        device,
        out_dir,
        args.face_epochs,
        3e-4,
        args.resume,
    )
    video_model, video_m = train_branch(
        "video",
        video_model,
        DataLoader(VideoDataset(v_tr), batch_size=4, shuffle=True, num_workers=2),
        DataLoader(VideoDataset(v_va), batch_size=4, shuffle=False, num_workers=2),
        device,
        out_dir,
        args.video_epochs,
        1e-4,
        args.resume,
    )

    # copy as latest outputs
    torch.save(audio_model.state_dict(), Path(args.out) / "audio_model.pt")
    torch.save(face_model.state_dict(), Path(args.out) / "face_model.pt")
    torch.save(video_model.state_dict(), Path(args.out) / "video_model.pt")
    (Path(args.out) / "audio_model.pt").write_bytes((out_dir / "audio_model.pt").read_bytes())
    (Path(args.out) / "face_model.pt").write_bytes((out_dir / "face_model.pt").read_bytes())
    (Path(args.out) / "video_model.pt").write_bytes((out_dir / "video_model.pt").read_bytes())

    fusion = {"labels": EMOTIONS, "fusion_weights": {"audio": 0.4, "face": 0.3, "video": 0.3}}
    (out_dir / "fusion_config.json").write_text(json.dumps(fusion, indent=2), encoding="utf-8")
    (Path(args.out) / "fusion_config.json").write_text(json.dumps(fusion, indent=2), encoding="utf-8")

    metrics = {
        "run_id": run_id,
        "device": str(device),
        "gpu_name": gpu_name,
        "capability": cap,
        "counts": {"audio_pairs": len(audio_pairs), "face_pairs": len(face_pairs), "video_pairs": len(video_pairs)},
        "audio_metrics": audio_m,
        "face_metrics": face_m,
        "video_metrics": video_m,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (Path(args.out) / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (Path(args.out) / "run_version.json").write_text(
        json.dumps({"run_id": run_id, "path": str(out_dir)}, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))

    if args.hf_repo_id:
        token = os.environ.get("HF_TOKEN", "")
        if not token:
            raise RuntimeError("HF_TOKEN not found for upload.")
        upload_to_hf(Path(args.out), args.hf_repo_id, token)
        print(f"Uploaded to https://huggingface.co/{args.hf_repo_id}")


if __name__ == "__main__":
    main()
