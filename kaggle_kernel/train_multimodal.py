import json
import os
import random
from pathlib import Path

import cv2
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.models as tvm
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad"]
CODE_MAP = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
    "01": "neutral",
    "02": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "neutral",
}


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
        z = self.features(x).flatten(1)
        return self.classifier(z)


def build_face_model(num_classes: int) -> nn.Module:
    model = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def build_video_model(num_classes: int) -> nn.Module:
    model = tvm.video.r3d_18(weights=tvm.video.R3D_18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def emotion_to_idx(emotion: str) -> int:
    return EMOTIONS.index(emotion)


def infer_emotion_from_name(path: str) -> str | None:
    name = Path(path).stem.upper()
    parts = name.split("-")
    if len(parts) >= 3 and parts[2] in CODE_MAP:
        return CODE_MAP[parts[2]]
    for code, emo in CODE_MAP.items():
        if f"_{code}_" in f"_{name}_" or name.endswith(f"_{code}"):
            return emo
    return None


def scan_files(root: str, suffixes: tuple[str, ...]) -> list[str]:
    out = []
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(suffixes):
                out.append(str(Path(r) / f))
    return out


def split_train_val(items, val_ratio=0.2):
    random.shuffle(items)
    cut = int(len(items) * (1 - val_ratio))
    return items[:cut], items[cut:]


class AudioDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path, label = self.pairs[idx]
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

    def __getitem__(self, idx):
        path, label = self.pairs[idx]
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

    def __getitem__(self, idx):
        path, label = self.pairs[idx]
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = np.linspace(0, max(total - 1, 0), self.num_frames, dtype=int)
        frames = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
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


def train_branch(model, train_loader, val_loader, device, epochs=1, lr=3e-4):
    model.to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=lr)
    best = {"acc": 0.0, "f1": 0.0}
    best_state = None

    for _ in range(epochs):
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
                x = x.to(device)
                logits = model(x)
                pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
                ys.extend(y.numpy().tolist())
                ps.extend(pred)
        acc = accuracy_score(ys, ps) if ys else 0.0
        f1w = f1_score(ys, ps, average="weighted") if ys else 0.0
        if acc > best["acc"]:
            best = {"acc": float(acc), "f1": float(f1w)}
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best


def gather_audio_pairs(max_items=5000):
    wavs = scan_files(str(ROOT), (".wav",))
    pairs = []
    for p in wavs:
        emo = infer_emotion_from_name(p)
        if emo in EMOTIONS:
            pairs.append((p, emotion_to_idx(emo)))
    return pairs[:max_items]


def gather_video_pairs(max_items=2000):
    vids = scan_files(str(ROOT), (".mp4", ".avi", ".mov"))
    pairs = []
    for p in vids:
        emo = infer_emotion_from_name(p)
        if emo in EMOTIONS:
            pairs.append((p, emotion_to_idx(emo)))
    return pairs[:max_items]


def gather_face_pairs(max_items=8000):
    images = scan_files(str(ROOT), (".jpg", ".jpeg", ".png"))
    pairs = []
    for p in images:
        parent = Path(p).parent.name.lower()
        if parent in EMOTIONS:
            pairs.append((p, emotion_to_idx(parent)))
    return pairs[:max_items]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    audio_pairs = gather_audio_pairs()
    face_pairs = gather_face_pairs()
    video_pairs = gather_video_pairs()
    print("audio_pairs:", len(audio_pairs), "face_pairs:", len(face_pairs), "video_pairs:", len(video_pairs))

    if len(audio_pairs) < 50 or len(face_pairs) < 50 or len(video_pairs) < 20:
        raise RuntimeError("Insufficient data discovered. Check dataset structure/paths.")

    a_tr, a_va = split_train_val(audio_pairs)
    f_tr, f_va = split_train_val(face_pairs)
    v_tr, v_va = split_train_val(video_pairs)

    audio_model = AudioCNN(len(EMOTIONS))
    face_model = build_face_model(len(EMOTIONS))
    video_model = build_video_model(len(EMOTIONS))

    audio_model, audio_metrics = train_branch(
        audio_model,
        DataLoader(AudioDataset(a_tr), batch_size=32, shuffle=True, num_workers=2),
        DataLoader(AudioDataset(a_va), batch_size=32, shuffle=False, num_workers=2),
        device,
        epochs=2,
        lr=3e-4,
    )
    face_model, face_metrics = train_branch(
        face_model,
        DataLoader(FaceDataset(f_tr), batch_size=32, shuffle=True, num_workers=2),
        DataLoader(FaceDataset(f_va), batch_size=32, shuffle=False, num_workers=2),
        device,
        epochs=2,
        lr=3e-4,
    )
    video_model, video_metrics = train_branch(
        video_model,
        DataLoader(VideoDataset(v_tr), batch_size=4, shuffle=True, num_workers=2),
        DataLoader(VideoDataset(v_va), batch_size=4, shuffle=False, num_workers=2),
        device,
        epochs=1,
        lr=1e-4,
    )

    torch.save(audio_model.state_dict(), WORK / "audio_model.pt")
    torch.save(face_model.state_dict(), WORK / "face_model.pt")
    torch.save(video_model.state_dict(), WORK / "video_model.pt")

    fusion_weights = {"audio": 0.4, "face": 0.3, "video": 0.3}
    config = {"labels": EMOTIONS, "fusion_weights": fusion_weights}
    (WORK / "fusion_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    metrics = {
        "device": str(device),
        "audio_metrics": audio_metrics,
        "face_metrics": face_metrics,
        "video_metrics": video_metrics,
        "counts": {
            "audio_pairs": len(audio_pairs),
            "face_pairs": len(face_pairs),
            "video_pairs": len(video_pairs),
        },
    }
    (WORK / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
