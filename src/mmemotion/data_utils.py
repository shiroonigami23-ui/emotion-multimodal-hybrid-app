from __future__ import annotations

import os
from pathlib import Path


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
