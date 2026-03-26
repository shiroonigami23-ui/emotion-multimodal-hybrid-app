from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo


SPACE_APP = r'''
import json
import tempfile
from pathlib import Path

import gradio as gr
from huggingface_hub import hf_hub_download, list_repo_files

from src.mmemotion.infer import MultiModalEmotionEngine

MODEL_REPO = "ShiroOnigami23/emotion-multimodal-engine"

def load_engine():
    try:
        files = list_repo_files(MODEL_REPO)
        required = ["audio_model.pt", "face_model.pt", "video_model.pt", "fusion_config.json"]
        for f in required:
            if f not in files:
                return None
        local = Path(tempfile.mkdtemp(prefix="mmemotion_"))
        for f in required:
            p = hf_hub_download(repo_id=MODEL_REPO, filename=f)
            (local / f).write_bytes(Path(p).read_bytes())
        return MultiModalEmotionEngine(str(local))
    except Exception:
        return None

ENGINE = load_engine()

def run(audio_file, image_file, video_file):
    if ENGINE is None:
        return {"status": "Model artifacts not uploaded yet. Training still running."}
    ap = audio_file if audio_file else None
    ip = image_file if image_file else None
    vp = video_file if video_file else None
    out = ENGINE.predict(ap, ip, vp)
    return out

with gr.Blocks(title="Emotion Multimodal App") as demo:
    gr.Markdown("# Emotion Multimodal App")
    gr.Markdown("Audio + Face + Video hybrid emotion detector.")
    gr.Markdown("Research tool only. Not a diagnostic/clinical decision system.")

    a = gr.Audio(type="filepath", label="Audio (.wav)")
    i = gr.Image(type="filepath", label="Face Image")
    v = gr.Video(label="Video Clip")
    btn = gr.Button("Run Multimodal Prediction", variant="primary")
    out = gr.JSON(label="Prediction")
    btn.click(run, inputs=[a, i, v], outputs=[out])

if __name__ == "__main__":
    demo.launch()
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space-id", default="ShiroOnigami23/emotion-multimodal-app")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF_TOKEN env var is required.")

    create_repo(args.space_id, token=token, repo_type="space", space_sdk="gradio", exist_ok=True)
    api = HfApi(token=token)

    readme = """---
title: Emotion Multimodal App
emoji: 🎭
colorFrom: blue
colorTo: pink
sdk: gradio
python_version: 3.11
app_file: app.py
pinned: false
---

# Emotion Multimodal App

Audio + face + video hybrid emotion detection.
"""

    tmp = Path("artifacts/space_tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "README.md").write_text(readme, encoding="utf-8")
    (tmp / "app.py").write_text(SPACE_APP, encoding="utf-8")
    (tmp / "requirements.txt").write_text(
        "gradio\ntorch\ntorchvision\ntorchaudio\nnumpy\nopencv-python-headless\nlibrosa\nhuggingface_hub\n",
        encoding="utf-8",
    )
    src_dir = tmp / "src" / "mmemotion"
    src_dir.mkdir(parents=True, exist_ok=True)
    for f in ["__init__.py", "infer.py", "models.py", "data_utils.py"]:
        (src_dir / f).write_text(Path("src/mmemotion").joinpath(f).read_text(encoding="utf-8"), encoding="utf-8")

    api.upload_folder(folder_path=str(tmp), repo_id=args.space_id, repo_type="space")
    print(f"Published Space: https://huggingface.co/spaces/{args.space_id}")


if __name__ == "__main__":
    main()
