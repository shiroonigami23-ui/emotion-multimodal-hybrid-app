from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo


SPACE_APP = r'''
import gradio as gr
import os
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

from src.mmemotion.infer import MultiModalEmotionEngine

MODEL_REPO = "ShiroOnigami23/emotion-multimodal-engine"

def load_engine():
    try:
        token = os.environ.get("HF_TOKEN", "")
        files = list_repo_files(MODEL_REPO, token=token or None)
        required = ["audio_model.pt", "face_model.pt", "video_model.pt", "fusion_config.json"]
        for f in required:
            if f not in files:
                return None
        local = Path(tempfile.mkdtemp(prefix="mmemotion_"))
        for f in required:
            p = hf_hub_download(repo_id=MODEL_REPO, filename=f, token=token or None)
            (local / f).write_bytes(Path(p).read_bytes())
        return MultiModalEmotionEngine(str(local))
    except Exception:
        return None

ENGINE = load_engine()

def run(video_file):
    if ENGINE is None:
        return {"status": "Model artifacts not uploaded yet. Training still running."}
    if not video_file:
        return {"status": "Upload a video file first."}
    return ENGINE.predict_from_video_hybrid(video_file)

with gr.Blocks(title="Hybrid Emotion Detector") as demo:
    gr.Markdown("# Hybrid Emotion Detector")
    gr.Markdown("Upload one video. App uses audio + face + video models together from the same clip.")
    gr.Markdown("Research tool only. Not a clinical/legal decision system.")

    v = gr.Video(label="Video Clip")
    btn = gr.Button("Run Unified Hybrid Detection", variant="primary")
    out = gr.JSON(label="Prediction")
    btn.click(run, inputs=[v], outputs=[out])

if __name__ == "__main__":
    # Disable experimental SSR in HF Spaces to avoid asyncio FD warnings at shutdown.
    os.environ["GRADIO_SSR_MODE"] = "false"
    demo.launch(ssr_mode=False)
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
title: Hybrid Emotion Detector
emoji: 🎭
colorFrom: blue
colorTo: pink
sdk: gradio
python_version: 3.11
app_file: app.py
pinned: false
---

# Hybrid Emotion Detector

Single-video multimodal emotion detection:
- audio branch from video audio track
- face branch from extracted key frame
- video temporal branch

All three branches fuse into one final emotion result.
"""

    tmp = Path("artifacts/space_tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "README.md").write_text(readme, encoding="utf-8")
    (tmp / "app.py").write_text(SPACE_APP, encoding="utf-8")
    (tmp / "requirements.txt").write_text(
        "gradio\ntorch\ntorchvision\ntorchaudio\nnumpy\nopencv-python-headless\nlibrosa\nhuggingface_hub\nimageio-ffmpeg\n",
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
