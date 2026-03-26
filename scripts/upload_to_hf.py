from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-repo-id", default="ShiroOnigami23/emotion-multimodal-engine")
    parser.add_argument("--outputs-dir", default="kaggle_pull")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF_TOKEN env var is required.")

    out = Path(args.outputs_dir)
    required = ["audio_model.pt", "face_model.pt", "video_model.pt", "fusion_config.json", "metrics.json"]
    for f in required:
        if not (out / f).exists():
            raise FileNotFoundError(f"Missing {f} in {out}")

    create_repo(args.model_repo_id, token=token, repo_type="model", exist_ok=True)
    api = HfApi(token=token)

    for f in required:
        api.upload_file(
            path_or_fileobj=str(out / f),
            path_in_repo=f,
            repo_id=args.model_repo_id,
            repo_type="model",
        )

    readme = out / "README_MODEL.md"
    readme.write_text(
        "# Emotion Multimodal Engine\n\n"
        "Audio + face + video hybrid model trained via Kaggle kernel.\n"
        "Use `app.py` from repo or HF Space to run multimodal inference.\n",
        encoding="utf-8",
    )
    api.upload_file(
        path_or_fileobj=str(readme),
        path_in_repo="README.md",
        repo_id=args.model_repo_id,
        repo_type="model",
    )
    print(f"Uploaded model to https://huggingface.co/{args.model_repo_id}")


if __name__ == "__main__":
    main()
