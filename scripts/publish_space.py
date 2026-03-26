from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space-id", default="ShiroOnigami23/emotion-multimodal-app")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF_TOKEN env var is required.")

    create_repo(args.space_id, token=token, repo_type="space", space_sdk="streamlit", exist_ok=True)
    api = HfApi(token=token)

    readme = """---
title: Emotion Multimodal App
emoji: 🎭
colorFrom: blue
colorTo: pink
sdk: streamlit
sdk_version: 1.37.0
app_file: app.py
pinned: false
---

# Emotion Multimodal App

Streamlit app for audio + face + video emotion detection.
"""

    tmp = Path("artifacts/space_tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "README.md").write_text(readme, encoding="utf-8")
    (tmp / "app.py").write_text(Path("app.py").read_text(encoding="utf-8"), encoding="utf-8")
    (tmp / "requirements.txt").write_text(Path("requirements.txt").read_text(encoding="utf-8"), encoding="utf-8")
    src_dir = tmp / "src" / "mmemotion"
    src_dir.mkdir(parents=True, exist_ok=True)
    for f in ["__init__.py", "infer.py", "models.py", "data_utils.py"]:
        (src_dir / f).write_text(Path("src/mmemotion") .joinpath(f).read_text(encoding="utf-8"), encoding="utf-8")

    api.upload_folder(folder_path=str(tmp), repo_id=args.space_id, repo_type="space")
    print(f"Published Space: https://huggingface.co/spaces/{args.space_id}")


if __name__ == "__main__":
    main()
