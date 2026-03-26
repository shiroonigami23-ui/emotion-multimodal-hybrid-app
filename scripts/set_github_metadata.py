from __future__ import annotations

import json
import os
import urllib.request


OWNER = "shiroonigami23-ui"
REPO = "emotion-multimodal-hybrid-app"
TOKEN = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")

if not TOKEN:
    raise SystemExit("Missing GITHUB_TOKEN/GH_TOKEN in environment.")

base = f"https://api.github.com/repos/{OWNER}/{REPO}"
headers = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {TOKEN}",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "metadata-updater",
}

patch_payload = {
    "description": "Unified multimodal emotion detector from one video using audio, face, and temporal fusion",
    "homepage": "https://huggingface.co/spaces/ShiroOnigami23/emotion-multimodal-app",
}
topics_payload = {
    "names": [
        "multimodal-emotion",
        "emotion-detection",
        "video-emotion-recognition",
        "audio-emotion-recognition",
        "facial-expression-recognition",
        "huggingface-space",
        "streamlit-app",
    ]
}

req = urllib.request.Request(
    base, data=json.dumps(patch_payload).encode("utf-8"), headers=headers, method="PATCH"
)
with urllib.request.urlopen(req) as r:
    print("Repo metadata updated:", r.status)

req = urllib.request.Request(
    f"{base}/topics",
    data=json.dumps(topics_payload).encode("utf-8"),
    headers=headers,
    method="PUT",
)
with urllib.request.urlopen(req) as r:
    print("Repo topics updated:", r.status)
