from __future__ import annotations

import json
import mimetypes
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


OWNER = "shiroonigami23-ui"
REPO = "emotion-multimodal-hybrid-app"
TAG = "v1.0.1"
APK_PATH = Path("android_app/build/outputs/apk/release/emotion-multimodal-v1.0.1-signed.apk")
TOKEN = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")

if not TOKEN:
    raise SystemExit("Missing GITHUB_TOKEN or GH_TOKEN in environment.")
if not APK_PATH.exists():
    raise SystemExit(f"APK not found: {APK_PATH}")


def req(url: str, method: str = "GET", data: bytes | None = None, headers: dict | None = None):
    h = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {TOKEN}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "apk-release-uploader",
    }
    if headers:
        h.update(headers)
    request = urllib.request.Request(url=url, method=method, data=data, headers=h)
    with urllib.request.urlopen(request) as r:
        body = r.read()
        return r.status, json.loads(body.decode("utf-8")) if body else {}


def get_or_create_release():
    api = f"https://api.github.com/repos/{OWNER}/{REPO}"
    payload = {
        "tag_name": TAG,
        "name": f"Emotion Multimodal {TAG}",
        "draft": False,
        "prerelease": False,
        "generate_release_notes": True,
    }
    try:
        status, data = req(
            f"{api}/releases",
            method="POST",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        print("Created release:", status)
        return data
    except urllib.error.HTTPError as exc:
        if exc.code != 422:
            raise
        status, data = req(f"{api}/releases/tags/{TAG}")
        print("Using existing release:", status)
        return data


def delete_asset_if_exists(release: dict, asset_name: str):
    api = f"https://api.github.com/repos/{OWNER}/{REPO}"
    for asset in release.get("assets", []):
        if asset.get("name") == asset_name:
            asset_id = asset["id"]
            req(f"{api}/releases/assets/{asset_id}", method="DELETE")
            print("Deleted old asset:", asset_name)
            break


def upload_asset(release_id: int, file_path: Path):
    name = file_path.name
    mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
    upload_url = (
        f"https://uploads.github.com/repos/{OWNER}/{REPO}/releases/{release_id}/assets?"
        + urllib.parse.urlencode({"name": name})
    )
    binary = file_path.read_bytes()
    status, data = req(
        upload_url,
        method="POST",
        data=binary,
        headers={"Content-Type": mime},
    )
    print("Uploaded asset:", status)
    print(data.get("browser_download_url", ""))


def main():
    release = get_or_create_release()
    delete_asset_if_exists(release, APK_PATH.name)
    upload_asset(release["id"], APK_PATH)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Failed: {e}")
        sys.exit(1)
