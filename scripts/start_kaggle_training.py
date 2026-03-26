import subprocess
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", default="aryanchande23l")
    parser.add_argument("--kernel", default="emotion-multimodal-hybrid-trainer-v1")
    args = parser.parse_args()

    meta_path = Path("kaggle_kernel/kernel-metadata.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["id"] = f"{args.owner}/{args.kernel}"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    subprocess.run(["kaggle", "kernels", "push", "-p", "kaggle_kernel"], check=True)
    print(f"Launched: https://www.kaggle.com/code/{args.owner}/{args.kernel}")
    print("Use `python scripts/check_kaggle_status.py` anytime to check progress.")


if __name__ == "__main__":
    main()
