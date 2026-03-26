import argparse
import subprocess
import time
from pathlib import Path


def run(cmd, check=True):
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def status(owner: str, kernel: str) -> str:
    cp = run(["kaggle", "kernels", "status", f"{owner}/{kernel}"])
    return cp.stdout.strip()


def fetch_log(owner: str, kernel: str, out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    run(["kaggle", "kernels", "output", f"{owner}/{kernel}", "-p", str(out_dir)], check=False)
    log = out_dir / f"{kernel}.log"
    return log.read_text(encoding="utf-8", errors="ignore") if log.exists() else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", default="aryanchande23l")
    parser.add_argument("--kernel", default="emotion-multimodal-hybrid-trainer-v1")
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--warmup-seconds", type=int, default=220)
    args = parser.parse_args()

    for attempt in range(1, args.max_attempts + 1):
        print(f"Attempt {attempt}/{args.max_attempts} launching kernel...")
        subprocess.run(
            ["python", "scripts/start_kaggle_training.py", "--owner", args.owner, "--kernel", args.kernel],
            check=True,
        )
        time.sleep(args.warmup_seconds)
        st = status(args.owner, args.kernel)
        print("Status:", st)
        if "RUNNING" in st.upper():
            print("Kernel is still running after warmup; likely passed GPU compatibility gate.")
            return
        if "ERROR" in st.upper():
            log_text = fetch_log(args.owner, args.kernel, Path("kaggle_retry_logs") / f"attempt_{attempt}")
            if "no kernel image is available for execution on the device" in log_text.lower():
                print("Incompatible GPU allocation (likely P100 with unsupported torch build). Retrying...")
                continue
            print("Kernel failed for another reason; inspect logs.")
            return
    print("Max attempts reached without stable compatible run.")


if __name__ == "__main__":
    main()
