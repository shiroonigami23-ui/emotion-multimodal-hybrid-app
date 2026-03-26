import subprocess
import time
import argparse

def get_status(kernel: str) -> str:
    cp = subprocess.run(
        ["kaggle", "kernels", "status", kernel],
        capture_output=True,
        text=True,
        check=True,
    )
    out = cp.stdout.strip()
    if "status" in out:
        return out.split("status", 1)[-1].strip().strip('"')
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", default="aryanchande23l")
    parser.add_argument("--kernel", default="emotion-multimodal-hybrid-trainer-v1")
    args = parser.parse_args()
    kernel = f"{args.owner}/{args.kernel}"
    while True:
        status = get_status(kernel)
        print(status)
        if "COMPLETE" in status.upper() or "ERROR" in status.upper():
            break
        time.sleep(20)


if __name__ == "__main__":
    main()
