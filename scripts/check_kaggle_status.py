import subprocess
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", default="aryanchande23l")
    parser.add_argument("--kernel", default="emotion-multimodal-hybrid-trainer-v1")
    args = parser.parse_args()
    kernel = f"{args.owner}/{args.kernel}"
    cp = subprocess.run(["kaggle", "kernels", "status", kernel], capture_output=True, text=True, check=True)
    print(cp.stdout.strip())


if __name__ == "__main__":
    main()
