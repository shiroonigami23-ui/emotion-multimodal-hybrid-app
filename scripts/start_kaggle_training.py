import subprocess


def main():
    subprocess.run(["kaggle", "kernels", "push", "-p", "kaggle_kernel"], check=True)
    print("Launched: https://www.kaggle.com/code/aryanchande23l/emotion-multimodal-hybrid-trainer-v1")
    print("Use `python scripts/check_kaggle_status.py` anytime to check progress.")


if __name__ == "__main__":
    main()
