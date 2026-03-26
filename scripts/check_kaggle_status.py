import subprocess


KERNEL = "aryanchande23l/emotion-multimodal-hybrid-trainer-v1"


def main():
    cp = subprocess.run(["kaggle", "kernels", "status", KERNEL], capture_output=True, text=True, check=True)
    print(cp.stdout.strip())


if __name__ == "__main__":
    main()
