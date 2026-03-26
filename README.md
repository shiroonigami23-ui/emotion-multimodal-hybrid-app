# Emotion Multimodal Hybrid App

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Hugging%20Face%20Space](https://img.shields.io/badge/Hugging%20Face-Space-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/spaces/ShiroOnigami23/emotion-multimodal-app)
[![Model](https://img.shields.io/badge/HF-Model-8A2BE2)](https://huggingface.co/ShiroOnigami23/emotion-multimodal-engine)
[![Android APK](https://img.shields.io/badge/Android-APK-3DDC84?logo=android&logoColor=white)](https://github.com/shiroonigami23-ui/emotion-multimodal-hybrid-app/releases)

New standalone project built from your voice-emotion baseline concept, without modifying:

- `emotion-voice-app` (GitHub)
- `ShiroOnigami23/emotion-voice-engine` (HF model)

## Goal

Unified hybrid emotion detection with three modalities from one video clip:

1. Audio signal extracted from video
2. Face expression from key frame extraction
3. Temporal video branch (frame sequence)

Final output is one fused emotion prediction from all three models working together.

Emotion classes (expanded): `angry, calm, disgust, fear, happy, neutral, sad, surprise`

## Live Demo

- Hugging Face Space (connected app): https://huggingface.co/spaces/ShiroOnigami23/emotion-multimodal-app
- Hugging Face model repo: https://huggingface.co/ShiroOnigami23/emotion-multimodal-engine

## Training (Kaggle)

Kernel path: `kaggle_kernel/`

Datasets used:

- `uwrfkaggler/ravdess-emotional-speech-audio`
- `ejlok1/cremad`
- `adrivg/ravdess-emotional-speech-video`
- `astraszab/facial-expression-dataset-image-folders-fer2013`

Note: FER2013 folder labels are numeric (`0..6`) and are mapped internally to emotion classes.

### Run

Start long run without waiting:

```bash
python scripts/start_kaggle_training.py --owner aryanchande23l
```

Check later (quick):

```bash
python scripts/check_kaggle_status.py --owner aryanchande23l
```

Auto-retry until compatible GPU allocation:

```bash
python scripts/relaunch_until_gpu_compatible.py --owner aryanchande23l --max-attempts 5
```

When complete:

```bash
kaggle kernels output aryanchande23l/emotion-multimodal-hybrid-trainer-v1 -p kaggle_pull
```

Expected outputs:

- `audio_model.pt`
- `face_model.pt`
- `video_model.pt`
- `fusion_config.json`
- `metrics.json`
- `run_version.json`

## Upload Model to Hugging Face

```bash
set HF_TOKEN=YOUR_TOKEN
python scripts/upload_to_hf.py --model-repo-id ShiroOnigami23/emotion-multimodal-engine --outputs-dir kaggle_pull
```

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Publish HF Space

```bash
set HF_TOKEN=YOUR_TOKEN
python scripts/publish_space.py --space-id ShiroOnigami23/emotion-multimodal-app
```

## Android APK Release

Android wrapper project is under `android_app/` and opens the deployed HF Space.

- CI workflow: `.github/workflows/android-release.yml`
- Trigger release by pushing a tag (example `v1.0.0`) or using workflow dispatch.
- Signed APK is uploaded to GitHub Releases automatically.

## Safety Notice

This is an affect-recognition research tool. It is not a diagnostic, clinical, legal, or hiring decision system.
