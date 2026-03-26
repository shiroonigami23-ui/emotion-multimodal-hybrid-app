# Emotion Multimodal Hybrid App

New standalone project built from your voice-emotion baseline concept, without modifying:

- `emotion-voice-app` (GitHub)
- `ShiroOnigami23/emotion-voice-engine` (HF model)

## Goal

Hybrid emotion detection with three modalities:

1. Audio speech signal
2. Face image expression
3. Video clip (frame-level temporal model)

Final output is fused across available modalities.

Emotion classes (expanded): `angry, calm, disgust, fear, happy, neutral, sad, surprise`

## Training (Kaggle)

Kernel path: `kaggle_kernel/`

Datasets used:

- `uwrfkaggler/ravdess-emotional-speech-audio`
- `ejlok1/cremad`
- `adrivg/ravdess-emotional-speech-video`
- `astraszab/facial-expression-dataset-image-folders-fer2013`

### Run

Start long run without waiting:

```bash
python scripts/start_kaggle_training.py
```

Check later (quick):

```bash
python scripts/check_kaggle_status.py
```

When complete:

```bash
kaggle kernels output aryansingh21fd/emotion-multimodal-hybrid-trainer-v1 -p kaggle_pull
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

## Safety Notice

This is an affect-recognition research tool. It is not a diagnostic, clinical, legal, or hiring decision system.
