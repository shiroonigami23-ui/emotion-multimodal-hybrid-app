# 🗺 Roadmap — Emotion Multimodal Hybrid

> Back to the [README](../README.md).

## Where the project stands

Status: **live**.

## Milestones

### v1 — Hybrid engine (done)

- [x] Audio, face and video branches
- [x] Weighted late fusion
- [x] Kaggle training automation, HF model + Space, Android app

### v1.1 — Accuracy

*Goal:* Learn the fusion instead of hand-setting it.

- [ ] Learned fusion head / attention over branch embeddings
- [ ] Per-dataset and per-class metrics published in the README
- [ ] Face detection + alignment before the face branch

### v1.2 — Real-time

- [ ] Webcam + microphone streaming mode
- [ ] On-device TFLite/ONNX models for the Android app

### v2.0

- [ ] Text branch (speech-to-text + sentiment) for a fourth modality
- [ ] Bias and fairness evaluation across demographics

## 🔭 Long-term Vision

Robust, explainable emotion recognition that still works when one signal is noisy — useful for accessibility,
education and human-computer interaction research.

## How to help

Pick any unchecked item, open an issue to discuss the approach, then send a PR. Items in the earliest unfinished milestone are the highest priority.


---
*Last updated: 2026-11-13*
