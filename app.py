from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st
from huggingface_hub import hf_hub_download, list_repo_files

from src.mmemotion import MultiModalEmotionEngine


st.set_page_config(page_title="Multimodal Emotion Engine", layout="wide", page_icon="🎭")

MODEL_REPO = "ShiroOnigami23/emotion-multimodal-engine"


@st.cache_resource
def load_engine():
    files = list_repo_files(MODEL_REPO)
    required = ["audio_model.pt", "face_model.pt", "video_model.pt", "fusion_config.json"]
    for f in required:
        if f not in files:
            raise RuntimeError(f"Missing {f} in {MODEL_REPO}")
    local = Path(tempfile.mkdtemp(prefix="mmemotion_"))
    for f in required:
        p = hf_hub_download(repo_id=MODEL_REPO, filename=f)
        (local / f).write_bytes(Path(p).read_bytes())
    return MultiModalEmotionEngine(str(local))


st.title("🎭 Multimodal Emotion Engine")
st.caption("Audio + Facial + Video fusion inference with abstain-safe presentation.")
st.warning(
    "Research tool only. Not for medical/clinical diagnosis or mental health treatment decisions."
)

engine = load_engine()

with st.sidebar:
    st.subheader("Input Modalities")
    audio_file = st.file_uploader("Upload Audio (.wav)", type=["wav"])
    image_file = st.file_uploader("Upload Face Image (.jpg/.png)", type=["jpg", "jpeg", "png"])
    video_file = st.file_uploader("Upload Video Clip (.mp4/.mov)", type=["mp4", "mov", "avi"])
    run = st.button("Run Multimodal Prediction", use_container_width=True)


def _persist_upload(upload, suffix: str):
    if not upload:
        return None
    tmp = Path(tempfile.mkstemp(suffix=suffix)[1])
    tmp.write_bytes(upload.getvalue())
    return str(tmp)


if run:
    try:
        ap = _persist_upload(audio_file, ".wav") if audio_file else None
        ip = _persist_upload(image_file, ".png") if image_file else None
        vp = _persist_upload(video_file, ".mp4") if video_file else None
        out = engine.predict(ap, ip, vp)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Predicted Emotion", out["predicted_emotion"])
            st.metric("Confidence", f"{out['confidence'] * 100:.2f}%")
        with c2:
            st.subheader("Fusion Probabilities")
            st.json(out["fusion_probs"])

        st.subheader("Branch-wise Outputs")
        st.json(out["branch_probs"])
        st.subheader("Raw JSON")
        st.code(json.dumps(out, indent=2), language="json")
    except Exception as exc:
        st.error(f"Inference failed: {exc}")
else:
    st.info("Upload one or more modalities and click `Run Multimodal Prediction`.")
