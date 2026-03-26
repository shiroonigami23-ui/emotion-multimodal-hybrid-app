from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st
from huggingface_hub import hf_hub_download, list_repo_files

from src.mmemotion import MultiModalEmotionEngine


st.set_page_config(page_title="Hybrid Emotion Detector", layout="wide", page_icon="🎭")
MODEL_REPO = "ShiroOnigami23/emotion-multimodal-engine"


@st.cache_resource
def load_engine():
    try:
        files = list_repo_files(MODEL_REPO)
        required = ["audio_model.pt", "face_model.pt", "video_model.pt", "fusion_config.json"]
        for f in required:
            if f not in files:
                return None
        local = Path(tempfile.mkdtemp(prefix="mmemotion_"))
        for f in required:
            p = hf_hub_download(repo_id=MODEL_REPO, filename=f)
            (local / f).write_bytes(Path(p).read_bytes())
        return MultiModalEmotionEngine(str(local))
    except Exception:
        return None


st.title("🎭 Hybrid Emotion Detector")
st.caption("One video input -> audio+face+video branches -> one fused emotion output.")
st.warning("Research tool only. Not for legal, clinical, or mental-health diagnosis.")

engine = load_engine()
video_file = st.file_uploader("Upload Video Clip (.mp4/.mov/.avi)", type=["mp4", "mov", "avi"])
run = st.button("Run Unified Hybrid Detection", use_container_width=True)


def _persist_upload(upload, suffix: str):
    tmp = Path(tempfile.mkstemp(suffix=suffix)[1])
    tmp.write_bytes(upload.getvalue())
    return str(tmp)


if run:
    if engine is None:
        st.error("Model artifacts are not available yet in Hugging Face model repo.")
        st.stop()
    if not video_file:
        st.error("Upload a video file first.")
        st.stop()
    try:
        vp = _persist_upload(video_file, ".mp4")
        out = engine.predict_from_video_hybrid(vp)
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Predicted Emotion", out["predicted_emotion"])
            st.metric("Confidence", f"{out['confidence'] * 100:.2f}%")
        with c2:
            st.subheader("Fusion Probabilities")
            st.json(out["fusion_probs"])

        st.subheader("All 3 Branches (from same video)")
        st.json(out["branch_probs"])
        st.subheader("Raw JSON")
        st.code(json.dumps(out, indent=2), language="json")
    except Exception as exc:
        st.error(f"Inference failed: {exc}")
else:
    st.info("Upload one video and click `Run Unified Hybrid Detection`.")
