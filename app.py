from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st
from huggingface_hub import hf_hub_download, list_repo_files

from src.mmemotion import MultiModalEmotionEngine


st.set_page_config(page_title="Hybrid Emotion Detector", layout="wide", page_icon="assets/icon-app.svg")
MODEL_REPO = "ShiroOnigami23/emotion-multimodal-engine"
SPACE_URL = "https://huggingface.co/spaces/ShiroOnigami23/emotion-multimodal-app"

st.markdown(
    """
<style>
.stApp {
  background:
    radial-gradient(circle at 10% 10%, rgba(72,163,255,0.10), transparent 45%),
    radial-gradient(circle at 90% 20%, rgba(90,212,145,0.10), transparent 40%),
    linear-gradient(180deg, #0c1017 0%, #121722 100%);
}
.hero {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 20px;
  padding: 20px;
  background: linear-gradient(135deg, rgba(28,36,52,0.92), rgba(19,24,35,0.95));
}
.mini-card {
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 16px;
  padding: 10px 12px;
  background: rgba(255,255,255,0.04);
  text-align: center;
}
.soft-note {
  border-left: 4px solid #5ad491;
  background: rgba(90,212,145,0.10);
  border-radius: 10px;
  padding: 12px;
}
</style>
""",
    unsafe_allow_html=True,
)


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


st.markdown(
    """
<div class="hero">
  <h1 style="margin-bottom:0;">Emotion Multimodal Hybrid</h1>
  <p style="margin-top:6px;">Upload one video clip. The app extracts audio + face + motion cues and produces one fused emotion.</p>
</div>
""",
    unsafe_allow_html=True,
)

head_l, head_r = st.columns([3, 2])
with head_l:
    st.caption("Connected model repo: `ShiroOnigami23/emotion-multimodal-engine`")
with head_r:
    st.link_button("Open Live HF Space", SPACE_URL, use_container_width=True)

icons = st.columns(3)
icons[0].markdown(Path("assets/icon-audio.svg").read_text(encoding="utf-8"), unsafe_allow_html=True)
icons[1].markdown(Path("assets/icon-face.svg").read_text(encoding="utf-8"), unsafe_allow_html=True)
icons[2].markdown(Path("assets/icon-video.svg").read_text(encoding="utf-8"), unsafe_allow_html=True)

st.markdown('<div class="soft-note">Research tool only. Not for legal, clinical, or mental-health diagnosis.</div>', unsafe_allow_html=True)
st.divider()

left, right = st.columns([2, 1])
with left:
    video_file = st.file_uploader("Upload Video Clip (.mp4/.mov/.avi)", type=["mp4", "mov", "avi"])
    run = st.button("Run Unified Hybrid Detection", use_container_width=True, type="primary")
    if video_file is not None:
        st.video(video_file)
with right:
    st.markdown("### Pipeline")
    st.markdown("- Extract audio track")
    st.markdown("- Detect face expression")
    st.markdown("- Analyze temporal video")
    st.markdown("- Fuse to one final emotion")

if run:
    if engine is None:
        st.error("Model artifacts are not available yet in Hugging Face model repo.")
        st.stop()
    if not video_file:
        st.error("Upload a video file first.")
        st.stop()
    try:
        with st.status("Running multimodal inference...", expanded=True) as status:
            st.write("Persisting upload")
            vp = _persist_upload(video_file, ".mp4")
            st.write("Running audio + face + video branches")
            out = engine.predict_from_video_hybrid(vp)
            st.write("Fusing branch probabilities")
            status.update(label="Inference complete", state="complete")

        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Emotion", out["predicted_emotion"])
        m2.metric("Confidence", f"{out['confidence'] * 100:.2f}%")
        m3.metric("Branches Used", "3 / 3")

        tab1, tab2, tab3 = st.tabs(["Fusion", "Branch Details", "Raw JSON"])
        with tab1:
            st.subheader("Fusion Probabilities")
            st.bar_chart(out["fusion_probs"])
        with tab2:
            st.subheader("Audio + Face + Video Branch Probabilities")
            st.json(out["branch_probs"])
        with tab3:
            st.code(json.dumps(out, indent=2), language="json")
    except Exception as exc:
        st.error(f"Inference failed: {exc}")
else:
    st.info("Upload one video and click `Run Unified Hybrid Detection`.")
