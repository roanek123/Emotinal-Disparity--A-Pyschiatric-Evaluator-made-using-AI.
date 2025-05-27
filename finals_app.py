import streamlit as st
import tempfile
import os
import re

from final_backend import (
    extract_audio,
    get_face,
    get_aud,
    get_s2t,
    get_sent,
    chain  # this is your LLM streaming chain
)

st.set_page_config(page_title="Video Emotion Analyzer", layout="wide")
st.title("📹 Video Emotion & Sentiment Analyzer")

st.write(
    """
    Upload a video file and click **Analyze**.  
    We'll extract the audio, detect facial and speech emotions, transcribe speech,
    classify overall sentiment, and generate a reflective LLM analysis—all in one go!
    """
)

uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])

if uploaded_video:
    if st.button("Analyze"):
        # 1) Save video to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid:
            tmp_vid.write(uploaded_video.read())
            video_path = tmp_vid.name

        audio_path = None
        try:
            # 2) Audio extraction
            with st.spinner("🔊 Extracting audio from video..."):
                audio_path = extract_audio(video_path)
                if not audio_path:
                    st.error("Failed to extract audio.")
                    raise RuntimeError("Audio extraction failed")
            st.success("Audio extracted successfully.")

            # 3) Facial emotion detection
            with st.spinner("😊 Detecting facial emotions..."):
                face_emotions = get_face(video_path)
            st.success(f"Detected facial emotions: {face_emotions}")

            # 4) Speech emotion detection
            with st.spinner("🗣️ Detecting speech emotions..."):
                speech_emotions = get_aud(audio_path)
            st.success(f"Detected speech emotions: {speech_emotions}")

            # 5) Transcription
            with st.spinner("✍️ Transcribing speech..."):
                transcription = get_s2t(audio_path)
            st.success("Transcription complete.")

            # 6) Sentiment classification
            with st.spinner("🔍 Classifying overall sentiment..."):
                sentiment = get_sent(transcription)
            st.success(f"Sentiment: **{sentiment}**")

            # 7) LLM analysis (streamed)
            st.subheader("💡 Reflective Analysis")
            analysis_placeholder = st.empty()

            # we’ll accumulate raw chunks in buffer,
            # then strip out <think>…</think> before showing
            buffer = ""
            analysis_text = ""

            with st.spinner("🤖 Generating analysis…"):
                for chunk in chain.stream({
                    "face_emotions": ", ".join(face_emotions),
                    "speech_emotion": ", ".join(speech_emotions),
                    "context": transcription,
                    "sentiment": sentiment
                }):
                    buffer += chunk
                    # Remove any <think>…</think> content
                    cleaned = re.sub(r"<think>.*?</think>", "", buffer, flags=re.DOTALL)
                    analysis_placeholder.markdown(cleaned)
                    analysis_text = cleaned

        except Exception as err:
            st.error(f"An error occurred: {err}")

        finally:
            # Clean up temp files
            if os.path.exists(video_path):
                os.unlink(video_path)
            if audio_path and os.path.exists(audio_path):
                os.unlink(audio_path)

else:
    st.info("Please upload a video file to get started.")
