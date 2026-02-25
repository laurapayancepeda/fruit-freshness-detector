# app.py
import streamlit as st
from model_loader import load_model
from PIL import Image
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from collections import deque

st.set_page_config(page_title="🍎 Fruit Freshness Detector", layout="centered")
st.title("🍎 How fresh is your fruit?")
st.write("Upload an image or use your webcam to detect fresh vs rotten fruits.")

# Load YOLO model
model = load_model()

# Input selection
input_method = st.radio("Select input method:", ("Upload Image", "Live Webcam"))

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img_array = np.array(img)

        results = model(img_array)
        annotated = results[0].plot(conf=0.5)
        st.image(annotated, caption="Detection Result", use_column_width=True)

elif input_method == "Live Webcam":
    st.write("Starting live webcam detection...")

    recent_classes = deque(maxlen=3)

    class WebcamProcessor(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = model(img)
            annotated_frame = results[0].plot(conf=0.5)

            # Smooth predicted classes
            pred_classes = [
                model.names[int(cls)] for cls in results[0].boxes.cls.tolist()
            ]
            recent_classes.append(pred_classes)
            all_recent = [c for sublist in recent_classes for c in sublist]
            if all_recent:
                most_common = max(set(all_recent), key=all_recent.count)
                st.caption(f"Most confident class (smoothed): {most_common}")

            return annotated_frame

    webrtc_streamer(key="fruit-detector", video_processor_factory=WebcamProcessor)
