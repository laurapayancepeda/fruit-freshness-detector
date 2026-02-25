# app.py
import streamlit as st
from model_loader import load_model
from PIL import Image
import numpy as np
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(page_title="🍎 Fruit Freshness Detector", layout="centered")
st.title("🍎 How fresh is your fruit?")
st.write(
    "Upload an image, take a photo, or use your webcam to detect fresh vs rotten fruits."
)

# YOLO model
model = load_model()


# Input selection

input_method = st.radio(
    "Select input method:", ("Upload Image", "Capture Photo", "Live Webcam")
)


# Upload Image

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img_array = np.array(img)
        results = model(img_array)
        annotated = results[0].plot(conf=0.5)
        st.image(annotated, caption="Detection Result", use_column_width=True)


# Capture Photo

elif input_method == "Capture Photo":
    st.write("Use your webcam to take a snapshot:")
    captured_image = st.camera_input("Take a photo")
    if captured_image is not None:
        img = Image.open(captured_image)
        img_array = np.array(img)
        results = model(img_array)
        annotated = results[0].plot(conf=0.5)
        st.image(annotated, caption="Detection Result", use_column_width=True)


# Live Webcam

elif input_method == "Live Webcam":
    st.write("Starting live webcam detection (local only)...")

    class WebcamProcessor(VideoProcessorBase):
        def __init__(self):
            self.recent_classes = deque(maxlen=3)
            self.current_class = ""

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = model(img)
            annotated_frame = results[0].plot(conf=0.5)

            # Smooth predicted classes
            pred_classes = [
                model.names[int(cls)] for cls in results[0].boxes.cls.tolist()
            ]
            self.recent_classes.append(pred_classes)
            all_recent = [c for sublist in self.recent_classes for c in sublist]
            if all_recent:
                self.current_class = max(set(all_recent), key=all_recent.count)

            return annotated_frame

    try:
        ctx = webrtc_streamer(
            key="fruit-detector",
            video_processor_factory=WebcamProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if ctx.video_processor:
            st.caption(
                f"Most confident class (smoothed): {ctx.video_processor.current_class}"
            )

    except Exception as e:
        st.warning("Live webcam failed. Falling back to Capture Photo.")

        captured_image = st.camera_input("Take a photo")
        if captured_image is not None:
            img = Image.open(captured_image)
            img_array = np.array(img)
            results = model(img_array)
            annotated = results[0].plot(conf=0.5)
            st.image(annotated, caption="Detection Result", use_column_width=True)
