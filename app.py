# app.py
import streamlit as st
from model_loader import load_model
from PIL import Image
import numpy as np
import cv2
import time
from collections import deque

st.set_page_config(page_title="🍎 Fruit Freshness Detector", layout="centered")
st.title("🍎 How fresh is your fruit?")
st.write("Upload an image or use your webcam to detect fresh vs rotten fruits.")

# Load model
model = load_model()

# Choose input
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
    frame_placeholder = st.empty()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Cannot access webcam.")
    else:
        recent_classes = deque(maxlen=3)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to grab frame.")
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (416, 416))

                results = model(frame_resized)
                annotated_frame = results[0].plot(conf=0.5)

                # Smoothed prediction
                pred_classes = [
                    model.names[int(cls)] for cls in results[0].boxes.cls.tolist()
                ]
                recent_classes.append(pred_classes)
                all_recent = [c for sublist in recent_classes for c in sublist]
                if all_recent:
                    most_common = max(set(all_recent), key=all_recent.count)
                    st.caption(f"Most confident class (smoothed): {most_common}")

                frame_placeholder.image(
                    annotated_frame, channels="RGB", use_column_width=True
                )
                time.sleep(0.05)
        finally:
            cap.release()
