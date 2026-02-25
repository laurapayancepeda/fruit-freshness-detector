import streamlit as st
from model_loader import load_model
from PIL import Image
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import cv2
from collections import deque

st.set_page_config(page_title="🍎 Fruit Freshness Detector", layout="centered")
st.title("🍎 Fruit Freshness Detector")
st.write("Upload an image or use your webcam to detect fresh vs rotten fruits.")

# Load YOLO model
model = load_model()

# Input method
input_method = st.radio("Select input method:", ["Upload Image", "Live Webcam"])

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        img_array = np.array(img)
        results = model(img_array)
        annotated = results[0].plot()
        st.image(annotated, caption="Detection Result", use_column_width=True)

elif input_method == "Live Webcam":
    st.write("Starting live webcam detection...")

    #  VideoProcessor class for streamlit-webrtc
    class YOLOProcessor(VideoProcessorBase):
        def __init__(self):
            self.model = model
            self.recent_classes = deque(maxlen=3)

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = self.model(img_rgb)
            annotated_frame = results[0].plot()

            # Predict classes and smooth
            pred_classes = [
                self.model.names[int(cls)] for cls in results[0].boxes.cls.tolist()
            ]
            self.recent_classes.append(pred_classes)
            all_recent = [c for sublist in self.recent_classes for c in sublist]
            if all_recent:
                most_common = max(set(all_recent), key=all_recent.count)
                cv2.putText(
                    annotated_frame,
                    f"{most_common}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

            return annotated_frame

    webrtc_streamer(
        key="yolo-webcam",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=YOLOProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )
