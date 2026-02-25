# model_loader.py
import os
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import streamlit as st


def load_model():
    MODEL_PATH = "best.pt"

    HF_REPO_ID = os.getenv("HF_REPO_ID")
    HF_FILENAME = os.getenv("HF_FILENAME", "best.pt")
    HF_TOKEN = os.getenv("HF_TOKEN")

    if not os.path.exists(MODEL_PATH):
        if HF_REPO_ID and HF_TOKEN:
            st.info("Downloading YOLO model from Hugging Face...")
            hf_hub_download(
                repo_id=HF_REPO_ID, filename=HF_FILENAME, local_dir=".", token=HF_TOKEN
            )
        else:
            st.warning("Model not found locally and no HF repo/token provided.")
            st.stop()

    model = YOLO(MODEL_PATH)
    return model
