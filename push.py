import os
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import streamlit as st


def load_model():
    """
    Load YOLO model for fruit freshness detection.
    Downloads from Hugging Face if missing.
    Uses HF_TOKEN for private repos.
    """
    MODEL_LOCAL = "best.pt"

    HF_REPO_ID = os.getenv("HF_REPO_ID", "laurapayancepe/fruit-freshness-model")
    HF_FILENAME = os.getenv("HF_FILENAME", "best.pt")
    HF_TOKEN = os.getenv("HF_TOKEN")

    if not os.path.exists(MODEL_LOCAL):
        st.info("Downloading YOLO model from Hugging Face...")
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            local_dir=".",
            token=HF_TOKEN,
            force_download=False,
        )

    model = YOLO(MODEL_LOCAL)
    return model
