# model_loader.py
import os
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import streamlit as st


def load_model():
    """
    Loads YOLO model from Hugging Face.
    Works for public or private repos using HF_TOKEN.
    """
    MODEL_LOCAL = "best.pt"
    HF_REPO_ID = "laurapayancepe/fruit-freshness-model"
    HF_FILENAME = "best.pt"
    HF_TOKEN = os.getenv("HF_TOKEN")  # secure, no hardcoding

    # Download model if missing
    if not os.path.exists(MODEL_LOCAL):
        st.info("Downloading YOLO model from Hugging Face...")
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            local_dir=".",
            token=HF_TOKEN,  # None if repo is public
        )

    # Load YOLO
    model = YOLO(MODEL_LOCAL)
    return model
