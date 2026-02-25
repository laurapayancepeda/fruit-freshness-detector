# push_model_simple.py
import os
from huggingface_hub import upload_file

# Get configuration from environment variables
HF_TOKEN = os.getenv("HF_TOKEN")  # Hugging Face token
HF_REPO_ID = os.getenv("HF_REPO_ID")  # e.g., "laurapayancepe/fruit-freshness-model"
MODEL_FILE = os.getenv("HF_FILENAME", "best.pt")  # local model file

# -----------------------------
# Checks
# -----------------------------
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not set!")
if not HF_REPO_ID:
    raise ValueError("HF_REPO_ID not set!")
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"Model file '{MODEL_FILE}' not found!")

# -----------------------------
# Upload file directly
# -----------------------------
upload_file(
    path_or_fileobj=MODEL_FILE,
    path_in_repo=MODEL_FILE,  # keep same name in repo
    repo_id=HF_REPO_ID,
    token=HF_TOKEN,
)

print(f"Successfully uploaded '{MODEL_FILE}' to Hugging Face repo '{HF_REPO_ID}'!")
