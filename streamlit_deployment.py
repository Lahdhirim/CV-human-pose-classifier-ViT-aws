"""
Specific file required only for streamlit prototype deployment
"""

import streamlit as st
import requests
from PIL import Image
import os
import pickle
import matplotlib.pyplot as plt
import time
from io import BytesIO
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from transformers import AutoModelForImageClassification
import torch

from src.config_loaders.inference_config_loader import inference_config_loader
from src.aws_services.s3_service import S3Manager
from src.utils.schema import SavingSchema


@st.cache_resource
def load_model_and_transforms():
    print("Loading configuration and model from S3 Bucket")

    # Load configuration
    config = inference_config_loader(config_path="config/inference_config.json")

    # Download model from S3
    s3_manager = S3Manager(bucket_name=config.bucket_name)
    s3_manager.download_directory(
        s3_prefix=config.s3_model_prefix, local_directory_path=config.local_model_dir
    )

    # Build Model
    model = AutoModelForImageClassification.from_pretrained(config.local_model_dir)
    model.eval()
    transforms_file_path = os.path.join(
        config.local_model_dir, SavingSchema.TRANSFORMS_PKL
    )
    with open(transforms_file_path, "rb") as f:
        _transforms = pickle.load(f)
    return model, _transforms, config.top_n


torch.manual_seed(42)  # fix seed to ensure reproductibility
model, _transforms, top_k = load_model_and_transforms()

# UI
st.set_page_config(page_title="Human Pose Classification", page_icon="🧍")
st.title("🧍Human Pose Classification App")

# --- Choose input method ---
input_method = st.radio("Choose input method:", ["Image URL", "Upload File"])

image_url = None
uploaded_file = None

if input_method == "Image URL":
    image_url = st.text_input("Enter the image URL:")
elif input_method == "Upload File":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Disable predict button if nothing is provided
button_disabled = (input_method == "Image URL" and not image_url) or (
    input_method == "Upload File" and not uploaded_file
)

if st.button("Predict", disabled=button_disabled):

    with st.spinner("🔄 Processing in progress... Please wait."):
        start_time = time.time()

        # Load image preview
        try:
            if image_url:
                response = requests.get(image_url, timeout=5)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Input Image", use_container_width=True)
        except Exception as e:
            st.error(f"Error loading image: {e}")
            st.stop()

        # Apply transforms and get prediction
        pixel_values = _transforms(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(pixel_values)
            probs = torch.softmax(outputs.logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)
            top_probs = top_probs.squeeze(0).tolist()
            top_indices = top_indices.squeeze(0).tolist()
            top_labels = [model.config.id2label[idx] for idx in top_indices]

        if top_labels and top_probs:
            st.success("✅ Prediction completed successfully.")

            # Histogram of top probabilities
            st.subheader("Top Predictions")
            fig, ax = plt.subplots()
            ax.bar(top_labels, top_probs)
            ax.set_xlabel("Probability")
            ax.set_ylabel("Class")
            ax.set_ylim(0, 1)
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)

            # Display calculation time
            prediction_time = int((time.time() - start_time) * 1000)
            st.caption(f"⏱ Prediction time: {prediction_time} ms")

        else:
            st.warning("⚠️ No predictions returned.")
