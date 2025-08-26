import uvicorn
from fastapi import FastAPI, HTTPException

import time
from colorama import Fore, Style
import pickle
from PIL import Image
import requests
from io import BytesIO
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from transformers import AutoModelForImageClassification
import torch

from src.web_app.data_model import ImageDataInput, ImageDataOutput
from src.aws_services.s3_service import S3Manager
from src.config_loaders.inference_config_loader import inference_config_loader
from src.utils.schema import SavingSchema

# Create FastAPI app
app = FastAPI()

# Global variables for model, transforms, config
model = None
_transforms = None
config = None


@app.on_event("startup")
def load_model():
    global model, _transforms, config

    print(f"{Fore.GREEN}Starting inference pipeline...{Style.RESET_ALL}")

    # Load configuration
    config_path = "config/inference_config.json"
    print(f"{Fore.YELLOW}Loading configuration from {config_path}...{Style.RESET_ALL}")
    config = inference_config_loader(config_path=config_path)

    # Download model from S3
    s3_manager = S3Manager(bucket_name=config.bucket_name)
    print(
        f"{Fore.YELLOW}Downloading trained models from S3 bucket {config.bucket_name}...{Style.RESET_ALL}"
    )
    s3_manager.download_directory(
        s3_prefix=config.s3_model_prefix, local_directory_path=config.local_model_dir
    )

    # Load model
    print(
        f"{Fore.YELLOW}Loading trained model from {config.local_model_dir}...{Style.RESET_ALL}"
    )
    model = AutoModelForImageClassification.from_pretrained(config.local_model_dir)
    model.eval()

    # Load _transforms
    transforms_file_path = os.path.join(
        config.local_model_dir, SavingSchema.TRANSFORMS_PKL
    )
    try:
        with open(transforms_file_path, "rb") as f:
            _transforms = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{Fore.RED}Could not find transforms at {transforms_file_path}{Style.RESET_ALL}"
        )

    print(f"{Fore.GREEN}Model and transforms loaded successfully!{Style.RESET_ALL}")


@app.post("/api/v1/pose_classifier")
def pose_classifier(data: ImageDataInput) -> ImageDataOutput:
    start = time.time()
    image_url = str(data.url)

    # Load image
    try:
        response = requests.get(image_url, timeout=5)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=408, detail=f"Timeout while fetching image: {image_url}"
        )
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=400, detail=f"Error fetching image {image_url}: {e}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing image {image_url}: {e}"
        )

    # Apply transforms
    torch.manual_seed(42)  # fix seed to ensure reproductibility
    pixel_values = _transforms(img).unsqueeze(0)

    # Predict
    with torch.no_grad():
        outputs = model(pixel_values)
        probs = torch.softmax(outputs.logits, dim=-1)

        # Get top_n probabilities and their classes
        top_probs, top_indices = torch.topk(probs, k=config.top_n, dim=-1)
        top_probs = top_probs.squeeze(0).tolist()
        top_indices = top_indices.squeeze(0).tolist()
        top_labels = [model.config.id2label[idx] for idx in top_indices]

    # Return prediction time in ms
    end = time.time()
    prediction_time = int((end - start) * 1000)

    output = ImageDataOutput(
        top_labels=top_labels, top_probs=top_probs, prediction_time=prediction_time
    )

    return output


if __name__ == "__main__":
    uvicorn.run(app="app:app", port=8502, reload=True, host="0.0.0.0")
