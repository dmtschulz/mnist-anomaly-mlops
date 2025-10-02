# app/main.py

import os
from huggingface_hub import hf_hub_download
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from torchvision import transforms as T
from src.train import load_model, loss_fn  # Load the model and loss function from src/train.py
import io
from io import BytesIO

import base64
import matplotlib.pyplot as plt

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI()

# Get Model from HF
HF_REPO_ID = os.environ.get("HF_REPO_ID", "dmtschulz/anomaly-detection-model")
HF_FILENAME = os.environ.get("HF_FILENAME", "autoencoder_mnist.pth")
HF_REVISION = os.environ.get("HF_REVISION", "main")


# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model once
try:
    # Dynamically download the model
    MODEL_PATH = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        revision=HF_REVISION,
        cache_dir="/app/hf_cache"  # Cache in folder inside container
    )
    model = load_model(MODEL_PATH)
    model.eval()
    log.info(f"Model {HF_FILENAME} loaded from Hugging Face with revision {HF_REVISION}.")
except Exception as e:
    raise RuntimeError(f"Failed to load model from Hugging Face: {e}")

model = load_model(MODEL_PATH)
model.eval()

# Transformations
transform = T.Compose([
    T.Grayscale(num_output_channels=1),
    T.Resize((28, 28)),
    T.ToTensor()
])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", summary="Predict anomaly score from image")
async def predict(file: UploadFile):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("L")
        image = transform(image)
        image = image.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            decoded = model(image)
            score = loss_fn(decoded, image).item()
            diff = (decoded - image).squeeze().cpu().numpy() ** 2

        # Create heatmap image
        fig, ax = plt.subplots()
        ax.imshow(diff, cmap="hot")
        ax.axis("off")
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)

        # Encode to base64
        heatmap_b64 = base64.b64encode(buf.read()).decode("utf-8")

        # Create reconstructed image
        decoded_img = decoded.squeeze().cpu().numpy()

        # Visualize the decoded image as png
        fig2, ax2 = plt.subplots()
        ax2.imshow(decoded_img, cmap="gray")
        ax2.axis("off")
        buf2 = BytesIO()
        plt.savefig(buf2, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig2)
        buf2.seek(0)

        # Decode to base64
        decoded_b64 = base64.b64encode(buf2.read()).decode("utf-8")

        # Return everything as JSON
        return JSONResponse(content={
            "anomaly_score": score,
            "heatmap": heatmap_b64,
            "decoded_image": decoded_b64
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")