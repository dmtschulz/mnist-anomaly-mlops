# app/main.py (ОБНОВЛЕННЫЙ ДЛЯ S3)

import os
import io
import base64
import logging
from io import BytesIO

import torch
import boto3
from torchvision import transforms as T

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from PIL import Image
from src.train import load_model, loss_fn # Предполагаем, что load_model и loss_fn доступны
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI()

# --- S3 Настройки (ВМЕСТО HF) ---
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "anomaly-mlops-mnist-data")
MODEL_S3_KEY = "models/best_model.pth"
LOCAL_MODEL_PATH = "/tmp/best_model.pth" # Временный путь для модели в контейнере

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = None
MODEL_LOADED_SUCCESSFULLY = False


def load_model_from_s3():
    """Скачивает best_model.pth из S3 и инициализирует модель."""
    global model
    global MODEL_LOADED_SUCCESSFULLY
    
    if not S3_BUCKET_NAME:
        log.error("Переменная среды S3_BUCKET_NAME не установлена.")
        return

    log.info(f"Запуск загрузки модели из S3: s3://{S3_BUCKET_NAME}/{MODEL_S3_KEY}")
    
    try:
        # Boto3 автоматически использует IAM Role, прикрепленную к EC2
        s3 = boto3.client('s3')
        
        # Скачиваем модель во временный файл
        s3.download_file(S3_BUCKET_NAME, MODEL_S3_KEY, LOCAL_MODEL_PATH)
        log.info("Модель успешно скачана. Инициализация...")

        # --- ЗАГРУЗКА МОДЕЛИ PYTORCH ---
        # NOTE: model.load_state_dict требует, чтобы класс модели был определен
        # Предполагаем, что load_model (из src.train) умеет инициализировать класс.
        
        # Загружаем state dict (используем map_location для совместимости CPU/GPU)
        state_dict = torch.load(LOCAL_MODEL_PATH, map_location=torch.device(DEVICE))
        
        # Инициализируем модель и загружаем веса
        model = load_model() # Предполагаем, что load_model создает пустой экземпляр
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        
        log.info(f"Модель {MODEL_S3_KEY} успешно инициализирована.")
        MODEL_LOADED_SUCCESSFULLY = True
        
    except Exception as e:
        log.error(f"❌ КРИТИЧЕСКАЯ ОШИБКА ЗАГРУЗКИ МОДЕЛИ: {e}")
        MODEL_LOADED_SUCCESSFULLY = False


# Выполняем загрузку модели при старте приложения
load_model_from_s3()

# Transformations
transform = T.Compose([
    T.Grayscale(num_output_channels=1),
    T.Resize((28, 28)),
    T.ToTensor()
])

# --- HEALTH CHECK ---
@app.get("/health")
def health_check():
    # Проверяем, удалось ли загрузить модель
    if not MODEL_LOADED_SUCCESSFULLY:
        raise HTTPException(status_code=503, detail="Model service unavailable: Failed to load best_model.pth from S3.")
    return {"status": "ok", "model_source": "S3", "model_key": MODEL_S3_KEY}

# --- PREDICT ENDPOINT ---
@app.post("/predict", summary="Predict anomaly score from image")
async def predict(file: UploadFile):
    # Добавляем проверку, чтобы избежать сбоя, если модель не загружена
    if not MODEL_LOADED_SUCCESSFULLY:
        raise HTTPException(status_code=503, detail="Model service unavailable. Cannot perform inference.")
        
    try:
        # ... (Остальной код инференса остается без изменений) ...
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
        log.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error during prediction: {str(e)}")