# src/train.py (Изменено для работы с AWS S3)

import os
import torch
import numpy as np
import argparse
import boto3
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import logging
import json

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --- GLOBAL CONSTANTS AND SETTINGS ---
# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
log.info("Using device: %s", DEVICE)

# Temporary directory for storing data and model inside the container
TEMP_DIR = "/tmp/training"
os.makedirs(TEMP_DIR, exist_ok=True)


# --- AUXILIARY S3 FUNCTIONS ---
def get_s3_client():
    """Возвращает S3-клиент. Он автоматически использует IAM Role, присвоенную EC2."""
    return boto3.client('s3')


def download_mnist_from_s3(s3_bucket: str, s3_prefix: str = 'raw/', local_path: str = f"{TEMP_DIR}/data"):
    """Скачивает все файлы MNIST из S3 в локальную временную папку."""
    s3 = get_s3_client()
    os.makedirs(local_path, exist_ok=True)

    log.info(f"Downloading MNIST data from s3://{s3_bucket}/{s3_prefix} to {local_path}...")

    # Список имен файлов MNIST (мы знаем, какие файлы нам нужны)
    mnist_files = [
        't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte',
        'train-images-idx3-ubyte', 'train-labels-idx1-ubyte'
    ]

    for filename in tqdm(mnist_files, desc="Downloading files"):
        try:
            s3_key = os.path.join(s3_prefix, filename)
            local_filepath = os.path.join(local_path, filename)

            s3.download_file(s3_bucket, s3_key, local_filepath)
            log.info(f"Successfully downloaded {filename}")
        except Exception as e:
            # Если файл не найден, это может быть проблемой.
            log.error(f"Error downloading {filename}: {e}")
            raise


def upload_file_to_s3(s3_bucket: str, local_file: str, s3_key: str):
    """Загружает файл (модель или метрики) в S3."""
    s3 = get_s3_client()
    log.info(f"Uploading {local_file} to s3://{s3_bucket}/{s3_key}")
    s3.upload_file(local_file, s3_bucket, s3_key)
    log.info("Upload complete.")


# --- МОДЕЛЬ И ФУНКЦИИ (без изменений) ---

# Autoencoder Model
class Autoencoder(nn.Module):
    # ... (Класс Autoencoder остается без изменений)
    def __init__(self, h: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 12000),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(12000, h),
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(h, 12000),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(12000, 784),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        x = x.view(x.size(0), -1)  # (N, 784)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.view(-1, 1, 28, 28)
        return decoded


# Loss function
loss_fn = nn.MSELoss()


# Train function (модифицирована для сохранения метрик в S3)
def train(loader,
            s3_bucket: str,
            h: int = 32,
            epochs: int = 1,
            model_key: str = "models/model.pth",
            metrics_key: str = "metrics/metrics.json"):

    model = Autoencoder(h).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        log.info("Epoch %d", epoch + 1)
        pbar = tqdm(loader)
        for batch, _ in pbar:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            prediction = model(batch)
            loss = loss_fn(prediction, batch)
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Epoch: {epoch} Loss: {loss.item():.4f}")

    # Оценка модели после тренировки
    _, test_loader = get_data_loaders(data_root=f"{TEMP_DIR}/data")  # Используем загруженные данные
    final_loss = evaluate_model(model, test_loader, loss_fn)
    log.info(f"Final Test Loss: {final_loss:.6f}")

    # 1. Сохранение модели локально
    local_model_path = f"{TEMP_DIR}/model.pth"
    torch.save(model.state_dict(), local_model_path)

    # 2. Сохранение метрик локально
    metrics_data = {"test_loss": final_loss, "epochs": epochs, "h_dim": h}
    local_metrics_path = f"{TEMP_DIR}/metrics.json"
    with open(local_metrics_path, "w") as f:
        json.dump(metrics_data, f)

    # 3. ЗАГРУЗКА на S3
    upload_file_to_s3(s3_bucket, local_model_path, model_key)
    upload_file_to_s3(s3_bucket, local_metrics_path, metrics_key)

    return model


# Save model (больше не используется напрямую в train, но сохраним)
def save_model(model, path):
    torch.save(model.state_dict(), path)
    log.info("Model saved to %s", path)


# Load model (полезно для локального тестирования)
def load_model(path, h=32):
    model = Autoencoder(h)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    log.info("Model loaded from %s", path)
    return model


# Get DataLoaders (модифицирована для работы с предварительно скачанными данными)
def get_data_loaders(batch_size=128, anomaly_class_threshold=None, data_root: str = f"{TEMP_DIR}/data"):
    transform = T.ToTensor()
    # download=False, т.к. данные уже должны быть в data_root (скачаны из S3)
    train_dataset = MNIST(data_root, train=True, download=False, transform=transform)
    test_dataset = MNIST(data_root, train=False, download=False, transform=transform)

    if anomaly_class_threshold is not None:
        indices = np.where(train_dataset.targets < anomaly_class_threshold)[0]
        train_dataset = Subset(train_dataset, indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    try:
        # Сохранение примера в TEMP_DIR, а не в корне
        img, _ = test_dataset[6]
        save_image(img, f"{TEMP_DIR}/sample_mnist.png")
    except Exception as e:
        log.warning(f"Could not save sample image: {e}")

    return train_loader, test_loader


# Predict Anomaly Score (без изменений)
def predict_anomaly_score(model, x):
    model.eval()
    with torch.no_grad():
        reconstruction = model(x.to(DEVICE))
        score = torch.mean((x - reconstruction.cpu()) ** 2, dim=(1, 2, 3))
    return score.cpu().numpy()


# Evaluate model (без изменений)
def evaluate_model(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> float:
    """Return mean loss over all samples in the loader."""
    model.eval()
    total_loss = 0.0
    n_samples = 0
    with torch.no_grad():
        for batch, _ in loader:
            batch = batch.to(DEVICE)
            recon = model(batch)
            loss = loss_fn(recon, batch)          # mean over the batch
            bs = batch.size(0)
            total_loss += loss.item() * bs        # weight by batch size
            n_samples += bs
    return total_loss / max(n_samples, 1)


# --- MAIN ENTRY POINT ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MNIST Autoencoder Training.")
    parser.add_argument("--s3-bucket", required=True, help="Name of the S3 bucket for data/artifact storage.")
    parser.add_argument("--s3-key-model", default="models/autoencoder_mnist.pth", help="S3 key for saving the model artifact.")
    parser.add_argument("--s3-key-metrics", default="metrics/metrics.json", help="S3 key for saving the metrics artifact.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--h-dim", type=int, default=32, help="Hidden dimension of the Autoencoder.")

    args = parser.parse_args()

    # 1. СКАЧИВАНИЕ ДАННЫХ ИЗ S3
    download_mnist_from_s3(args.s3_bucket)

    # 2. ЗАПУСК ТРЕНИРОВКИ
    train_loader, _ = get_data_loaders(data_root=f"{TEMP_DIR}/data")

    # 3. ТРЕНИРОВКА и СОХРАНЕНИЕ В S3
    train(
        train_loader,
        s3_bucket=args.s3_bucket,
        epochs=args.epochs,
        h=args.h_dim,
        model_key=args.s3_key_model,
        metrics_key=args.s3_key_metrics
    )
