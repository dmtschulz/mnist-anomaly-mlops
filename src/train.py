# src/train.py

import os
import torch
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T
from torchvision.datasets import MNIST
from torchvision.utils import save_image

import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
log.info("Using device: %s", DEVICE)

# Constants
DATA_ROOT = "./data"
MODEL_DIR = "./models"

os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# Autoencoder Model
class Autoencoder(nn.Module):
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


# Train function
def train(loader, h=32, epochs=5, save_path=None):
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
            pbar.set_description(f"Loss: {loss.item():.4f}")

    if save_path:
        save_model(model, save_path)

    return model


# Save model
def save_model(model, path):
    torch.save(model.state_dict(), path)
    log.info("Model saved to %s", path)


# Load model
def load_model(path, h=32):
    model = Autoencoder(h)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    log.info("Model loaded from %s", path)
    return model


# Get DataLoaders
def get_data_loaders(batch_size=128, anomaly_class_threshold=None):
    transform = T.ToTensor()
    train_dataset = MNIST(DATA_ROOT, train=True, download=True, transform=transform)
    test_dataset = MNIST(DATA_ROOT, train=False, download=True, transform=transform)

    if anomaly_class_threshold is not None:
        indices = np.where(train_dataset.targets < anomaly_class_threshold)[0]
        train_dataset = Subset(train_dataset, indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    img, _ = test_dataset[6]
    save_image(img, "data/sample_mnist.png")

    return train_loader, test_loader


# Predict Anomaly Score
def predict_anomaly_score(model, x):
    model.eval()
    with torch.no_grad():
        reconstruction = model(x.to(DEVICE))
        score = torch.mean((x - reconstruction.cpu()) ** 2, dim=(1, 2, 3))
    return score.cpu().numpy()


# Main for getting data and testing the model
if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders()
    model = train(train_loader, h=32, epochs=5, save_path=f"{MODEL_DIR}/autoencoder_mnist.pth")
