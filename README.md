# 🧠 Anomaly Detection as a Service

A minimalistic anomaly detection microservice powered by an autoencoder trained on MNIST-style grayscale images.  
This project provides both a backend API and a web interface to analyze images and identify abnormal patterns.

## 🚀 Features

- 🧪 FastAPI backend for anomaly score prediction
- 📸 Streamlit frontend with image upload and heatmap visualization
- 🔥 Visual output includes anomaly score, reconstruction, and heatmap
- ⚙️ Pretrained autoencoder (trained on MNIST-style digit images)
- 🧰 Easily extendable to new datasets and architectures

## ⚙️ Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model (optional — a pretrained model will be used if found):
```
python src/train.py
```

### 🧪 Running the Backend (FastAPI)

```
uvicorn app.main:app --reload
```

### 🎛️ Running the Frontend (Streamlit)
```
streamlit run frontend/app.py
```
This launches the Streamlit UI for image upload, anomaly score visualization, and heatmap generation.

## 📝 Notes
- Input images must be grayscale and 28x28 pixels.
- You can retrain the model using a different dataset by modifying src/train.py.

- This is a toy project for educational/demo purposes. It is not intended for production use.

### 📚 API Docs
Visit the interactive API docs at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)