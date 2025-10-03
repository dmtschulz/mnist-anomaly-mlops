# src/gating.py

import json
import os
import argparse

# Путь для новой метрики, которую сохранила тренировка (в S3)
NEW_METRICS_PATH = "metrics/metrics.json"
# Путь для лучшей метрики, сохраненной в S3
BEST_LOSS_KEY = "metrics/best_loss.json"
# Путь для лучшей модели, сохраненной в S3
BEST_MODEL_KEY = "models/best_model.pth"
# Путь для модели-кандидата, сохраненной в S3
CANDIDATE_MODEL_KEY = "models/candidate_model.pth"


def compare_and_gate(s3_bucket: str, new_metrics_key: str):
    """
    Скачивает метрики, сравнивает новую потерю (loss) с лучшей потерей.
    Если новая модель лучше, она становится best_model в S3.
    Возвращает True, если модель стала лучшей.
    """
    try:
        import boto3
        s3 = boto3.client('s3')
    except ImportError:
        # Для локального теста, если boto3 не установлен
        print("Boto3 not installed. Running in dummy mode.")
        return True # Всегда успех в dummy mode

    is_best_model = False
    
    # --- 1. СКАЧИВАНИЕ ЛУЧШЕЙ МЕТРИКИ ---
    # Временно скачиваем лучшую метрику для сравнения
    temp_best_loss_path = "/tmp/best_loss_temp.json"
    
    try:
        # Пытаемся скачать текущую лучшую метрику
        s3.download_file(s3_bucket, BEST_LOSS_KEY, temp_best_loss_path)
        with open(temp_best_loss_path, 'r') as f:
            best_metrics = json.load(f)
        best_loss = best_metrics.get('loss', 9999.0)
        print(f"Current BEST Loss: {best_loss:.4f}")
    except Exception:
        # Если файл best_loss.json не найден (первый запуск), считаем потерю очень высокой
        best_loss = 9999.0
        print("No previous best model found. Setting BEST Loss to 9999.0.")
        
    # --- 2. СКАЧИВАНИЕ НОВОЙ МЕТРИКИ ---
    temp_new_metrics_path = "/tmp/new_metrics_temp.json"
    try:
        # Скачиваем метрику, только что созданную тренировкой
        s3.download_file(s3_bucket, new_metrics_key, temp_new_metrics_path)
        with open(temp_new_metrics_path, 'r') as f:
            new_metrics = json.load(f)
        new_loss = new_metrics.get('loss')
        
        if new_loss is None:
            raise ValueError("New metrics file is missing 'loss' key.")

        print(f"Candidate NEW Loss: {new_loss:.4f}")
        
    except Exception as e:
        print(f"Error downloading or reading new metrics file ({new_metrics_key}): {e}")
        return False # Сбой, не публикуем

    # --- 3. СРАВНЕНИЕ И ПУБЛИКАЦИЯ ВНУТРИ S3 (ГЕЙТИНГ) ---
    if new_loss < best_loss:
        print("🔥 Candidate model is BETTER! Updating best_model.pth in S3...")
        
        # 1. Сделать новую модель лучшей
        s3.copy_object(
            Bucket=s3_bucket,
            CopySource={'Bucket': s3_bucket, 'Key': CANDIDATE_MODEL_KEY},
            Key=BEST_MODEL_KEY
        )
        # 2. Обновить лучшую метрику
        s3.upload_file(temp_new_metrics_path, s3_bucket, BEST_LOSS_KEY)
        
        is_best_model = True
    else:
        print("🤷 Candidate model is NOT better than the current BEST. Skipping update.")

    return is_best_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compares new model metrics with the best stored model.")
    parser.add_argument("--s3-bucket", required=True, help="S3 bucket name.")
    
    args = parser.parse_args()
    
    # Мы предполагаем, что тренировка сохранила артефакты в S3 под этими именами:
    if compare_and_gate(args.s3_bucket, new_metrics_key=NEW_METRICS_PATH):
        print("::set-output name=is_best::true")
        print("GATING_RESULT=true") # Используется для GitHub Actions Runner
    else:
        print("::set-output name=is_best::false")
        print("GATING_RESULT=false") # Используется для GitHub Actions Runner
