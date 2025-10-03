# src/gating.py
import json
import os
import argparse
import boto3

# Пути в S3
NEW_METRICS_KEY = "metrics/metrics.json"
BEST_LOSS_KEY = "metrics/best_loss.json"
CANDIDATE_MODEL_KEY = "models/candidate_model.pth"
BEST_MODEL_KEY = "models/best_model.pth"

def compare_and_gate(s3_bucket: str):
    """Сравнивает метрики и публикует лучшую модель в S3."""
    s3 = boto3.client('s3')
    is_best_model = False
    
    # 1. СКАЧАТЬ ТЕКУЩУЮ ЛУЧШУЮ МЕТРИКУ
    temp_best_loss_path = "/tmp/best_loss_temp.json"
    try:
        s3.download_file(s3_bucket, BEST_LOSS_KEY, temp_best_loss_path)
        with open(temp_best_loss_path, 'r') as f:
            best_metrics = json.load(f)
        # Мы ищем минимальный loss, поэтому начинаем с большого числа
        best_loss = best_metrics.get('loss', 99999.0) 
        print(f"Current BEST Loss: {best_loss:.4f}")
    except Exception:
        best_loss = 99999.0
        print("No previous best model found. Starting fresh.")
        
    # 2. СКАЧАТЬ НОВУЮ МЕТРИКУ
    temp_new_metrics_path = "/tmp/new_metrics_temp.json"
    try:
        s3.download_file(s3_bucket, NEW_METRICS_KEY, temp_new_metrics_path)
        with open(temp_new_metrics_path, 'r') as f:
            new_metrics = json.load(f)
        new_loss = new_metrics.get('loss')
        if new_loss is None: raise ValueError("New metrics file is missing 'loss' key.")
        print(f"Candidate NEW Loss: {new_loss:.4f}")
        
    except Exception as e:
        print(f"❌ Error reading new metrics: {e}")
        return False

    # 3. СРАВНЕНИЕ И ПУБЛИКАЦИЯ В S3
    if new_loss < best_loss:
        print("🔥 Candidate model is BETTER! Promoting to best_model.pth in S3...")
        
        # 1. Промоутировать модель
        s3.copy_object(
            Bucket=s3_bucket,
            CopySource={'Bucket': s3_bucket, 'Key': CANDIDATE_MODEL_KEY},
            Key=BEST_MODEL_KEY
        )
        # 2. Обновить лучшую метрику
        s3.upload_file(temp_new_metrics_path, s3_bucket, BEST_LOSS_KEY)
        is_best_model = True
    else:
        print("🤷 Candidate model is NOT better than the current BEST. Skipping promotion.")

    return is_best_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compares new model metrics with the best stored model.")
    parser.add_argument("--s3-bucket", required=True, help="S3 bucket name.")
    args = parser.parse_args()
    
    result = compare_and_gate(args.s3_bucket)
    
    # Сохраняем результат для следующего шага в GitHub Actions
    print(f"GATING_RESULT={result}")
    # Устанавливаем выходную переменную для доступа в if:
    print(f"::set-output name=is_best::{'true' if result else 'false'}")