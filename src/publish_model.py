# src/publish.py

import os
from huggingface_hub import HfApi

# 1. Считываем все необходимые данные из переменных окружения
#    (они будут установлены в job'е train_and_publish)
HF_TOKEN = os.environ.get('HF_TOKEN_WRITE')
NEW_TAG = os.environ.get('NEW_TAG')
REPO_ID = os.environ.get('REPO_ID')

MODEL_PATH = 'models/autoencoder_mnist.pth' 

if __name__ == "__main__":
    if not HF_TOKEN or not NEW_TAG or not REPO_ID:
        raise EnvironmentError("Не установлены обязательные переменные окружения: HF_TOKEN, NEW_TAG, REPO_ID.")
        
    print(f"-> Начинаем публикацию модели с тегом: {NEW_TAG}...")

    api = HfApi(token=HF_TOKEN) 
    
    # 2. Загрузка файла
    try:
        api.upload_file(
            path_or_fileobj=MODEL_PATH,
            path_in_repo=os.path.basename(MODEL_PATH),
            repo_id=REPO_ID,
            repo_type='model',
            revision=NEW_TAG, # Используем тег в качестве ветки/ревизии
            commit_message=f'Model {NEW_TAG} trained via GitHub Actions'
        )
        print("✅ Публикация на Hugging Face завершена.")
        print("ВАЖНО: Обновите BEST_MODEL_LOSS вручную в настройках GitHub, чтобы 'закрыть' гейт.")
    except Exception as e:
        print(f"❌ Ошибка при публикации: {e}")
        # Это заставит job упасть, если публикация не удалась
        exit(1)