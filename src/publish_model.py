# src/publish.py

import os
from huggingface_hub import HfApi

# 1. Считываем все необходимые данные из переменных окружения
#    (они будут установлены в job'е train_and_publish)
HF_TOKEN = os.environ.get('HF_TOKEN_WRITE')
NEW_TAG = os.environ.get('NEW_TAG')
REPO_ID = os.environ.get('REPO_ID')

TARGET_REVISION = 'candidate'
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
            revision=TARGET_REVISION,
            create_pr=False,
            commit_message=f'Model {NEW_TAG} trained via GitHub Actions',
            create_tag=NEW_TAG
        )
        print(f"✅ Публикация в ветку '{TARGET_REVISION}' завершена.")
        print(f"ВАЖНО: Теперь вам нужно вручную создать Pull Request на Hugging Face, чтобы слить '{TARGET_REVISION}' в 'main'.")
        
    except Exception as e:
        print(f"❌ Ошибка при публикации: {e}")
        # Это заставит job упасть, если публикация не удалась
        exit(1)