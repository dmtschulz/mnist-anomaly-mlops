# src/publish.py

import os
from huggingface_hub import HfApi, CommitOperationAdd

# 1. Read all variables from env
HF_TOKEN = os.environ.get('HF_TOKEN_WRITE')
NEW_TAG = os.environ.get('NEW_TAG')
REPO_ID = os.environ.get('REPO_ID')

TARGET_REVISION = 'candidate'
MODEL_PATH = 'models/autoencoder_mnist.pth' 

if __name__ == "__main__":
    if not HF_TOKEN or not NEW_TAG or not REPO_ID or not MODEL_PATH or not TARGET_REVISION:
        raise EnvironmentError("Env Variables not set: HF_TOKEN, NEW_TAG, REPO_ID, MODEL_PATH, TARGET_REVISION.")
        
    print(f"-> Starting publishing the model with tag: {NEW_TAG}...")

    api = HfApi(token=HF_TOKEN)

    try:
        # 1. Commit model in TARGET_REVISION branch
        api.create_commit(
            repo_id=REPO_ID,
            repo_type="model",
            operations=[
                CommitOperationAdd(
                    path_in_repo=os.path.basename(MODEL_PATH),
                    path_or_fileobj=MODEL_PATH
                )
            ],
            commit_message=f"Model {NEW_TAG} trained via GitHub Actions",
            revision=TARGET_REVISION
        )

        print(f"✅ Publishing to '{TARGET_REVISION}' done.")
        print(f"-> Creating discussion for merging '{TARGET_REVISION}' into 'main'...")

        # 2. Create Discussion
        # Create discussion with Pull Request type
        api.create_discussion(
            repo_id=REPO_ID,
            repo_type="model",
            title=f"CANDIDATE: Model Candidate {NEW_TAG}",
            description=f"Proposing to merge branch `{TARGET_REVISION}` into `main`.\n Model is trained and is ready to be published.",
            pull_request=True
        )

        print(f"✅ Published and created discussion for PR.")
        print(f"IMPORTANT: Check discussion on Hugging Face, to merge '{TARGET_REVISION}' into 'main'.")

    except Exception as e:
        print(f"❌ Error in : {e}")
        exit(1)
