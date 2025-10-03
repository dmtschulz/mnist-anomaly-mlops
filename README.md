# 🧠 MNIST Anomaly Detection MLOps Project: CI/CD Pipeline

This project demonstrates an **end-to-end MLOps pipeline** for a containerized **PyTorch AutoEncoder** used for anomaly detection in MNIST images. The core achievement is the fully automated and secure Continuous Delivery (CD) pipeline to **AWS EC2**, showcasing mastery of cloud, container, and orchestration technologies.

---

## ✨ Core Technology Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Model** | PyTorch AutoEncoder (NN with linear layers) | Unsupervised model for anomaly scoring. |
| **Backend** | **FastAPI**, Docker | Inference API, image preprocessing, and S3 model loading. |
| **Frontend** | **Streamlit**, Docker | User interface for interaction and visualization (score/heatmap). |
| **Container Registry** | GitHub Container Registry (GHCR) | Stores all built Docker images. |
| **CI/CD Orchestration** | **GitHub Actions** | Drives the entire training and deployment workflow. |
| **Cloud Infrastructure** | AWS EC2, AWS S3 | EC2 (Deployment Host), S3 (Artifact Store, Model and Dataset). |
| **Secure Deployment** | **AWS Systems Manager (SSM)** | Securely pushes commands to EC2 without exposed SSH ports. |
| **Deployment Logic** | **Docker Compose** | Defines and manages the multi-container application on the EC2 host. |

---

## 🏗️ Architecture and Pipeline Flow

The project is structured around two interconnected GitHub Actions pipelines, ensuring reliable and repeatable operations:

### 1. Training & Artifact Pipeline (CI/CT)

This pipeline handles model governance and artifact storage:

1.  **Training:** Executes the PyTorch training script.
2.  **Evaluation & Logging:** Assesses model quality.
3.  **Artifact Storage:** Uploads the final `best_model.pth` artifact to **AWS S3**.

### 2. Deployment Pipeline (Continuous Delivery)

This is the core achievement, demonstrating secure delivery to the production host:

1.  **Containerization:** Builds and pushes the latest `backend` (FastAPI) and `frontend` (Streamlit) images to **GHCR**.
2.  **Targeting:** Dynamically retrieves the **EC2 Instance ID** via AWS tags.
3.  **Secure Command Delivery:** Sends a complex shell command sequence to the EC2 host via **AWS SSM**.
4.  **On-Host Execution:** The EC2 host executes the following steps as the secure `ec2-user`:
    * Creates and sets permissions for the deployment directory (`/opt/mlops-deploy`).
    * Securely logs into GHCR using a **GitHub Token**.
    * **Exports all required AWS/S3 environment variables** to the Docker session.
    * Runs **`docker-compose pull`** and **`docker-compose up -d --remove-orphans`** to automatically roll out the new containers.

---

## 🛠️ Key Technical Achievements

The successful deployment of this pipeline required solving several complex, real-world MLOps challenges:

* **IAM & Authorization:** Configured fine-grained AWS IAM roles for the EC2 host to allow secure, credential-less access to S3 artifacts using Boto3, while maintaining the principle of least privilege.
* **SSM & Permission Handling:** Debugged and finalized the **SSM shell execution logic** to correctly handle Linux permissions (`sudo mkdir`, `chown`) and user switching (`--user ec2-user`), which was crucial for granting the deployed containers necessary filesystem access.
* **Secrets Management:** Mastered the complex transfer of sensitive data: successfully passing **GitHub Secrets** (e.g., `GITHUB_TOKEN`, `AWS_REGION`) from the GitHub Actions runner environment into the executed **Docker Compose** environment variables on the remote EC2 host.
* **Application Health:** Diagnosed and corrected application startup flaws, ensuring the FastAPI backend correctly initializes the model from S3 within its Uvicorn **`@app.on_event("startup")`** sequence before accepting traffic.
* **Networking Security:** Secured the deployment host by relying on **AWS SSM** for command execution and limiting external access solely to the necessary **Streamlit port (8501)** via the **Security Group**.