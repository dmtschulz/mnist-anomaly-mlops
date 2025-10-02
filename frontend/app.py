# frontend/app.py

import streamlit as st
import requests
import base64
import os

# Read API address from variable or default (if without Docker).
API_HOST = os.environ.get("API_HOST", "127.0.0.1")
API_PORT = os.environ.get("API_PORT", "8000") # Default port for local launch

# Generate full URL
API_URL = f"http://{API_HOST}:{API_PORT}/predict"
# Now API_URL will be 'http://127.0.0.1:8000/predict' (localy) OR 'http://backend:80/predict' (in Docker)

st.set_page_config(page_title="Anomaly Detection", page_icon="🔍")
st.title("🧠 Anomaly Detection as a Service")
st.markdown("Load a MNIST (28x28, b/w) grayscale image - get the anomaly score!")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded image", width=150)

    if st.button("Check"):
        with st.spinner("Analyzing..."):
            try:
                response = requests.post(API_URL, files={"file": uploaded_file})
                response.raise_for_status()
                result = response.json()

                score = result["anomaly_score"]
                heatmap_b64 = result["heatmap"]
                decoded_b64 = result["decoded_image"]

                st.success(f"🧪 Anomaly Score: **{score:.5f}**")
                if score > 0.02:
                    st.error("🚨 Anomaly detected!")
                else:
                    st.info("✅ Everything looks normal.")

                # Decode images
                reconstructed_img = base64.b64decode(decoded_b64)
                heatmap_img = base64.b64decode(heatmap_b64)

                # Display Original and Reconstructed side-by-side
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(uploaded_file, caption="Original", width=150)
                with col2:
                    st.image(reconstructed_img, caption="Reconstruction", width=150)
                with col3:
                    st.image(heatmap_img, caption="Anomaly Heatmap", width=150)

            except requests.exceptions.JSONDecodeError:
                st.error("Invalid response from the API. Please check backend logs.")
