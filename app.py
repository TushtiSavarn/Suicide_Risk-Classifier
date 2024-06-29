import streamlit as st
import requests
import os
from tensorflow.keras.models import load_model

# Function to download file from GitHub using raw URL
def download_file_from_github(url, destination):
    response = requests.get(url, stream=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

# GitHub raw URL and local path
github_url = 'https://github.com/TushtiSavarn/Suicide_Risk-Classifier/raw/main/sentiment_model.h5'
model_path = 'sentiment_model.h5'

# Download the model if not already present
if not os.path.exists(model_path):
    st.info("Downloading the model...")
    download_file_from_github(github_url, model_path)
    st.success("Model downloaded successfully.")

# Load the Keras model
try:
    st.info("Loading the model...")
    model = load_model(model_path)
    st.success("Model loaded successfully.")
except OSError as e:
    st.error(f"Error loading the model: {str(e)}")
except Exception as e:
    st.error(f"Unknown error loading the model: {str(e)}")

# Streamlit app
st.title("Suicide Risk Classifier")
input_text = st.text_area("Enter your message here")

if st.button('Check'):
    if 'model' in locals():
        # Perform inference with the loaded model
        st.info("Model is loaded and ready.")
    else:
        st.warning("Model not loaded correctly. Please check your setup.")
