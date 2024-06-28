import streamlit as st
import requests
from tensorflow.keras.models import load_model
import os

# Function to download file from Google Drive
def download_file_from_google_drive(id, destination):
    URL = "https://drive.google.com/uc?id=" + id
    session = requests.Session()

    response = session.get(URL, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

# Google Drive file ID and destination
file_id = '1l01TbtTSCHZpgk9r3UJlCPE2VmduPolb'
destination = 'sentiment_model.h5'

# Download the model if not already present
if not os.path.exists(destination):
    st.info("Downloading the model...")
    download_file_from_google_drive(file_id, destination)
    st.success("Model downloaded successfully.")

# Check if the model file exists and load it
if os.path.exists(destination):
    try:
        st.info("Loading the model...")
        model = load_model(destination)
        st.success("Model loaded successfully.")
    except OSError as e:
        st.error(f"Error loading the model: {str(e)}")
    except Exception as e:
        st.error(f"Unknown error loading the model: {str(e)}")
else:
    st.error(f"Model file '{destination}' does not exist or could not be downloaded.")

# Streamlit app
st.title("Suicide Risk Classifier")
input_text = st.text_area("Enter your message here")

if st.button('Check'):
    if 'model' in locals():
        # Perform inference with the loaded model
        st.info("Model is loaded and ready.")
    else:
        st.warning("Model not loaded correctly. Please check your setup.")
