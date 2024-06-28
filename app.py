import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import requests
from tensorflow.keras.models import load_model
import re
import os

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Function to download model from Google Drive
def download_file_from_google_drive(id, destination):
    URL = "https://drive.google.com/uc?id=" + id  # Direct download link format
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

# Google Drive file ID
file_id = '1l01TbtTSCHZpgk9r3UJlCPE2VmduPolb'
destination = 'sentiment_model.h5'

# Download the model if not already present
if not os.path.exists(destination):
    st.info("Downloading the model...")
    download_file_from_google_drive(file_id, destination)
    st.info("Download completed.")

# Check if the downloaded file is valid
if os.path.exists(destination):
    st.info(f"Model file '{destination}' exists.")
    if os.path.getsize(destination) > 0:
        st.success("Model file is non-empty.")
    else:
        st.error("Model file is empty.")
else:
    st.error(f"Model file '{destination}' does not exist or could not be downloaded.")

# Load the Keras model with error handling
try:
    st.info("Loading the model...")
    model = load_model(destination)
    st.success("Model loaded successfully.")
except OSError as e:
    st.error(f"Error loading the model: {str(e)}")
except Exception as e:
    st.error(f"Unknown error loading the model: {str(e)}")

# Initialize PorterStemmer
ps = PorterStemmer()

# Text preprocessing function
def clean_text(text):
    # Remove special characters and emojis
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    # Remove non-alphanumeric characters
    text = [word for word in text if word.isalnum()]
    
    # Remove stopwords
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    
    # Stemming
    text = [ps.stem(word) for word in text]
    
    return " ".join(text)

# Streamlit app
st.title("Suicide Risk Classifier")
input_text = st.text_area("Enter your message here")

if st.button('Check'):
    if 'model' in locals():
        transformed_text = clean_text(input_text)
        # Perform vectorization and prediction
        st.info("Model is loaded and ready for predictions.")
    else:
        st.warning("Model not loaded correctly. Please check your setup.")
