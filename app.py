import streamlit as st
import requests
import os
from tensorflow.keras.models import load_model
import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

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
    st.info("Downloading model...")
    download_file_from_google_drive(file_id, destination)
    st.success("Model downloaded successfully.")

# Load the saved vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf = pickle.load(file)

# Load the Keras model
try:
    model = load_model(destination)
    st.success("Model loaded successfully.")
except:
    st.warning("Model loading failed. Please try again later.")

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
        vector_input = tfidf.transform([transformed_text])
        result = model.predict(vector_input)[0][0]

        if result > 0.5:
            st.header("The message indicates a risk of suicide")
        else:
            st.header("The message does not indicate a risk of suicide")
    else:
        st.warning("Model not loaded. Please wait for the model to download and load.")
