import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from tensorflow.keras.models import load_model
import re

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the TF-IDF vectorizer and Keras model
with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf = pickle.load(file)

model = load_model('sentiment_model.h5')

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
    transformed_text = clean_text(input_text)
    vector_input = tfidf.transform([transformed_text])
    result = model.predict(vector_input)[0][0]
    
    if result > 0.5:
        st.header("The message indicates a risk of suicide")
    else:
        st.header("The message does not indicate a risk of suicide")
