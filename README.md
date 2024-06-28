# Suicide Risk Classifier

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-v3.7%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.0.0-orange.svg)
![Keras](https://img.shields.io/badge/keras-v2.6.0-red.svg)

## Overview

Suicide Risk Classifier is a machine learning project aimed at detecting suicide risk in text data. Using Natural Language Processing (NLP) techniques and a neural network model, this project provides a tool to identify texts that indicate a risk of suicide. The classifier is built with TensorFlow and Keras and deployed using Streamlit for an easy-to-use web interface.

## Table of Contents

- [Dataset used](#Dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Streamlit Application](#streamlit-application)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Dataset
 link:-(https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch)

 ## Features

- **Text Cleaning**: Removes special characters, emojis, and stopwords.
- **Text Preprocessing**: Includes tokenization, stemming, and TF-IDF vectorization.
- **Model Training**: Trains a neural network model to classify text data.
- **Web Interface**: Uses Streamlit for an interactive user interface.
- **Real-time Prediction**: Provides instant feedback on the risk level of the input text.

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/suicide-risk-classifier.git
    cd suicide-risk-classifier
    ```

2. **Create a virtual environment**:
    ```sh
    python -m venv env
    source env/bin/activate   # On Windows, use `env\Scripts\activate`
    ```

3. **Install the dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Download the necessary NLTK resources**:
    ```sh
    python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
    ```

2. **Run the Streamlit application**:
    ```sh
    streamlit run app.py
    ```

3. **Interact with the web interface**:
    - Enter a text message in the text area.
    - Click the "Check" button to see the prediction.

## Data Preprocessing

The preprocessing pipeline includes the following steps:

1. **Cleaning**: Remove special characters and emojis.
2. **Tokenization**: Split text into individual words.
3. **Stopwords Removal**: Eliminate common words that do not contribute to the model.
4. **Stemming**: Reduce words to their base form.
5. **TF-IDF Vectorization**: Transform text data into numerical features.

## Model Training

The model is built using TensorFlow and Keras with the following architecture:

- Input Layer
- Dense Layer with 128 neurons and ReLU activation
- Dense Layer with 64 neurons and ReLU activation
- Dense Layer with 32 neurons and ReLU activation
- Output Layer with 1 neuron and sigmoid activation

The model is trained with binary cross-entropy loss and the Adam optimizer.

## Streamlit Application

The Streamlit app provides a simple interface to interact with the classifier. The app allows users to input text and get real-time predictions about the risk of suicide in the text.

## Contributing

We welcome contributions to enhance this project! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature-name`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

- This project uses [Streamlit](https://www.streamlit.io/) for the web interface.
- The model is built with [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/).
- The data preprocessing leverages the [NLTK](https://www.nltk.org/) library.

---

We hope this tool helps in identifying and preventing suicide risks through timely intervention. If you have any questions or suggestions, feel free to open an issue or contact us.

