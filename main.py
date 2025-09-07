# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import re
import streamlit as st

# ---- Constants ----
VOCAB_SIZE = 10000   # same as training
INDEX_FROM = 3       # IMDB shifts word indices by 3

# Load the IMDB dataset word index
word_index = imdb.get_word_index()

# Apply the +3 offset to match imdb.load_data()
word_index = {k: (v + INDEX_FROM) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<OOV>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model
model = load_model("Simple_RNN_imdb.h5")

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    # tokenize text (keep only words)
    words = re.findall(r"[a-z']+", text.lower())

    encoded_review = [1]  # <START> token
    for word in words:
        idx = word_index.get(word, 2)  # use <OOV> if not found
        if idx >= VOCAB_SIZE:          # keep words only in top vocab
            idx = 2
        encoded_review.append(idx)

    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# ---- Streamlit App ----
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('Movie Review')

if st.button('Evaluate'):
    preprocessed_input = preprocess_text(user_input)

    # Make prediction
    prediction = model.predict(preprocessed_input, verbose=0)
    sentiment = 'Positive' if prediction[0][0] > 0.8 else 'Negative'

    # Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]:.4f}')
else:
    st.write('Please enter a movie review.')
