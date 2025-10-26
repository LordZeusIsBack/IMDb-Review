import streamlit as st
from keras.models import load_model
from keras.datasets import imdb
from keras.preprocessing import sequence


word_index = imdb.get_word_index()
reversed_word_index = {v: k for k, v in word_index.items()}

model = load_model('RNN.keras')

def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    model_prediction = model.predict(preprocessed_input)
    output_sentiment = 'Positive' if model_prediction[0][0] > 0.5 else 'Negative'
    return output_sentiment, model_prediction[0][0]

def preprocess_text(text: str):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

st.title('IMDb Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

user_input = st.text_area('Movie Review')

if st.button('Predict'):
    sentiment, prediction = predict_sentiment(user_input)
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction: {prediction}')
else: st.write('Please enter a movie review.')
