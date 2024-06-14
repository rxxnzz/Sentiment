import streamlit as st
from test import predict

# Judul aplikasi
st.title("Sentiment Analysis for PILKADA 2017 Tweets")

# Input text dari pengguna
user_input = st.text_area("Enter the tweet text")

# List untuk menyimpan hasil prediksi

# Tombol untuk memprediksi
if st.button("Predict Sentiment"):
    if user_input:
        sentiment = predict(user_input)
        # Menambahkan hasil prediksi ke list
        st.write(f"The sentiment of the tweet is: **{sentiment}**")
    else:
        st.write("Please enter a tweet text.")
