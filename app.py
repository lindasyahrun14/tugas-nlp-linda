import streamlit as st
import joblib
import pandas as pd

# Load otak AI yang tadi kamu download
model = joblib.load('model_ai.pkl')
vec = joblib.load('vectorizer_ai.pkl')

st.title("Aplikasi Analisis Sentimen Layanan AI")
st.write("Tugas NLP Pipeline - Linda")

# Input teks dari user
input_user = st.text_area("Masukkan ulasan/chat di sini:")

if st.button("Cek Sentimen"):
    if input_user:
        # Prediksi menggunakan model
        prediksi = model.predict(vec.transform([input_user]))
        st.success(f"Hasil Analisis: Sentimen {prediksi[0]}")
    else:
        st.warning("Silakan ketik teks dulu ya!")