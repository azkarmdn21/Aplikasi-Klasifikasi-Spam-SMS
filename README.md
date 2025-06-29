# 📩 Spam SMS Classifier – Streamlit App

Aplikasi machine learning berbasis Streamlit untuk klasifikasi pesan SMS menjadi **Spam** atau **Bukan Spam (Ham)**.

## 🚀 Fitur
- Input manual untuk deteksi spam
- Upload file CSV untuk prediksi massal
- Statistik dataset (jumlah spam/ham, akurasi model)
- WordCloud untuk visualisasi kata populer
- Confusion Matrix dan Classification Report
- Dibangun dengan: `scikit-learn`, `Streamlit`, `pandas`

## 📦 Dataset
Menggunakan dataset SMS spam yang sudah dilabeli (`spam.csv`), dengan 2 kolom:
- `label`: spam / ham
- `text`: isi pesan
