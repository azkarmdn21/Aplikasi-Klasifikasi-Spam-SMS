import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df.rename(columns={"v1": "label", "v2": "text"})[['label', 'text']]
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Model
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))

# App config
st.set_page_config(page_title="Spam Classifier", layout="centered")
st.title("📩 Aplikasi Klasifikasi Spam SMS")

# Sidebar: Statistik
st.sidebar.header("📊 Statistik Dataset")
st.sidebar.write(f"Jumlah Data: {len(df)}")
st.sidebar.write(f"Spam: {df['label'].sum()}")
st.sidebar.write(f"Ham: {len(df) - df['label'].sum()}")
st.sidebar.write(f"🎯 Akurasi Model: **{accuracy:.2%}**")

# Sidebar: Pie Chart
st.sidebar.subheader("📈 Distribusi Kelas")
fig1, ax1 = plt.subplots()
ax1.pie(df['label'].value_counts(), labels=['Ham', 'Spam'], autopct='%1.1f%%', colors=['#36A2EB', '#FF6384'], startangle=90)
ax1.axis('equal')
st.sidebar.pyplot(fig1)

# Input user
st.subheader("🔍 Cek Pesan SMS")
user_input = st.text_area("Masukkan pesan di sini:")

if "history" not in st.session_state:
    st.session_state.history = []

if st.button("Prediksi"):
    prediction = model.predict([user_input])[0]
    prob = model.predict_proba([user_input])[0][prediction]
    result = "SPAM" if prediction == 1 else "BUKAN SPAM"
    
    if prediction == 1:
        st.error(f"❌ Pesan ini adalah **{result}** ({prob:.2f})")
    else:
        st.success(f"✅ Pesan ini adalah **{result}** ({prob:.2f})")
    
    st.session_state.history.append({
        "Pesan": user_input,
        "Hasil": result,
        "Probabilitas": f"{prob:.2f}"
    })

# Riwayat prediksi
if st.session_state.history:
    st.subheader("🕒 Riwayat Prediksi")
    st.table(pd.DataFrame(st.session_state.history))

# Batch Prediction (dengan perbaikan encoding)
st.subheader("📤 Upload CSV untuk Batch Prediksi")
uploaded_file = st.file_uploader("Upload file dengan kolom 'text'", type=["csv"])
if uploaded_file:
    try:
        batch_df = pd.read_csv(uploaded_file, encoding='latin-1')  # fix UnicodeDecodeError
        if "text" in batch_df.columns:
            batch_pred = model.predict(batch_df["text"])
            batch_df["Prediksi"] = ["SPAM" if x == 1 else "BUKAN SPAM" for x in batch_pred]
            st.success("✅ Prediksi selesai!")
            st.write(batch_df[["text", "Prediksi"]])
            st.download_button("⬇️ Download Hasil", batch_df.to_csv(index=False), file_name="hasil_prediksi.csv", mime="text/csv")
        else:
            st.warning("⚠️ File harus memiliki kolom bernama 'text'")
    except UnicodeDecodeError:
        st.error("❌ Gagal membaca file. Pastikan file Anda dalam format CSV dengan encoding UTF-8 atau Latin-1.")

# WordCloud
st.subheader("🌥️ WordCloud dari SMS")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**🔵 Ham (Bukan Spam)**")
    ham_words = " ".join(df[df.label == 0]["text"])
    wordcloud_ham = WordCloud(width=300, height=300, background_color="white").generate(ham_words)
    st.image(wordcloud_ham.to_array())

with col2:
    st.markdown("**🔴 Spam**")
    spam_words = " ".join(df[df.label == 1]["text"])
    wordcloud_spam = WordCloud(width=300, height=300, background_color="white").generate(spam_words)
    st.image(wordcloud_spam.to_array())

# Evaluasi Model
if st.checkbox("📋 Tampilkan Evaluasi Model"):
    st.subheader("📌 Confusion Matrix")
    cm = confusion_matrix(y_test, model.predict(X_test))
    fig2, ax2 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
    st.pyplot(fig2)

    st.subheader("📌 Classification Report")
    report = classification_report(y_test, model.predict(X_test), output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())
