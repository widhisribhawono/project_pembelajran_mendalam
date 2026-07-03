import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re, emoji
import os
import gdown

# 1. Konfigurasi Halaman & Tema Elegan
st.set_page_config(
    page_title="Sentimen Analisis Pembelajaran Mendalam", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS untuk tampilan premium dan bersih
st.markdown("""
    <style>
    .main {
        background-color: #fcfcfc;
    }
    h1 {
        color: #1e293b;
        font-weight: 700;
        font-size: 26pt !important;
        margin-bottom: 5px;
    }
    .subtitle {
        color: #64748b;
        font-size: 11pt;
        margin-bottom: 30px;
    }
    .stButton>button {
        background-color: #0f172a;
        color: white;
        border-radius: 6px;
        padding: 10px 24px;
        font-weight: 500;
        border: none;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #1e293b;
        color: white;
        border: none;
    }
    /* Card UI untuk Hasil */
    .result-card {
        padding: 20px;
        border-radius: 8px;
        margin-top: 20px;
        border-left: 5px solid #cbd5e1;
        background-color: #ffffff;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    .card-positif { border-left-color: #10b981; background-color: #f0fdf4; }
    .card-negatif { border-left-color: #ef4444; background-color: #fef2f2; }
    .card-netral { border-left-color: #f59e0b; background-color: #fffbeb; }
    
    .label-title {
        font-size: 10pt;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 2px;
    }
    .label-value {
        font-size: 16pt;
        font-weight: 600;
        color: #0f172a;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Header Aplikasi (Minimalis & Profesional)
st.markdown("<h1>Analisis Sentimen Komentar</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Sistem Klasifikasi Teks Komentar YouTube Mengenai Kebijakan Pembelajaran Mendalam Menggunakan Model IndoBERT</p>", unsafe_allow_html=True)

# 3. Fungsi Mengunduh & Memuat Model dari Google Drive
@st.cache_resource
def load_trained_model():
    folder_lokal = "model_sentimen_indobert"
    
    # Jika folder model belum ada di server Streamlit, unduh otomatis dari Google Drive
    if not os.path.exists(folder_lokal):
        id_folder_drive = "1xd67jClhI8Yc81_xeM7LNL7FVq2hx0ZW"
        url_drive = f"https://drive.google.com/drive/folders/{id_folder_drive}?usp=drive_link"
        
        with st.spinner("Sedang mengunduh model IndoBERT kelompok Anda dari Google Drive (ini hanya dilakukan sekali pada awal deploy)..."):
            gdown.download_folder(url_drive, output=folder_lokal, quiet=True, remaining_ok=True)
            
    # Muat model dan tokenizer dari folder lokal yang sudah diunduh
    tokenizer = AutoTokenizer.from_pretrained(folder_lokal)
    model = AutoModelForSequenceClassification.from_pretrained(folder_lokal, num_labels=3)
    return tokenizer, model

try:
    tokenizer, model = load_trained_model()
except Exception as e:
    st.error("Gagal memuat model klasifikasi dari Google Drive. Pastikan folder Drive dapat diakses oleh siapa saja yang memiliki link.")

# 4. Fungsi Preprocessing Teks
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 5. Form Input Komentar
user_input = st.text_area(
    "Masukkan Teks Komentar:", 
    placeholder="Salin atau ketik komentar di sini untuk menguji sentimen...",
    height=120
)

# Layout tombol
col1, col2 = st.columns([1, 4])
with col1:
    submit_button = st.button("Analisis")

# 6. Logika Eksekusi & Prediksi Model IndoBERT
if submit_button:
    if user_input.strip() != "":
        # Proses Data Teks
        teks_bersih = clean_text(user_input)
        inputs = tokenizer(teks_bersih, return_tensors="pt", padding=True, truncation=True, max_length=128)
        
        # Prediksi Menggunakan Model IndoBERT Hasil Latihan
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  
        
        # Menentukan Indeks Prediksi Tertinggi
        prediction_idx = torch.argmax(logits, dim=1).item()
        
        # Pemetaan Indeks Hasil Sesuai Pelatihan (0=Positif, 1=Negatif, 2=Netral)
        label_labels = {0: "Positif", 1: "Negatif", 2: "Netral"}
        hasil_sentimen = label_labels[prediction_idx]
        
        # Penentuan Desain Card Berdasarkan Hasil Klasifikasi
        if hasil_sentimen == "Positif":
            card_style = "card-positif"
            text_color = "#16a34a"
        elif hasil_sentimen == "Negatif":
            card_style = "card-negatif"
            text_color = "#dc2626"
        else:
            card_style = "card-netral"
            text_color = "#d97706"
            
        # Tampilan Hasil UI berbentuk Card Dashboard Eksklusif
        st.markdown(f"""
            <div class="result-card {card_style}">
                <div class="label-title">Hasil Analisis Sentimen</div>
                <div class="label-value" style="color: {text_color};">{hasil_sentimen}</div>
            </div>
        """, unsafe_allow_html=True)
        
    else:
        st.caption("Peringatan: Kolom input teks tidak boleh kosong.")
