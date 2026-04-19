import streamlit as st
import pickle
import numpy as np

# 1. Konfigurasi Tampilan Halaman Web
st.set_page_config(page_title="Smart QR Guard", page_icon="🛡️")
st.title("🛡️ Smart QR Guard: Deteksi Quishing")
st.write("Sistem ini mengekstrak fitur tautan (URL) dari QR Code dan menganalisisnya menggunakan algoritma Random Forest.")

# 2. Load Model Random Forest
# (Pastikan file model_rf.pkl ada di folder yang sama)
@st.cache_resource
def load_model():
    with open('model_rf.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# 3. Input dari Pengguna (Kasir/Frontend)
st.subheader("Masukkan URL dari hasil scan QR Code:")
url_input = st.text_input("URL:", placeholder="Contoh: https://bca.login-secure.com")

# 4. Fungsi Ekstraksi Fitur Sederhana (Asisten Koki)
def extract_features(url):
    # Fitur 1: Panjang URL
    url_length = len(url)
    
    # Fitur 2: Jumlah Titik
    num_dots = url.count('.')
    
    # Fitur 3: Pengecekan Karakter Spesial (@, -, ?)
    special_chars = sum(url.count(c) for c in ['@', '-', '?'])
    
    # Fitur 4: Pengecekan HTTPS (1 jika iya, 0 jika tidak)
    has_https = 1 if url.startswith('https://') else 0
    
    # Fitur 5: Kata Kunci Mencurigakan
    suspicious_words = ['login', 'verify', 'update', 'secure', 'bank']
    has_suspicious = 1 if any(word in url.lower() for word in suspicious_words) else 0
    
    return np.array([[url_length, num_dots, special_chars, has_https, has_suspicious]])

# 5. Tombol Prediksi (Koki Kepala Memasak)
if st.button("Analisis Keamanan"):
    if url_input:
        # Ekstrak fitur dari URL yang diketik
        fitur_ekstrak = extract_features(url_input)
        
        # Lakukan prediksi dengan model Random Forest
        prediksi = model.predict(fitur_ekstrak)
        
        # 6. Tampilkan Hasil
        st.divider()
        if prediksi[0] == 1: # Asumsi 1 adalah Phishing
            st.error("🚨 **BAHAYA: Tautan Phishing Terdeteksi!**")
            st.write("Sistem mendeteksi anomali pada struktur URL ini. Jangan lanjutkan!")
        else:
            st.success("✅ **AMAN: Tautan Bersih.**")
            st.write("Struktur URL tampak normal dan aman untuk dikunjungi.")
    else:
        st.warning("Mohon masukkan URL terlebih dahulu.")