import streamlit as st
import pickle
import numpy as np
import cv2
from pyzbar.pyzbar import decode
from PIL import Image

st.set_page_config(page_title="Smart QR Guard", page_icon="🛡️")
st.title("🛡️ Smart QR Guard: Deteksi Quishing")
st.write("Unggah gambar QR Code Anda, dan sistem akan menganalisis keamanan tautan di dalamnya.")

# 1. Load Model Random Forest
@st.cache_resource
def load_model():
    with open('model_rf.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# 2. Fungsi Ekstraksi Fitur (Tetap Sama)
def extract_features(url):
    url_length = len(url)
    num_dots = url.count('.')
    special_chars = sum(url.count(c) for c in ['@', '-', '?'])
    has_https = 1 if url.startswith('https://') else 0
    suspicious_words = ['login', 'verify', 'update', 'secure', 'bank']
    has_suspicious = 1 if any(word in url.lower() for word in suspicious_words) else 0
    return np.array([[url_length, num_dots, special_chars, has_https, has_suspicious]])

# 3. Input Pengguna: UPLOAD GAMBAR
uploaded_file = st.file_uploader("Unggah Gambar QR Code (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Tampilkan gambar yang diunggah
    image = Image.open(uploaded_file)
    st.image(image, caption="QR Code yang diunggah", width=250)
    
    # 4. Proses Computer Vision: Membaca isi QR Code
    # Konversi gambar agar bisa dibaca oleh OpenCV & Pyzbar
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    
    st.write("⏳ Sedang memindai gambar...")
    decoded_objects = decode(opencv_image)
    
    if decoded_objects:
        # Jika QR berhasil dibaca
        for obj in decoded_objects:
            url_terdeteksi = obj.data.decode('utf-8')
            
            st.success("Tautan berhasil diekstrak dari gambar!")
            st.code(url_terdeteksi, language="text")
            
            # 5. Prediksi AI
            st.write("🧠 Menjalankan analisis Random Forest...")
            fitur_ekstrak = extract_features(url_terdeteksi)
            prediksi = model.predict(fitur_ekstrak)
            
            st.divider()
            if prediksi[0] == 1:
                st.error("🚨 **BAHAYA: Tautan Phishing Terdeteksi!**")
                st.write("QR Code ini kemungkinan besar mengarah ke situs penipuan.")
            else:
                st.success("✅ **AMAN: Tautan Bersih.**")
                st.write("QR Code ini aman untuk dipindai.")
    else:
        # Jika gambar bukan QR Code atau buram
        st.error("❌ Gagal membaca QR Code. Pastikan gambar jelas dan tidak terpotong.")
