import streamlit as st
import pickle
import numpy as np
from pyzbar.pyzbar import decode
from PIL import Image

# 1. Konfigurasi Halaman 
st.set_page_config(page_title="VeriQR AI", page_icon="🛡️", layout="centered")

# Penggunaan CSS untuk Background
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        color: #ffffff;
    }
    
    .stFileUploader > div > div {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        border: 2px dashed #4facfe;
        padding: 20px;
    }
    
    h1, h2, h3, p {
        color: #f8f9fa !important;
    }
    
    .stAlert {
        border-radius: 10px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ VeriQR AI")
st.write("### Sistem Analisis & Deteksi QR Phishing")
st.write("Unggah gambar QR Code Anda, dan sistem AI kami akan menganalisis keamanan tautan di dalamnya secara real-time.")
st.markdown("---")

# 2. Load Model Random Forest
@st.cache_resource
def load_model():
    with open('model_rf.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# 3. Fungsi Ekstraksi Fitur 
def extract_features(url):
    url_length = len(url)
    num_dots = url.count('.')
    special_chars = sum(url.count(c) for c in ['@', '-', '?'])
    has_https = 1 if url.startswith('https://') else 0
    suspicious_words = ['login', 'verify', 'update', 'secure', 'bank']
    has_suspicious = 1 if any(word in url.lower() for word in suspicious_words) else 0
    return np.array([[url_length, num_dots, special_chars, has_https, has_suspicious]])

# 4. Input Pengguna: UPLOAD GAMBAR
uploaded_file = st.file_uploader("📂 Tarik & Lepas Gambar QR Code (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Buat dua kolom agar tampilan rapi (Kiri: Gambar, Kanan: Hasil Analisis)
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        # Membaca gambar menggunakan Pillow Hanya Sekali
        image = Image.open(uploaded_file)
        st.image(image, caption="QR Code", use_container_width=True)
    
    with col2:
        st.write("⏳ **Memindai matriks gambar...**")
        
        # Menggunakan objek gambar dari Pillow
        decoded_objects = decode(image)
        
        if decoded_objects:
            for obj in decoded_objects:
                url_terdeteksi = obj.data.decode('utf-8')
                
                st.success("Tautan berhasil diekstrak!")
                st.code(url_terdeteksi, language="text")
                
                st.write("🧠 **Menjalankan analisis Random Forest...**")
                fitur_ekstrak = extract_features(url_terdeteksi)
                prediksi = model.predict(fitur_ekstrak)
                
                st.markdown("### Hasil Keputusan AI:")
                if prediksi[0] == 1:
                    st.error("🚨 BAHAYA: TAUTAN PHISHING TERDETEKSI!")
                    st.write("Model mengidentifikasi pola anomali leksikal pada URL ini. Jangan kunjungi tautan tersebut.")
                else:
                    st.success("✅ AMAN: TAUTAN BERSIH.")
                    st.write("Struktur URL tampak normal dan lolos uji keamanan.")
        else:
            st.error("❌ Gagal membaca QR Code. Pastikan gambar tidak buram atau terpotong.")
