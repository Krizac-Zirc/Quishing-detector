import streamlit as st
import pickle
import numpy as np
from pyzbar.pyzbar import decode
from PIL import Image

# 1. Konfigurasi Visual
st.set_page_config(page_title="VeriQR AI", page_icon="🛡️", layout="centered")

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
    }
    h1, h2, h3, p { color: #f8f9fa !important; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ VeriQR AI")
st.write("### Deteksi Phishing QR Code Real-Time")
st.markdown("---")

# 2. Load Model
@st.cache_resource
def load_model():
    with open('model_rf.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

def extract_features(url):
    return np.array([[
        len(url), 
        url.count('.'), 
        sum(url.count(c) for c in ['@', '-', '?']),
        1 if url.startswith('https://') else 0,
        1 if any(w in url.lower() for w in ['login', 'verify', 'update', 'secure', 'bank']) else 0
    ]])

# 3. Input & Logika Deteksi
uploaded_file = st.file_uploader("📂 Unggah Gambar QR Code", type=["png", "jpg", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns([1, 1.5])
    img = Image.open(uploaded_file)
    
    with col1:
        st.image(img, caption="Preview QR", use_container_width=True)
    
    with col2:
        decoded = decode(img)
        if decoded:
            for obj in decoded:
                url = obj.data.decode('utf-8')
                st.success("Tautan Terdeteksi!")
                st.code(url, language="text")
                
                # Pelindung Whitelist Institusi
                whitelist = ['upi.edu', 'siak.upi.edu']
                is_safe_zone = any(domain in url.lower() for domain in whitelist)
                
                if is_safe_zone:
                    st.balloons()
                    st.success("✅ AMAN: Domain Terpercaya (Institusi).")
                else:
                    # Prediksi dengan Probabilitas (Threshold 60%)
                    features = extract_features(url)
                    prob = model.predict_proba(features)[0][1] # Ambil skor Phishing
                    
                    st.markdown("### Analisis Kecerdasan Buatan:")
                    if prob >= 0.60:
                        st.error(f"🚨 BAHAYA: TERDETEKSI PHISHING! (Skor Ancaman: {prob*100:.1f}%)")
                    else:
                        st.success(f"✅ AMAN: TAUTAN BERSIH. (Skor Keamanan: {(1-prob)*100:.1f}%)")
        else:
            st.error("❌ Gambar tidak terbaca sebagai QR Code.")
