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
        decoded_objects = decode(img)
        
        if decoded_objects:
            for obj in decoded_objects:
                url_terdeteksi = obj.data.decode('utf-8')
                
                st.success("Tautan berhasil diekstrak!")
                st.code(url_terdeteksi, language="text")
                
                # ==========================================
                # PERBAIKAN 1: SMART WHITELIST 
                # ==========================================
                # Kita pisahkan Whitelist menjadi dua level: Institusi & Publik
                domain_institusi = ['upi.edu', 'siak.upi.edu']
                domain_publik = ['google.com', 'docs.google.com', 'drive.google.com', 'linktr.ee', 'wa.me', 'instagram.com']
                
                url_lower = url_terdeteksi.lower()
                is_institusi = any(domain in url_lower for domain in domain_institusi)
                is_publik = any(domain in url_lower for domain in domain_publik)

                st.markdown("### Hasil Keputusan AI:")
                
                if is_institusi:
                    st.balloons()
                    st.success("✅ AMAN: Domain Internal Institusi.")
                    st.write("Sistem mengenali ini sebagai tautan resmi kampus.")
                elif is_publik:
                    st.info("ℹ️ AMAN (DENGAN CATATAN): Domain Publik Terpercaya.")
                    st.write("Domain ini sah, namun sering digunakan oleh pihak ketiga (seperti Google Form/Linktree). Harap tetap berhati-hati jika diminta memasukkan password.")
                else:
                    # ==========================================
                    # PERBAIKAN 2: THRESHOLD DIKEMBALIKAN KE 80%
                    # ==========================================
                    st.write("🧠 **Menjalankan analisis Random Forest...**")
                    fitur_ekstrak = extract_features(url_terdeteksi)
                    probabilitas = model.predict_proba(fitur_ekstrak)[0]
                    persentase_phishing = probabilitas[1] 
                    
                    # Naikkan batas toleransi menjadi 80%
                    batas_toleransi = 0.80 
                    
                    if persentase_phishing >= batas_toleransi:
                        st.error(f"🚨 BAHAYA: TAUTAN PHISHING TERDETEKSI! (Skor Ancaman: {persentase_phishing*100:.1f}%)")
                        st.write("Model mengidentifikasi pola anomali leksikal tingkat tinggi. JANGAN kunjungi tautan tersebut.")
                    else:
                        st.success(f"✅ AMAN: TAUTAN BERSIH. (Skor Keamanan: {(1-persentase_phishing)*100:.1f}%)")
                        st.write("Struktur URL ini dinilai aman oleh sistem.")
        else:
            st.error("❌ Gagal membaca QR Code. Pastikan gambar tidak buram atau terpotong.")
