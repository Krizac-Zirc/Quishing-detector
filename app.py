import streamlit as st
import pickle
import numpy as np
import re
from pyzbar.pyzbar import decode
from PIL import Image

# ── Konfigurasi Halaman ──────────────────────────────────────────────────────
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
        padding: 20px;
    }
    h1, h2, h3, p { color: #f8f9fa !important; }
    .stAlert { border-radius: 10px; font-weight: bold; }
    .feature-box {
        background: rgba(255,255,255,0.07);
        border-radius: 10px;
        padding: 12px 16px;
        margin-top: 10px;
        font-size: 0.85em;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ VeriQR AI")
st.write("### Sistem Analisis & Deteksi QR Phishing")
st.write(
    "Unggah gambar QR Code, dan sistem AI kami akan menganalisis keamanan "
    "tautan di dalamnya secara real-time."
)
st.markdown("---")

# ── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model_rf.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ── Ekstraksi Fitur (10 fitur — HARUS sinkron dengan train_model.py) ─────────
SUSPICIOUS_WORDS = [
    "login", "verify", "update", "secure", "bank",
    "account", "confirm", "signin", "password", "reset",
    "paypal", "ebay", "amazon", "apple", "microsoft",
    "free", "winner", "click", "limited", "urgent",
]

def extract_features(url: str) -> np.ndarray:
    url_lower = url.lower()
    parts     = url.split("/")

    f1  = len(url)                                                          # panjang URL
    f2  = url.count(".")                                                    # jumlah titik
    f3  = sum(url.count(c) for c in ["@", "-", "?", "=", "&"])            # karakter khusus
    f4  = 1 if url_lower.startswith("https://") else 0                     # pakai HTTPS
    f5  = 1 if any(w in url_lower for w in SUSPICIOUS_WORDS) else 0       # kata mencurigakan
    f6  = max(url.count("/") - 2, 0)                                       # kedalaman path
    f7  = 1 if re.search(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", url) else 0  # alamat IP
    f8  = len(re.findall(r"[^a-zA-Z0-9]", url))                           # karakter non-alfanumerik
    f9  = 1 if url.count("//") > 1 else 0                                 # double-slash (redirect)
    f10 = len(parts[2]) if len(parts) > 2 else 0                          # panjang domain

    return np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10], dtype=float)

FEATURE_LABELS = [
    "Panjang URL", "Jumlah titik", "Karakter khusus",
    "HTTPS", "Kata mencurigakan", "Kedalaman path",
    "Alamat IP", "Karakter non-alfanumerik", "Double-slash", "Panjang domain",
]

# ── Domain Whitelist ─────────────────────────────────────────────────────────
TRUSTED_DOMAINS = ["upi.edu", "siak.upi.edu"]

# ── Antarmuka Unggah ─────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📂 Tarik & Lepas Gambar QR Code (PNG/JPG)",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1.5])

    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="QR Code", use_container_width=True)

    with col2:
        st.write("⏳ **Memindai matriks gambar...**")
        decoded_objects = decode(image)

        if not decoded_objects:
            st.error(
                "❌ QR Code tidak terdeteksi. "
                "Pastikan gambar jelas, tidak buram, dan mengandung QR Code yang valid."
            )
        else:
            for obj in decoded_objects:
                url_terdeteksi = obj.data.decode("utf-8")

                st.success("✔️ Tautan berhasil diekstrak!")
                st.code(url_terdeteksi, language="text")

                st.write("🧠 **Menjalankan analisis Random Forest...**")
                fitur = extract_features(url_terdeteksi)

                # Whitelist check
                is_whitelisted = any(
                    domain in url_terdeteksi.lower() for domain in TRUSTED_DOMAINS
                )

                st.markdown("### Hasil Keputusan AI:")

                if is_whitelisted:
                    st.success("✅ AMAN — Domain Tepercaya (Whitelist)")
                    st.write("Sistem mengenali domain ini sebagai entitas resmi yang aman.")
                else:
                    proba             = model.predict_proba(fitur.reshape(1, -1))[0]
                    pct_phishing      = proba[1]          # indeks 1 = phishing
                    pct_aman          = proba[0]          # indeks 0 = benign
                    BATAS_TOLERANSI   = 0.50              # threshold default 50%

                    if pct_phishing >= BATAS_TOLERANSI:
                        st.error(
                            f"🚨 BAHAYA — PHISHING TERDETEKSI! "
                            f"(Keyakinan: {pct_phishing * 100:.1f}%)"
                        )
                        st.write(
                            "Model mendeteksi pola leksikal anomali yang khas pada URL phishing/quishing."
                        )
                    else:
                        st.success(
                            f"✅ AMAN — Tautan Bersih. "
                            f"(Skor Keamanan: {pct_aman * 100:.1f}%)"
                        )

                    # Progress bar visual
                    st.markdown("**Probabilitas Phishing:**")
                    st.progress(float(pct_phishing))
                    st.caption(f"{pct_phishing * 100:.1f}% kemungkinan phishing")

                # Detail fitur (expandable)
                with st.expander("🔍 Detail Fitur yang Dianalisis"):
                    st.markdown('<div class="feature-box">', unsafe_allow_html=True)
                    for label, val in zip(FEATURE_LABELS, fitur):
                        icon = "✅" if label == "HTTPS" and val == 1 else \
                               "⚠️" if label in ("Alamat IP", "Kata mencurigakan", "Double-slash") and val == 1 else "•"
                        st.write(f"{icon} **{label}:** {int(val)}")
                    st.markdown("</div>", unsafe_allow_html=True)
