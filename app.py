import streamlit as st
import pickle
import numpy as np
import re
import math
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
        line-height: 1.8;
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

# ── Load Model (format dict) ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model_rf.pkl", "rb") as f:
        data = pickle.load(f)
    return data

model_data        = load_model()
model             = model_data["model"]
FEATURE_LABELS    = model_data["feature_labels"]
URL_SHORTENERS    = model_data["url_shorteners"]
SUSPICIOUS_TLDS   = model_data["suspicious_tlds"]
SUSPICIOUS_KEYWORDS = model_data["suspicious_keywords"]
KNOWN_BRANDS      = model_data["known_brands"]
THRESHOLD         = model_data.get("threshold", 0.50)

# ── Helper ───────────────────────────────────────────────────────────────────
def get_domain(url: str) -> str:
    try:
        without_scheme = re.sub(r"^https?://", "", url)
        return without_scheme.split("/")[0].split(":")[0].lower()
    except Exception:
        return ""

def get_tld(domain: str) -> str:
    parts = domain.split(".")
    return parts[-1].lower() if parts else ""

def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq: dict = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in freq.values())

# ── Ekstraksi Fitur — 18 fitur ───────────────────────────────────────────────
def extract_features(url: str) -> np.ndarray:
    url_lower    = url.lower()
    domain       = get_domain(url)
    tld          = get_tld(domain)
    domain_parts = domain.split(".")
    registrable  = ".".join(domain_parts[-2:]) if len(domain_parts) >= 2 else domain

    f01 = len(url)
    f02 = url.count(".")
    f03 = sum(url.count(c) for c in ["@", "-", "?", "=", "&", "%", "_"])
    f04 = 1 if url_lower.startswith("https://") else 0
    f05 = 1 if any(w in url_lower for w in SUSPICIOUS_KEYWORDS) else 0
    f06 = max(url.count("/") - 2, 0)
    f07 = 1 if re.search(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", url) else 0
    f08 = 1 if registrable in URL_SHORTENERS else 0
    f09 = 1 if tld in SUSPICIOUS_TLDS else 0
    f10 = max(len(domain_parts) - 2, 0)
    f11 = round(shannon_entropy(domain_parts[0] if domain_parts else ""), 4)
    f12 = sum(c.isdigit() for c in domain) / max(len(domain), 1)
    brand_in_sub = (
        any(brand in ".".join(domain_parts[:-2]).lower() for brand in KNOWN_BRANDS)
        if len(domain_parts) > 2 else False
    )
    f13 = 1 if brand_in_sub else 0
    path_part = "/".join(url.split("/")[3:]) if len(url.split("/")) > 3 else ""
    f14 = 1 if any(brand in path_part.lower() for brand in KNOWN_BRANDS) else 0
    f15 = 1 if url.count("//") > 1 else 0
    f16 = url.lower().count("%")
    f17 = len(url.split("?")[1].split("&")) if "?" in url else 0
    f18 = round(shannon_entropy(url), 4)

    return np.array(
        [f01, f02, f03, f04, f05, f06, f07,
         f08, f09, f10, f11, f12, f13, f14,
         f15, f16, f17, f18],
        dtype=float,
    )

# ── Domain Whitelist ──────────────────────────────────────────────────────────
TRUSTED_DOMAINS = ["upi.edu", "siak.upi.edu"]

# ── UI Utama ─────────────────────────────────────────────────────────────────
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
                "Pastikan gambar jelas dan mengandung QR Code yang valid."
            )
        else:
            for obj in decoded_objects:
                url_terdeteksi = obj.data.decode("utf-8")

                st.success("✔️ Tautan berhasil diekstrak!")
                st.code(url_terdeteksi, language="text")

                st.write("🧠 **Menjalankan analisis AI (Ensemble RF + GBM)...**")
                fitur = extract_features(url_terdeteksi)

                is_whitelisted = any(
                    domain in url_terdeteksi.lower() for domain in TRUSTED_DOMAINS
                )

                st.markdown("### Hasil Keputusan AI:")

                if is_whitelisted:
                    st.success("✅ AMAN — Domain Tepercaya (Whitelist)")
                    st.write("Sistem mengenali domain ini sebagai entitas resmi yang aman.")
                else:
                    proba        = model.predict_proba(fitur.reshape(1, -1))[0]
                    pct_phishing = float(proba[1])
                    pct_aman     = float(proba[0])

                    if pct_phishing >= THRESHOLD:
                        st.error(
                            f"🚨 BAHAYA — PHISHING/QUISHING TERDETEKSI! "
                            f"(Keyakinan: {pct_phishing * 100:.1f}%)"
                        )
                        st.write(
                            "Model mendeteksi pola leksikal dan struktural yang khas "
                            "pada URL phishing/quishing."
                        )
                    else:
                        st.success(
                            f"✅ AMAN — Tautan Bersih. "
                            f"(Skor Keamanan: {pct_aman * 100:.1f}%)"
                        )

                    st.markdown("**Probabilitas Phishing:**")
                    st.progress(pct_phishing)
                    st.caption(f"{pct_phishing * 100:.1f}% kemungkinan phishing")

                # Detail fitur
                with st.expander("🔍 Detail Fitur yang Dianalisis"):
                    st.markdown('<div class="feature-box">', unsafe_allow_html=True)
                    WARNING_FEATURES = {
                        "URL Shortener", "Suspicious TLD", "Alamat IP",
                        "Double-slash", "Brand di subdomain", "Kata mencurigakan",
                        "Percent-encoding",
                    }
                    for label, val in zip(FEATURE_LABELS, fitur):
                        if label in WARNING_FEATURES and val > 0:
                            icon = "⚠️"
                        elif label == "HTTPS" and val == 1:
                            icon = "🔒"
                        else:
                            icon = "•"
                        st.write(f"{icon} **{label}:** {round(float(val), 3)}")
                    st.markdown("</div>", unsafe_allow_html=True)
