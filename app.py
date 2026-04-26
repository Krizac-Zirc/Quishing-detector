"""
VeriQR AI — Sistem Deteksi QR Phishing (Quishing)
Arsitektur: Rule Engine (primer) + ML Ensemble (sekunder)
Versi: 4.0
"""

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
        background-color: rgba(255,255,255,0.1);
        border-radius: 15px;
        border: 2px dashed #4facfe;
        padding: 20px;
    }
    h1, h2, h3, p { color: #f8f9fa !important; }
    .stAlert { border-radius: 10px; font-weight: bold; }
    .reason-box {
        background: rgba(255,255,255,0.06);
        border-left: 3px solid #f0a500;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.9em;
    }
    .reason-danger { border-left-color: #ff4b4b; }
    .reason-safe   { border-left-color: #21c55d; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ VeriQR AI")
st.write("### Sistem Analisis & Deteksi QR Phishing (Quishing)")
st.write("Unggah gambar QR Code — sistem AI akan menganalisis URL dan mendeteksi ancaman secara real-time.")
st.markdown("---")

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model_rf.pkl", "rb") as f:
        return pickle.load(f)

md = load_model()
model               = md["model"]
FEATURE_LABELS      = md["feature_labels"]
URL_SHORTENERS      = md["url_shorteners"]
SUSPICIOUS_TLDS     = md["suspicious_tlds"]
SUSPICIOUS_KEYWORDS = md["suspicious_keywords"]
KNOWN_BRANDS        = md["known_brands"]
OFFICIAL_DOMAINS    = md["official_domains"]
REDIRECT_PLATFORMS  = md["redirect_platforms"]

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_domain(url: str) -> str:
    try:
        ws = re.sub(r"^https?://", "", url.strip())
        return ws.split("/")[0].split(":")[0].lower()
    except Exception:
        return ""

def get_tld(domain: str) -> str:
    parts = domain.split(".")
    return parts[-1].lower() if parts else ""

def get_registrable(domain: str) -> str:
    parts = domain.split(".")
    if len(parts) >= 3 and parts[-2] in ("co","com","net","org","go","ac"):
        return ".".join(parts[-3:])
    return ".".join(parts[-2:]) if len(parts) >= 2 else domain

def shannon_entropy(s: str) -> float:
    if not s: return 0.0
    freq: dict = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((v/n) * math.log2(v/n) for v in freq.values())

def is_redirect_platform(url: str, domain: str, path: str) -> bool:
    for plat_domain, plat_paths in REDIRECT_PLATFORMS.items():
        if domain == plat_domain or domain.endswith("." + plat_domain):
            for pp in plat_paths:
                if path.startswith(pp):
                    return True
    return False

def has_brand_in_domain_not_official(domain: str, registrable: str) -> bool:
    if registrable in OFFICIAL_DOMAINS:
        return False
    domain_body = re.sub(r"[^a-z]", "", registrable.split(".")[0])
    full_flat   = re.sub(r"[^a-z]", "", domain)
    for brand in KNOWN_BRANDS:
        if brand in domain_body or brand in full_flat:
            return True
    return False

def is_typosquatting(domain: str) -> bool:
    no_hyphen = domain.replace("-", ".")
    if no_hyphen == domain:
        return False
    for brand in KNOWN_BRANDS:
        if brand in no_hyphen:
            return True
    return False

# ── Rule Engine (primer) ──────────────────────────────────────────────────────
def rule_based_score(url: str):
    """
    Kembalikan (score 0–100, list alasan).
    >= 50 → BERBAHAYA | 25–49 → MENCURIGAKAN | < 25 → AMAN
    """
    score   = 0
    reasons = []

    url_lower    = url.lower().strip()
    domain       = get_domain(url)
    tld          = get_tld(domain)
    registrable  = get_registrable(domain)
    domain_parts = domain.split(".")
    path         = "/" + "/".join(url.split("/")[3:]) if len(url.split("/")) > 3 else "/"
    params       = url.split("?")[1] if "?" in url else ""

    # Whitelist domain resmi
    if registrable in OFFICIAL_DOMAINS and not is_redirect_platform(url, domain, path):
        return 5, ["✅ Domain resmi terverifikasi"]

    # [80] IP Address
    if re.search(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", domain):
        score += 80
        reasons.append("🔴 Menggunakan alamat IP langsung (bukan nama domain)")

    # [70] URL Shortener
    if registrable in URL_SHORTENERS or domain in URL_SHORTENERS:
        score += 70
        reasons.append(f"🔴 URL Shortener terdeteksi ({domain}) — tujuan asli tersembunyi")

    # [65] Redirect melalui platform resmi
    if is_redirect_platform(url, domain, path):
        score += 65
        reasons.append(f"🔴 Redirect melalui platform ({domain}) — taktik menyembunyikan URL tujuan")

    # [60] Nama brand/bank palsu di domain
    if has_brand_in_domain_not_official(domain, registrable):
        score += 60
        reasons.append("🔴 Nama brand/bank palsu di domain (bukan website resmi)")

    # [55] Brand di subdomain (impersonation)
    if len(domain_parts) > 2:
        subdomains = ".".join(domain_parts[:-2])
        for brand in KNOWN_BRANDS:
            if brand in subdomains and registrable not in OFFICIAL_DOMAINS:
                score += 55
                reasons.append(f"🔴 Impersonasi brand '{brand}' di subdomain")
                break

    # [50] Typosquatting
    if is_typosquatting(domain):
        score += 50
        reasons.append("🔴 Typosquatting — meniru domain resmi dengan tanda hubung")

    # [45] TLD mencurigakan
    if tld in SUSPICIOUS_TLDS:
        score += 45
        reasons.append(f"🟠 TLD mencurigakan (.{tld}) — sering dipakai phisher gratis")

    # [35] Kata ancaman di nama domain
    domain_body = domain_parts[0] if domain_parts else ""
    kw_dom = [k for k in SUSPICIOUS_KEYWORDS if k in domain_body]
    if kw_dom:
        score += 35
        reasons.append(f"🟠 Kata mencurigakan di nama domain: {', '.join(kw_dom[:3])}")

    # [25] Double-slash redirect trick
    if url.count("//") > 1:
        score += 25
        reasons.append("🟠 Double-slash (teknik redirect tersembunyi)")

    # [20] Banyak tanda hubung di domain
    if domain.count("-") >= 2:
        score += 20
        reasons.append(f"🟡 Banyak tanda hubung di domain ({domain.count('-')}x)")

    # [15] Tidak pakai HTTPS
    if not url_lower.startswith("https://"):
        score += 15
        reasons.append("🟡 Tidak menggunakan HTTPS (koneksi tidak terenkripsi)")

    # [15] Banyak kata mencurigakan di path/param
    kw_path = [k for k in SUSPICIOUS_KEYWORDS if k in path.lower() or k in params.lower()]
    if len(kw_path) >= 2:
        score += 15
        reasons.append(f"🟡 Banyak kata mencurigakan di URL: {', '.join(kw_path[:3])}")

    # [10] URL sangat panjang
    if len(url) > 100:
        score += 10
        reasons.append(f"🟡 URL sangat panjang ({len(url)} karakter)")

    return min(score, 100), reasons

# ── Feature Extraction (untuk ML) ────────────────────────────────────────────
def extract_features(url: str) -> np.ndarray:
    url_lower    = url.lower()
    domain       = get_domain(url)
    tld          = get_tld(domain)
    registrable  = get_registrable(domain)
    domain_parts = domain.split(".")
    path         = "/" + "/".join(url.split("/")[3:]) if len(url.split("/")) > 3 else "/"

    f01 = len(url)
    f02 = url.count(".")
    f03 = sum(url.count(c) for c in ["@","-","?","=","&","%","_"])
    f04 = 1 if url_lower.startswith("https://") else 0
    f05 = 1 if any(w in url_lower for w in SUSPICIOUS_KEYWORDS) else 0
    f06 = max(url.count("/") - 2, 0)
    f07 = 1 if re.search(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", url) else 0
    f08 = 1 if (registrable in URL_SHORTENERS or domain in URL_SHORTENERS) else 0
    f09 = 1 if tld in SUSPICIOUS_TLDS else 0
    f10 = max(len(domain_parts) - 2, 0)
    f11 = round(shannon_entropy(domain_parts[0] if domain_parts else ""), 4)
    f12 = sum(c.isdigit() for c in domain) / max(len(domain), 1)
    brand_in_sub = any(b in ".".join(domain_parts[:-2]).lower() for b in KNOWN_BRANDS) if len(domain_parts) > 2 else False
    f13 = 1 if brand_in_sub else 0
    f14 = 1 if has_brand_in_domain_not_official(domain, registrable) else 0
    f15 = 1 if url.count("//") > 1 else 0
    f16 = url.lower().count("%")
    f17 = len(url.split("?")[1].split("&")) if "?" in url else 0
    f18 = round(shannon_entropy(url), 4)
    f19 = 1 if is_typosquatting(domain) else 0
    f20 = 1 if is_redirect_platform(url, domain, path) else 0
    f21 = domain.count("-")
    f22 = 1 if any(k in (domain_parts[0] if domain_parts else "") for k in SUSPICIOUS_KEYWORDS) else 0

    return np.array(
        [f01,f02,f03,f04,f05,f06,f07,f08,f09,f10,
         f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22],
        dtype=float,
    )

# ── Verdict Helper ────────────────────────────────────────────────────────────
def get_verdict(final_score: float):
    if final_score >= 50:
        return "BERBAHAYA", "🚨", "error"
    elif final_score >= 25:
        return "MENCURIGAKAN", "⚠️", "warning"
    else:
        return "AMAN", "✅", "success"

# ── Whitelist ─────────────────────────────────────────────────────────────────
TRUSTED_DOMAINS = ["upi.edu", "siak.upi.edu"]

# ── UI Utama ──────────────────────────────────────────────────────────────────
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

                # Whitelist
                if any(d in url_terdeteksi.lower() for d in TRUSTED_DOMAINS):
                    st.success("✅ AMAN — Domain Tepercaya Institusi")
                    st.write("Domain ini dikenali sebagai domain resmi yang aman.")
                    st.stop()

                st.write("🧠 **Menganalisis URL dengan Rule Engine + AI...**")

                # Rule Engine
                rule_score, reasons = rule_based_score(url_terdeteksi)

                # ML Score
                fitur    = extract_features(url_terdeteksi)
                proba    = model.predict_proba(fitur.reshape(1, -1))[0]
                ml_score = float(proba[1]) * 100  # phishing probability 0-100

                # Skor akhir: Rule Engine dominan, ML sebagai booster
                final_score = max(rule_score, ml_score * 0.6)
                final_score = min(final_score, 100.0)

                verdict, icon, alert_type = get_verdict(final_score)

                st.markdown("---")
                st.markdown(f"### {icon} Hasil Analisis: **{verdict}**")

                # Progress bar
                bar_color  = "red" if final_score >= 50 else ("orange" if final_score >= 25 else "green")
                st.progress(final_score / 100)
                st.caption(
                    f"Skor Risiko: **{final_score:.0f}/100**  |  "
                    f"Rule Engine: {rule_score}/100  |  ML: {ml_score:.1f}%"
                )

                if alert_type == "error":
                    st.error(
                        f"🚨 **PHISHING/QUISHING TERDETEKSI!** "
                        f"Skor Risiko: {final_score:.0f}/100\n\n"
                        "**JANGAN** kunjungi tautan ini. Laporkan kepada pihak terkait."
                    )
                elif alert_type == "warning":
                    st.warning(
                        f"⚠️ **URL INI MENCURIGAKAN.** "
                        f"Skor Risiko: {final_score:.0f}/100\n\n"
                        "Berhati-hatilah. Verifikasi keaslian QR Code sebelum melanjutkan."
                    )
                else:
                    st.success(
                        f"✅ **URL TERLIHAT AMAN.** "
                        f"Skor Risiko: {final_score:.0f}/100\n\n"
                        "Tidak ditemukan indikator phishing yang kuat. "
                        "Tetap waspada — AI tidak menjamin keamanan 100%."
                    )

                # Alasan
                if reasons:
                    with st.expander(f"📋 Mengapa? — {len(reasons)} indikator ditemukan"):
                        for r in reasons:
                            css_class = "reason-danger" if "🔴" in r else (
                                "reason-safe" if "✅" in r else "reason-box")
                            st.markdown(
                                f'<div class="reason-box {css_class}">{r}</div>',
                                unsafe_allow_html=True,
                            )

                # Detail fitur teknis
                with st.expander("🔬 Detail Fitur Teknis (22 fitur)"):
                    WARNING_FEAT = {
                        "URL Shortener","Suspicious TLD","Alamat IP","Double-slash",
                        "Brand di subdomain","Brand palsu di domain","Kata mencurigakan",
                        "Percent-encoding","Typosquatting","Platform redirect",
                        "Kata ancaman di domain",
                    }
                    for label, val in zip(FEATURE_LABELS, fitur):
                        icon_f = "⚠️" if (label in WARNING_FEAT and val > 0) else (
                                 "🔒" if label == "HTTPS" and val == 1 else "•")
                        st.write(f"{icon_f} **{label}:** {round(float(val), 3)}")

                st.markdown("---")
                st.caption(
                    "⚠️ *VeriQR AI menganalisis struktur URL secara leksikal. "
                    "Phishing melalui platform sah (Google Forms, Linktree) mungkin tidak terdeteksi. "
                    "Selalu verifikasi sumber QR Code secara manual.*"
                )
