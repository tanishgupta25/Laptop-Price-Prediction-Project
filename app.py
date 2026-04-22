"""
app.py
------
Streamlit Web App for Laptop Price Prediction.
Premium dark-themed UI with glassmorphism effects.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LaptopAI — Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255,255,255,0.1);
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* Glass card */
.glass-card {
    background: rgba(255,255,255,0.07);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

/* Hero */
.hero-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.hero-subtitle {
    color: #94a3b8;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

/* Metric cards */
.metric-card {
    background: rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 22px 20px;
    border: 1px solid rgba(255,255,255,0.12);
    text-align: center;
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-4px); }
.metric-label {
    color: #94a3b8;
    font-size: 0.82rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #f1f5f9;
}

/* Price badge */
.price-badge {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 16px;
    padding: 30px;
    text-align: center;
    box-shadow: 0 0 40px rgba(99,102,241,0.4);
}
.price-badge .price-text {
    font-size: 3.2rem;
    font-weight: 800;
    color: white;
}
.price-badge .price-label {
    color: rgba(255,255,255,0.7);
    font-size: 0.9rem;
    margin-top: 4px;
}

/* Quality badges */
.badge-low    { background: linear-gradient(135deg,#f59e0b,#fbbf24); }
.badge-medium { background: linear-gradient(135deg,#3b82f6,#60a5fa); }
.badge-high   { background: linear-gradient(135deg,#10b981,#34d399); }
.quality-badge {
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    color: white;
    font-weight: 700;
    font-size: 1.2rem;
}

/* Recommend tag */
.rec-budget  { border-left: 4px solid #f59e0b; background: rgba(245,158,11,0.12); }
.rec-mid     { border-left: 4px solid #3b82f6; background: rgba(59,130,246,0.12); }
.rec-premium { border-left: 4px solid #10b981; background: rgba(16,185,129,0.12); }
.rec-box {
    border-radius: 12px;
    padding: 16px 20px;
    color: #e2e8f0;
    font-size: 1rem;
    font-weight: 500;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.1);
    margin: 20px 0;
}

/* Inputs */
.stSelectbox > div > div { background: rgba(255,255,255,0.05) !important; }
.stSlider > div { color: #e2e8f0 !important; }
label, .stSelectbox label { color: #cbd5e1 !important; font-weight: 500 !important; }
</style>
""", unsafe_allow_html=True)


# ── Load Model & Metadata ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


@st.cache_data
def load_meta():
    with open("model_meta.json") as f:
        return json.load(f)


try:
    model = load_model()
    meta = load_meta()
except FileNotFoundError:
    st.error("⚠️ model.pkl not found. Please run `python train_model.py` first.")
    st.stop()


# ── Feature Engineering (mirrors train_model.py) ──────────────────────────────
def get_cpu_tier(cpu_str):
    if any(x in cpu_str for x in ["i9", "i7", "Ryzen 7"]):
        return "High"
    elif any(x in cpu_str for x in ["i5", "Ryzen 5"]):
        return "Mid"
    return "Entry"


def get_gpu_tier(gpu_str):
    if any(x in gpu_str for x in ["RTX", "Quadro"]):
        return "High"
    elif any(x in gpu_str for x in ["GTX 1060", "GTX 1070", "GTX 1080", "RX 580"]):
        return "Mid-High"
    elif any(x in gpu_str for x in ["GTX 1050", "940MX", "Radeon"]):
        return "Mid"
    return "Integrated"


def get_resolution_cat(res_str):
    if "4K" in res_str or "3840" in res_str:
        return "4K"
    elif any(x in res_str for x in ["2560", "2880", "2304", "2960"]):
        return "QHD"
    elif "1920" in res_str or "Full HD" in res_str:
        return "FHD"
    return "HD"


def predict_price(company, type_name, inches, screen_res, cpu_brand, cpu_tier,
                  cpu_ghz, ram_gb, ssd_gb, hdd_gb, gpu_tier, os_name, weight_kg):
    input_df = pd.DataFrame([{
        "Company": company,
        "TypeName": type_name,
        "CPU_Brand": cpu_brand,
        "CPU_Tier": cpu_tier,
        "GPU_Tier": gpu_tier,
        "Resolution_Cat": get_resolution_cat(screen_res),
        "OpSys": os_name,
        "Inches": inches,
        "RAM_GB": ram_gb,
        "Weight_kg": weight_kg,
        "CPU_GHz": cpu_ghz,
        "SSD_GB": ssd_gb,
        "HDD_GB": hdd_gb,
    }])
    log_pred = model.predict(input_df)[0]
    return np.expm1(log_pred)


def quality_score(price):
    if price < 700:
        return "Low", "badge-low", "💛"
    elif price < 1500:
        return "Medium", "badge-medium", "💙"
    return "High", "badge-high", "💚"


def recommendation(price):
    if price < 700:
        return "Budget", "rec-budget", "🎯 Great entry-level pick for everyday use."
    elif price < 1500:
        return "Mid-range", "rec-mid", "⚡ Solid performance for professionals."
    return "Premium", "rec-premium", "🚀 Top-tier power for demanding workloads."


# ── Sidebar Inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configure Laptop")
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    companies = ["Apple", "HP", "Dell", "Lenovo", "Asus", "Acer",
                 "MSI", "Toshiba", "Samsung", "Razer", "Microsoft", "Huawei"]
    company = st.selectbox("🏢 Brand / Company", companies)

    type_opts = ["Ultrabook", "Gaming", "Notebook",
                 "2 in 1 Convertible", "Workstation", "Netbook"]
    type_name = st.selectbox("💼 Laptop Type", type_opts)

    inches = st.select_slider("📏 Screen Size (inches)",
                               options=[11.6, 12.5, 13.3, 13.5, 14.0, 14.1,
                                        15.4, 15.6, 16.0, 17.3],
                               value=15.6)

    res_opts = ["Full HD 1920x1080", "IPS Panel Full HD 1920x1080",
                "4K Ultra HD 3840x2160", "IPS Panel 2560x1440",
                "IPS Panel Retina Display 2560x1600", "1366x768"]
    screen_res = st.selectbox("🖥️ Screen Resolution", res_opts)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("**🧠 Processor**")
    cpu_brand = st.selectbox("CPU Brand", ["Intel", "AMD", "Other"])
    cpu_tier = st.selectbox("CPU Tier", ["High", "Mid", "Entry"],
                             help="High = i7/i9/Ryzen7, Mid = i5/Ryzen5, Entry = i3/AMD E-series")
    cpu_ghz = st.slider("CPU Speed (GHz)", 1.0, 4.0, 2.5, step=0.1)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("**💾 Memory & Storage**")
    ram_gb = st.select_slider("RAM (GB)", options=[2, 4, 8, 12, 16, 32, 64], value=8)
    ssd_gb = st.select_slider("SSD Storage (GB)", options=[0, 64, 128, 256, 512, 1024, 2048], value=256)
    hdd_gb = st.select_slider("HDD Storage (GB)", options=[0, 500, 1024, 2048], value=0)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("**🎮 Graphics & OS**")
    gpu_tier = st.selectbox("GPU Tier",
                             ["Integrated", "Mid", "Mid-High", "High"],
                             help="High = RTX/Quadro, Mid-High = GTX 1060+, Mid = GTX 1050, Integrated = Intel HD")
    os_opts = ["Windows 10", "macOS", "Linux", "No OS", "Chrome OS", "Windows 10 S"]
    os_name = st.selectbox("🖥️ Operating System", os_opts)
    weight_kg = st.slider("⚖️ Weight (kg)", 0.9, 4.5, 2.0, step=0.1)

    predict_btn = st.button("🔮 Predict Price", use_container_width=True, type="primary")

# ── Main Page ─────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">💻 LaptopAI Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">AI-powered laptop price estimation with quality scoring & recommendations</div>', unsafe_allow_html=True)

# Summary cards
c1, c2, c3, c4 = st.columns(4)
specs = [
    ("🏢 Brand", company),
    ("💼 Type", type_name),
    ("🧠 Processor", f"{cpu_brand} {cpu_tier} {cpu_ghz}GHz"),
    ("💾 RAM / SSD", f"{ram_gb}GB / {ssd_gb}GB"),
]
for col, (label, val) in zip([c1, c2, c3, c4], specs):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="font-size:1.1rem">{val}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ── Results ───────────────────────────────────────────────────────────────────
if predict_btn:
    with st.spinner("🤖 Running prediction model..."):
        price = predict_price(
            company, type_name, inches, screen_res,
            cpu_brand, cpu_tier, cpu_ghz,
            ram_gb, ssd_gb, hdd_gb,
            gpu_tier, os_name, weight_kg
        )
        qs, qs_class, qs_emoji = quality_score(price)
        rec, rec_class, rec_desc = recommendation(price)

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.markdown(f"""
        <div class="price-badge">
            <div class="price-label">💰 PREDICTED PRICE</div>
            <div class="price-text">€{price:,.0f}</div>
            <div class="price-label">≈ ${price * 1.08:,.0f} USD</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="quality-badge {qs_class}">
            <div style="font-size:2rem">{qs_emoji}</div>
            <div>Quality Score</div>
            <div style="font-size:1.5rem">{qs}</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="rec-box {rec_class}" style="height:100%;display:flex;flex-direction:column;justify-content:center">
            <div style="font-size:1.4rem;font-weight:700;color:#f1f5f9">{rec}</div>
            <div style="color:#94a3b8;font-size:0.9rem;margin-top:8px">{rec_desc}</div>
        </div>""", unsafe_allow_html=True)

    # Price breakdown gauge
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("### 📊 Price Breakdown Analysis")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**💡 What drives this price?**")
        factors = {
            "RAM": ram_gb * 12,
            "SSD Storage": ssd_gb * 0.8,
            "CPU Tier": {"High": 350, "Mid": 120, "Entry": 40}[cpu_tier],
            "GPU Tier": {"High": 350, "Mid-High": 240, "Mid": 80, "Integrated": 0}[gpu_tier],
            "Brand Premium": {"Apple": 250, "Razer": 220, "MSI": 150,
                               "Dell": 80, "HP": 60, "Lenovo": 40,
                               "Asus": 40, "Acer": 0}.get(company, 20),
        }
        total = sum(factors.values()) or 1
        for k, v in sorted(factors.items(), key=lambda x: -x[1]):
            pct = v / total * 100
            st.markdown(f"**{k}** — {pct:.0f}%")
            st.progress(min(pct / 100, 1.0))
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**🎯 Market Positioning**")
        ranges = [
            ("Budget", "< €700", price < 700),
            ("Mid-range", "€700 – €1,500", 700 <= price < 1500),
            ("Premium", "> €1,500", price >= 1500),
        ]
        for rname, rrange, active in ranges:
            icon = "✅" if active else "⬜"
            st.markdown(f"{icon} **{rname}** ({rrange})")
        st.markdown("---")
        st.markdown(f"**Model Accuracy (R²): {meta.get('final_r2', 'N/A')}**")
        st.markdown("Trained on 2,200+ real & synthetic laptop records.")
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="glass-card" style="text-align:center;padding:60px 40px">
        <div style="font-size:4rem">🤖</div>
        <div style="color:#a78bfa;font-size:1.4rem;font-weight:700;margin:12px 0">
            Configure your laptop specs in the sidebar
        </div>
        <div style="color:#64748b;font-size:1rem">
            Then click <strong>Predict Price</strong> to get an instant AI-powered price estimate,
            quality score, and market recommendation.
        </div>
    </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#475569;font-size:0.8rem;padding:12px">
    🤖 LaptopAI | Built with Streamlit & Scikit-learn |
    Model: Random Forest / Gradient Boosting | Dataset: 2,200+ records
</div>""", unsafe_allow_html=True)
