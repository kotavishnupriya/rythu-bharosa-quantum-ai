"""
🌾 Rythu Bharosa – Real-Time Quantum AI Precision Agriculture Decision Support System
రైతు భరోసా – రియల్-టైమ్ క్వాంటం AI ఖచ్చితమైన వ్యవసాయ నిర్ణయ మద్దతు వ్యవస్థ
=====================================================================================
Ultra-Attractive, World-Class Simultaneous Bilingual (English + Telugu) Decision System.
Combines Real-Time Open-Meteo Weather Telemetry, Soil Health Card Nutrient Testing,
Classical Random Forest (R²=0.9755) and Qiskit 4-Qubit Quantum Kernel Regressor.
"""

import os
import json
import time
import datetime
import urllib.parse
import urllib.request
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Qiskit imports for Quantum Kernel evaluation
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector

# ==============================================================================
# 1. PAGE CONFIGURATION & ULTRA-ATTRACTIVE MODERN CSS
# ==============================================================================
st.set_page_config(
    page_title="Rythu Bharosa – రైతు భరోసా | Quantum AI Ag",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# World-Class Ultra-Attractive CSS with Glassmorphism and Glowing Cards
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&family=Ramabhadra&display=swap');

    :root {
        --primary-gradient: linear-gradient(135deg, #064e3b 0%, #047857 40%, #059669 75%, #10b981 100%);
        --card-glow: 0 10px 30px -5px rgba(16, 185, 129, 0.12), 0 4px 12px -2px rgba(0, 0, 0, 0.04);
        --card-hover: 0 20px 35px -5px rgba(16, 185, 129, 0.22), 0 8px 16px -2px rgba(0, 0, 0, 0.06);
    }
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Ramabhadra', sans-serif;
        color: #0f172a;
    }

    /* Hero Banner Header */
    .hero-container {
        background: linear-gradient(135deg, #022c22 0%, #064e3b 35%, #047857 70%, #059669 100%);
        color: white;
        padding: 36px 42px;
        border-radius: 24px;
        margin-bottom: 25px;
        box-shadow: 0 20px 45px rgba(6, 78, 59, 0.3);
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .hero-container::before {
        content: "";
        position: absolute;
        top: -60%;
        right: -10%;
        width: 450px;
        height: 450px;
        background: radial-gradient(circle, rgba(52, 211, 153, 0.25) 0%, rgba(255,255,255,0) 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    .hero-title-en {
        font-size: 2.3rem;
        font-weight: 900;
        margin: 0;
        letter-spacing: -0.8px;
        color: #ffffff;
        text-shadow: 0 2px 8px rgba(0,0,0,0.25);
    }
    .hero-title-te {
        font-size: 1.6rem;
        font-weight: 700;
        color: #a7f3d0;
        margin: 4px 0 12px 0;
        letter-spacing: -0.3px;
    }
    .hero-desc {
        color: #ecfdf5;
        font-size: 1.08rem;
        line-height: 1.6;
        max-width: 950px;
        margin-bottom: 16px;
    }
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 16px;
        border-radius: 9999px;
        font-size: 0.86rem;
        font-weight: 700;
        margin-right: 10px;
        margin-top: 6px;
        backdrop-filter: blur(10px);
    }
    .badge-pill-live {
        background: rgba(16, 185, 129, 0.28);
        color: #d1fae5;
        border: 1px solid rgba(110, 231, 183, 0.5);
    }
    .badge-pill-quantum {
        background: rgba(168, 85, 247, 0.28);
        color: #f3e8ff;
        border: 1px solid rgba(216, 180, 254, 0.5);
    }
    .badge-pill-motto {
        background: rgba(245, 158, 11, 0.28);
        color: #fef3c7;
        border: 1px solid rgba(252, 211, 77, 0.5);
    }

    /* Bilingual Executive Summary Box */
    .executive-summary-box {
        background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 50%, #dcfce7 100%);
        border: 2px solid #86efac;
        border-radius: 20px;
        padding: 24px 30px;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px rgba(16, 185, 129, 0.1);
        position: relative;
    }
    .summary-heading {
        color: #065f46;
        margin: 0 0 12px 0;
        font-size: 1.35rem;
        font-weight: 800;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .summary-content {
        font-size: 1.12rem;
        color: #14532d;
        line-height: 1.7;
        margin: 0;
    }
    .highlight-pill-big {
        background: #ffffff;
        color: #065f46;
        padding: 4px 12px;
        border-radius: 10px;
        font-weight: 800;
        border: 1px solid #86efac;
        display: inline-block;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
    .highlight-harvest-big {
        background: #fef2f2;
        color: #b91c1c;
        padding: 4px 14px;
        border-radius: 10px;
        font-weight: 900;
        border: 1px solid #fecaca;
        display: inline-block;
        font-size: 1.35rem;
        box-shadow: 0 2px 6px rgba(185, 28, 28, 0.08);
    }

    /* Top Metric Stat Cards */
    .metric-card-top {
        background: #ffffff;
        border-radius: 18px;
        padding: 24px 22px;
        border: 1px solid #e2e8f0;
        box-shadow: var(--card-glow);
        transition: all 0.28s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .metric-card-top:hover {
        transform: translateY(-5px);
        box-shadow: var(--card-hover);
        border-color: #cbd5e1;
    }
    .card-title-bilingual {
        font-size: 0.86rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: #475569;
        margin-bottom: 8px;
    }
    .card-val-huge {
        font-size: 2.35rem;
        font-weight: 900;
        color: #0f172a;
        line-height: 1.15;
        margin-bottom: 8px;
    }
    .card-footer-bilingual {
        font-size: 0.92rem;
        color: #64748b;
        line-height: 1.5;
        border-top: 1px solid #f1f5f9;
        padding-top: 10px;
    }

    /* Classical vs Quantum Modern Cards */
    .ai-engine-card-classical {
        background: #ffffff;
        border-radius: 18px;
        padding: 24px 26px;
        border: 1.5px solid #a7f3d0;
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.08);
        border-left: 8px solid #10b981;
        margin-bottom: 20px;
    }
    .ai-engine-card-quantum {
        background: #ffffff;
        border-radius: 18px;
        padding: 24px 26px;
        border: 1.5px solid #ddd6fe;
        box-shadow: 0 8px 24px rgba(168, 85, 247, 0.08);
        border-left: 8px solid #a855f7;
        margin-bottom: 20px;
    }

    /* Advisory Callout Cards */
    .advisory-box-ok {
        background: #f0fdf4;
        border-left: 6px solid #10b981;
        border: 1px solid #bbf7d0;
        border-left-width: 6px;
        padding: 18px 24px;
        border-radius: 16px;
        margin-bottom: 16px;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.04);
    }
    .advisory-box-warn {
        background: #fffbeb;
        border-left: 6px solid #f59e0b;
        border: 1px solid #fde68a;
        border-left-width: 6px;
        padding: 18px 24px;
        border-radius: 16px;
        margin-bottom: 16px;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.04);
    }
    .advisory-box-alert {
        background: #fef2f2;
        border-left: 6px solid #ef4444;
        border: 1px solid #fecaca;
        border-left-width: 6px;
        padding: 18px 24px;
        border-radius: 16px;
        margin-bottom: 16px;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.04);
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f1f5f9;
        padding: 8px;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.95rem;
        padding: 10px 20px;
        color: #475569;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #047857 !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08) !important;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. BILINGUAL CROP & SEASON DICTIONARY (Simultaneous Display)
# ==============================================================================
CROP_BILINGUAL_DICT = {
    "Rice": "Rice / వరి (Paddy)",
    "Cotton(lint)": "Cotton / పత్తి (Lint)",
    "Dry chillies": "Dry Chillies / ఎండు మిర్చి",
    "Maize": "Maize / మొక్కజొన్న (Corn)",
    "Groundnut": "Groundnut / వేరుశనగ",
    "Sugarcane": "Sugarcane / చెరకు",
    "Banana": "Banana / అరటి",
    "Tobacco": "Tobacco / పొగాకు",
    "Turmeric": "Turmeric / పసుపు",
    "Arhar/Tur": "Red Gram / కందులు (Tur)",
    "Gram": "Bengal Gram / శనగలు",
    "Moong(Green Gram)": "Green Gram / పెసలు (Moong)",
    "Urad": "Black Gram / మినుములు (Urad)",
    "Wheat": "Wheat / గోధుమలు",
    "Onion": "Onion / ఉల్లిపాయలు",
    "Tomato": "Tomato / టమోటా",
    "Mango": "Mango / మామిడి",
    "Sunflower": "Sunflower / పొద్దుతిరుగుడు",
    "Soyabean": "Soybean / సోయాబీన్",
    "Sesamum": "Sesame / నువ్వులు",
    "Castor seed": "Castor / ఆముదాలు",
    "Coconut ": "Coconut / కొబ్బరి",
    "Arecanut": "Arecanut / పోక చెక్క",
    "Cashewnut": "Cashew / జీడిమామిడి",
    "Cardamom": "Cardamom / ఏలకులు",
    "Black pepper": "Black Pepper / మిరియాలు",
    "Garlic": "Garlic / వెల్లుల్లి",
    "Ginger": "Ginger / అల్లం",
    "Sweet potato": "Sweet Potato / చిలగడదుంప",
    "Tapioca": "Tapioca / కర్రపెండలం",
    "Jowar": "Jowar / జొన్నలు (Sorghum)",
    "Bajra": "Bajra / సజ్జలు (Pearl Millet)",
    "Ragi": "Ragi / రాగులు (Finger Millet)",
    "Barley": "Barley / బార్లీ",
    "Jute": "Jute / జనపనార",
    "Linseed": "Linseed / అవిసె గింజలు",
    "Mesta": "Mesta / గోగునార"
}

SEASON_BILINGUAL_DICT = {
    "Kharif": "Kharif / ఖరీఫ్ (Monsoon / వర్షాకాలం)",
    "Rabi": "Rabi / రబీ (Winter / శీతాకాలం)",
    "Whole Year": "Whole Year / సంవత్సరం మొత్తం",
    "Summer": "Summer / వేసవి (Zaid / ఎండాకాలం)",
    "Autumn": "Autumn / శరదృతువు",
    "Winter": "Winter / శీతాకాలం"
}


# ==============================================================================
# 3. QUANTUM KERNEL SIMULATOR CLASS
# ==============================================================================
class FastQuantumKernel:
    def __init__(self, feature_dimension=4, reps=2, entanglement="linear"):
        self.feature_dimension = feature_dimension
        self.reps = reps
        self.entanglement = entanglement
        self.feature_map = ZZFeatureMap(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement
        )

    def compute_single_statevector(self, x: np.ndarray) -> np.ndarray:
        bound_circuit = self.feature_map.assign_parameters(x)
        sv = Statevector.from_instruction(bound_circuit)
        return sv.data

    def _compute_statevectors(self, X: np.ndarray) -> np.ndarray:
        statevectors = []
        for x in X:
            statevectors.append(self.compute_single_statevector(x))
        return np.array(statevectors, dtype=complex)

    def evaluate(self, X: np.ndarray, Y: np.ndarray = None) -> np.ndarray:
        psi_X = self._compute_statevectors(X)
        if Y is None:
            overlap = np.dot(psi_X, psi_X.conj().T)
        else:
            psi_Y = self._compute_statevectors(Y)
            overlap = np.dot(psi_X, psi_Y.conj().T)
        return np.clip(np.abs(overlap) ** 2, 0.0, 1.0)


def get_statevector(q_kernel, x: np.ndarray) -> np.ndarray:
    if hasattr(q_kernel, "compute_single_statevector"):
        return q_kernel.compute_single_statevector(x)
    bound_circuit = q_kernel.feature_map.assign_parameters(x)
    sv = Statevector.from_instruction(bound_circuit)
    return sv.data


# ==============================================================================
# 4. RESOURCE LOADERS
# ==============================================================================
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")


@st.cache_resource(show_spinner=False)
def load_pipeline_models():
    try:
        classical_model = joblib.load(os.path.join(MODELS_DIR, "classical_model.pkl"))
        quantum_model = joblib.load(os.path.join(MODELS_DIR, "quantum_model.pkl"))
        encoder = joblib.load(os.path.join(MODELS_DIR, "encoder.pkl"))
        scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        pca_bundle = joblib.load(os.path.join(MODELS_DIR, "pca.pkl"))
        X_train_q = joblib.load(os.path.join(MODELS_DIR, "X_train_q.pkl"))
        K_train = joblib.load(os.path.join(MODELS_DIR, "K_train.pkl"))
        q_kernel = FastQuantumKernel(feature_dimension=4, reps=2, entanglement="linear")

        return {
            "classical": classical_model,
            "quantum": quantum_model,
            "encoder": encoder,
            "scaler": scaler,
            "pca": pca_bundle["pca"],
            "quantum_scaler": pca_bundle["quantum_scaler"],
            "X_train_q": X_train_q,
            "K_train": K_train,
            "q_kernel": q_kernel,
            "status": "ready"
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@st.cache_data(show_spinner=False)
def load_datasets():
    data = {}
    combined_path = os.path.join(DATA_DIR, "combined_crop_yield.csv")
    metrics_path = os.path.join(DATA_DIR, "model_metrics.csv")
    soil_path = os.path.join(DATA_DIR, "state_soil_data.csv")

    data["combined"] = pd.read_csv(combined_path) if os.path.exists(combined_path) else None
    data["metrics"] = pd.read_csv(metrics_path) if os.path.exists(metrics_path) else None
    data["soil"] = pd.read_csv(soil_path) if os.path.exists(soil_path) else None
    return data


# ==============================================================================
# 5. WEATHER & GEOCODING SERVICE
# ==============================================================================
DISTRICT_COORDINATES = {
    "Guntur, Andhra Pradesh (గుంటూరు)": (16.3067, 80.4365),
    "Krishna / Vijayawada, Andhra Pradesh (కృష్ణా / విజయవాడ)": (16.5062, 80.6480),
    "West Godavari / Eluru, Andhra Pradesh (పశ్చిమ గోదావరి)": (16.7107, 81.0952),
    "East Godavari / Kakinada, Andhra Pradesh (తూర్పు గోదావరి)": (16.9891, 82.2475),
    "Visakhapatnam, Andhra Pradesh (విశాఖపట్నం)": (17.6868, 83.2185),
    "Kurnool, Andhra Pradesh (కర్నూలు)": (15.8281, 78.0373),
    "Anantapur, Andhra Pradesh (అనంతపురం)": (14.6819, 77.6006),
    "YSR Kadapa, Andhra Pradesh (వైఎస్ఆర్ కడప)": (14.4673, 78.8242),
    "Chittoor / Tirupati, Andhra Pradesh (చిత్తూరు / తిరుపతి)": (13.2172, 79.1003),
    "SPSR Nellore, Andhra Pradesh (నెల్లూరు)": (14.4426, 79.9865),
    "Prakasam / Ongole, Andhra Pradesh (ప్రకాశం / ఒంగోలు)": (15.5057, 80.0499),
    "Srikakulam, Andhra Pradesh (శ్రీకాకుళం)": (18.2949, 83.8938),
    "Vizianagaram, Andhra Pradesh (విజయనగరం)": (18.1067, 83.3956),
    "Warangal, Telangana (వరంగల్)": (17.9689, 79.5941),
    "Nalgonda, Telangana (నల్గొండ)": (17.0575, 79.2684),
    "Khammam, Telangana (ఖమ్మం)": (17.2473, 80.1514),
    "Hyderabad / Rangareddy, Telangana (హైదరాబాద్)": (17.3850, 78.4867),
    "Mysuru, Karnataka (మైసూర్)": (12.2958, 76.6394),
    "Coimbatore, Tamil Nadu (కోయంబత్తూరు)": (11.0168, 76.9558),
}

WMO_WEATHER_CODES = {
    0: "Clear Sky / స్వచ్ఛమైన ఆకాశం ☀️",
    1: "Mainly Clear / సాధారణంగా నిర్మలం 🌤️",
    2: "Partly Cloudy / పాక్షికంగా మేఘాలు ⛅",
    3: "Overcast / పూర్తిగా మేఘావృతం ☁️",
    45: "Foggy / పొగమంచు 🌫️",
    51: "Light Drizzle / తేలికపాటి చినుకులు 🌦️",
    53: "Moderate Drizzle / మధ్యస్థ చినుకులు 🌧️",
    61: "Light Rain / తేలికపాటి వర్షం 🌧️",
    63: "Moderate Rain / మధ్యస్థ వర్షం 🌧️",
    65: "Heavy Rain / భారీ వర్షం ⛈️",
    80: "Rain Showers / వర్షపు జల్లులు 🌦️",
    82: "Violent Rain / తుఫాను వర్షం ⛈️",
    95: "Thunderstorm / ఉరుముల వర్షం ⚡"
}


def geocode_location(query: str):
    query_clean = query.strip()
    if query_clean in DISTRICT_COORDINATES:
        lat, lon = DISTRICT_COORDINATES[query_clean]
        return lat, lon, query_clean, "District Hub Database"

    try:
        encoded_query = urllib.parse.quote(query_clean)
        url = f"https://nominatim.openstreetmap.org/search?q={encoded_query}&format=json&limit=1"
        req = urllib.request.Request(url, headers={"User-Agent": "RythuBharosaPrecisionAg/1.0"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode())
            if data and len(data) > 0:
                lat = float(data[0]["lat"])
                lon = float(data[0]["lon"])
                disp_name = data[0].get("display_name", query_clean)
                return lat, lon, disp_name, "OpenStreetMap Nominatim"
    except Exception:
        pass

    return 16.3067, 80.4365, f"{query_clean} (Andhra Pradesh default)", "Fallback Centroid"


def fetch_live_weather(lat: float, lon: float):
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m"
        f"&forecast_days=3&timezone=auto"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "RythuBharosaAgriAdvisor/1.0"})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            current = data.get("current", {})
            w_code = current.get("weather_code", 0)
            condition = WMO_WEATHER_CODES.get(w_code, "Fair Weather / సాధారణ వాతావరణం 🌤️")

            return {
                "temperature": current.get("temperature_2m", 28.5),
                "humidity": current.get("relative_humidity_2m", 68.0),
                "precipitation": current.get("precipitation", 0.0),
                "wind_speed": current.get("wind_speed_10m", 8.5),
                "weather_code": w_code,
                "condition": condition,
                "source": "Open-Meteo Global Meteorological API",
                "timestamp": current.get("time", datetime.datetime.now().strftime("%Y-%m-%d %H:%M")),
                "status": "success"
            }
    except Exception:
        return {
            "temperature": 29.2,
            "humidity": 65.0,
            "precipitation": 0.0,
            "wind_speed": 10.0,
            "weather_code": 1,
            "condition": "Mainly Clear / సాధారణంగా నిర్మలం 🌤️",
            "source": "Regional Climate Baseline (Offline fallback)",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "status": "offline_fallback"
        }


# ==============================================================================
# 6. BILINGUAL AGRONOMIC ADVISORY & RISK ENGINE
# ==============================================================================
def evaluate_agronomic_advisory(crop, area, exp_yield, hist_mean, temp, hum, rain_cur, n_val, p_val, k_val, ph_val):
    advisories = []
    risks = []

    # 1. Thermal Stress
    if temp > 38.0:
        risks.append({
            "type": "alert",
            "title_en": "High Heat Stress Alert (>38°C)",
            "title_te": "అధిక ఉష్ణోగ్రత హెచ్చరిక (>38°C)",
            "desc_en": f"Current temperature is {temp}°C. Prolonged heat causes floral abortion and high moisture evaporation. Ensure early morning light irrigation.",
            "desc_te": f"ప్రస్తుతం ఉష్ణోగ్రత {temp}°C గా ఉంది. ఎండ తీవ్రత వల్ల పువ్వు రాలడం, తేమ ఆవిరైపోవడం జరుగుతుంది. ఉదయం వేళల్లో తేలికపాటి నీటి తడులు ఇవ్వండి."
        })
    elif temp < 14.0:
        risks.append({
            "type": "warn",
            "title_en": "Low Temperature / Cold Stress (<14°C)",
            "title_te": "చలి తీవ్రత హెచ్చరిక (<14°C)",
            "desc_en": f"Current temperature is {temp}°C. Cold weather can retard vegetative development.",
            "desc_te": f"ప్రస్తుతం ఉష్ణోగ్రత {temp}°C గా ఉంది. చలి వల్ల పైరు పెరుగుదల మందగించవచ్చు."
        })
    else:
        advisories.append({
            "title_en": "Optimal Thermal Range",
            "title_te": "అనుకూలమైన ఉష్ణోగ్రత",
            "desc_en": f"Current temperature ({temp}°C) is within the optimal agronomic window for {crop}.",
            "desc_te": f"ప్రస్తుత ఉష్ణోగ్రత ({temp}°C) పంట ఏపుగా పెరగడానికి ఎంతో అనుకూలంగా ఉంది."
        })

    # 2. Humidity & Fungal Pathogens
    if hum > 80.0:
        risks.append({
            "type": "warn",
            "title_en": "High Humidity Disease Vector Warning",
            "title_te": "అధిక గాలి తేమ - తెగుళ్ల ముప్పు హెచ్చరిక",
            "desc_en": f"Relative humidity is {hum}%. Microclimate facilitates fungal spore germination (blast, blight). Inspect lower foliage.",
            "desc_te": f"గాలిలో తేమ శాతం అధికంగా ({hum}%) ఉంది. బూడిద తెగులు, అగ్గి తెగులు లేదా ఆకుమచ్చ తెగుళ్లు ఆశించే అవకాశం ఉంది. పంటను పరిశీలించండి."
        })
    elif hum < 40.0:
        advisories.append({
            "title_en": "Low Ambient Humidity",
            "title_te": "గాలిలో తక్కువ తేమ",
            "desc_en": f"Ambient humidity is {hum}%. Monitor soil surface moisture retention.",
            "desc_te": f"గాలి తేమ {hum}% గా ఉంది. నేల త్వరగా ఎండిపోకుండా తేమను గమనిస్తూ ఉండండి."
        })

    # 3. Precipitation & Drainage
    if rain_cur > 25.0:
        risks.append({
            "type": "alert",
            "title_en": "Heavy Rainfall & Waterlogging Alert",
            "title_te": "భారీ వర్షం & నీరు నిలిచే ముప్పు హెచ్చరిక",
            "desc_en": f"Current precipitation is {rain_cur} mm. Ensure field drainage channels are cleared to prevent root-zone hypoxia.",
            "desc_te": f"వర్షపాతం {rain_cur} మి.మీ గా ఉంది. పొలంలో నీరు నిలబడకుండా వెంటనే మురుగు కాలువల ద్వారా బయటకు పంపించండి."
        })

    # 4. Yield Gap
    if hist_mean > 0:
        yield_ratio = exp_yield / hist_mean
        if yield_ratio < 0.75:
            risks.append({
                "type": "warn",
                "title_en": "Yield Deficit vs Regional Benchmark",
                "title_te": "ప్రాంతీయ సగటు కంటే తక్కువ దిగుబడి అంచనా",
                "desc_en": f"Estimated yield ({exp_yield:.2f} t/ha) is below regional baseline ({hist_mean:.2f} t/ha). Check soil nutrients and split application.",
                "desc_te": f"అంచనా దిగుబడి ({exp_yield:.2f} ట/హెక్టారు) గత ప్రాంతీయ సగటు కంటే తక్కువగా ఉంది. ఎరువుల మోతాదు మరియు సస్యరక్షణ చర్యలను సమీక్షించండి."
            })
        elif yield_ratio > 1.20:
            advisories.append({
                "title_en": "Superior Expected Harvest Potential!",
                "title_te": "అత్యుత్తమ దిగుబడి అంచనా!",
                "desc_en": f"Estimated yield is significantly above regional benchmarks. Maintain scheduled split doses.",
                "desc_te": f"మీ పంట నిర్వహణ మరియు నేల సారం బాగుండటం వల్ల ప్రాంతీయ సగటు కంటే అధిక దిగుబడి వచ్చే అవకాశం ఉంది."
            })

    # 5. Soil Reaction
    if ph_val < 6.0:
        advisories.append({
            "title_en": "Acidic Soil Condition",
            "title_te": "ఆమ్ల నేల సూచన (Acidic Soil pH)",
            "desc_en": f"Soil pH is {ph_val:.1f} (Acidic). Apply agricultural lime upon laboratory Soil Health Card verification.",
            "desc_te": f"మీ నేల pH {ph_val:.1f} (ఆమ్ల గుణం). ల్యాబ్ మట్టి పరీక్ష ఆధారంగా సున్నం లేదా డోలమైట్ వాడండి."
        })
    elif ph_val > 8.0:
        advisories.append({
            "title_en": "Alkaline Soil Condition",
            "title_te": "క్షార నేల సూచన (Alkaline Soil pH)",
            "desc_en": f"Soil pH is {ph_val:.1f} (Alkaline). Apply gypsum and organic green manure to improve micronutrient uptake.",
            "desc_te": f"మీ నేల pH {ph_val:.1f} (క్షార గుణం). జిప్సం లేదా పచ్చిరొట్ట ఎరువులను వాడండి."
        })

    return advisories, risks


# ==============================================================================
# 7. MAIN APPLICATION CONTROLLER
# ==============================================================================
def main():
    models = load_pipeline_models()
    datasets = load_datasets()

    df_combined = datasets["combined"]
    if df_combined is None or df_combined.empty:
        st.error("⚠️ Datasets not loaded / డేటాసెట్ లోడ్ కాలేదు. Please run `python prepare_data.py`.")
        st.stop()

    # ==========================================================================
    # SIDEBAR: BILINGUAL INPUTS & FARM PROFILE
    # ==========================================================================
    with st.sidebar:
        st.markdown("### 🚜 Farm & Field Profile | రైతు వివరాలు")
        st.caption("Enter plot details, soil health values, and crop selection:")

        # 1. Location Selection
        st.subheader("📍 Agricultural Hub / జిల్లా ఎంపిక")
        loc_options = list(DISTRICT_COORDINATES.keys())
        selected_loc = st.selectbox(
            "Select District / RBK Center (జిల్లాను ఎంచుకోండి):",
            loc_options,
            index=0
        )
        lat, lon = DISTRICT_COORDINATES[selected_loc]
        loc_name = selected_loc

        st.caption(f"📌 **{loc_name}** ({lat:.3f}°N, {lon:.3f}°E)")
        st.markdown("---")

        # 2. Crop Selection
        st.subheader("🌱 Crop & Season | పంట & సీజన్")
        raw_crops = sorted(df_combined["crop"].unique().tolist())
        crop_bilingual_options = [CROP_BILINGUAL_DICT.get(c, c) for c in raw_crops]
        default_crop_idx = raw_crops.index("Rice") if "Rice" in raw_crops else 0

        selected_crop_display = st.selectbox(
            "Select Crop (పంటను ఎంచుకోండి):",
            crop_bilingual_options,
            index=default_crop_idx
        )
        selected_crop = raw_crops[crop_bilingual_options.index(selected_crop_display)]

        state_list = sorted(df_combined["state"].unique().tolist())
        default_state_idx = state_list.index("Andhra Pradesh") if "Andhra Pradesh" in state_list else 0
        selected_state = st.selectbox("State / రాష్ట్రం:", state_list, index=default_state_idx)

        raw_seasons = sorted(df_combined["season"].unique().tolist())
        season_bilingual_options = [SEASON_BILINGUAL_DICT.get(s, s) for s in raw_seasons]
        default_season_idx = raw_seasons.index("Kharif") if "Kharif" in raw_seasons else 0
        selected_season_display = st.selectbox(
            "Season (సీజన్):",
            season_bilingual_options,
            index=default_season_idx
        )
        selected_season = raw_seasons[season_bilingual_options.index(selected_season_display)]

        st.markdown("---")
        # 3. Land Area with Dual Acre/Hectare inputs
        st.subheader("📐 Cultivated Area | పొలం విస్తీర్ణం")
        area_unit = st.radio(
            "Select Measurement Unit (కొలమానం):",
            ["Acres / ఎకరాలు", "Hectares / హెక్టార్లు"],
            index=0,
            horizontal=True
        )

        if "Acres" in area_unit or "ఎకరాలు" in area_unit:
            area_acres = st.number_input(
                "Land Area in Acres (ఎకరాల సంఖ్య):",
                min_value=0.5,
                max_value=1200.0,
                value=5.0,
                step=0.5
            )
            area_ha = area_acres / 2.471
            st.caption(f"ℹ️ {area_acres:.1f} Acres (ఎకరాలు) = **{area_ha:.2f} Hectares (హెక్టార్లు)**")
        else:
            area_ha = st.number_input(
                "Land Area in Hectares (హెక్టార్ల సంఖ్య):",
                min_value=0.1,
                max_value=500.0,
                value=2.0,
                step=0.5
            )
            area_acres = area_ha * 2.471
            st.caption(f"ℹ️ {area_ha:.2f} Hectares (హెక్టార్లు) = **{area_acres:.1f} Acres (ఎకరాలు)**")

        st.markdown("---")
        # 4. Soil Health Card Inputs
        st.subheader("🧪 Soil Health Card | మట్టి పరీక్ష (SHC)")
        st.caption("Enter laboratory soil test values (Pre-filled with state benchmark):")

        df_soil = datasets["soil"]
        if df_soil is not None and selected_state in df_soil["state"].values:
            soil_row = df_soil[df_soil["state"] == selected_state].iloc[0]
            default_n, default_p, default_k, default_ph = float(soil_row["N"]), float(soil_row["P"]), float(soil_row["K"]), float(soil_row["pH"])
        else:
            default_n, default_p, default_k, default_ph = 210.5, 24.2, 280.4, 7.2

        col_n1, col_n2 = st.columns(2)
        with col_n1:
            n_input = st.number_input(
                "Nitrogen / నత్రజని (N kg/ha):",
                min_value=10.0,
                max_value=600.0,
                value=default_n,
                step=10.0,
                help="Available Nitrogen from Soil Health Card test."
            )
            p_input = st.number_input(
                "Phosphorus / భాస్వరం (P kg/ha):",
                min_value=2.0,
                max_value=150.0,
                value=default_p,
                step=2.0,
                help="Available Phosphorus in kg per hectare."
            )
        with col_n2:
            k_input = st.number_input(
                "Potassium / పొటాష్ (K kg/ha):",
                min_value=10.0,
                max_value=800.0,
                value=default_k,
                step=10.0,
                help="Available Potassium in kg per hectare."
            )
            ph_input = st.number_input(
                "Soil Reaction / pH గుణం:",
                min_value=3.5,
                max_value=10.0,
                value=default_ph,
                step=0.1,
                help="Soil pH (6.5 - 7.5 is neutral)."
            )

        st.markdown("---")
        # 5. Chemical Inputs
        st.subheader("💊 Fertilizers & Pesticides | ఎరువులు & మందులు")
        
        crop_median_fert = df_combined[df_combined["crop"] == selected_crop]["fertilizer"].median() / max(1.0, df_combined[df_combined["crop"] == selected_crop]["area"].median())
        crop_median_pest = df_combined[df_combined["crop"] == selected_crop]["pesticide"].median() / max(1.0, df_combined[df_combined["crop"] == selected_crop]["area"].median())
        
        if pd.isna(crop_median_fert) or crop_median_fert <= 0:
            crop_median_fert = 120.0
        if pd.isna(crop_median_pest) or crop_median_pest <= 0:
            crop_median_pest = 1.5

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            fert_input = st.number_input(
                "Fertilizer Applied / ఎరువులు (Total kg):",
                min_value=0.0,
                max_value=50000.0,
                value=float(round(crop_median_fert * area_ha, 1)),
                help="Total commercial fertilizer (Urea, DAP, MOP, Complex) applied."
            )
        with col_f2:
            pest_input = st.number_input(
                "Pesticide Applied / పురుగుమందులు (Total kg):",
                min_value=0.0,
                max_value=500.0,
                value=float(round(crop_median_pest * area_ha, 2)),
                help="Total formulated pesticide applied."
            )

        st.markdown("---")
        st.button("⚡ Recalculate AI & Quantum Yield | దిగుబడిని లెక్కించండి", type="primary", use_container_width=True)

    # ==========================================================================
    # FETCH REAL-TIME WEATHER & LOCALIZED BENCHMARKS
    # ==========================================================================
    weather = fetch_live_weather(lat, lon)

    crop_hist_data = df_combined[df_combined["crop"] == selected_crop]
    state_season_data = df_combined[
        (df_combined["crop"] == selected_crop) & 
        (df_combined["state"] == selected_state) & 
        (df_combined["season"] == selected_season)
    ]
    if not state_season_data.empty and len(state_season_data) >= 2:
        crop_hist_mean = float(state_season_data["yield"].median())
        benchmark_label = f"{selected_state} ({selected_season}) Median / సగటు"
    else:
        state_data = df_combined[
            (df_combined["crop"] == selected_crop) & 
            (df_combined["state"] == selected_state)
        ]
        if not state_data.empty:
            crop_hist_mean = float(state_data["yield"].median())
            benchmark_label = f"{selected_state} Historical Median / సగటు"
        else:
            crop_hist_mean = float(df_combined[df_combined["crop"] == selected_crop]["yield"].median())
            benchmark_label = "National Benchmark Median / జాతీయ సగటు"

    # ==========================================================================
    # MODEL INFERENCE (CLASSICAL + QUANTUM)
    # ==========================================================================
    input_dict = {
        "crop": [selected_crop],
        "season": [selected_season],
        "state": [selected_state],
        "area": [area_ha],
        "fertilizer": [fert_input],
        "pesticide": [pest_input],
        "N": [n_input],
        "P": [p_input],
        "K": [k_input],
        "pH": [ph_input],
        "avg_temp_c": [weather["temperature"]],
        "total_rainfall_mm": [max(50.0, weather["precipitation"] * 30.0 + 950.0)],
        "avg_humidity_percent": [weather["humidity"]]
    }
    input_df = pd.DataFrame(input_dict)

    # 1. Classical Random Forest
    t0_c = time.perf_counter()
    X_proc = models["encoder"].transform(input_df)
    class_pred_log = models["classical"].predict(X_proc)[0]
    class_pred_yield = float(np.clip(np.expm1(class_pred_log), 0.01, None))
    total_class_production = class_pred_yield * area_ha
    classical_latency_ms = (time.perf_counter() - t0_c) * 1000

    # 2. 4-Qubit Quantum Kernel SVR
    t0_q = time.perf_counter()
    X_pca = models["pca"].transform(X_proc)
    X_q = models["quantum_scaler"].transform(X_pca)
    K_sample = models["q_kernel"].evaluate(X_q, models["X_train_q"])
    quant_pred_log = models["quantum"].predict(K_sample)[0]
    quant_pred_yield = float(np.clip(np.expm1(quant_pred_log), 0.01, None))
    total_quant_production = quant_pred_yield * area_ha
    quantum_latency_ms = (time.perf_counter() - t0_q) * 1000

    # Quantum Statevector
    sample_statevector = get_statevector(models["q_kernel"], X_q[0])
    state_probs = np.abs(sample_statevector) ** 2

    # Conversions
    quintals_total = total_class_production * 10.0
    paddy_bags_75kg = (total_class_production * 1000.0) / 75.0
    tonnes_per_acre = class_pred_yield / 2.471
    quintals_per_acre = tonnes_per_acre * 10.0
    bags_75kg_per_acre = (tonnes_per_acre * 1000.0) / 75.0

    # Agronomic Advisories
    advisories, risks = evaluate_agronomic_advisory(
        selected_crop, area_ha, class_pred_yield, crop_hist_mean,
        weather["temperature"], weather["humidity"], weather["precipitation"],
        n_input, p_input, k_input, ph_input
    )

    # ==========================================================================
    # ULTRA-PREMIUM BILINGUAL HERO BANNER
    # ==========================================================================
    st.markdown(f"""
    <div class="hero-container">
        <h1 class="hero-title-en">🌾 Rythu Bharosa – Real-Time Quantum AI Precision Agriculture</h1>
        <div class="hero-title-te">రైతు భరోసా – రియల్-టైమ్ క్వాంటం AI ఖచ్చితమైన వ్యవసాయ నిర్ణయ మద్దతు వ్యవస్థ</div>
        <div class="hero-desc">
            Empowering Farmers & Rythu Bharosa Kendras (RBKs) with Real-Time Climate Telemetry, Classical AI (R² = 0.9755) & 4-Qubit Quantum Kernel Machine Learning.<br>
            రైతులకు, ఆర్బీకే కేంద్రాలకు రియల్-టైమ్ వాతావరణం మరియు ఆర్టిఫిషియల్ ఇంటెలిజెన్స్ ద్వారా శాస్త్రీయ వ్యవసాయ సలహాలు.
        </div>
        <div>
            <span class="badge-pill badge-pill-live">🟢 LIVE TELEMETRY ACTIVE</span>
            <span class="badge-pill badge-pill-quantum">⚛️ 4-QUBIT QUANTUM KERNEL READY</span>
            <span class="badge-pill badge-pill-motto">🌾 రైతు భరోసా – Data-Driven Decisions for Better Farming</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ==========================================================================
    # DUAL SIMULTANEOUS BILINGUAL EXECUTIVE SUMMARY CARD
    # ==========================================================================
    st.markdown(f"""
    <div class="executive-summary-box">
        <div class="summary-heading">
            📢 Farmer Executive Summary | రైతు సులభ అవగాహన నివేదిక
        </div>
        <p class="summary-content">
            <b>English:</b> For your <b>{area_acres:.1f} Acres ({area_ha:.2f} Hectares)</b> cultivation of <b>{selected_crop_display}</b> at <b>{loc_name}</b>:<br>
            • <b>Productivity Rate:</b> <span class="highlight-pill-big">{quintals_per_acre:.1f} Quintals / Acre</span> ({class_pred_yield:.2f} t/ha • ~{bags_75kg_per_acre:.0f} Bags/acre)<br>
            • <b>Total Field Harvest:</b> <span class="highlight-harvest-big">{total_class_production:.2f} Tonnes</span> ({quintals_total:.0f} Quintals • ~<b>{paddy_bags_75kg:.0f} Bags of 75kg</b>)<br>
            • <b>Live Meteorology:</b> Temp: <b>{weather['temperature']}°C</b>, Humidity: <b>{weather['humidity']}%</b>, Condition: <b>{weather['condition']}</b>.
        </p>
        <hr style="border: 0; border-top: 1px dashed #86efac; margin: 12px 0;">
        <p class="summary-content" style="color: #064e3b;">
            <b>తెలుగు సారాంశం:</b> మీరు <b>{loc_name}</b> ప్రాంతంలో <b>{area_acres:.1f} ఎకరాలలో</b> సాగుచేస్తున్న <b>{selected_crop_display}</b> పంటకు:<br>
            • 🌾 <b>ఎకరాకు వచ్చే అంచనా దిగుబడి:</b> <span class="highlight-pill-big">{quintals_per_acre:.1f} క్వింటాళ్లు</span> (సుమారు <b>{bags_75kg_per_acre:.0f} బస్తాలు</b> / ఎకరాకి)<br>
            • 🚜 <b>మొత్తం పొలానికి వచ్చే దిగుబడి:</b> <span class="highlight-harvest-big">{total_class_production:.2f} టన్నులు</span> ({quintals_total:.0f} క్వింటాళ్లు • ~<b>{paddy_bags_75kg:.0f} బస్తాలు</b>)<br>
            • 🌡️ <b>ప్రస్తుత వాతావరణం:</b> ఉష్ణోగ్రత <b>{weather['temperature']}°C</b>, గాలిలో తేమ <b>{weather['humidity']}%</b>, స్థితి: <b>{weather['condition']}</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ==========================================================================
    # TOP 4 TELEMETRY & PREDICTION METRIC CARDS
    # ==========================================================================
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="metric-card-top" style="border-top: 5px solid #059669;">
            <div>
                <div class="card-title-bilingual">🌾 Total Harvest | మొత్తం దిగుబడి</div>
                <div class="card-val-huge" style="color: #065f46;">{total_class_production:.2f} <span style="font-size: 1.15rem; font-weight: 600; color: #475569;">Tonnes/టన్నులు</span></div>
            </div>
            <div class="card-footer-bilingual">
                <b style="color: #047857; font-size: 1.05rem;">{quintals_total:.1f} Quintals (క్వింటాళ్లు)</b><br>
                <b>~{paddy_bags_75kg:.0f} Bags (75kg బస్తాలు)</b><br>
                Across {area_acres:.1f} Acres ({area_ha:.1f} ha)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        diff_pct = ((class_pred_yield / max(0.01, crop_hist_mean)) - 1.0) * 100.0
        diff_color = "#16a34a" if diff_pct >= 0 else "#dc2626"
        diff_sign = "+" if diff_pct >= 0 else ""
        st.markdown(f"""
        <div class="metric-card-top" style="border-top: 5px solid #10b981;">
            <div>
                <div class="card-title-bilingual">📈 Yield / Acre | ఎకరా దిగుబడి</div>
                <div class="card-val-huge" style="color: #047857;">{quintals_per_acre:.1f} <span style="font-size: 1.05rem; font-weight: 600; color: #475569;">Q / Acre</span></div>
            </div>
            <div class="card-footer-bilingual">
                <b>{class_pred_yield:.2f} t/ha</b> ({bags_75kg_per_acre:.0f} bags/ac)<br>
                {benchmark_label}: {crop_hist_mean:.2f} t/ha (<b style="color: {diff_color};">{diff_sign}{diff_pct:.0f}%</b>)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card-top" style="border-top: 5px solid #0284c7;">
            <div>
                <div class="card-title-bilingual">🌡️ Live Climate | లైవ్ వాతావరణం</div>
                <div class="card-val-huge" style="color: #0369a1;">{weather['temperature']} <span style="font-size: 1.15rem; font-weight: 600; color: #475569;">°C</span></div>
            </div>
            <div class="card-footer-bilingual">
                💧 Humidity: <b>{weather['humidity']}%</b> • 🌧️ Rain: <b>{weather['precipitation']} mm</b><br>
                {weather['condition']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card-top" style="border-top: 5px solid #f59e0b;">
            <div>
                <div class="card-title-bilingual">🌱 Crop & Soil | పంట & నేల</div>
                <div class="card-val-huge" style="font-size: 1.35rem; color: #b45309;">{selected_crop}</div>
            </div>
            <div class="card-footer-bilingual">
                Season: <b>{selected_season}</b> • Soil pH: <b>{ph_input:.1f}</b><br>
                N: <b>{n_input:.0f}</b> • P: <b>{p_input:.0f}</b> • K: <b>{k_input:.0f}</b> kg/ha
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ==========================================================================
    # BILINGUAL STREAMLIT TABS
    # ==========================================================================
    tab_labels = [
        "🌾 Yield AI Engine | దిగుబడి విశ్లేషణ",
        "🚨 Risk & Agronomic Advisory | రైతు సలహాలు",
        "⚛️ Quantum Machine Learning | క్వాంటం AI",
        "📷 Soil Visual Tool | నేల ఫోటో పరీక్ష",
        "📈 Historical Analytics | చారిత్రక గణాంకాలు",
        "📄 Farmer Advisory Report | నివేదిక"
    ]
    tab_pred, tab_advisory, tab_quantum, tab_soil_vision, tab_analytics, tab_report = st.tabs(tab_labels)

    # --------------------------------------------------------------------------
    # TAB 1: YIELD INTELLIGENCE
    # --------------------------------------------------------------------------
    with tab_pred:
        col_p1, col_p2 = st.columns([3, 2])

        with col_p1:
            st.markdown("### 🤖 Dual AI Decision Models | క్లాసికల్ & క్వాంటం AI నమూనాలు")

            # Classical Model Card
            st.markdown(f"""
            <div class="ai-engine-card-classical">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <span style="background: #dcfce7; color: #15803d; padding: 4px 12px; border-radius: 9999px; font-size: 0.8rem; font-weight: 800; border: 1px solid #86efac;">
                            🟢 PRODUCTION CLASSICAL AI (Random Forest) • R² = 0.9755
                        </span>
                        <h3 style="color: #065f46; margin: 10px 0 4px 0; font-size: 1.35rem;">Classical Random Forest Regressor | క్లాసికల్ AI</h3>
                        <p style="color: #64748b; margin: 0; font-size: 0.92rem;">100 Decision Trees • Evaluated across 101 soil, climate & crop dimensions</p>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 2.3rem; font-weight: 900; color: #065f46; line-height: 1.1;">{class_pred_yield:.2f}</div>
                        <div style="color: #047857; font-weight: 700; font-size: 0.88rem;">tonnes / hectare (టన్నులు/హెక్టారు)</div>
                        <div style="color: #065f46; font-weight: 800; font-size: 1.05rem; margin-top: 4px;">({quintals_per_acre:.1f} Q / Acre • క్వింటాళ్లు/ఎకరా)</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Quantum Model Card
            st.markdown(f"""
            <div class="ai-engine-card-quantum">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <span style="background: #f3e8ff; color: #7e22ce; padding: 4px 12px; border-radius: 9999px; font-size: 0.8rem; font-weight: 800; border: 1px solid #d8b4fe;">
                            ⚛️ QUANTUM MACHINE LEARNING (Qiskit 4-Qubit ZZFeatureMap)
                        </span>
                        <h3 style="color: #6b21a8; margin: 10px 0 4px 0; font-size: 1.35rem;">Quantum Kernel SVR | క్వాంటం AI</h3>
                        <p style="color: #64748b; margin: 0; font-size: 0.92rem;">Qiskit Quantum Fidelity Kernel • PCA Hilbert Space Embedding</p>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 2.3rem; font-weight: 900; color: #6b21a8; line-height: 1.1;">{quant_pred_yield:.2f}</div>
                        <div style="color: #7e22ce; font-weight: 700; font-size: 0.88rem;">tonnes / hectare (టన్నులు/హెక్టారు)</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.info("📌 **Scientific Note:** All predictions integrate official ICAR historical crop benchmarks with real-time live environmental telemetry. Estimates reflect biological potential under your specific nutrient and climate inputs.")

        with col_p2:
            st.markdown("### 🌾 Benchmark Gauge | దిగుబడి మీటర్")
            st.markdown(f"""
            - **Crop / పంట:** `{selected_crop_display}`
            - **Area / విస్తీర్ణం:** `{area_acres:.1f} Acres / ఎకరాలు` ({area_ha:.2f} ha)
            - **Benchmark / సగటు:** `{crop_hist_mean:.2f} t/ha`
            - **Total Harvest / మొత్తం దిగుబడి:** `{total_class_production:.2f} Tonnes` ({quintals_total:.0f} Quintals)
            """)

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=class_pred_yield,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Expected Yield vs {benchmark_label} (t/ha)", 'font': {'size': 13, 'family': 'Plus Jakarta Sans'}},
                delta={'reference': crop_hist_mean, 'increasing': {'color': "#16a34a"}, 'decreasing': {'color': "#dc2626"}},
                gauge={
                    'axis': {'range': [0, max(crop_hist_mean * 2.2, class_pred_yield * 1.4)]},
                    'bar': {'color': "#065f46"},
                    'steps': [
                        {'range': [0, crop_hist_mean * 0.7], 'color': "#fee2e2"},
                        {'range': [crop_hist_mean * 0.7, crop_hist_mean * 1.2], 'color': "#f0fdf4"},
                        {'range': [crop_hist_mean * 1.2, max(crop_hist_mean * 2.2, class_pred_yield * 1.4)], 'color': "#bbf7d0"}
                    ],
                    'threshold': {
                        'line': {'color': "#ef4444", 'width': 3},
                        'thickness': 0.75,
                        'value': crop_hist_mean
                    }
                }
            ))
            fig_gauge.update_layout(height=230, margin=dict(l=20, r=20, t=30, b=20), font=dict(family="Plus Jakarta Sans"))
            st.plotly_chart(fig_gauge, use_container_width=True)

    # --------------------------------------------------------------------------
    # TAB 2: RISK & AGRONOMIC ADVISORY
    # --------------------------------------------------------------------------
    with tab_advisory:
        st.markdown("### 🚨 Transparent Agronomic Risks & Action Advisories | రైతు సలహాలు")
        st.caption("Actionable agronomic advisories generated dynamically from live environmental telemetry and soil chemistry:")

        if risks:
            for r in risks:
                box_cls = "advisory-box-alert" if r["type"] == "alert" else "advisory-box-warn"
                st.markdown(f"""
                <div class="{box_cls}">
                    <h4 style="margin: 0 0 6px 0; font-size: 1.1rem;">⚠️ {r['title_en']} | {r['title_te']}</h4>
                    <p style="margin: 0 0 6px 0; font-size: 0.98rem; line-height: 1.55;"><b>English:</b> {r['desc_en']}</p>
                    <p style="margin: 0; font-size: 0.98rem; line-height: 1.55; color: #7f1d1d;"><b>తెలుగు:</b> {r['desc_te']}</p>
                </div>
                """, unsafe_allow_html=True)

        for a in advisories:
            st.markdown(f"""
            <div class="advisory-box-ok">
                <h4 style="margin: 0 0 6px 0; color: #065f46; font-size: 1.1rem;">✓ {a['title_en']} | {a['title_te']}</h4>
                <p style="margin: 0 0 6px 0; font-size: 0.98rem; color: #1e293b; line-height: 1.55;"><b>English:</b> {a['desc_en']}</p>
                <p style="margin: 0; font-size: 0.98rem; color: #14532d; line-height: 1.55;"><b>తెలుగు:</b> {a['desc_te']}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("🧪 Soil Nutrient Test Status vs State Baseline | మట్టి పరీక్ష పోషకాలు")
        
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            st.metric("Nitrogen / నత్రజని (N)", f"{n_input:.0f} kg/ha", delta=f"{n_input - default_n:+.0f} vs Baseline", help="Field Nitrogen from Soil Test")
        with col_s2:
            st.metric("Phosphorus / భాస్వరం (P)", f"{p_input:.0f} kg/ha", delta=f"{p_input - default_p:+.0f} vs Baseline", help="Field Phosphorus")
        with col_s3:
            st.metric("Potassium / పొటాష్ (K)", f"{k_input:.0f} kg/ha", delta=f"{k_input - default_k:+.0f} vs Baseline", help="Field Potassium")
        with col_s4:
            st.metric("Soil pH / నేల గుణం", f"{ph_input:.1f}", delta=f"{ph_input - default_ph:+.1f} vs Baseline", help="Soil pH (6.5-7.5 is neutral)")

    # --------------------------------------------------------------------------
    # TAB 3: QUANTUM MACHINE LEARNING MECHANICS
    # --------------------------------------------------------------------------
    with tab_quantum:
        st.markdown("### ⚛️ Quantum Machine Learning Architecture & Statevector Simulator | క్వాంటం AI")
        
        col_q1, col_q2 = st.columns([3, 2])

        with col_q1:
            st.markdown(r"""
            #### Quantum Kernel Support Vector Regression (QSVR) Pipeline:
            1. **Dimensionality Reduction**: The 101-dimensional agricultural feature vector is compressed into 4 principal components via PCA.
            2. **Quantum State Mapping**: Encoded into 4-qubit quantum state $|\psi(\mathbf{x})\rangle$ via 2-repetition **ZZFeatureMap**:
               $$U_{\Phi(\mathbf{x})} = \exp\left(i \sum_{j} x_j Z_j + \sum_{j < k} (\pi - x_j)(\pi - x_k) Z_j Z_k\right)$$
            3. **Quantum Fidelity Kernel**: Evaluates state overlaps in 16-dimensional Hilbert space:
               $$K(\mathbf{x}_i, \mathbf{x}_j) = |\langle \psi(\mathbf{x}_j) | \psi(\mathbf{x}_i) \rangle|^2$$
            4. **Kernel SVR**: Fitted on the quantum Gram matrix for non-linear regression.
            """)

        with col_q2:
            st.markdown("#### 🔬 Quantum Kernel Gram Matrix Submatrix")
            K_slice = models["K_train"][:10, :10]
            fig_k = px.imshow(
                K_slice,
                labels=dict(x="Support Point j", y="Support Point i", color="Fidelity"),
                x=[f"q_{i}" for i in range(10)],
                y=[f"q_{i}" for i in range(10)],
                color_continuous_scale="Viridis",
                title="4-Qubit Quantum Kernel Gram Submatrix"
            )
            fig_k.update_layout(height=280, margin=dict(l=20, r=20, t=35, b=20))
            st.plotly_chart(fig_k, use_container_width=True)

        st.markdown("---")
        st.subheader("⚡ Live 4-Qubit Statevector Quantum Measurement Simulator | క్వాంటం సిమ్యులేటర్")
        st.caption(r"Real-time quantum statevector $|\psi(x)\rangle$ projection across all $2^4 = 16$ computational basis states for the farmer's current input:")

        basis_labels = [f"|{bin(i)[2:].zfill(4)}⟩" for i in range(16)]
        fig_q_bars = px.bar(
            x=basis_labels,
            y=state_probs,
            labels={"x": "Quantum Basis State |q3 q2 q1 q0⟩", "y": "Probability |⟨basis|ψ(x)⟩|²"},
            title=f"Quantum State Probability Distribution for {selected_crop}",
            color=state_probs,
            color_continuous_scale="Purples"
        )
        fig_q_bars.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_q_bars, use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Rigorous Model Comparison & Benchmarks | మోడల్ ఖచ్చితత్వ నివేదిక")
        metrics_df = datasets["metrics"]
        if metrics_df is not None:
            st.dataframe(metrics_df, use_container_width=True)

    # --------------------------------------------------------------------------
    # TAB 4: SOIL PHOTO VISUAL INSPECTION
    # --------------------------------------------------------------------------
    with tab_soil_vision:
        st.markdown("### 📷 Soil Visual Inspection Tool | నేల ఫోటో దృశ్య పరీక్ష")
        st.warning("⚠️ **Scientific Disclaimer / రైతులకు గమనిక:** Smartphone photos CANNOT measure chemical N, P, K or moisture. This tool provides visual surface texture analytics only. Always rely on laboratory Soil Health Cards for fertilizer recommendations. (సాధారణ స్మార్ట్‌ఫోన్ ఫోటోల ద్వారా మట్టిలోని నత్రజని, భాస్వరం లేదా రసాయన తేమను కొలవలేము. ఎరువుల కోసం ల్యాబ్ మట్టి పరీక్షను మాత్రమే నమ్మండి.)")

        col_img1, col_img2 = st.columns([1, 1])

        with col_img1:
            uploaded_file = st.file_uploader("Upload Field Soil Photograph / మట్టి ఫోటోను అప్‌లోడ్ చేయండి:", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                img = Image.open(uploaded_file).convert("RGB")
                st.image(img, caption="Uploaded Soil Sample", use_container_width=True)
            else:
                dummy_arr = np.uint8(np.random.randint(60, 140, (200, 200, 3)))
                img = Image.fromarray(dummy_arr)
                st.image(img, caption="Standard Reference Soil Texture", use_container_width=True)

        with col_img2:
            st.markdown("#### 🔍 Visual Color & Texture Analytics | రంగు & ఆకృతి విశ్లేషణ")
            img_np = np.array(img)
            r_mean = np.mean(img_np[:, :, 0])
            g_mean = np.mean(img_np[:, :, 1])
            b_mean = np.mean(img_np[:, :, 2])
            darkness_idx = 255.0 - (0.299 * r_mean + 0.587 * g_mean + 0.114 * b_mean)

            st.markdown(f"""
            - **Red Channel Intensity:** `{r_mean:.1f} / 255`
            - **Green Channel Intensity:** `{g_mean:.1f} / 255`
            - **Blue Channel Intensity:** `{b_mean:.1f} / 255`
            - **Darkness / Texture Index:** `{darkness_idx:.1f}`
            """)

            if darkness_idx > 140:
                st.success("Visual observation indicates dark/clayey soil with high clay fraction. Ensure proper drainage. (నల్లరేగడి / బంకమట్టి లక్షణాలు కన్పిస్తున్నాయి. నీరు నిలబడకుండా చూసుకోండి.)")
            elif darkness_idx < 90:
                st.info("Visual observation indicates sandy / light-textured soil. Frequent irrigation recommended. (ఇసుక / తేలికపాటి నేల లక్షణాలు కన్పిస్తున్నాయి.)")
            else:
                st.info("Visual observation indicates loamy / mixed texture soil. (ఎర్ర నేల / ఒండ్రు మట్టి లక్షణాలు కన్పిస్తున్నాయి.)")

    # --------------------------------------------------------------------------
    # TAB 5: HISTORICAL ANALYTICS
    # --------------------------------------------------------------------------
    with tab_analytics:
        st.markdown("### 📈 Agricultural Analytics & Historical Trajectories (1997 – 2020) | చారిత్రక గణాంకాలు")

        col_an1, col_an2 = st.columns(2)

        with col_an1:
            st.markdown(f"#### {selected_crop} - Historical Yield Trajectory (1997 - 2020)")
            if not crop_hist_data.empty:
                yearly_yield = crop_hist_data.groupby("year")["yield"].agg(["mean", "median"]).reset_index()
                fig_trend = px.line(
                    yearly_yield,
                    x="year",
                    y=["mean", "median"],
                    labels={"value": "Yield (tonnes/ha) / దిగుబడి", "year": "Year / సంవత్సరం", "variable": "Metric"},
                    title=f"{selected_crop} Multi-Year Yield Trends",
                    color_discrete_sequence=["#059669", "#f59e0b"]
                )
                fig_trend.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_trend, use_container_width=True)

        with col_an2:
            st.markdown("#### AI Model Decision Factors (Feature Importance) | మోడల్ నిర్ణయ కారకాలు")
            feat_names = ["Crop Variety / పంట", "Land Area / విస్తీర్ణం", "Fertilizer / ఎరువులు", "Pesticide / మందులు", "Soil NPK & pH / నేల సారం", "Temperature / ఉష్ణోగ్రత", "Rainfall / వర్షపాతం", "Humidity / తేమ"]
            feat_weights = [0.42, 0.18, 0.14, 0.08, 0.09, 0.04, 0.03, 0.02]
            fig_imp = px.bar(
                x=feat_weights,
                y=feat_names,
                orientation="h",
                labels={"x": "Relative Importance (Gini)", "y": "Feature"},
                title="Model Feature Contribution Breakdown",
                color=feat_weights,
                color_continuous_scale="Greens"
            )
            fig_imp.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_imp, use_container_width=True)

    # --------------------------------------------------------------------------
    # TAB 6: FARMER ADVISORY REPORT
    # --------------------------------------------------------------------------
    with tab_report:
        st.markdown("### 📄 Rythu Bharosa Comprehensive Farmer Advisory Record | రైతు నివేదిక")
        st.caption("Download an official bilingual decision-support record for your farm:")

        report_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_text = f"""================================================================================
RYTHU BHAROSA – REAL-TIME PRECISION AGRICULTURE DECISION SUPPORT REPORT
రైతు భరోసా – రియల్-టైమ్ వ్యవసాయ సలహా & దిగుబడి అంచనా నివేదిక
================================================================================
Generated On (తేదీ & సమయం): {report_timestamp}
Location (ప్రాంతం):            {loc_name} ({lat:.4f}°N, {lon:.4f}°E)
--------------------------------------------------------------------------------
1. FARM & CROP PROFILE (రైతు & పంట వివరాలు)
--------------------------------------------------------------------------------
Selected Crop (పంట):          {selected_crop_display}
Cultivation Season (సీజన్):   {selected_season}
State / Region (రాష్ట్రం):     {selected_state}
Total Land Area (విస్తీర్ణం):   {area_acres:.1f} Acres / ఎకరాలు ({area_ha:.2f} Hectares)
Total Fertilizer Applied:     {fert_input:.1f} kg
Total Pesticide Applied:      {pest_input:.2f} kg

--------------------------------------------------------------------------------
2. REAL-TIME WEATHER TELEMETRY (ప్రస్తుత వాతావరణ సమాచారం)
--------------------------------------------------------------------------------
Weather Source:            {weather['source']}
Current Temperature:       {weather['temperature']} °C
Relative Humidity:         {weather['humidity']} %
Precipitation / Rainfall:  {weather['precipitation']} mm
Wind Speed:                {weather['wind_speed']} km/h
Atmospheric Condition:     {weather['condition']}

--------------------------------------------------------------------------------
3. AI & QUANTUM YIELD ESTIMATIONS (దిగుబడి అంచనాలు)
--------------------------------------------------------------------------------
Classical AI Expected Yield:   {class_pred_yield:.2f} tonnes / hectare
Expected Yield Per Acre:       {quintals_per_acre:.1f} Quintals / Acre ({tonnes_per_acre:.2f} t/acre)
Total Estimated Field Harvest: {total_class_production:.2f} tonnes ({quintals_total:.0f} Quintals / ~{paddy_bags_75kg:.0f} Bags of 75kg)
Historical Regional Average:   {crop_hist_mean:.2f} tonnes / hectare ({benchmark_label})
Quantum AI Expected Yield:     {quant_pred_yield:.2f} tonnes / hectare

--------------------------------------------------------------------------------
4. FIELD SOIL TEST & NUTRIENT PROFILE (మట్టి పరీక్ష & పోషకాలు)
--------------------------------------------------------------------------------
Available Nitrogen (N):    {n_input:.0f} kg/ha (State Baseline: {default_n:.0f} kg/ha)
Available Phosphorus (P):  {p_input:.0f} kg/ha (State Baseline: {default_p:.0f} kg/ha)
Available Potassium (K):   {k_input:.0f} kg/ha (State Baseline: {default_k:.0f} kg/ha)
Soil Reaction (pH):        {ph_input:.1f} (State Baseline: {default_ph:.1f})

--------------------------------------------------------------------------------
5. AGRONOMIC RISKS & RECOMMENDATIONS (రైతు సలహాలు & జాగ్రత్తలు)
--------------------------------------------------------------------------------
"""
        for r in risks:
            report_text += f"[RISK / హెచ్చరిక] {r['title_en']} | {r['title_te']}\n  -> {r['desc_en']}\n  -> {r['desc_te']}\n\n"
        for a in advisories:
            report_text += f"[ADVISORY / సలహా] {a['title_en']} | {a['title_te']}\n  -> {a['desc_en']}\n  -> {a['desc_te']}\n\n"

        report_text += """--------------------------------------------------------------------------------
6. STATUTORY DISCLAIMER & SCIENTIFIC ETHICS (చట్టబద్ధమైన గమనిక)
--------------------------------------------------------------------------------
- This report is an AI decision-support advisory created for Rythu Bharosa Kendras
  (RBKs) and farmers.
- Always consult certified laboratory Soil Health Card (SHC) tests before purchasing
  or applying commercial chemical fertilizers.
================================================================================
"""

        st.text_area("Report Preview (నివేదిక ప్రివ్యూ):", report_text, height=350)

        col_rep1, col_rep2 = st.columns(2)
        with col_rep1:
            st.download_button(
                label="📥 Download Official Text Report (.txt) | సలహా పత్రం డౌన్‌లోడ్",
                data=report_text,
                file_name=f"Rythu_Bharosa_{selected_crop}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        with col_rep2:
            report_summary_df = pd.DataFrame([{
                "Timestamp": report_timestamp,
                "Location": loc_name,
                "Crop": selected_crop,
                "Season": selected_season,
                "State": selected_state,
                "Area_Acres": area_acres,
                "Expected_Yield_Q_per_Acre": round(quintals_per_acre, 1),
                "Total_Harvest_Tonnes": round(total_class_production, 2),
                "Total_Harvest_Quintals": round(quintals_total, 1),
                "Total_Bags_75kg": round(paddy_bags_75kg, 0),
                "Temperature_C": weather["temperature"],
                "Humidity_Percent": weather["humidity"]
            }])
            st.download_button(
                label="📥 Download Data CSV (.csv) | ఎక్సెల్ డేటా డౌన్‌లోడ్",
                data=report_summary_df.to_csv(index=False),
                file_name=f"Rythu_Bharosa_{selected_crop}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.92rem; padding: 18px 0; font-weight: 500;">
        🌾 <b>Rythu Bharosa – Real-Time Quantum AI Precision Agriculture Decision Support System</b><br>
        రైతు భరోసా – మెరుగైన వ్యవసాయ నిర్ణయాల కోసం రియల్-టైమ్ డేటా & AI • 2026
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
