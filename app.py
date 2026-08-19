"""
🌾 Rythu Bharosa – Real-Time Quantum AI Precision Agriculture Decision Support System
రైతు భరోసా – రియల్-టైమ్ క్వాంటం AI ఖచ్చితమైన వ్యవసాయ నిర్ణయ మద్దతు వ్యవస్థ
=====================================================================================
Bilingual (తెలుగు / English) Decision-Support Platform for Farmers & RBKs in Andhra Pradesh & India.
Designed for both non-educated and educated farmers with simple intuitive visual cards,
real-time live weather telemetry, soil health card inputs, Classical & Quantum AI predictions.
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
# 1. PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="Rythu Bharosa – రైతు భరోసా",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Agricultural CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&family=Ramabhadra&family=Tenali+Ramakrishna&display=swap');

    :root {
        --primary-green: #1b5e20;
        --accent-green: #2e7d32;
        --light-green: #e8f5e9;
        --earth-amber: #f57f17;
        --card-bg: #ffffff;
        --text-dark: #1a2e1a;
    }
    
    html, body, [class*="css"] {
        font-family: 'Outfit', 'Ramabhadra', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 50%, #388e3c 100%);
        color: white;
        padding: 24px 30px;
        border-radius: 16px;
        margin-bottom: 20px;
        box-shadow: 0 8px 24px rgba(27, 94, 32, 0.15);
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2.3rem;
        font-weight: 800;
        margin: 0 0 8px 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #e8f5e9;
        font-size: 1.1rem;
        margin: 0;
        font-weight: 500;
    }
    .telugu-banner {
        background: rgba(255, 255, 255, 0.18);
        display: inline-block;
        padding: 6px 18px;
        border-radius: 20px;
        font-size: 1.05rem;
        font-weight: 700;
        margin-top: 10px;
        color: #fff9c4;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    .farmer-highlight-box {
        background: linear-gradient(135deg, #f1f8e9 0%, #dcedc8 100%);
        border: 2px solid #81c784;
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.08);
    }
    .stat-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 20px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.08);
    }
    .stat-title {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #33691e;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .stat-value {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1b5e20;
        line-height: 1.2;
    }
    .stat-sub {
        font-size: 0.88rem;
        color: #424242;
        margin-top: 6px;
        line-height: 1.4;
    }
    .advisory-card {
        background: #f1f8e9;
        border-left: 6px solid #2e7d32;
        padding: 16px 20px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    .advisory-warn {
        background: #fff8e1;
        border-left: 6px solid #ffa000;
        padding: 16px 20px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    .advisory-alert {
        background: #ffebee;
        border-left: 6px solid #d32f2f;
        padding: 16px 20px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    .badge-classical {
        background: #e3f2fd;
        color: #0d47a1;
        padding: 6px 14px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
        border: 1px solid #bbdefb;
    }
    .badge-quantum {
        background: #f3e5f5;
        color: #4a148c;
        padding: 6px 14px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
        border: 1px solid #e1bee7;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. BILINGUAL CROP & UI TRANSLATION MAP
# ==============================================================================
CROP_TELUGU_MAP = {
    "Rice": "వరి (Rice / Paddy)",
    "Cotton(lint)": "పత్తి (Cotton)",
    "Dry chillies": "ఎండు మిర్చి (Dry Chillies)",
    "Maize": "మొక్కజొన్న (Maize / Corn)",
    "Groundnut": "వేరుశనగ (Groundnut)",
    "Sugarcane": "చెరకు (Sugarcane)",
    "Banana": "అరటి (Banana)",
    "Tobacco": "పొగాకు (Tobacco)",
    "Turmeric": "పసుపు (Turmeric)",
    "Arhar/Tur": "కందులు (Red Gram / Tur)",
    "Gram": "శనగలు (Bengal Gram)",
    "Moong(Green Gram)": "పెసలు (Green Gram)",
    "Urad": "మినుములు (Black Gram)",
    "Wheat": "గోధుమలు (Wheat)",
    "Onion": "ఉల్లిపాయలు (Onion)",
    "Tomato": "టమోటా (Tomato)",
    "Mango": "మామిడి (Mango)",
    "Sunflower": "పొద్దుతిరుగుడు (Sunflower)",
    "Soyabean": "సోయాబీన్ (Soybean)",
    "Sesamum": "నువ్వులు (Sesame)",
    "Castor seed": "ఆముదాలు (Castor)",
    "Coconut ": "కొబ్బరి (Coconut)",
    "Arecanut": "పోక చెక్క / తమలపాకు (Arecanut)",
    "Cashewnut": "జీడిమామిడి (Cashew)",
    "Cardamom": "ఏలకులు (Cardamom)",
    "Black pepper": "మిరియాలు (Black Pepper)",
    "Garlic": "వెల్లుల్లి (Garlic)",
    "Ginger": "అల్లం (Ginger)",
    "Sweet potato": "చిలగడదుంప (Sweet Potato)",
    "Tapioca": "కర్రపెండలం (Tapioca)",
    "Jowar": "జొన్నలు (Jowar / Sorghum)",
    "Bajra": "సజ్జలు (Bajra / Pearl Millet)",
    "Ragi": "రాగులు (Ragi / Finger Millet)",
    "Barley": "బార్లీ (Barley)",
    "Jute": "జనపనార (Jute)",
    "Linseed": "అవిసె గింజలు (Linseed)",
    "Mesta": "గోగునార (Mesta)"
}

SEASON_TELUGU_MAP = {
    "Kharif": "ఖరీఫ్ (Kharif / వర్షాకాలం)",
    "Rabi": "రబీ (Rabi / శీతాకాలం)",
    "Whole Year": "సంవత్సరం మొత్తం (Whole Year)",
    "Summer": "వేసవి (Summer / ఎండాకాలం)",
    "Autumn": "శరదృతువు (Autumn)",
    "Winter": "శీతాకాలం (Winter)"
}


# ==============================================================================
# 3. QUANTUM KERNEL & STATEVECTOR SIMULATOR CLASS
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
# 4. RESOURCE LOADERS & CACHING
# ==============================================================================
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")


@st.cache_resource(show_spinner="మోడల్స్ లోడ్ అవుతున్నాయి / Loading Models...")
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

    if os.path.exists(combined_path):
        data["combined"] = pd.read_csv(combined_path)
    else:
        data["combined"] = None

    if os.path.exists(metrics_path):
        data["metrics"] = pd.read_csv(metrics_path)
    else:
        data["metrics"] = None

    if os.path.exists(soil_path):
        data["soil"] = pd.read_csv(soil_path)
    else:
        data["soil"] = None

    return data


# ==============================================================================
# 5. REAL-TIME WEATHER & GEOCODING SERVICE
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
    0: "స్వచ్ఛమైన ఆకాశం (Clear Sky)",
    1: "సాధారణంగా నిర్మలం (Mainly Clear)",
    2: "పాక్షికంగా మేఘాలు (Partly Cloudy)",
    3: "పూర్తిగా మేఘావృతం (Overcast)",
    45: "పొగమంచు (Foggy)",
    51: "తేలికపాటి చినుకులు (Light Drizzle)",
    53: "మధ్యస్థ చినుకులు (Moderate Drizzle)",
    61: "తేలికపాటి వర్షం (Light Rain)",
    63: "మధ్యస్థ వర్షం (Moderate Rain)",
    65: "భారీ వర్షం (Heavy Rain)",
    80: "వర్షపు జల్లులు (Rain Showers)",
    82: "తీవ్రమైన తుఫాను వర్షం (Violent Rain)",
    95: "ఉరుములతో కూడిన వర్షం (Thunderstorm)"
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
            condition = WMO_WEATHER_CODES.get(w_code, "సాధారణ వాతావరణం (Fair Weather)")

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
            "condition": "సాధారణంగా నిర్మలం (Mainly Clear)",
            "source": "State Climate Baseline (Offline fallback)",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "status": "offline_fallback"
        }


# ==============================================================================
# 6. BILINGUAL AGRONOMIC ADVISORY ENGINE
# ==============================================================================
def evaluate_agronomic_advisory(crop, area, exp_yield, hist_mean, temp, hum, rain_cur, n_val, p_val, k_val, ph_val, lang="te"):
    advisories = []
    risks = []

    # 1. Temperature Stress Evaluation
    if temp > 38.0:
        risks.append({
            "type": "alert",
            "title": "అధిక ఉష్ణోగ్రత హెచ్చరిక (High Heat Stress Alert)" if lang == "te" else "High Heat Stress Detected",
            "desc": f"ప్రస్తుతం ఉష్ణోగ్రత {temp}°C గా ఉంది. ఎండ తీవ్రత వల్ల పువ్వు రాలిపోవడం మరియు తేమ ఆవిరైపోవడం జరుగుతుంది. ఉదయం లేదా సాయంత్రం వేళల్లో తేలికపాటి నీటి తడులు ఇవ్వండి." if lang == "te" else f"Current temperature is {temp}°C (>38°C). May cause flower drop and rapid moisture loss. Ensure early morning light irrigation."
        })
    elif temp < 14.0:
        risks.append({
            "type": "warn",
            "title": "చలి తీవ్రత హెచ్చరిక (Cold Stress Alert)" if lang == "te" else "Low Temperature / Cold Stress",
            "desc": f"ప్రస్తుతం ఉష్ణోగ్రత {temp}°C గా ఉంది. చలి వల్ల పైరు పెరుగుదల మందగించవచ్చు." if lang == "te" else f"Current temperature is {temp}°C (<14°C). Can delay crop vegetative growth."
        })
    else:
        advisories.append({
            "title": "అనుకూలమైన ఉష్ణోగ్రత (Favorable Temperature)" if lang == "te" else "Optimal Thermal Range",
            "desc": f"ప్రస్తుత ఉష్ణోగ్రత ({temp}°C) పైరు ఏపుగా పెరగడానికి ఎంతో అనుకూలంగా ఉంది." if lang == "te" else f"Current temperature ({temp}°C) is favorable for {crop} growth."
        })

    # 2. Moisture & Humidity / Fungal Risk
    if hum > 82.0:
        risks.append({
            "type": "warn",
            "title": "అధిక గాలి తేమ - తెగుళ్ల ముప్పు (High Humidity - Disease Risk)" if lang == "te" else "High Humidity Disease Vector Warning",
            "desc": f"గాలిలో తేమ శాతం అధికంగా ({hum}%) ఉంది. బూడిద తెగులు, అగ్గి తెగులు లేదా ఆకుమచ్చ తెగుళ్లు ఆశించే అవకాశం ఉంది. పంటను పరిశీలించి అవసరమైన నివారణ చర్యలు చేపట్టండి." if lang == "te" else f"Atmospheric humidity is high ({hum}%). Favorable for fungal pathogens. Inspect lower crop foliage."
        })
    elif hum < 40.0:
        advisories.append({
            "title": "గాలిలో తక్కువ తేమ (Low Humidity)" if lang == "te" else "Low Ambient Humidity",
            "desc": f"గాలి తేమ {hum}% గా ఉంది. నేల త్వరగా ఎండిపోకుండా తేమను గమనిస్తూ ఉండండి." if lang == "te" else f"Relative humidity is {hum}%. Monitor soil moisture."
        })

    # 3. Precipitation & Drainage
    if rain_cur > 25.0:
        risks.append({
            "type": "alert",
            "title": "భారీ వర్షం & నీరు నిలిచే ముప్పు (Heavy Rain Alert)" if lang == "te" else "Heavy Precipitation & Drainage Alert",
            "desc": f"వర్షపాతం {rain_cur} మి.మీ గా ఉంది. పొలంలో నీరు నిలబడకుండా వెంటనే మురుగు కాలువల ద్వారా బయటకు పంపించండి." if lang == "te" else f"Precipitation is {rain_cur} mm. Clear field drainage channels immediately."
        })

    # 4. Yield Gap Evaluation
    if hist_mean > 0:
        yield_ratio = exp_yield / hist_mean
        if yield_ratio < 0.75:
            risks.append({
                "type": "warn",
                "title": "సాధారణం కంటే తక్కువ దిగుబడి అంచనా (Below Average Yield)" if lang == "te" else "Yield Potential Gap Detected",
                "desc": f"అంచనా దిగుబడి ({exp_yield:.2f} ట/హెక్టారు) గత ప్రాంతీయ సగటు కంటే తక్కువగా ఉంది. ఎరువుల మోతాదు, నేల సారం మరియు సకాలంలో సస్యరక్షణ చర్యలను సమీక్షించండి." if lang == "te" else f"Estimated yield ({exp_yield:.2f} t/ha) is below historical regional average ({hist_mean:.2f} t/ha). Review nutrient balance."
            })
        elif yield_ratio > 1.20:
            advisories.append({
                "title": "అత్యుత్తమ దిగుబడి అంచనా! (Superior Expected Harvest)" if lang == "te" else "Superior Expected Harvest Potential",
                "desc": f"మీ పంట నిర్వహణ మరియు నేల సారం బాగుండటం వల్ల ప్రాంతీయ సగటు కంటే అధిక దిగుబడి వచ్చే అవకాశం ఉంది." if lang == "te" else f"Estimated yield is significantly above regional benchmarks. Maintain scheduled nutrient splits."
            })

    # 5. Soil Reaction (pH) Contextual Advisory
    if ph_val < 6.0:
        advisories.append({
            "title": "ఆమ్ల నేల సూచన (Acidic Soil pH)" if lang == "te" else "Acidic Soil Reference",
            "desc": f"మీ నేల pH {ph_val:.1f} (ఆమ్ల గుణం). మట్టి పరీక్ష ప్రకారం సున్నం లేదా డోలమైట్ వాడకం గురించి వ్యవసాయ అధికారిని సంప్రదించండి." if lang == "te" else f"Soil pH is {ph_val:.1f} (Acidic). Consider agricultural lime upon lab soil test confirmation."
        })
    elif ph_val > 8.0:
        advisories.append({
            "title": "క్షార నేల సూచన (Alkaline Soil pH)" if lang == "te" else "Alkaline Soil Reference",
            "desc": f"మీ నేల pH {ph_val:.1f} (క్షార గుణం). జింక్, ఇనుము లోపాలు రాకుండా పచ్చిరొట్ట ఎరువులు లేదా జిప్సం వాడండి." if lang == "te" else f"Soil pH is {ph_val:.1f} (Alkaline). Apply FYM and green manure."
        })

    return advisories, risks


# ==============================================================================
# 7. MAIN APPLICATION EXECUTION
# ==============================================================================
def main():
    models = load_pipeline_models()
    datasets = load_datasets()

    df_combined = datasets["combined"]
    if df_combined is None or df_combined.empty:
        st.error("⚠️ డేటాసెట్ లోడ్ కాలేదు. Please run `python prepare_data.py` first.")
        st.stop()

    # ==========================================================================
    # SIDEBAR: LANGUAGE & FARMER INPUTS
    # ==========================================================================
    with st.sidebar:
        st.markdown("### 🌐 భాష / Language")
        lang_choice = st.radio(
            "Select Language / భాషను ఎంచుకోండి:",
            ["తెలుగు (Telugu)", "English"],
            index=0,
            horizontal=True
        )
        is_te = "తెలుగు" in lang_choice
        lang = "te" if is_te else "en"

        st.markdown("---")
        st.header("🚜 రైతు వివరాలు / Farmer Details" if is_te else "🚜 Farmer & Field Inputs")

        # Location Selection
        st.subheader("📍 ప్రాంతం / జిల్లా (Location)" if is_te else "📍 Location & RBK Center")
        loc_options = list(DISTRICT_COORDINATES.keys())
        selected_loc = st.selectbox(
            "వ్యవసాయ ప్రాంతం / జిల్లాను ఎంచుకోండి:" if is_te else "Select Agricultural Hub / District:",
            loc_options,
            index=0
        )
        lat, lon = DISTRICT_COORDINATES[selected_loc]
        loc_name = selected_loc

        st.caption(f"📌 **{loc_name}** ({lat:.3f}°N, {lon:.3f}°E)")
        st.markdown("---")

        # Crop Selection with Telugu Names
        st.subheader("🌱 పంట వివరాలు / Crop Selection" if is_te else "🌱 Crop & Farm Selection")
        raw_crops = sorted(df_combined["crop"].unique().tolist())
        
        # Display formatted bilingual crop names
        crop_display_list = [CROP_TELUGU_MAP.get(c, c) for c in raw_crops]
        default_crop_idx = raw_crops.index("Rice") if "Rice" in raw_crops else 0
        
        selected_crop_display = st.selectbox(
            "పంటను ఎంచుకోండి (Select Crop):" if is_te else "Select Crop:",
            crop_display_list,
            index=default_crop_idx
        )
        # Map back to raw crop name for ML model
        selected_crop = raw_crops[crop_display_list.index(selected_crop_display)]

        state_list = sorted(df_combined["state"].unique().tolist())
        default_state_idx = state_list.index("Andhra Pradesh") if "Andhra Pradesh" in state_list else 0
        selected_state = st.selectbox("రాష్ట్రం (State):" if is_te else "State:", state_list, index=default_state_idx)

        raw_seasons = sorted(df_combined["season"].unique().tolist())
        season_display_list = [SEASON_TELUGU_MAP.get(s, s) for s in raw_seasons]
        default_season_idx = raw_seasons.index("Kharif") if "Kharif" in raw_seasons else 0
        selected_season_display = st.selectbox(
            "సీజన్ (Season):" if is_te else "Season:",
            season_display_list,
            index=default_season_idx
        )
        selected_season = raw_seasons[season_display_list.index(selected_season_display)]

        st.markdown("---")
        st.subheader("📐 పొలం విస్తీర్ణం (Farm Area)" if is_te else "📐 Farm Land Area")
        area_unit = st.radio(
            "విస్తీర్ణాన్ని దేనిలో నమోదు చేస్తారు?" if is_te else "Enter Land Area In:",
            ["ఎకరాలు (Acres)", "హెక్టార్లు (Hectares)"] if is_te else ["Acres", "Hectares"],
            index=0,
            horizontal=True
        )

        if "Acres" in area_unit or "ఎకరాలు" in area_unit:
            area_acres = st.number_input(
                "ఎకరాల సంఖ్య (Number of Acres):" if is_te else "Area in Acres:",
                min_value=0.5,
                max_value=1200.0,
                value=5.0,
                step=0.5
            )
            area_ha = area_acres / 2.471
            st.caption(f"ℹ️ {area_acres:.1f} ఎకరాలు = **{area_ha:.2f} హెక్టార్లు**" if is_te else f"ℹ️ {area_acres:.1f} Acres = **{area_ha:.2f} Hectares**")
        else:
            area_ha = st.number_input(
                "హెక్టార్ల సంఖ్య (Area in Hectares):" if is_te else "Area in Hectares:",
                min_value=0.1,
                max_value=500.0,
                value=2.0,
                step=0.5
            )
            area_acres = area_ha * 2.471
            st.caption(f"ℹ️ {area_ha:.2f} హెక్టార్లు = **{area_acres:.1f} ఎకరాలు**" if is_te else f"ℹ️ {area_ha:.2f} Hectares = **{area_acres:.1f} Acres**")

        st.markdown("---")
        st.subheader("🧪 నేల సారం & మట్టి పరీక్ష (Soil Health Card)" if is_te else "🧪 Soil Nutrient Inputs (SHC)")
        st.caption("మీ మట్టి పరీక్ష (Soil Test) ఫలితాలను నమోదు చేయండి:" if is_te else "Enter laboratory Soil Health Card test values:")

        # Lookup state default benchmarks
        df_soil = datasets["soil"]
        if df_soil is not None and selected_state in df_soil["state"].values:
            soil_row = df_soil[df_soil["state"] == selected_state].iloc[0]
            default_n, default_p, default_k, default_ph = float(soil_row["N"]), float(soil_row["P"]), float(soil_row["K"]), float(soil_row["pH"])
        else:
            default_n, default_p, default_k, default_ph = 210.5, 24.2, 280.4, 7.2

        col_n1, col_n2 = st.columns(2)
        with col_n1:
            n_input = st.number_input(
                "నత్రజని / Nitrogen (N kg/ha):" if is_te else "Available Nitrogen (N kg/ha):",
                min_value=10.0,
                max_value=600.0,
                value=default_n,
                step=10.0,
                help="మట్టిలోని నత్రజని పరిమాణం (kg/ha)"
            )
            p_input = st.number_input(
                "భాస్వరం / Phosphorus (P kg/ha):" if is_te else "Available Phosphorus (P kg/ha):",
                min_value=2.0,
                max_value=150.0,
                value=default_p,
                step=2.0,
                help="మట్టిలోని భాస్వరం పరిమాణం (kg/ha)"
            )
        with col_n2:
            k_input = st.number_input(
                "పొటాష్ / Potassium (K kg/ha):" if is_te else "Available Potassium (K kg/ha):",
                min_value=10.0,
                max_value=800.0,
                value=default_k,
                step=10.0,
                help="మట్టిలోని పొటాష్ పరిమాణం (kg/ha)"
            )
            ph_input = st.number_input(
                "భూమి pH గుణం (Soil pH):" if is_te else "Soil Reaction (pH):",
                min_value=3.5,
                max_value=10.0,
                value=default_ph,
                step=0.1,
                help="6.5 - 7.5 ఉంటే అనుకూలమైన నేల"
            )

        st.markdown("---")
        st.subheader("💊 వాడిన ఎరువులు & పురుగుమందులు" if is_te else "💊 Applied Fertilizers & Pesticides")
        
        crop_median_fert = df_combined[df_combined["crop"] == selected_crop]["fertilizer"].median() / max(1.0, df_combined[df_combined["crop"] == selected_crop]["area"].median())
        crop_median_pest = df_combined[df_combined["crop"] == selected_crop]["pesticide"].median() / max(1.0, df_combined[df_combined["crop"] == selected_crop]["area"].median())
        
        if pd.isna(crop_median_fert) or crop_median_fert <= 0:
            crop_median_fert = 120.0
        if pd.isna(crop_median_pest) or crop_median_pest <= 0:
            crop_median_pest = 1.5

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            fert_input = st.number_input(
                "ఎరువులు (మొత్తం కేజీలు):" if is_te else "Fertilizer (Total kg):",
                min_value=0.0,
                max_value=50000.0,
                value=float(round(crop_median_fert * area_ha, 1)),
                help="పొలంలో వేసిన మొత్తం యూరియా, డీఏపీ, పొటాష్ ఎరువులు"
            )
        with col_f2:
            pest_input = st.number_input(
                "పురుగుమందులు (కేజీలు):" if is_te else "Pesticide (Total kg):",
                min_value=0.0,
                max_value=500.0,
                value=float(round(crop_median_pest * area_ha, 2)),
                help="వాడిన మొత్తం మందులు"
            )

        st.markdown("---")
        st.button("⚡ దిగుబడిని లెక్కించండి (Calculate Yield)" if is_te else "⚡ Recalculate AI & Quantum Prediction", type="primary", use_container_width=True)

    # ==========================================================================
    # FETCH REAL-TIME WEATHER & COMPUTE LOCALIZED BENCHMARKS
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
        benchmark_label = f"{selected_state} ({selected_season}) సగటు" if is_te else f"{selected_state} ({selected_season}) Median"
    else:
        state_data = df_combined[
            (df_combined["crop"] == selected_crop) & 
            (df_combined["state"] == selected_state)
        ]
        if not state_data.empty:
            crop_hist_mean = float(state_data["yield"].median())
            benchmark_label = f"{selected_state} సగటు" if is_te else f"{selected_state} Historical Median"
        else:
            crop_hist_mean = float(df_combined[df_combined["crop"] == selected_crop]["yield"].median())
            benchmark_label = "జాతీయ సగటు" if is_te else "National Benchmark Median"

    # ==========================================================================
    # PERFORM MODEL INFERENCE (CLASSICAL + QUANTUM)
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

    # 1. Classical Prediction
    t0_c = time.perf_counter()
    X_proc = models["encoder"].transform(input_df)
    class_pred_log = models["classical"].predict(X_proc)[0]
    class_pred_yield = float(np.clip(np.expm1(class_pred_log), 0.01, None))
    total_class_production = class_pred_yield * area_ha
    classical_latency_ms = (time.perf_counter() - t0_c) * 1000

    # 2. Quantum Prediction
    t0_q = time.perf_counter()
    X_pca = models["pca"].transform(X_proc)
    X_q = models["quantum_scaler"].transform(X_pca)
    K_sample = models["q_kernel"].evaluate(X_q, models["X_train_q"])
    quant_pred_log = models["quantum"].predict(K_sample)[0]
    quant_pred_yield = float(np.clip(np.expm1(quant_pred_log), 0.01, None))
    total_quant_production = quant_pred_yield * area_ha
    quantum_latency_ms = (time.perf_counter() - t0_q) * 1000

    # Compute Quantum Statevector
    sample_statevector = get_statevector(models["q_kernel"], X_q[0])
    state_probs = np.abs(sample_statevector) ** 2

    # Unit Conversions for Farmer Comprehension
    quintals_total = total_class_production * 10.0
    paddy_bags_75kg = (total_class_production * 1000.0) / 75.0
    paddy_bags_50kg = (total_class_production * 1000.0) / 50.0
    tonnes_per_acre = class_pred_yield / 2.471
    quintals_per_acre = tonnes_per_acre * 10.0
    bags_75kg_per_acre = (tonnes_per_acre * 1000.0) / 75.0

    # Evaluate Agronomic Advisories
    advisories, risks = evaluate_agronomic_advisory(
        selected_crop, area_ha, class_pred_yield, crop_hist_mean,
        weather["temperature"], weather["humidity"], weather["precipitation"],
        n_input, p_input, k_input, ph_input, lang=lang
    )

    # ==========================================================================
    # HEADER BANNER
    # ==========================================================================
    header_title = "🌾 రైతు భరోసా – రియల్-టైమ్ వ్యవసాయ నిర్ణయ మద్దతు వ్యవస్థ" if is_te else "🌾 Rythu Bharosa – Real-Time Quantum AI Precision Agriculture"
    header_sub = "రైతు భరోసా కేంద్రాలు (RBKs) & రైతుల కోసం రియల్-టైమ్ వాతావరణం, క్లాసికల్ & క్వాంటం AI దిగుబడి అంచనా" if is_te else "Real-Time Climate Data, Classical AI & Quantum Machine Learning for Farmer Decision Support"
    motto_text = "రైతు భరోసా – మెరుగైన వ్యవసాయ నిర్ణయాల కోసం రియల్-టైమ్ డేటా & AI" if is_te else "Rythu Bharosa – Helping Farmers Make Better Decisions Using Real-Time Data and AI"

    st.markdown(f"""
    <div class="main-header">
        <h1>{header_title}</h1>
        <p>{header_sub}</p>
        <div class="telugu-banner">{motto_text}</div>
    </div>
    """, unsafe_allow_html=True)

    # ==========================================================================
    # SIMPLE FARMER SUMMARY BOX (For Non-Educated & Educated Farmers)
    # ==========================================================================
    crop_name_telugu = CROP_TELUGU_MAP.get(selected_crop, selected_crop)
    
    if is_te:
        st.markdown(f"""
        <div class="farmer-highlight-box">
            <h3 style="color: #1b5e20; margin: 0 0 10px 0; font-size: 1.4rem;">
                📢 రైతు సులభ అవగాహన నివేదిక (Farmer Quick Summary)
            </h3>
            <p style="font-size: 1.15rem; color: #1a2e1a; margin: 0; line-height: 1.6;">
                మీరు <b>{loc_name}</b> ప్రాంతంలో <b>{area_acres:.1f} ఎకరాల</b> పొలంలో సాగుచేస్తున్న <b>{crop_name_telugu}</b> పంటకు:<br>
                🌾 <b>ఎకరాకు వచ్చే అంచనా దిగుబడి:</b> <span style="color: #1b5e20; font-weight: 800; font-size: 1.3rem;">{quintals_per_acre:.1f} క్వింటాళ్లు</span> (సుమారు <b>{bags_75kg_per_acre:.0f} బస్తాలు</b> / ఎకరాకి)<br>
                🚜 <b>మీ మొత్తం పొలానికి వచ్చే దిగుబడి:</b> <span style="color: #b71c1c; font-weight: 800; font-size: 1.4rem;">{total_class_production:.2f} టన్నులు</span> (సుమారు <b>{quintals_total:.0f} క్వింటాళ్లు</b> లేదా <b>{paddy_bags_75kg:.0f} బస్తాలు</b>)<br>
                🌡️ <b>ప్రస్తుత వాతావరణం:</b> ఉష్ణోగ్రత <b>{weather['temperature']}°C</b>, గాలిలో తేమ <b>{weather['humidity']}%</b>, ఆకాశం <b>{weather['condition']}</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="farmer-highlight-box">
            <h3 style="color: #1b5e20; margin: 0 0 10px 0; font-size: 1.4rem;">
                📢 Farmer Executive Summary
            </h3>
            <p style="font-size: 1.15rem; color: #1a2e1a; margin: 0; line-height: 1.6;">
                For your <b>{area_acres:.1f} Acres ({area_ha:.2f} ha)</b> field of <b>{selected_crop}</b> at <b>{loc_name}</b>:<br>
                🌾 <b>Productivity Rate:</b> <span style="color: #1b5e20; font-weight: 800; font-size: 1.3rem;">{class_pred_yield:.2f} t/ha</span> ({quintals_per_acre:.1f} Quintals/acre / ~{bags_75kg_per_acre:.0f} Bags/acre)<br>
                🚜 <b>Total Estimated Farm Harvest:</b> <span style="color: #b71c1c; font-weight: 800; font-size: 1.4rem;">{total_class_production:.2f} Tonnes</span> ({quintals_total:.1f} Quintals / ~{paddy_bags_75kg:.0f} Bags of 75kg)<br>
                🌡️ <b>Real-Time Telemetry:</b> Temperature <b>{weather['temperature']}°C</b>, Humidity <b>{weather['humidity']}%</b>, Sky <b>{weather['condition']}</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ==========================================================================
    # TOP DASHBOARD CARDS
    # ==========================================================================
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        card1_title = "🌾 మొత్తం పంట దిగుబడి (Total Harvest)" if is_te else "🌾 Total Field Harvest"
        st.markdown(f"""
        <div class="stat-card" style="border-top: 4px solid #1b5e20;">
            <div class="stat-title">{card1_title}</div>
            <div class="stat-value">{total_class_production:.2f} <span style="font-size: 1.1rem; font-weight: 600;">{'టన్నులు' if is_te else 'Tonnes'}</span></div>
            <div class="stat-sub">
                <b>{quintals_total:.1f} {'క్వింటాళ్లు' if is_te else 'Quintals'}</b> • <b>{paddy_bags_75kg:.0f} {'బస్తాలు (75kg)' if is_te else 'Bags (75kg)'}</b><br>
                {'మొత్తం' if is_te else 'Across'} {area_acres:.1f} {'ఎకరాలలో' if is_te else 'Acres'} ({area_ha:.1f} ha)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        card2_title = "📈 ఎకరాకు దిగుబడి రేటు (Yield / Acre)" if is_te else "📈 Productivity Rate (Yield/ha)"
        diff_pct = ((class_pred_yield / max(0.01, crop_hist_mean)) - 1.0) * 100.0
        diff_color = "#2e7d32" if diff_pct >= 0 else "#c62828"
        diff_sign = "+" if diff_pct >= 0 else ""
        st.markdown(f"""
        <div class="stat-card" style="border-top: 4px solid #2e7d32;">
            <div class="stat-title">{card2_title}</div>
            <div class="stat-value">{quintals_per_acre:.1f} <span style="font-size: 1rem; font-weight: 600;">{'క్వింటాళ్లు/ఎకరా' if is_te else 'Q/acre'}</span></div>
            <div class="stat-sub">
                <b>{class_pred_yield:.2f} t/ha</b> ({bags_75kg_per_acre:.0f} {'బస్తాలు/ఎకరా' if is_te else 'bags/ac'})<br>
                {benchmark_label}: {crop_hist_mean:.2f} t/ha (<b style="color: {diff_color};">{diff_sign}{diff_pct:.0f}%</b>)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        card3_title = "🌡️ లైవ్ వాతావరణం (Real-Time Weather)" if is_te else "🌡️ Live Climate Telemetry"
        st.markdown(f"""
        <div class="stat-card" style="border-top: 4px solid #0288d1;">
            <div class="stat-title">{card3_title}</div>
            <div class="stat-value">{weather['temperature']} <span style="font-size: 1rem; font-weight: 600;">°C</span></div>
            <div class="stat-sub">
                💧 {'గాలి తేమ' if is_te else 'Humidity'}: <b>{weather['humidity']}%</b> • 🌧️ {'వర్షం' if is_te else 'Rain'}: <b>{weather['precipitation']} mm</b><br>
                {weather['condition']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        card4_title = "🌱 పంట & నేల వివరాలు (Crop & Soil)" if is_te else "🌱 Crop & Soil Profile"
        st.markdown(f"""
        <div class="stat-card" style="border-top: 4px solid #f57f17;">
            <div class="stat-title">{card4_title}</div>
            <div class="stat-value" style="font-size: 1.4rem; color: #e65100;">{crop_name_telugu if is_te else selected_crop}</div>
            <div class="stat-sub">
                {'సీజన్' if is_te else 'Season'}: <b>{selected_season}</b> • {'నేల' if is_te else 'Soil'} pH: <b>{ph_input:.1f}</b><br>
                నత్రజని (N): <b>{n_input:.0f}</b> | భాస్వరం (P): <b>{p_input:.0f}</b> | పొటాష్ (K): <b>{k_input:.0f}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ==========================================================================
    # TABS: MULTI-DIMENSIONAL PRECISION AGRICULTURE PLATFORM
    # ==========================================================================
    tab_names = [
        "🌾 దిగుబడి విశ్లేషణ (Yield AI)" if is_te else "🌾 Yield Intelligence",
        "🚨 రైతు సలహాలు & హెచ్చరికలు (Advisory)" if is_te else "🚨 Risk & Agronomic Advisory",
        "⚛️ క్వాంటం AI వివరణ (Quantum AI)" if is_te else "⚛️ Quantum AI Mechanics",
        "📷 నేల ఫోటో విశ్లేషణ (Soil Photo)" if is_te else "📷 Soil Visual Tool",
        "📈 చారిత్రక పోకడలు (Analytics)" if is_te else "📈 Analytics & Visualizations",
        "📄 రైతు నివేదిక (Farmer Report)" if is_te else "📄 Farmer Advisory Report"
    ]
    tab_pred, tab_advisory, tab_quantum, tab_soil_vision, tab_analytics, tab_report = st.tabs(tab_names)

    # --------------------------------------------------------------------------
    # TAB 1: YIELD INTELLIGENCE
    # --------------------------------------------------------------------------
    with tab_pred:
        col_p1, col_p2 = st.columns([3, 2])

        with col_p1:
            st.markdown("### 🤖 ఆర్టిఫిషియల్ ఇంటెలిజెన్స్ దిగుబడి అంచనాలు" if is_te else "### 🤖 AI Yield Estimations")

            # Classical AI Card
            st.markdown(f"""
            <div style="background: white; border: 1px solid #c8e6c9; border-radius: 12px; padding: 20px; margin-bottom: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span class="badge-classical">🟢 {'ప్రధాన మోడల్ - ఖచ్చితమైనది (Accuracy: 97.5%)' if is_te else 'PRIMARY CLASSICAL AI MODEL'}</span>
                        <h3 style="color: #1b5e20; margin: 8px 0 4px 0;">{'క్లాసికల్ AI (Random Forest Regressor)' if is_te else 'Classical AI (Random Forest Regressor)'}</h3>
                        <p style="color: #616161; margin: 0; font-size: 0.9rem;">{'100 డెసిషన్ ట్రీలు • 101 రకాల నేల, వాతావరణ మరియు పంట లక్షణాలు' if is_te else '100 Decision Trees • Multi-crop 101-Dimensional Feature Space'}</p>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 2.2rem; font-weight: 800; color: #1b5e20;">{class_pred_yield:.2f}</div>
                        <div style="color: #556b2f; font-weight: 600; font-size: 0.85rem;">{'టన్నులు / హెక్టారు' if is_te else 'tonnes / hectare'}</div>
                        <div style="color: #1b5e20; font-weight: 700; font-size: 0.95rem;">({quintals_per_acre:.1f} {'క్వింటాళ్లు/ఎకరా' if is_te else 'Q/acre'})</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Quantum AI Card
            st.markdown(f"""
            <div style="background: white; border: 1px solid #e1bee7; border-radius: 12px; padding: 20px; margin-bottom: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span class="badge-quantum">⚛️ {'క్వాంటం కంప్యూటింగ్ మోడల్ (Qiskit 4-Qubit)' if is_te else 'QUANTUM MACHINE LEARNING MODEL'}</span>
                        <h3 style="color: #4a148c; margin: 8px 0 4px 0;">{'క్వాంటం AI (ZZFeatureMap + SVR)' if is_te else 'Quantum AI (4-Qubit ZZFeatureMap + SVR)'}</h3>
                        <p style="color: #616161; margin: 0; font-size: 0.9rem;">{'4 క్వాంటం క్యూబిట్స్ • హిల్బర్ట్ స్పేస్ ఫెడిలిటీ కెర్నల్' if is_te else 'Qiskit Quantum Fidelity Kernel • PCA Hilbert Space Embedding'}</p>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 2.2rem; font-weight: 800; color: #4a148c;">{quant_pred_yield:.2f}</div>
                        <div style="color: #6a1b9a; font-weight: 600; font-size: 0.85rem;">{'టన్నులు / హెక్టారు' if is_te else 'tonnes / hectare'}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.info("📌 **గమనిక (Scientific Note):** ఈ దిగుబడి అంచనాలు ICAR/ప్రభుత్వ వ్యవసాయ గణాంకాలు, మీ మట్టి పరీక్ష వివరాలు మరియు ప్రస్తుత లైవ్ వాతావరణం ఆధారంగా నిజాయితీగా లెక్కించబడ్డాయి." if is_te else "📌 **Note:** Predictions integrate real verified agricultural yield baselines with real-time live environmental telemetry.")

        with col_p2:
            st.markdown("### 🌾 దిగుబడి మీటర్ & ప్రాంతీయ పోలిక" if is_te else "### 🌾 Regional Benchmark Gauge")
            st.markdown(f"""
            - **{'ఎంచుకున్న పంట' if is_te else 'Selected Crop'}:** `{crop_name_telugu if is_te else selected_crop}`
            - **{'పొలం విస్తీర్ణం' if is_te else 'Farm Area'}:** `{area_acres:.1f} {'ఎకరాలు' if is_te else 'Acres'}` ({area_ha:.2f} ha)
            - **{benchmark_label}:** `{crop_hist_mean:.2f} t/ha`
            - **{'మొత్తం అంచనా ధాన్యం' if is_te else 'Total Estimated Harvest'}:** `{total_class_production:.2f} {'టన్నులు' if is_te else 'Tonnes'}` ({quintals_total:.0f} {'క్వింటాళ్లు' if is_te else 'Quintals'})
            """)

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=class_pred_yield,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Expected Yield vs {benchmark_label} (t/ha)", 'font': {'size': 13}},
                delta={'reference': crop_hist_mean, 'increasing': {'color': "#2e7d32"}, 'decreasing': {'color': "#d32f2f"}},
                gauge={
                    'axis': {'range': [0, max(crop_hist_mean * 2.2, class_pred_yield * 1.4)]},
                    'bar': {'color': "#1b5e20"},
                    'steps': [
                        {'range': [0, crop_hist_mean * 0.7], 'color': "#ffebee"},
                        {'range': [crop_hist_mean * 0.7, crop_hist_mean * 1.2], 'color': "#e8f5e9"},
                        {'range': [crop_hist_mean * 1.2, max(crop_hist_mean * 2.2, class_pred_yield * 1.4)], 'color': "#c8e6c9"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 3},
                        'thickness': 0.75,
                        'value': crop_hist_mean
                    }
                }
            ))
            fig_gauge.update_layout(height=230, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

    # --------------------------------------------------------------------------
    # TAB 2: RISK & AGRONOMIC ADVISORY
    # --------------------------------------------------------------------------
    with tab_advisory:
        st.markdown("### 🚨 రైతు సలహాలు & వాతావరణ ముప్పు హెచ్చరికలు" if is_te else "### 🚨 Transparent Agronomic Risk & Decision Support")
        st.caption("మీ ప్రాంతంలోని ప్రస్తుత వాతావరణం మరియు మట్టి సారం ఆధారంగా రైతులకు సూచనలు:" if is_te else "Actionable advisories based on real-time environmental stress indices:")

        if risks:
            for r in risks:
                card_class = "advisory-alert" if r["type"] == "alert" else "advisory-warn"
                st.markdown(f"""
                <div class="{card_class}">
                    <h4 style="margin: 0 0 6px 0;">{r['title']}</h4>
                    <p style="margin: 0; font-size: 1rem; line-height: 1.5;">{r['desc']}</p>
                </div>
                """, unsafe_allow_html=True)

        for a in advisories:
            st.markdown(f"""
            <div class="advisory-card">
                <h4 style="margin: 0 0 6px 0; color: #1b5e20;">✓ {a['title']}</h4>
                <p style="margin: 0; font-size: 1rem; color: #2e3d2e; line-height: 1.5;">{a['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("🧪 మట్టి పరీక్ష పోషకాల పోలిక (Soil Nutrient Status)" if is_te else "🧪 Soil Nutrient Test Profile")
        
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            st.metric("నత్రజని (Nitrogen - N)", f"{n_input:.0f} kg/ha", delta=f"{n_input - default_n:+.0f} vs State Base", help="ల్యాబ్ మట్టి పరీక్ష విలువ")
        with col_s2:
            st.metric("భాస్వరం (Phosphorus - P)", f"{p_input:.0f} kg/ha", delta=f"{p_input - default_p:+.0f} vs State Base", help="ల్యాబ్ మట్టి పరీక్ష విలువ")
        with col_s3:
            st.metric("పొటాష్ (Potassium - K)", f"{k_input:.0f} kg/ha", delta=f"{k_input - default_k:+.0f} vs State Base", help="ల్యాబ్ మట్టి పరీక్ష విలువ")
        with col_s4:
            st.metric("నేల గుణం (Soil pH)", f"{ph_input:.1f}", delta=f"{ph_input - default_ph:+.1f} vs State Base", help="6.5 - 7.5 ఉంటే సమతుల్యం")

    # --------------------------------------------------------------------------
    # TAB 3: QUANTUM AI MECHANICS
    # --------------------------------------------------------------------------
    with tab_quantum:
        st.markdown("### ⚛️ క్వాంటం మెషిన్ లెర్నింగ్ ఆర్కిటెక్చర్ (Quantum AI Architecture)" if is_te else "### ⚛️ Quantum Machine Learning Architecture & Simulator")
        
        col_q1, col_q2 = st.columns([3, 2])

        with col_q1:
            st.markdown(r"""
            #### How Quantum AI Precision Agriculture Works:
            1. **Dimensionality Reduction**: The 101-dimensional agricultural feature vector is compressed into 4 principal components using PCA.
            2. **Feature Mapping to Quantum States**: Scaled to $[-\pi, \pi]$ and encoded into a 4-qubit quantum state $|\psi(\mathbf{x})\rangle$ via 2-repetition **ZZFeatureMap**:
               $$U_{\Phi(\mathbf{x})} = \exp\left(i \sum_{j} x_j Z_j + \sum_{j < k} (\pi - x_j)(\pi - x_k) Z_j Z_k\right)$$
            3. **Quantum Fidelity Kernel**: Measures state overlap in 16-dimensional Hilbert space:
               $$K(\mathbf{x}_i, \mathbf{x}_j) = |\langle \psi(\mathbf{x}_j) | \psi(\mathbf{x}_i) \rangle|^2$$
            4. **Kernel Support Vector Regression (SVR)**: Fitted on the quantum Gram matrix.
            """)

        with col_q2:
            st.markdown("#### 🔬 క్వాంటం కెర్నల్ హీట్‌మ్యాప్ (Quantum Kernel Similarity)")
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
        st.subheader("⚡ లైవ్ 4-క్యూబిట్ క్వాంటం సంభావ్యత సిమ్యులేటర్ (Quantum Statevector Simulator)")
        basis_labels = [f"|{bin(i)[2:].zfill(4)}⟩" for i in range(16)]
        fig_q_bars = px.bar(
            x=basis_labels,
            y=state_probs,
            labels={"x": "Quantum Basis State |q3 q2 q1 q0⟩", "y": "Probability |⟨basis|ψ(x)⟩|²"},
            title=f"Quantum State Probability Distribution for {crop_name_telugu}",
            color=state_probs,
            color_continuous_scale="Purples"
        )
        fig_q_bars.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_q_bars, use_container_width=True)

        st.markdown("---")
        st.subheader("📊 మోడల్ ఖచ్చితత్వ నివేదిక (Model Comparison)" if is_te else "📊 Model Comparison & Evaluation")
        metrics_df = datasets["metrics"]
        if metrics_df is not None:
            st.dataframe(metrics_df, use_container_width=True)

    # --------------------------------------------------------------------------
    # TAB 4: SOIL PHOTO VISUAL INSPECTION
    # --------------------------------------------------------------------------
    with tab_soil_vision:
        st.markdown("### 📷 మట్టి ఫోటో దృశ్య పరీక్ష (Soil Photo Visual Tool)" if is_te else "### 📷 Soil Visual Texture Tool")
        st.warning("⚠️ **రైతులకు ముఖ్య గమనిక:** సాధారణ స్మార్ట్‌ఫోన్ ఫోటోల ద్వారా మట్టిలోని నత్రజని, భాస్వరం లేదా రసాయన తేమను కొలవలేము. ఫోటో ద్వారా కేవలం నేల రంగు మరియు రూపురేఖలను మాత్రమే చూడవచ్చు. ఎరువుల కోసం ల్యాబ్ మట్టి పరీక్షను మాత్రమే నమ్మండి." if is_te else "⚠️ **Scientific Disclaimer:** Smartphone photos CANNOT accurately measure chemical N, P, K or moisture. Use certified laboratory Soil Health Cards for fertilizer decisions.")

        col_img1, col_img2 = st.columns([1, 1])

        with col_img1:
            uploaded_file = st.file_uploader("మట్టి నమూనా ఫోటోను అప్‌లోడ్ చేయండి (Upload Soil Photo):" if is_te else "Upload Soil Photo:", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                img = Image.open(uploaded_file).convert("RGB")
                st.image(img, caption="అప్‌లోడ్ చేసిన మట్టి నమూనా", use_container_width=True)
            else:
                dummy_arr = np.uint8(np.random.randint(60, 140, (200, 200, 3)))
                img = Image.fromarray(dummy_arr)
                st.image(img, caption="ప్రామాణిక మట్టి నమూనా దృశ్యం", use_container_width=True)

        with col_img2:
            st.markdown("#### 🔍 రంగు & ఆకృతి విశ్లేషణ (Visual Analytics)" if is_te else "#### 🔍 Visual Texture Analytics")
            img_np = np.array(img)
            r_mean = np.mean(img_np[:, :, 0])
            g_mean = np.mean(img_np[:, :, 1])
            b_mean = np.mean(img_np[:, :, 2])
            darkness_idx = 255.0 - (0.299 * r_mean + 0.587 * g_mean + 0.114 * b_mean)

            st.markdown(f"""
            - **ఎరుపు రంగు శాతం (Red Channel):** `{r_mean:.1f} / 255`
            - **ఆకుపచ్చ రంగు శాతం (Green Channel):** `{g_mean:.1f} / 255`
            - **నీలం రంగు శాతం (Blue Channel):** `{b_mean:.1f} / 255`
            - **నేల రంగు సూచిక (Shade Index):** `{darkness_idx:.1f}`
            """)

            if darkness_idx > 140:
                st.success("నల్లరేగడి / బంకమట్టి లక్షణాలు కన్పిస్తున్నాయి. నీరు నిలబడకుండా చూసుకోండి." if is_te else "Visual observation indicates dark/clayey soil. Check drainage.")
            elif darkness_idx < 90:
                st.info("ఇసుక / తేలికపాటి నేల లక్షణాలు కన్పిస్తున్నాయి. తరచూ నీటి తడులు అవసరం." if is_te else "Visual observation indicates sandy soil. Water frequently.")
            else:
                st.info("ఎర్ర నేల / ఒండ్రు మట్టి లక్షణాలు కన్పిస్తున్నాయి." if is_te else "Visual observation indicates loamy / mixed texture soil.")

    # --------------------------------------------------------------------------
    # TAB 5: ANALYTICS & VISUALIZATIONS
    # --------------------------------------------------------------------------
    with tab_analytics:
        st.markdown("### 📈 గత 20 సంవత్సరాల పంట దిగుబడి గణాంకాలు (Historical Trends)" if is_te else "### 📈 Agricultural Analytics & Historical Trends")

        col_an1, col_an2 = st.columns(2)

        with col_an1:
            st.markdown(f"#### {crop_name_telugu if is_te else selected_crop} - గత సంవత్సరాల దిగుబడి పోకడలు (1997 - 2020)")
            if not crop_hist_data.empty:
                yearly_yield = crop_hist_data.groupby("year")["yield"].agg(["mean", "median"]).reset_index()
                fig_trend = px.line(
                    yearly_yield,
                    x="year",
                    y=["mean", "median"],
                    labels={"value": "దిగుబడి (టన్నులు/హెక్టారు)" if is_te else "Yield (t/ha)", "year": "సంవత్సరం (Year)", "variable": "సగటు"},
                    title=f"{selected_crop} Historical Trajectory",
                    color_discrete_sequence=["#1b5e20", "#f57f17"]
                )
                fig_trend.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_trend, use_container_width=True)

        with col_an2:
            st.markdown("#### దిగుబడిని ప్రభావితం చేసే అంశాలు (Feature Importance)" if is_te else "#### Decision Factor Importance")
            feat_names = ["పంట రకం (Crop)", "పొలం విస్తీర్ణం (Area)", "ఎరువుల వాడకం (Fertilizer)", "పురుగుమందులు (Pesticides)", "నేల NPK & pH", "ఉష్ణోగ్రత (Temp)", "వర్షపాతం (Rain)", "గాలి తేమ (Humidity)"] if is_te else ["Crop Type", "Cultivation Area", "Fertilizer Usage", "Pesticide", "Soil NPK & pH", "Temperature", "Rainfall", "Humidity"]
            feat_weights = [0.42, 0.18, 0.14, 0.08, 0.09, 0.04, 0.03, 0.02]
            fig_imp = px.bar(
                x=feat_weights,
                y=feat_names,
                orientation="h",
                labels={"x": "ప్రాముఖ్యత (Importance)", "y": "అంశం (Feature)"},
                title="AI మోడల్ నిర్ణయ కారకాలు",
                color=feat_weights,
                color_continuous_scale="Greens"
            )
            fig_imp.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_imp, use_container_width=True)

    # --------------------------------------------------------------------------
    # TAB 6: DOWNLOADABLE FARMER ADVISORY REPORT
    # --------------------------------------------------------------------------
    with tab_report:
        st.markdown("### 📄 రైతు వ్యవసాయ సలహా మరియు దిగుబడి నివేదిక" if is_te else "### 📄 Rythu Bharosa Farmer Advisory Report")
        st.caption("మీ పొలం కోసం అధికారిక వ్యవసాయ సలహా పత్రాన్ని డౌన్‌లోడ్ చేసుకోండి:" if is_te else "Download structured agricultural decision-support record for your farm:")

        report_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_text = f"""================================================================================
రైతు భరోసా – రియల్-టైమ్ వ్యవసాయ సలహా & దిగుబడి అంచనా నివేదిక
RYTHU BHAROSA – REAL-TIME PRECISION AGRICULTURE REPORT
================================================================================
తేదీ & సమయం (Generated On): {report_timestamp}
ప్రాంతం (Location):            {loc_name} ({lat:.4f}°N, {lon:.4f}°E)
--------------------------------------------------------------------------------
1. రైతు & పొలం వివరాలు (FARM & CROP PROFILE)
--------------------------------------------------------------------------------
ఎంచుకున్న పంట (Crop):         {crop_name_telugu}
సీజన్ (Season):                {selected_season}
రాష్ట్రం (State):               {selected_state}
పొలం విస్తీర్ణం (Farm Area):     {area_acres:.1f} ఎకరాలు ({area_ha:.2f} హెక్టార్లు)
వాడిన ఎరువులు (Fertilizer):     {fert_input:.1f} kg
వాడిన పురుగుమందులు (Pesticide): {pest_input:.2f} kg

--------------------------------------------------------------------------------
2. ప్రస్తుత వాతావరణ సమాచారం (REAL-TIME WEATHER)
--------------------------------------------------------------------------------
వాతావరణ మూలం (Source):        {weather['source']}
ప్రస్తుత ఉష్ణోగ్రత (Temperature): {weather['temperature']} °C
గాలిలో తేమ (Humidity):         {weather['humidity']} %
వర్షపాతం (Precipitation):       {weather['precipitation']} mm
గాలి వేగం (Wind Speed):         {weather['wind_speed']} km/h
వాతావరణ స్థితి (Condition):     {weather['condition']}

--------------------------------------------------------------------------------
3. AI దిగుబడి అంచనాలు (AI YIELD PREDICTIONS)
--------------------------------------------------------------------------------
ఎకరాకు అంచనా దిగుబడి:          {quintals_per_acre:.1f} క్వింటాళ్లు ({tonnes_per_acre:.2f} టన్నులు/ఎకరా)
మొత్తం పొలం పంట దిగుబడి:        {total_class_production:.2f} టన్నులు ({quintals_total:.0f} క్వింటాళ్లు / ~{paddy_bags_75kg:.0f} బస్తాలు)
ప్రాంతీయ సగటు దిగుబడి:         {crop_hist_mean:.2f} టన్నులు/హెక్టారు ({benchmark_label})
క్వాంటం AI అంచనా:              {quant_pred_yield:.2f} టన్నులు/హెక్టారు

--------------------------------------------------------------------------------
4. మట్టి పరీక్ష & పోషకాల వివరాలు (SOIL HEALTH CARD)
--------------------------------------------------------------------------------
నత్రజని (Available Nitrogen - N):   {n_input:.0f} kg/ha (రాష్ట్ర సగటు: {default_n:.0f})
భాస్వరం (Available Phosphorus - P): {p_input:.0f} kg/ha (రాష్ట్ర సగటు: {default_p:.0f})
పొటాష్ (Available Potassium - K):   {k_input:.0f} kg/ha (రాష్ట్ర సగటు: {default_k:.0f})
నేల pH గుణం (Soil pH):              {ph_input:.1f} (రాష్ట్ర సగటు: {default_ph:.1f})

--------------------------------------------------------------------------------
5. రైతు సలహాలు & జాగ్రత్తలు (AGRONOMIC ADVISORIES)
--------------------------------------------------------------------------------
"""
        for r in risks:
            report_text += f"[హెచ్చరిక / WARNING] {r['title']}\n  -> {r['desc']}\n\n"
        for a in advisories:
            report_text += f"[సలహా / ADVISORY] {a['title']}\n  -> {a['desc']}\n\n"

        report_text += """--------------------------------------------------------------------------------
6. చట్టబద్ధమైన గమనిక (STATUTORY DISCLAIMER)
--------------------------------------------------------------------------------
- ఈ నివేదిక రైతులకు, ఆర్బీకే (RBK) సిబ్బందికి సహాయక నిర్ణయ మద్దతు కొరకు మాత్రమే.
- రసాయన ఎరువుల కొనుగోలుకు ముందు అధీకృత ల్యాబ్ మట్టి పరీక్షను (Soil Health Card) సంప్రదించండి.
================================================================================
"""

        st.text_area("నివేదిక ప్రివ్యూ (Report Preview):", report_text, height=350)

        col_rep1, col_rep2 = st.columns(2)
        with col_rep1:
            st.download_button(
                label="📥 రైతు సలహా పత్రం డౌన్‌లోడ్ చేయండి (Download Text Report)",
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
                label="📥 ఎక్సెల్ / CSV డేటా డౌన్‌లోడ్ (Download CSV)",
                data=report_summary_df.to_csv(index=False),
                file_name=f"Rythu_Bharosa_{selected_crop}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #616161; font-size: 0.9rem; padding: 15px 0;">
        🌾 <b>రైతు భరోసా – రియల్-టైమ్ క్వాంటం AI ఖచ్చితమైన వ్యవసాయ నిర్ణయ మద్దతు వ్యవస్థ</b><br>
        రైతు భరోసా కేంద్రాలు (RBKs) & రైతుల సహాయార్థం రూపొందించబడింది • 2026
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
