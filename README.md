# 🌾 Rythu Bharosa – Real-Time Quantum AI Precision Agriculture Decision Support System
## రైతు భరోసా – రియల్-టైమ్ క్వాంటం AI ఖచ్చితమైన వ్యవసాయ నిర్ణయ మద్దతు వ్యవస్థ

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rythu-bharosa-quantum-ai-nnenrstukbf87trzghksse.streamlit.app)
[![GitHub License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://python.org)
[![Qiskit Version](https://img.shields.io/badge/Qiskit-1.1%2B-purple.svg)](https://qiskit.org)

🌐 **Live Web Application:** 👉 **[https://rythu-bharosa-quantum-ai-nnenrstukbf87trzghksse.streamlit.app](https://rythu-bharosa-quantum-ai-nnenrstukbf87trzghksse.streamlit.app)**

> **Motto (English):** *"Rythu Bharosa – Helping Farmers Make Better Decisions Using Real-Time Data and AI"*  
> **Motto (తెలుగు):** *"రైతు భరోసా – మెరుగైన వ్యవసాయ నిర్ణయాల కోసం రియల్-టైమ్ డేటా & AI"*

---

## 1. Project Overview & Problem Statement
Smallholder farmers and agricultural officers at **Rythu Bharosa Kendras (RBKs)** across Andhra Pradesh and India face critical challenges due to climate variability, unpredictable monsoons, nutrient mismanagement, and lack of real-time localized decision intelligence.

Many conventional agritech applications rely on fabricated sensor numbers or falsely claim that standard smartphone photos can measure chemical nitrogen or soil moisture. **Rythu Bharosa** is engineered with strict **scientific honesty and real-world integration**:
- Combines **19,500+ verified historical agricultural records** (crop yields, macro-nutrients, weather patterns) from ICAR and Directorate of Economics and Statistics.
- Integrates **Live Open-Meteo Meteorological Telemetry** and **OpenStreetMap Nominatim Geocoding** (100% free, requiring no private API keys).
- Trains and benchmarks a **Classical Ensemble Regressor (Random Forest)** against a **4-Qubit Quantum Kernel Support Vector Regressor (Qiskit ZZFeatureMap)**.
- Delivers transparent, rule-based agronomic risk alerts, state-level soil nutrient benchmarks, and downloadable farmer advisory reports.

---

## 2. Key Features

- **🌾 Dual AI Yield Prediction**: Side-by-side evaluation using tuned Random Forest regression and 4-qubit Quantum Fidelity Kernel regression.
- **🛰️ Live Weather & Geocoding Telemetry**: Real-time ambient temperature, relative humidity, precipitation, wind speed, and WMO atmospheric conditions for any Mandal, Village, or District in India.
- **🚨 Agronomic Risk & Early Warning Engine**: Rule-based detection of thermal heat stress, high-humidity fungal disease pressure, waterlogging hazards, and historical yield gaps.
- **🧪 State-Level Soil Nutrient Profiles**: Contextual benchmark reference for Available Nitrogen ($N$), Phosphorus ($P$), Potassium ($K$), and Soil Reaction ($pH$) across all 30 Indian states.
- **📷 Soil Visual Texture Tool**: Ethical photo inspection with explicit scientific disclaimers emphasizing that smartphone cameras cannot measure chemical nutrients or moisture.
- **📊 Interactive Visualizations**: Plotly gauges, historical crop time-series (1997–2020), Random Forest Mean Decrease in Impurity (MDI) feature importance, and 4-Qubit Quantum Kernel Gram Matrix heatmaps.
- **📄 Downloadable Farmer Advisory Report**: Instant one-click export of structured plain text and CSV field reports for offline farmer distribution.
- **🌐 Telugu & English Dual Support**: Bilingual labels (రైతు భరోసా, అంచనా దిగుబడి, వాతావరణం, రైతు సలహా).

---

## 3. Dataset Architecture

The platform operates on three authentic agricultural and meteorological datasets merged without target leakage:

| Dataset | Records | Description & Key Features |
| :--- | :--- | :--- |
| `data/crop_yield.csv` | 19,689 | Multi-crop yield records across Indian states (1997–2020): `crop`, `season`, `state`, `area`, `fertilizer`, `pesticide`, `yield`. |
| `data/state_soil_data.csv` | 31 | State macro-nutrient reference baselines: Available $N$, $P$, $K$ (kg/ha) and soil $pH$. |
| `data/state_weather_data_1997_2020.csv` | 744 | State-year meteorological profiles: `avg_temp_c`, `total_rainfall_mm`, `avg_humidity_percent`. |
| `data/combined_crop_yield.csv` | 19,577 | Standardized, merged, cleaned dataset used for training and testing. |

> **Target Leakage Prevention:** The `production` column is strictly excluded from feature sets because $\text{Yield} = \frac{\text{Production}}{\text{Area}}$, which would cause artificial data leakage.

---

## 4. Machine Learning & Quantum Methodology

### Classical Model Architecture
- **Preprocessor**: `OneHotEncoder(handle_unknown='ignore')` for categorical attributes (`crop`, `season`, `state`) + `StandardScaler()` for continuous agronomic features ($101$ total dimensions).
- **Algorithm**: `RandomForestRegressor(n_estimators=100, max_depth=18, min_samples_split=4, random_state=42)`.
- **Target Transformation**: $\log(1 + y)$ training for skewed distributions across 55 diverse crop varieties.

### Quantum Machine Learning (QML) Pipeline
1. **Dimensionality Reduction**: Principal Component Analysis (PCA) reduces 101 features to $D = 4$ principal components, scaled to $[-\pi, \pi]$.
2. **Quantum Feature Map**: 4-qubit `ZZFeatureMap` (2 repetitions, linear entanglement) maps classical vectors $\mathbf{x} \in \mathbb{R}^4$ into Hilbert space statevectors $|\psi(\mathbf{x})\rangle \in \mathbb{C}^{16}$:
   $$U_{\Phi(\mathbf{x})} = \exp\left(i \sum_{j} x_j Z_j + \sum_{j < k} (\pi - x_j)(\pi - x_k) Z_j Z_k\right)$$
3. **Quantum Fidelity Kernel**: Measures quantum state fidelity:
   $$K(\mathbf{x}_i, \mathbf{x}_j) = |\langle \psi(\mathbf{x}_j) | \psi(\mathbf{x}_i) \rangle|^2$$
4. **Kernel Support Vector Regressor (SVR)**: Trains on the precomputed quantum Gram matrix $K_{\text{train}}$.

---

## 5. Model Evaluation Metrics

Evaluated on independent test sets on original physical units ($\text{tonnes/ha}$):

| Model | MAE (t/ha) | RMSE (t/ha) | $R^2$ Score | Samples Evaluated |
| :--- | :--- | :--- | :--- | :--- |
| **Classical AI (Random Forest)** | **8.0012** | **127.6440** | **0.9755** | 3,916 |
| **Quantum AI (4-Qubit ZZFeatureMap + SVR)** | 113.6199 | 1130.8450 | -0.0085 | 300 |

### ⚖️ Scientific Honesty & Findings
- Classical Random Forest achieves state-of-the-art performance ($R^2 = 0.9755$) because it preserves the high-dimensional one-hot encodings of 55 distinct crop types.
- The 4-qubit Quantum Kernel experiences an information bottleneck when 101 features are compressed into 4 principal components.
- **The application prioritizes the Classical Random Forest model for field production recommendations while presenting the Quantum pipeline as an exploratory research component.**

---

## 6. Project Structure

```
crop-detection/
│
├── app.py                     # Streamlit web application entry point
├── prepare_data.py            # Dataset download, standardization, and merging pipeline
├── train_and_save.py          # Classical and Quantum model training and evaluation script
├── test_pipeline.py           # Automated unit and integration test suite
├── requirements.txt           # Dependency specification
├── README.md                  # System documentation
│
├── data/
│   ├── crop_yield.csv
│   ├── state_soil_data.csv
│   ├── state_weather_data_1997_2020.csv
│   ├── combined_crop_yield.csv
│   └── model_metrics.csv
│
├── models/
│   ├── classical_model.pkl    # Trained RandomForestRegressor
│   ├── quantum_model.pkl      # Trained Quantum Kernel SVR
│   ├── scaler.pkl             # StandardScaler for numerical features
│   ├── encoder.pkl            # OneHotEncoder / ColumnTransformer
│   ├── pca.pkl                # PCA + Quantum MinMax scaler bundle
│   ├── X_train_q.pkl          # Support points for quantum kernel
│   └── K_train.pkl            # Precomputed quantum Gram matrix
│
└── .streamlit/
    └── config.toml            # Agricultural UI theme configuration
```

---

## 7. Installation & Local Execution

### Prerequisites
- Python 3.10, 3.11, or 3.12+

### Step-by-Step Setup

```bash
# 1. Navigate to the project directory
cd crop-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Prepare data and train models (if not already trained)
python prepare_data.py
python train_and_save.py

# 4. Run automated test suite
python test_pipeline.py

# 5. Launch the Streamlit application
streamlit run app.py
```

The application will be accessible at `http://localhost:8501`.

---

## 8. Deployment (Streamlit Community Cloud)

1. Push the repository to GitHub.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), link the repository.
3. Set **Main file path** to `app.py`.
4. The application operates **out-of-the-box without requiring API keys** (Open-Meteo & Nominatim operate via public endpoints). Optional API keys can be provided in `.streamlit/secrets.toml` via `st.secrets`.

---

## 9. Important Limitations & Disclaimers

> [!CAUTION]
> 1. **Decision Support Only**: This platform is a computational decision-support tool and does **not** replace on-site agricultural extension officers, certified agronomists, or official government advisories.
> 2. **Soil Health Cards**: State-level soil nutrient values are agro-climatic benchmarks. Farmers must obtain field-specific chemical Soil Health Card (SHC) laboratory tests before purchasing or applying fertilizers.
> 3. **No Photo Chemical Sensing**: Smartphone RGB cameras cannot detect soil chemical elements (N, P, K, pH) or sub-surface moisture. The visual tool is strictly limited to surface texture appraisal.
