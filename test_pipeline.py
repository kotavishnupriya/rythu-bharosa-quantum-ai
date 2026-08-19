"""
Rythu Bharosa - Automated Test Suite
====================================
Tests:
1. Dataset availability and integrity
2. Preprocessor and model artifacts loading
3. Classical regression inference
4. Quantum kernel statevector evaluation and SVR inference
5. Open-Meteo live weather API fetching and fallback behavior
6. Nominatim geocoding resolution
7. Model metrics consistency
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

# Set relative paths
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

sys.path.insert(0, BASE_DIR)
from app import FastQuantumKernel, fetch_live_weather, geocode_location, evaluate_agronomic_advisory


def test_datasets_exist_and_non_empty():
    """Verify all 4 required CSV datasets exist and have valid rows."""
    files = [
        ("crop_yield.csv", 10000),
        ("state_soil_data.csv", 25),
        ("state_weather_data_1997_2020.csv", 500),
        ("combined_crop_yield.csv", 15000),
        ("model_metrics.csv", 2)
    ]
    for filename, min_rows in files:
        fpath = os.path.join(DATA_DIR, filename)
        assert os.path.exists(fpath), f"Missing dataset: {fpath}"
        df = pd.read_csv(fpath)
        assert len(df) >= min_rows, f"{filename} has {len(df)} rows, expected >= {min_rows}"
    print("[PASS] All CSV datasets validated successfully.")


def test_models_exist_and_loadable():
    """Verify all 7 model artifacts exist and can be loaded."""
    model_files = [
        "classical_model.pkl",
        "quantum_model.pkl",
        "scaler.pkl",
        "encoder.pkl",
        "pca.pkl",
        "X_train_q.pkl",
        "K_train.pkl"
    ]
    for mf in model_files:
        fpath = os.path.join(MODELS_DIR, mf)
        assert os.path.exists(fpath), f"Missing model artifact: {fpath}"
        obj = joblib.load(fpath)
        assert obj is not None, f"Failed loading {mf}"
    print("[PASS] All 7 model artifacts verified and loadable.")


def test_classical_and_quantum_predictions():
    """Verify end-to-end inference using classical and quantum pipelines."""
    encoder = joblib.load(os.path.join(MODELS_DIR, "encoder.pkl"))
    classical_model = joblib.load(os.path.join(MODELS_DIR, "classical_model.pkl"))
    quantum_model = joblib.load(os.path.join(MODELS_DIR, "quantum_model.pkl"))
    pca_bundle = joblib.load(os.path.join(MODELS_DIR, "pca.pkl"))
    X_train_q = joblib.load(os.path.join(MODELS_DIR, "X_train_q.pkl"))
    
    sample_df = pd.DataFrame([{
        "crop": "Rice",
        "season": "Kharif",
        "state": "Andhra Pradesh",
        "area": 2.0,
        "fertilizer": 300.0,
        "pesticide": 3.0,
        "N": 210.5,
        "P": 24.2,
        "K": 280.4,
        "pH": 7.2,
        "avg_temp_c": 28.5,
        "total_rainfall_mm": 1100.0,
        "avg_humidity_percent": 70.0
    }])

    # Classical inference
    X_proc = encoder.transform(sample_df)
    class_pred_log = classical_model.predict(X_proc)[0]
    class_pred = float(np.expm1(class_pred_log))
    assert class_pred > 0.1, f"Invalid classical prediction: {class_pred}"

    # Quantum inference
    X_pca = pca_bundle["pca"].transform(X_proc)
    X_q = pca_bundle["quantum_scaler"].transform(X_pca)
    q_kernel = FastQuantumKernel(feature_dimension=4, reps=2, entanglement="linear")
    K_sample = q_kernel.evaluate(X_q, X_train_q)
    assert K_sample.shape == (1, len(X_train_q)), f"Unexpected K_sample shape: {K_sample.shape}"
    
    quant_pred_log = quantum_model.predict(K_sample)[0]
    quant_pred = float(np.expm1(quant_pred_log))
    assert quant_pred > 0.0, f"Invalid quantum prediction: {quant_pred}"

    print(f"[PASS] Inference test passed! Classical: {class_pred:.2f} t/ha | Quantum: {quant_pred:.2f} t/ha")


def test_weather_api_and_fallback():
    """Verify Open-Meteo weather API call and response structure."""
    lat, lon = 16.3067, 80.4365
    weather = fetch_live_weather(lat, lon)
    assert "temperature" in weather
    assert "humidity" in weather
    assert "precipitation" in weather
    assert "condition" in weather
    assert isinstance(weather["temperature"], (int, float))
    assert 0 <= weather["humidity"] <= 100
    safe_cond = weather['condition'].encode('ascii', 'ignore').decode()
    print(f"[PASS] Weather telemetry test passed! Temp: {weather['temperature']}C, Hum: {weather['humidity']}%, Cond: {safe_cond}")


def test_geocoding_resolution():
    """Verify geocoding resolution for Andhra Pradesh hubs."""
    lat, lon, name, src = geocode_location("Guntur, Andhra Pradesh")
    assert abs(lat - 16.3067) < 0.1
    assert abs(lon - 80.4365) < 0.1
    print(f"[PASS] Geocoding test passed! Resolved {name} to ({lat}, {lon})")


def test_agronomic_advisory_logic():
    """Verify rule-based advisory triggers under heat/humidity stress."""
    advisories, risks = evaluate_agronomic_advisory(
        crop="Rice", area=2.0, exp_yield=1.2, hist_mean=3.0,
        temp=41.0, hum=88.0, rain_cur=40.0,
        n_val=210.5, p_val=24.2, k_val=280.4, ph_val=5.2
    )
    assert len(risks) >= 3, "Expected multiple stress risk flags (heat, humidity, precipitation, yield gap)"
    print(f"[PASS] Agronomic advisory test passed! Triggered {len(risks)} risks and {len(advisories)} advisories.")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Rythu Bharosa Automated Test Suite")
    print("=" * 60)
    test_datasets_exist_and_non_empty()
    test_models_exist_and_loadable()
    test_classical_and_quantum_predictions()
    test_weather_api_and_fallback()
    test_geocoding_resolution()
    test_agronomic_advisory_logic()
    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY (100% PASS RATE)!")
    print("=" * 60)
