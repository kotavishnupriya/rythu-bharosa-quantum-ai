"""
Rythu Bharosa - Live Output Generator & Visual Plotter
"""
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from app import FastQuantumKernel, fetch_live_weather, geocode_location, evaluate_agronomic_advisory

MODELS_DIR = "models"
DATA_DIR = "data"
OUTPUT_DIR = "output_previews"
os.makedirs(OUTPUT_DIR, exist_ok=True)

encoder = joblib.load(os.path.join(MODELS_DIR, "encoder.pkl"))
classical_model = joblib.load(os.path.join(MODELS_DIR, "classical_model.pkl"))
quantum_model = joblib.load(os.path.join(MODELS_DIR, "quantum_model.pkl"))
pca_bundle = joblib.load(os.path.join(MODELS_DIR, "pca.pkl"))
X_train_q = joblib.load(os.path.join(MODELS_DIR, "X_train_q.pkl"))
K_train = joblib.load(os.path.join(MODELS_DIR, "K_train.pkl"))
q_kernel = FastQuantumKernel(feature_dimension=4, reps=2, entanglement="linear")

scenarios = [
    ("Rice (Paddy)", "Rice", "Kharif", "Andhra Pradesh", "Guntur, Andhra Pradesh", 2.5),
    ("Cotton (Lint)", "Cotton(lint)", "Kharif", "Andhra Pradesh", "Kurnool, Andhra Pradesh", 4.0),
    ("Dry Chillies", "Dry chillies", "Rabi", "Andhra Pradesh", "Guntur, Andhra Pradesh", 1.5),
    ("Maize (Corn)", "Maize", "Kharif", "Telangana", "Warangal, Telangana", 3.0),
    ("Groundnut", "Groundnut", "Kharif", "Andhra Pradesh", "Anantapur, Andhra Pradesh", 5.0)
]

print("=" * 80)
print("  RYTHU BHAROSA - LIVE REAL-TIME AI & QUANTUM YIELD OUTPUTS")
print("  Data-Driven Decisions for Better Farming")
print("=" * 80)

results = []

for label, crop, season, state, loc, area in scenarios:
    lat, lon, name, src = geocode_location(loc)
    w = fetch_live_weather(lat, lon)
    
    input_df = pd.DataFrame([{
        "crop": crop, "season": season, "state": state, "area": area,
        "fertilizer": 150.0 * area, "pesticide": 2.0 * area,
        "N": 210.5, "P": 24.2, "K": 280.4, "pH": 7.2,
        "avg_temp_c": w["temperature"],
        "total_rainfall_mm": 1100.0,
        "avg_humidity_percent": w["humidity"]
    }])
    
    X_proc = encoder.transform(input_df)
    c_pred = float(np.clip(np.expm1(classical_model.predict(X_proc)[0]), 0.01, None))
    
    X_pca = pca_bundle["pca"].transform(X_proc)
    X_q = pca_bundle["quantum_scaler"].transform(X_pca)
    K_sample = q_kernel.evaluate(X_q, X_train_q)
    q_pred = float(np.clip(np.expm1(quantum_model.predict(K_sample)[0]), 0.01, None))
    
    tot_harvest = c_pred * area
    
    safe_cond = w["condition"].encode("ascii", "ignore").decode()
    print(f"\n[SCENARIO] {label}")
    print(f"  * Location:     {name} ({lat:.3f}N, {lon:.3f}E)")
    print(f"  * Live Weather: {w['temperature']}C | Humidity: {w['humidity']}% | Rain: {w['precipitation']}mm | {safe_cond}")
    print(f"  * Crop / Area:  {crop} ({season}) across {area} ha ({area * 2.471:.1f} acres)")
    print(f"  +----------------------------------------------------------------+")
    print(f"  | Classical AI Yield (Random Forest) : {c_pred:6.2f} t/ha | Harvest: {tot_harvest:6.2f} t |")
    print(f"  | Quantum AI Yield (ZZFeatureMap+SVR): {q_pred:6.2f} t/ha                     |")
    print(f"  +----------------------------------------------------------------+")

    results.append({
        "label": label,
        "crop": crop,
        "location": loc,
        "classical_yield": c_pred,
        "quantum_yield": q_pred,
        "harvest_tonnes": tot_harvest,
        "temp": w["temperature"],
        "humidity": w["humidity"]
    })

print("\n" + "=" * 80)

# Generate Quantum Kernel and Evaluation Summary Charts
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 1. Quantum Kernel Heatmap
sns.heatmap(K_train[:12, :12], cmap="viridis", annot=False, ax=axes[0], cbar_kws={'label': 'Quantum State Overlap |<ψ_i|ψ_j>|²'})
axes[0].set_title("4-Qubit ZZFeatureMap Quantum Fidelity Gram Matrix", fontsize=11, fontweight='bold')
axes[0].set_xlabel("Quantum Support Vector j")
axes[0].set_ylabel("Quantum Support Vector i")

# 2. Classical vs Quantum Predictions Bar Plot
crops = [r["crop"] for r in results]
x_pos = np.arange(len(crops))
width = 0.35

axes[1].bar(x_pos - width/2, [r["classical_yield"] for r in results], width, label="Classical AI (Random Forest)", color="#2e7d32")
axes[1].bar(x_pos + width/2, [r["quantum_yield"] for r in results], width, label="Quantum AI (ZZFeatureMap+SVR)", color="#7b1fa2")
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(crops, rotation=15)
axes[1].set_ylabel("Expected Yield (tonnes/hectare)", fontweight='bold')
axes[1].set_title("Live Yield Estimations Across AP / Telangana Crops", fontsize=11, fontweight='bold')
axes[1].legend()
axes[1].grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "quantum_classical_comparison.png")
plt.savefig(plot_path, dpi=150)
print(f"Saved visual comparison chart to: {plot_path}")
