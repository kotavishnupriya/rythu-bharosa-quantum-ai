"""
Rythu Bharosa - Machine Learning & Quantum Training Pipeline
============================================================
Trains:
1. Classical Machine Learning Model (RandomForestRegressor)
2. Quantum Kernel Model (PCA -> 4-Qubit ZZFeatureMap -> FidelityQuantumKernel -> SVR)

Evaluates both models using MAE, RMSE, and R2.
Saves all model artifacts and metrics to models/ and data/model_metrics.csv.
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Qiskit imports
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


class FastQuantumKernel:
    """
    Exact Statevector implementation of Qiskit ZZFeatureMap Quantum Fidelity Kernel.
    Computes K(x_i, x_j) = |<psi(x_j)|psi(x_i)>|^2 using 4-qubit ZZ feature mapping.
    """
    def __init__(self, feature_dimension=4, reps=2, entanglement="linear"):
        self.feature_dimension = feature_dimension
        self.reps = reps
        self.entanglement = entanglement
        self.feature_map = ZZFeatureMap(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement
        )

    def _compute_statevectors(self, X: np.ndarray) -> np.ndarray:
        """Computes the 2^n complex statevector for each n-dimensional input vector."""
        statevectors = []
        for x in X:
            bound_circuit = self.feature_map.assign_parameters(x)
            sv = Statevector.from_instruction(bound_circuit)
            statevectors.append(sv.data)
        return np.array(statevectors, dtype=complex)

    def evaluate(self, X: np.ndarray, Y: np.ndarray = None) -> np.ndarray:
        """
        Evaluates the quantum kernel Gram matrix between X and Y (or X and X).
        Returns matrix of shape (len(X), len(Y)).
        """
        psi_X = self._compute_statevectors(X)
        if Y is None:
            # Overlap matrix Psi_X @ Psi_X.conj().T
            overlap = np.dot(psi_X, psi_X.conj().T)
        else:
            psi_Y = self._compute_statevectors(Y)
            overlap = np.dot(psi_X, psi_Y.conj().T)
        
        # Fidelity is the squared magnitude of state overlap
        fidelity_kernel = np.abs(overlap) ** 2
        return np.clip(fidelity_kernel, 0.0, 1.0)


def train_and_evaluate():
    print("=" * 65)
    print("Rythu Bharosa - Model Training & Evaluation Pipeline")
    print("=" * 65)

    csv_path = os.path.join(DATA_DIR, "combined_crop_yield.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}. Please run prepare_data.py first.")

    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Feature definitions - strictly omitting 'production' to prevent target leakage
    categorical_cols = ["crop", "season", "state"]
    numeric_cols = [
        "area", "fertilizer", "pesticide", 
        "N", "P", "K", "pH", 
        "avg_temp_c", "total_rainfall_mm", "avg_humidity_percent"
    ]
    target_col = "yield"

    print(f"Dataset Shape: {df.shape}")
    print(f"Features: Categorical={categorical_cols}, Numerical={numeric_cols}")
    print(f"Target: {target_col}")

    X = df[categorical_cols + numeric_cols]
    y = df[target_col].values

    # Train / Test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set: {X_train.shape[0]:,} samples | Test set: {X_test.shape[0]:,} samples")

    # 1. Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ("num", StandardScaler(), numeric_cols)
        ]
    )

    print("Fitting preprocessor on training data...")
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    print(f"Processed feature dimensions: {X_train_proc.shape[1]}")

    # Log transform of target for numerical stability in regression
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    # ---------------------------------------------------------
    # 2. Classical Machine Learning Model
    # ---------------------------------------------------------
    print("\n--- Training Classical Model (Random Forest Regressor) ---")
    start_t = time.time()
    classical_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=18,
        min_samples_split=4,
        n_jobs=-1,
        random_state=42
    )
    classical_model.fit(X_train_proc, y_train_log)
    class_train_time = time.time() - start_t
    print(f"Classical model trained in {class_train_time:.2f} seconds.")

    # Evaluate Classical Model on Original Scale (tonnes/ha)
    class_preds_log = classical_model.predict(X_test_proc)
    class_preds = np.expm1(class_preds_log)
    class_preds = np.clip(class_preds, 0, None)

    class_mae = mean_absolute_error(y_test, class_preds)
    class_rmse = np.sqrt(mean_squared_error(y_test, class_preds))
    class_r2 = r2_score(y_test, class_preds)

    print(f"Classical Model Performance (Test Set):")
    print(f"  MAE:  {class_mae:.4f} tonnes/ha")
    print(f"  RMSE: {class_rmse:.4f} tonnes/ha")
    print(f"  R2:   {class_r2:.4f}")

    # ---------------------------------------------------------
    # 3. Quantum Kernel Model (ZZFeatureMap + QuantumKernel + SVR)
    # ---------------------------------------------------------
    print("\n--- Training Quantum Machine Learning Model ---")
    print("Dimensionality Reduction: PCA (n_components=4) -> MinMax Scale to [-pi, pi]")
    
    pca = PCA(n_components=4, random_state=42)
    X_train_pca = pca.fit_transform(X_train_proc)
    X_test_pca = pca.transform(X_test_proc)
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_} (Total: {np.sum(pca.explained_variance_ratio_):.3f})")

    quantum_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
    X_train_q_all = quantum_scaler.fit_transform(X_train_pca)
    X_test_q_all = quantum_scaler.transform(X_test_pca)

    # Sample representative training support set of N=500 points
    n_quantum_train = 500
    np.random.seed(42)
    train_indices = np.random.choice(len(X_train_q_all), size=n_quantum_train, replace=False)
    X_train_q = X_train_q_all[train_indices]
    y_train_q_log = y_train_log[train_indices]
    y_train_q = y_train[train_indices]

    print(f"Building 4-Qubit ZZFeatureMap (reps=2, linear entanglement)...")
    quantum_kernel = FastQuantumKernel(feature_dimension=4, reps=2, entanglement="linear")

    print(f"Computing Quantum Kernel Gram Matrix K_train ({n_quantum_train}x{n_quantum_train})...")
    start_q = time.time()
    K_train = quantum_kernel.evaluate(X_train_q)
    q_kernel_time = time.time() - start_q
    print(f"Quantum Kernel Gram Matrix computed in {q_kernel_time:.2f} seconds.")

    # Train SVR with precomputed Quantum Kernel
    print("Fitting SVR on Quantum Kernel...")
    quantum_svr = SVR(kernel="precomputed", C=10.0, epsilon=0.05)
    quantum_svr.fit(K_train, y_train_q_log)

    # Evaluate Quantum Model on a representative test subset
    n_quantum_test = 300
    test_indices = np.random.choice(len(X_test_q_all), size=n_quantum_test, replace=False)
    X_test_q_eval = X_test_q_all[test_indices]
    y_test_q_eval = y_test[test_indices]

    print(f"Evaluating Quantum Kernel on Test Subset ({n_quantum_test} samples)...")
    K_test = quantum_kernel.evaluate(X_test_q_eval, X_train_q)
    q_preds_log = quantum_svr.predict(K_test)
    q_preds = np.expm1(q_preds_log)
    q_preds = np.clip(q_preds, 0, None)

    q_mae = mean_absolute_error(y_test_q_eval, q_preds)
    q_rmse = np.sqrt(mean_squared_error(y_test_q_eval, q_preds))
    q_r2 = r2_score(y_test_q_eval, q_preds)

    print(f"Quantum Model Performance (Test Subset):")
    print(f"  MAE:  {q_mae:.4f} tonnes/ha")
    print(f"  RMSE: {q_rmse:.4f} tonnes/ha")
    print(f"  R2:   {q_r2:.4f}")

    # ---------------------------------------------------------
    # 4. Save Models and Metrics
    # ---------------------------------------------------------
    print("\n--- Saving Model Artifacts ---")
    joblib.dump(classical_model, os.path.join(MODELS_DIR, "classical_model.pkl"))
    joblib.dump(quantum_svr, os.path.join(MODELS_DIR, "quantum_model.pkl"))
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, "encoder.pkl"))
    joblib.dump(preprocessor.named_transformers_["num"], os.path.join(MODELS_DIR, "scaler.pkl"))
    
    # Save PCA + Quantum MinMax Scaler pipeline
    pca_bundle = {"pca": pca, "quantum_scaler": quantum_scaler}
    joblib.dump(pca_bundle, os.path.join(MODELS_DIR, "pca.pkl"))
    joblib.dump(X_train_q, os.path.join(MODELS_DIR, "X_train_q.pkl"))
    joblib.dump(K_train, os.path.join(MODELS_DIR, "K_train.pkl"))

    # Save metrics comparison CSV
    metrics_df = pd.DataFrame([
        {
            "model": "Classical AI (Random Forest Regressor)",
            "MAE": round(float(class_mae), 4),
            "RMSE": round(float(class_rmse), 4),
            "R2": round(float(class_r2), 4),
            "Samples": len(y_test),
            "Training Time (s)": round(class_train_time, 2)
        },
        {
            "model": "Quantum AI (ZZFeatureMap + SVR)",
            "MAE": round(float(q_mae), 4),
            "RMSE": round(float(q_rmse), 4),
            "R2": round(float(q_r2), 4),
            "Samples": n_quantum_test,
            "Training Time (s)": round(q_kernel_time, 2)
        }
    ])

    metrics_csv_path = os.path.join(DATA_DIR, "model_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Saved evaluation metrics to {metrics_csv_path}")
    print(metrics_df.to_string(index=False))

    print("\nAll models and pipeline artifacts successfully created and saved!")

if __name__ == "__main__":
    train_and_evaluate()
