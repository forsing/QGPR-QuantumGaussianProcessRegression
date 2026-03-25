# Quantum Gaussian Process Regression (QGPR) for Lottery Prediction
# Lottery prediction generated using a manual Quantum Gaussian Process implementation.
# Quantum Regression Model with Qiskit

# v2: df.copy(); jitter na kernel; train_window/reps; clip po poziciji (sortirana 7/39).

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector

from qiskit_machine_learning.utils import algorithm_globals
import random

# ================= SEED PARAMETERS =================
SEED = 39
random.seed(SEED)
np.random.seed(SEED)
algorithm_globals.random_seed = SEED
# ==================================================


# Use the existing dataframe
df_raw = pd.read_csv('/Users/4c/Desktop/GHQ/data/loto7hh_4586_k24.csv')
# 4548 historical draws of Lotto 7/39 (Serbia)

_MIN_POS = np.array([1, 2, 3, 4, 5, 6, 7], dtype=int)
_MAX_POS = np.array([33, 34, 35, 36, 37, 38, 39], dtype=int)


def compute_quantum_kernel_matrix(X1, X2, feature_map):
    """
    Computes the quantum kernel matrix K(i, j) = |<phi(x1_i)|phi(x2_j)>|^2
    """
    n1 = len(X1)
    n2 = len(X2)
    kernel_matrix = np.zeros((n1, n2))

    # Pre-compute statevectors for efficiency
    sv1 = [Statevector.from_instruction(feature_map.assign_parameters(x)) for x in X1]
    sv2 = [Statevector.from_instruction(feature_map.assign_parameters(x)) for x in X2]

    for i in range(n1):
        for j in range(n2):
            # Fidelity = |<psi|phi>|^2
            fidelity = np.abs(np.vdot(sv1[i].data, sv2[j].data)) ** 2
            kernel_matrix[i, j] = fidelity

    return kernel_matrix


def quantum_gaussian_process_predict(df):
    df = df.copy()
    cols = ['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6', 'Num7']
    predictions = {}

    # Model Hyperparameters
    num_lags = 2
    num_qubits = 2
    train_window = 22  # Small window for computational efficiency
    alpha = 0.15  # Noise variance (regularization)
    kernel_jitter = 1e-9

    # Define a ZZFeatureMap for 2 qubits
    feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=2, entanglement='linear')

    for idx, col in enumerate(cols):
        # 1. Feature Engineering: 2 Lags
        df_col = pd.DataFrame(df[col])
        for i in range(1, num_lags + 1):
            df_col[f'lag_{i}'] = df_col[col].shift(i)

        df_col = df_col.dropna().tail(train_window + 1)

        X = df_col[[f'lag_{i}' for i in range(1, num_lags + 1)]].values
        y = df_col[col].values.astype(np.float64)

        X_train = X[:-1]
        y_train = y[:-1]
        X_next = X[-1:]

        # 2. Scaling to [0, 2*pi] for the quantum feature map
        scaler_x = MinMaxScaler(feature_range=(0, 2 * np.pi))
        X_train_scaled = scaler_x.fit_transform(X_train)
        X_next_scaled = scaler_x.transform(X_next)

        # 3. Compute Quantum Kernel Matrices
        K_train = compute_quantum_kernel_matrix(X_train_scaled, X_train_scaled, feature_map)
        K_test = compute_quantum_kernel_matrix(X_next_scaled, X_train_scaled, feature_map)

        # 4. Manual Gaussian Process Prediction Logic
        # y_pred = K_test * (K_train + alpha*I)^-1 * y_train
        # We solve (K_train + alpha*I) * beta = y_train for beta
        n = len(K_train)
        K_reg = K_train + alpha * np.eye(n) + kernel_jitter * np.eye(n)
        try:
            beta = np.linalg.solve(K_reg, y_train)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(K_reg, y_train, rcond=None)[0]

        y_pred = np.dot(K_test, beta)

        # 5. Result Extraction
        lo, hi = int(_MIN_POS[idx]), int(_MAX_POS[idx])
        predictions[col] = int(round(np.clip(y_pred[0], lo, hi)))

    return predictions


print()
print("Computing predictions using Quantum Gaussian Process Regression (QGPR) ...")
print()
q_gpr_results = quantum_gaussian_process_predict(df_raw)

# Format for display
q_gpr_df = pd.DataFrame([q_gpr_results])
# q_gpr_df.index = ['Quantum Gaussian Process Regression (QGPR) Prediction']

print()
print("Lottery prediction generated using a manual Quantum Gaussian Process implementation.")
print()
print("Quantum Gaussian Process Regression (QGPR) Results:")
print(q_gpr_df.to_string(index=True))
print()
"""
Computing predictions using Quantum Gaussian Process Regression (QGPR) ...

Lottery prediction generated using a manual Quantum Gaussian Process implementation.

Quantum Gaussian Process Regression (QGPR) Results:
   Num1  Num2  Num3  Num4  Num5  Num6  Num7
0     x     y    13    26    29    25    z
"""



"""
Quantum Gaussian Process Regression (QGPR).

v2: df.copy(); num_lags = 2 i num_qubits = 2 ostaju usklađeni sa ZZFeatureMap; reps 1→2, train_window 15→22, alpha 0.1→0.15 + kernel_jitter na dijagonali; np.linalg.solve sa lstsq rezervom; predikcija clip na dozvoljen opseg po poziciji.

(v2: num_qubits=num_lags=2; ZZFeatureMap reps=2; alpha=0.15 + kernel jitter; solve sa lstsq fallback; clip po poziciji.)
"""
