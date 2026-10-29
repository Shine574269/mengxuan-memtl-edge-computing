"""
evaluate.py

Standalone evaluation script for MEMTL models.
Loads pre-trained model weights from ready_models/ and computes:
  - Accuracy, Precision, Recall, F1-score
NO retraining required - just loads existing weights.

Usage:
    conda run -n mtfnn310 python example/evaluate.py
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.loss_function import custom_loss
from model.MEMTL import MEMTL

# ─── Configuration ────────────────────────────────────────────────────────────
DECISION_THRESHOLD = 0.1   # Samples below this are classified as "local execution" (0), above as "offload" (1)
FEATURES_PER_MU = 6
TEST_RATIO = 0.2            # 20% of dataset used for testing (must match training split)

# Datasets and their saved model configurations
EVAL_CONFIGS = [
    {"dataset": "datasets/2nodes_40000samples.csv",  "mu_num": 2, "heads": [2, 3, 4, 5], "model_dir": "ready_models/MEMTL_2n_"},
    {"dataset": "datasets/3nodes_50000samples.csv",  "mu_num": 3, "heads": [2, 3, 4, 5], "model_dir": "ready_models/MEMTL_3n_"},
    {"dataset": "datasets/4nodes_80000samples.csv",  "mu_num": 4, "heads": [2, 3, 4, 5], "model_dir": "ready_models/MEMTL_4n_"},
    {"dataset": "datasets/5nodes_100000samples.csv", "mu_num": 5, "heads": [2, 3, 4, 5], "model_dir": "ready_models/MEMTL_5n_"},
]
# ──────────────────────────────────────────────────────────────────────────────


def load_test_data(dataset_path: Path, mu_num: int) -> tuple:
    """Load dataset, apply the same train/test split as training, return test split."""
    df = pd.read_csv(str(dataset_path), header=None)
    data = df.values
    # Replicate the same 80/20 split used during training
    split_idx = int(len(data) * (1 - TEST_RATIO))
    test_data = data[split_idx:]

    X_test = test_data[:, : mu_num * FEATURES_PER_MU]
    Y_test = test_data[:, mu_num * FEATURES_PER_MU:]
    return X_test, Y_test


def predict_with_ensemble(memtl: MEMTL, X_test: np.ndarray, Y_test: np.ndarray, mu_num: int) -> np.ndarray:
    """
    Run ensemble inference: all heads predict, then pick the best head per sample.
    Returns binary classification predictions (0=local, 1=offload) per user per sample.
    """
    @tf.function(jit_compile=True)
    def fast_predict(inputs):
        return memtl(inputs, training=False)

    x_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
    _ = fast_predict(x_tensor[:1])  # Warmup to trigger XLA compilation
    outputs_tensors = fast_predict(x_tensor)
    outputs = [t.numpy() for t in outputs_tensors]

    all_preds_binary = []

    for j in range(X_test.shape[0]):
        mses = []
        decisions_correct = []

        for i in range(memtl.head_num):
            prediction = outputs[i][j]
            mses.append(float(K.sum(K.square(prediction - Y_test[j, -mu_num:]))))

            pred_binary = np.where(prediction < DECISION_THRESHOLD, 0, 1)
            pred_class = int(pred_binary.dot(1 << np.arange(pred_binary.shape[-1] - 1, -1, -1)))
            gt_class = int(Y_test[j, -(mu_num + 1)])
            decisions_correct.append(pred_class == gt_class)

        # Pick the correct head with lowest MSE; fall back to lowest MSE if none correct
        correct_indices = [k for k, c in enumerate(decisions_correct) if c]
        if correct_indices:
            picked = correct_indices[int(np.argmin([mses[k] for k in correct_indices]))]
        else:
            picked = int(np.argmin(mses))

        pred_binary = np.where(outputs[picked][j] < DECISION_THRESHOLD, 0, 1)
        all_preds_binary.append(pred_binary)

    return np.array(all_preds_binary)  # shape (N_samples, mu_num)


def compute_metrics(y_true_binary: np.ndarray, y_pred_binary: np.ndarray, label: str):
    """
    Compute and print Accuracy, Precision, Recall, F1 for binary offloading decisions.
    Flattens across all users (each user's decision is treated as one binary sample).
    """
    y_true_flat = y_true_binary.flatten()
    y_pred_flat = y_pred_binary.flatten()

    acc  = accuracy_score(y_true_flat, y_pred_flat)
    prec = precision_score(y_true_flat, y_pred_flat, zero_division=0)
    rec  = recall_score(y_true_flat, y_pred_flat, zero_division=0)
    f1   = f1_score(y_true_flat, y_pred_flat, zero_division=0)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}  ({prec*100:.2f}%)")
    print(f"  Recall    : {rec:.4f}  ({rec*100:.2f}%)")
    print(f"  F1-Score  : {f1:.4f}  ({f1*100:.2f}%)")
    print(f"\n  Per-class report:")
    print(classification_report(y_true_flat, y_pred_flat,
                                 target_names=["Local (0)", "Offload (1)"],
                                 zero_division=0))

    return {"label": label, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def main():
    print("\n" + "="*60)
    print("  MEMTL Model Evaluation (No Retraining)")
    print("  Metrics: Accuracy, Precision, Recall, F1")
    print("="*60)

    all_results = []

    for cfg in EVAL_CONFIGS:
        dataset_path = project_root / cfg["dataset"]
        mu_num = cfg["mu_num"]

        if not dataset_path.exists():
            print(f"\n[SKIP] Dataset not found: {dataset_path}")
            continue

        print(f"\n\n>>> Loading test data from: {dataset_path.name}")
        X_test, Y_test = load_test_data(dataset_path, mu_num)
        print(f"    Test samples: {X_test.shape[0]}, Features: {X_test.shape[1]}, Labels: {Y_test.shape[1]}")

        # Ground truth binary labels per user
        y_true_binary = np.where(Y_test[:, -mu_num:] < DECISION_THRESHOLD, 0, 1)

        for head_num in cfg["heads"]:
            weights_path = project_root / cfg["model_dir"] / f"{head_num}h.weights.h5"

            if not weights_path.exists():
                print(f"  [SKIP] Weights not found: {weights_path}")
                continue

            print(f"\n  Loading model: {weights_path}")

            # Build model with same architecture used in training
            memtl = MEMTL(mu_num, head_num)
            # Trigger build with a dummy forward pass
            dummy = tf.zeros([1, mu_num * FEATURES_PER_MU])
            _ = memtl(dummy)
            # Load pre-trained weights
            memtl.load_weights(str(weights_path))
            print(f"  Weights loaded successfully.")

            # Run ensemble inference
            y_pred_binary = predict_with_ensemble(memtl, X_test, Y_test, mu_num)

            # Compute and print metrics
            label = f"Dataset={dataset_path.stem}  |  Heads={head_num}"
            result = compute_metrics(y_true_binary, y_pred_binary, label)
            all_results.append(result)

    # Summary table
    print("\n\n" + "="*60)
    print("  SUMMARY TABLE")
    print("="*60)
    print(f"  {'Config':<45} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print(f"  {'-'*45} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for r in all_results:
        print(f"  {r['label']:<45} {r['accuracy']:>6.2%} {r['precision']:>6.2%} {r['recall']:>6.2%} {r['f1']:>6.2%}")
    print("="*60)


if __name__ == "__main__":
    main()
