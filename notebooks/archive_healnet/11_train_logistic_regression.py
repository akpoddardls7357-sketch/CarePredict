import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

# ============================================================
# CAREPREDICT - STEP 14
# Logistic Regression Baseline
# ============================================================

# ------------------------------------------------------------
# 1. Load preprocessed data
# ------------------------------------------------------------
X_train = np.load("../data/processed/X_train.npy")
X_val = np.load("../data/processed/X_validation.npy")

y_train = np.load("../data/processed/y_train.npy")
y_val = np.load("../data/processed/y_validation.npy")

print("\n" + "=" * 70)
print("CAREPREDICT - LOGISTIC REGRESSION")
print("=" * 70)

print(f"\nTraining features   : {X_train.shape}")
print(f"Validation features : {X_val.shape}")

# ------------------------------------------------------------
# 2. Create model
# ------------------------------------------------------------
model = LogisticRegression(
    class_weight="balanced",
    max_iter=2000,
    random_state=42
)

print("\n" + "=" * 70)
print("1. MODEL CONFIGURATION")
print("=" * 70)

print("Model          : Logistic Regression")
print("Class weighting: balanced")
print("Max iterations : 2000")

# ------------------------------------------------------------
# 3. Train
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("2. TRAINING")
print("=" * 70)

model.fit(X_train, y_train)

print("✓ Logistic Regression training complete.")

# ------------------------------------------------------------
# 4. Predictions
# ------------------------------------------------------------
y_pred = model.predict(X_val)
y_prob = model.predict_proba(X_val)[:, 1]

# ------------------------------------------------------------
# 5. Metrics
# ------------------------------------------------------------
accuracy = accuracy_score(y_val, y_pred)

precision = precision_score(
    y_val,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_val,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_val,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_val,
    y_prob
)

pr_auc = average_precision_score(
    y_val,
    y_prob
)

# ------------------------------------------------------------
# 6. Display results
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("3. VALIDATION PERFORMANCE")
print("=" * 70)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print(f"PR-AUC    : {pr_auc:.4f}")

# ------------------------------------------------------------
# 7. Save model
# ------------------------------------------------------------
model_path = "../models/logistic_regression.pkl"

joblib.dump(
    model,
    model_path
)

print("\n" + "=" * 70)
print("4. MODEL SAVED")
print("=" * 70)

print(f"✓ {model_path}")

# ------------------------------------------------------------
# 8. Save validation predictions
# ------------------------------------------------------------
np.save(
    "../results/logistic_val_predictions.npy",
    y_pred
)

np.save(
    "../results/logistic_val_probabilities.npy",
    y_prob
)

print("✓ Validation predictions saved.")

# ------------------------------------------------------------
# 9. Final summary
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 14 COMPLETE")
print("=" * 70)

print(
    f"""
Logistic Regression baseline:

Accuracy  : {accuracy:.4f}
Precision : {precision:.4f}
Recall    : {recall:.4f}
F1-Score  : {f1:.4f}
ROC-AUC   : {roc_auc:.4f}
PR-AUC    : {pr_auc:.4f}
"""
)

print("=" * 70)
