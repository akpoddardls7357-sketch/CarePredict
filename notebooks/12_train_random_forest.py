import os
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# CAREPREDICT - RANDOM FOREST
# ============================================================

DATA_DIR = "data/processed/preprocessed"
MODEL_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 70)
print("CAREPREDICT - RANDOM FOREST")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

X_train = np.load(
    f"{DATA_DIR}/X_train.npy"
)

X_validation = np.load(
    f"{DATA_DIR}/X_validation.npy"
)

y_train = np.load(
    f"{DATA_DIR}/y_train.npy"
)

y_validation = np.load(
    f"{DATA_DIR}/y_validation.npy"
)

print("\nData loaded:")

print(
    f"Training:   {X_train.shape}"
)

print(
    f"Validation: {X_validation.shape}"
)

# ------------------------------------------------------------
# 2. TRAIN RANDOM FOREST
# ------------------------------------------------------------

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=14,
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

print("Training complete.")

# ------------------------------------------------------------
# 3. PREDICTIONS
# ------------------------------------------------------------

y_pred = model.predict(
    X_validation
)

y_probability = model.predict_proba(
    X_validation
)[:, 1]

# ------------------------------------------------------------
# 4. METRICS
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_validation,
    y_pred
)

precision = precision_score(
    y_validation,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_validation,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_validation,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_validation,
    y_probability
)

pr_auc = average_precision_score(
    y_validation,
    y_probability
)

cm = confusion_matrix(
    y_validation,
    y_pred
)

print("\n" + "=" * 70)
print("VALIDATION PERFORMANCE")
print("=" * 70)

print(
    f"\nAccuracy:  {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall:    {recall:.4f}"
)

print(
    f"F1 Score:  {f1:.4f}"
)

print(
    f"ROC-AUC:   {roc_auc:.4f}"
)

print(
    f"PR-AUC:    {pr_auc:.4f}"
)

print("\nConfusion Matrix:")

print(cm)

print("\nClassification Report:")

print(
    classification_report(
        y_validation,
        y_pred,
        target_names=[
            "No Readmission",
            "30-Day Readmission"
        ],
        zero_division=0
    )
)

# ------------------------------------------------------------
# 5. FEATURE IMPORTANCE
# ------------------------------------------------------------

feature_names = np.genfromtxt(
    f"{DATA_DIR}/feature_names.csv",
    delimiter=",",
    dtype=str,
    skip_header=1
)

importances = model.feature_importances_

importance_order = np.argsort(
    importances
)[::-1]

print("\nTop 20 Feature Importances:")

for rank, index in enumerate(
    importance_order[:20],
    start=1
):

    print(
        f"{rank:02d}. "
        f"{feature_names[index]}: "
        f"{importances[index]:.6f}"
    )

# ------------------------------------------------------------
# 6. SAVE MODEL
# ------------------------------------------------------------

model_path = (
    f"{MODEL_DIR}/random_forest.pkl"
)

joblib.dump(
    model,
    model_path
)

# ------------------------------------------------------------
# 7. SAVE PREDICTIONS
# ------------------------------------------------------------

np.save(
    f"{RESULTS_DIR}/rf_validation_predictions.npy",
    y_pred
)

np.save(
    f"{RESULTS_DIR}/rf_validation_probabilities.npy",
    y_probability
)

# ------------------------------------------------------------
# 8. SAVE FEATURE IMPORTANCE
# ------------------------------------------------------------

importance_df = []

for index in importance_order:

    importance_df.append([
        feature_names[index],
        importances[index]
    ])

import pandas as pd

importance_df = pd.DataFrame(
    importance_df,
    columns=[
        "feature",
        "importance"
    ]
)

importance_df.to_csv(
    f"{RESULTS_DIR}/random_forest_feature_importance.csv",
    index=False
)

# ------------------------------------------------------------
# 9. SAVE METRICS
# ------------------------------------------------------------

with open(
    f"{RESULTS_DIR}/random_forest_metrics.txt",
    "w"
) as f:

    f.write(
        "CAREPREDICT - RANDOM FOREST\n"
    )

    f.write("=" * 60 + "\n\n")

    f.write(
        f"Accuracy: {accuracy:.6f}\n"
    )

    f.write(
        f"Precision: {precision:.6f}\n"
    )

    f.write(
        f"Recall: {recall:.6f}\n"
    )

    f.write(
        f"F1: {f1:.6f}\n"
    )

    f.write(
        f"ROC-AUC: {roc_auc:.6f}\n"
    )

    f.write(
        f"PR-AUC: {pr_auc:.6f}\n"
    )

    f.write(
        "\nConfusion Matrix:\n"
    )

    f.write(
        str(cm)
    )

print(
    "\nModel saved to:"
)

print(model_path)

print(
    "\nMetrics saved to:"
)

print(
    f"{RESULTS_DIR}/random_forest_metrics.txt"
)

print("\n" + "=" * 70)
print("RANDOM FOREST COMPLETE")
print("=" * 70)
