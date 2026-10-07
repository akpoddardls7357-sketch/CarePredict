import os
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)


# ============================================================
# CAREPREDICT - THRESHOLD OPTIMIZATION
# ============================================================

DATA_DIR = "data/processed/preprocessed"
MODEL_PATH = "models/xgboost_focused_tuned.pkl"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 70)
print("CAREPREDICT - THRESHOLD OPTIMIZATION")
print("=" * 70)


# ------------------------------------------------------------
# 1. LOAD VALIDATION DATA
# ------------------------------------------------------------

X_validation = np.load(
    f"{DATA_DIR}/X_validation.npy"
)

y_validation = np.load(
    f"{DATA_DIR}/y_validation.npy"
)

print("\nValidation data:")
print(f"Features: {X_validation.shape}")
print(f"Samples:  {len(y_validation)}")


# ------------------------------------------------------------
# 2. LOAD BEST XGBOOST MODEL
# ------------------------------------------------------------

print("\nLoading tuned XGBoost model...")

model = joblib.load(MODEL_PATH)

print("Model loaded:")
print(MODEL_PATH)


# ------------------------------------------------------------
# 3. GET PREDICTED PROBABILITIES
# ------------------------------------------------------------

print("\nGenerating validation probabilities...")

probabilities = model.predict_proba(
    X_validation
)[:, 1]

roc_auc = roc_auc_score(
    y_validation,
    probabilities
)

pr_auc = average_precision_score(
    y_validation,
    probabilities
)

print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC:  {pr_auc:.4f}")


# ------------------------------------------------------------
# 4. TEST MULTIPLE THRESHOLDS
# ------------------------------------------------------------

thresholds = np.arange(
    0.10,
    0.71,
    0.01
)

results = []


for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_validation,
        predictions
    )

    precision = precision_score(
        y_validation,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_validation,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_validation,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        y_validation,
        predictions
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp)

    results.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp
    })


results_df = pd.DataFrame(results)


# ------------------------------------------------------------
# 5. FIND BEST THRESHOLDS
# ------------------------------------------------------------

best_f1 = results_df.loc[
    results_df["f1"].idxmax()
]

best_recall = results_df.loc[
    results_df["recall"].idxmax()
]

best_precision = results_df.loc[
    results_df["precision"].idxmax()
]


# ------------------------------------------------------------
# 6. PRINT RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("THRESHOLD RESULTS")
print("=" * 70)

print(
    "\n{:<10} {:<12} {:<12} {:<12} {:<12} {:<12}".format(
        "Threshold",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Specificity"
    )
)

print("-" * 70)

for _, row in results_df.iterrows():

    print(
        "{:<10.2f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}".format(
            row["threshold"],
            row["accuracy"],
            row["precision"],
            row["recall"],
            row["f1"],
            row["specificity"]
        )
    )


# ------------------------------------------------------------
# 7. BEST F1 THRESHOLD
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BEST F1 THRESHOLD")
print("=" * 70)

print(
    f"\nThreshold:   {best_f1['threshold']:.2f}"
)

print(
    f"Accuracy:    {best_f1['accuracy']:.4f}"
)

print(
    f"Precision:   {best_f1['precision']:.4f}"
)

print(
    f"Recall:      {best_f1['recall']:.4f}"
)

print(
    f"F1:          {best_f1['f1']:.4f}"
)

print(
    f"Specificity: {best_f1['specificity']:.4f}"
)


# ------------------------------------------------------------
# 8. BEST RECALL
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BEST RECALL THRESHOLD")
print("=" * 70)

print(
    f"\nThreshold:   {best_recall['threshold']:.2f}"
)

print(
    f"Precision:   {best_recall['precision']:.4f}"
)

print(
    f"Recall:      {best_recall['recall']:.4f}"
)

print(
    f"F1:          {best_recall['f1']:.4f}"
)


# ------------------------------------------------------------
# 9. BEST PRECISION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BEST PRECISION THRESHOLD")
print("=" * 70)

print(
    f"\nThreshold:   {best_precision['threshold']:.2f}"
)

print(
    f"Precision:   {best_precision['precision']:.4f}"
)

print(
    f"Recall:      {best_precision['recall']:.4f}"
)

print(
    f"F1:          {best_precision['f1']:.4f}"
)


# ------------------------------------------------------------
# 10. SAVE RESULTS
# ------------------------------------------------------------

results_path = (
    f"{RESULTS_DIR}/threshold_optimization.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# ------------------------------------------------------------
# 11. SAVE RECOMMENDED THRESHOLD
# ------------------------------------------------------------

threshold_path = (
    f"{RESULTS_DIR}/optimal_threshold.txt"
)

with open(
    threshold_path,
    "w"
) as f:

    f.write(
        "CAREPREDICT - OPTIMAL THRESHOLD\n"
    )

    f.write("=" * 50 + "\n\n")

    f.write(
        f"ROC-AUC: {roc_auc:.6f}\n"
    )

    f.write(
        f"PR-AUC: {pr_auc:.6f}\n\n"
    )

    f.write(
        "Recommended threshold based on maximum F1:\n"
    )

    f.write(
        f"{best_f1['threshold']:.2f}\n\n"
    )

    f.write(
        f"Accuracy: {best_f1['accuracy']:.6f}\n"
    )

    f.write(
        f"Precision: {best_f1['precision']:.6f}\n"
    )

    f.write(
        f"Recall: {best_f1['recall']:.6f}\n"
    )

    f.write(
        f"F1: {best_f1['f1']:.6f}\n"
    )

    f.write(
        f"Specificity: {best_f1['specificity']:.6f}\n"
    )


print("\n" + "=" * 70)
print("THRESHOLD OPTIMIZATION COMPLETE")
print("=" * 70)

print(
    f"\nFull results saved to:"
    f"\n{results_path}"
)

print(
    f"\nRecommended threshold saved to:"
    f"\n{threshold_path}"
)
