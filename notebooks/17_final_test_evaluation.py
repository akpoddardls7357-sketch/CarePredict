import os
import joblib
import numpy as np
import pandas as pd

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
# CONFIGURATION
# ============================================================

DATA_DIR = "data/processed/preprocessed"
MODEL_PATH = "models/xgboost_focused_tuned.pkl"
RESULTS_DIR = "results"

# LOCKED threshold selected using validation set
THRESHOLD = 0.51


# ============================================================
# SETUP
# ============================================================

os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 70)
print("CAREPREDICT - FINAL TEST EVALUATION")
print("=" * 70)

print("\nIMPORTANT:")
print("This evaluation uses the untouched TEST set.")
print("No model tuning or threshold optimization is performed here.")
print(f"Locked classification threshold: {THRESHOLD}")


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\n[1/5] Loading test data...")

X_test = np.load(
    os.path.join(DATA_DIR, "X_test.npy")
)

y_test = np.load(
    os.path.join(DATA_DIR, "y_test.npy")
)

print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

print(f"Positive cases: {np.sum(y_test == 1):,}")
print(f"Negative cases: {np.sum(y_test == 0):,}")
print(f"Positive rate: {np.mean(y_test):.4f}")


# ============================================================
# LOAD LOCKED MODEL
# ============================================================

print("\n[2/5] Loading locked XGBoost model...")

model = joblib.load(MODEL_PATH)

print(f"Model loaded from: {MODEL_PATH}")


# ============================================================
# GENERATE TEST PREDICTIONS
# ============================================================

print("\n[3/5] Generating predictions...")

test_probabilities = model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= THRESHOLD
).astype(int)

print("Predictions generated successfully.")


# ============================================================
# CALCULATE FINAL METRICS
# ============================================================

print("\n[4/5] Calculating final test metrics...")

accuracy = accuracy_score(
    y_test,
    test_predictions
)

precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    test_probabilities
)

pr_auc = average_precision_score(
    y_test,
    test_probabilities
)

cm = confusion_matrix(
    y_test,
    test_predictions
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST SET RESULTS")
print("=" * 70)

print(f"\nClassification Threshold : {THRESHOLD:.2f}")
print(f"Accuracy                 : {accuracy:.4f}")
print(f"Precision                : {precision:.4f}")
print(f"Recall                   : {recall:.4f}")
print(f"F1 Score                 : {f1:.4f}")
print(f"ROC-AUC                  : {roc_auc:.4f}")
print(f"PR-AUC                   : {pr_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_predictions,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

print("\n[5/5] Saving final evaluation artifacts...")

# Save probabilities
np.save(
    os.path.join(
        RESULTS_DIR,
        "final_test_probabilities.npy"
    ),
    test_probabilities
)

# Save predictions
np.save(
    os.path.join(
        RESULTS_DIR,
        "final_test_predictions.npy"
    ),
    test_predictions
)

# Save confusion matrix
cm_df = pd.DataFrame(
    cm,
    index=["Actual_0", "Actual_1"],
    columns=["Predicted_0", "Predicted_1"]
)

cm_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "final_test_confusion_matrix.csv"
    )
)


# Save metrics CSV
metrics_df = pd.DataFrame({
    "Metric": [
        "Threshold",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
        "PR-AUC"
    ],
    "Value": [
        THRESHOLD,
        accuracy,
        precision,
        recall,
        f1,
        roc_auc,
        pr_auc
    ]
})

metrics_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "final_test_results.csv"
    ),
    index=False
)


# Save human-readable report
with open(
    os.path.join(
        RESULTS_DIR,
        "final_test_metrics.txt"
    ),
    "w"
) as f:

    f.write("CAREPREDICT - FINAL TEST EVALUATION\n")
    f.write("=" * 60 + "\n\n")

    f.write(
        "Model: XGBoost Focused Tuned\n"
    )

    f.write(
        f"Classification Threshold: {THRESHOLD:.2f}\n\n"
    )

    f.write(
        f"Accuracy:  {accuracy:.4f}\n"
    )

    f.write(
        f"Precision: {precision:.4f}\n"
    )

    f.write(
        f"Recall:    {recall:.4f}\n"
    )

    f.write(
        f"F1 Score:  {f1:.4f}\n"
    )

    f.write(
        f"ROC-AUC:   {roc_auc:.4f}\n"
    )

    f.write(
        f"PR-AUC:    {pr_auc:.4f}\n\n"
    )

    f.write("Confusion Matrix:\n")
    f.write(str(cm))
    f.write("\n\n")

    f.write("Classification Report:\n")
    f.write(
        classification_report(
            y_test,
            test_predictions,
            digits=4,
            zero_division=0
        )
    )


print("\n" + "=" * 70)
print("FINAL EVALUATION COMPLETE")
print("=" * 70)

print("\nSaved files:")

print("  results/final_test_probabilities.npy")
print("  results/final_test_predictions.npy")
print("  results/final_test_confusion_matrix.csv")
print("  results/final_test_results.csv")
print("  results/final_test_metrics.txt")

print("\nThe test set is now considered FINAL and must not be used for tuning.")
