import numpy as np
import pandas as pd

# ============================================================
# CAREPREDICT - STEP 13
# Class Imbalance Analysis & Strategy
# ============================================================

TRAIN_PATH = "../data/processed/train.csv"

TARGET = "readmitted_30_days"

# ------------------------------------------------------------
# 1. Load training data
# ------------------------------------------------------------
df = pd.read_csv(TRAIN_PATH)

print("\n" + "=" * 70)
print("CAREPREDICT - CLASS IMBALANCE HANDLING")
print("=" * 70)

# ------------------------------------------------------------
# 2. Target distribution
# ------------------------------------------------------------
counts = df[TARGET].value_counts().sort_index()

print("\n" + "=" * 70)
print("1. TRAINING TARGET DISTRIBUTION")
print("=" * 70)

for cls, count in counts.items():

    percentage = count / len(df) * 100

    print(
        f"Class {cls}: "
        f"{count:,} patients "
        f"({percentage:.2f}%)"
    )

# ------------------------------------------------------------
# 3. Calculate imbalance ratio
# ------------------------------------------------------------
negative_count = counts.get(0, 0)
positive_count = counts.get(1, 0)

imbalance_ratio = negative_count / positive_count

print("\n" + "=" * 70)
print("2. IMBALANCE RATIO")
print("=" * 70)

print(
    f"Class 0 / Class 1 ratio: "
    f"{imbalance_ratio:.4f}:1"
)

# ------------------------------------------------------------
# 4. Calculate balanced class weights
# ------------------------------------------------------------
total_samples = len(df)
number_of_classes = 2

weight_0 = total_samples / (
    number_of_classes * negative_count
)

weight_1 = total_samples / (
    number_of_classes * positive_count
)

class_weights = {
    0: weight_0,
    1: weight_1
}

print("\n" + "=" * 70)
print("3. BALANCED CLASS WEIGHTS")
print("=" * 70)

print(f"Class 0 weight: {weight_0:.4f}")
print(f"Class 1 weight: {weight_1:.4f}")

# ------------------------------------------------------------
# 5. XGBoost scale_pos_weight
# ------------------------------------------------------------
scale_pos_weight = (
    negative_count / positive_count
)

print("\n" + "=" * 70)
print("4. XGBOOST CLASS WEIGHT")
print("=" * 70)

print(
    f"scale_pos_weight: "
    f"{scale_pos_weight:.4f}"
)

# ------------------------------------------------------------
# 6. Explain strategy
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("5. SELECTED STRATEGY")
print("=" * 70)

print("""
CarePredict will initially use class weighting rather
than synthetic oversampling.

Logistic Regression:
    class_weight = "balanced"

Random Forest:
    class_weight = "balanced"

XGBoost:
    scale_pos_weight = negative_count / positive_count

The original patient records will remain unchanged.

Model performance will be evaluated using:
    • Precision
    • Recall
    • F1-score
    • ROC-AUC
    • PR-AUC
""")

# ------------------------------------------------------------
# 7. Final conclusion
# ------------------------------------------------------------
print("=" * 70)
print("STEP 13 COMPLETE")
print("=" * 70)

print(
    f"""
Training samples : {total_samples:,}
Class 0          : {negative_count:,}
Class 1          : {positive_count:,}

Imbalance ratio  : {imbalance_ratio:.2f}:1

Class weighting strategy selected.
"""
)

print("=" * 70)
