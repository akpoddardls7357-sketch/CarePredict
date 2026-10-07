import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# ============================================================
# CAREPREDICT - CLASS IMBALANCE ANALYSIS
# ============================================================

DATA_DIR = "data/processed/preprocessed"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 70)
print("CAREPREDICT - CLASS IMBALANCE ANALYSIS")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD TRAINING TARGET
# ------------------------------------------------------------

y_train = np.load(
    f"{DATA_DIR}/y_train.npy"
)

print("\nTraining target distribution:")

classes, counts = np.unique(
    y_train,
    return_counts=True
)

for cls, count in zip(classes, counts):
    percentage = (
        count / len(y_train)
    ) * 100

    print(
        f"Class {cls}: "
        f"{count:,} "
        f"({percentage:.2f}%)"
    )

# ------------------------------------------------------------
# 2. IMBALANCE RATIO
# ------------------------------------------------------------

negative_count = counts[0]
positive_count = counts[1]

imbalance_ratio = (
    negative_count / positive_count
)

print(
    f"\nClass imbalance ratio: "
    f"{imbalance_ratio:.2f}:1"
)

# ------------------------------------------------------------
# 3. BALANCED CLASS WEIGHTS
# ------------------------------------------------------------

balanced_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

print("\nBalanced class weights:")

for cls, weight in zip(
    classes,
    balanced_weights
):
    print(
        f"Class {cls}: "
        f"{weight:.4f}"
    )

# ------------------------------------------------------------
# 4. XGBOOST SCALE POSITIVE WEIGHT
# ------------------------------------------------------------

scale_pos_weight = (
    negative_count / positive_count
)

print(
    "\nXGBoost scale_pos_weight:"
)

print(
    f"{scale_pos_weight:.4f}"
)

# ------------------------------------------------------------
# 5. BASELINE ACCURACY
# ------------------------------------------------------------

majority_accuracy = (
    negative_count / len(y_train)
)

print(
    "\nMajority-class baseline accuracy:"
)

print(
    f"{majority_accuracy:.4f} "
    f"({majority_accuracy * 100:.2f}%)"
)

# ------------------------------------------------------------
# 6. SAVE PARAMETERS
# ------------------------------------------------------------

with open(
    f"{RESULTS_DIR}/class_imbalance_summary.txt",
    "w"
) as f:

    f.write(
        "CAREPREDICT - CLASS IMBALANCE SUMMARY\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    f.write(
        f"Negative class count: "
        f"{negative_count}\n"
    )

    f.write(
        f"Positive class count: "
        f"{positive_count}\n"
    )

    f.write(
        f"Imbalance ratio: "
        f"{imbalance_ratio:.4f}:1\n\n"
    )

    f.write(
        "Balanced class weights:\n"
    )

    for cls, weight in zip(
        classes,
        balanced_weights
    ):

        f.write(
            f"Class {cls}: "
            f"{weight:.6f}\n"
        )

    f.write(
        f"\nXGBoost scale_pos_weight: "
        f"{scale_pos_weight:.6f}\n"
    )

    f.write(
        f"\nMajority baseline accuracy: "
        f"{majority_accuracy:.6f}\n"
    )

print(
    "\nSummary saved to:"
)

print(
    f"{RESULTS_DIR}/class_imbalance_summary.txt"
)

print("\n" + "=" * 70)
print("CLASS IMBALANCE ANALYSIS COMPLETE")
print("=" * 70)
