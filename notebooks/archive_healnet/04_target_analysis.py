import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CAREPREDICT - STEP 7
# Target Analysis & Class Imbalance
# Dataset: Kaggle HealNet | Vitals & Variables
# ============================================================

DATA_PATH = "../data/processed/cleaned_data.csv"
TARGET = "readmitted_30_days"

# ------------------------------------------------------------
# 1. Load processed dataset
# ------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

print("\n" + "=" * 70)
print("CAREPREDICT - TARGET ANALYSIS")
print("=" * 70)

print(f"\nDataset shape: {df.shape}")

# ------------------------------------------------------------
# 2. Target distribution
# ------------------------------------------------------------
target_counts = df[TARGET].value_counts().sort_index()

target_percentages = (
    df[TARGET]
    .value_counts(normalize=True)
    .sort_index() * 100
)

print("\n" + "=" * 70)
print("1. TARGET DISTRIBUTION")
print("=" * 70)

print("\nPatient counts:")
print(target_counts)

print("\nPatient percentages:")
for value in target_counts.index:
    print(
        f"Class {value}: "
        f"{target_counts[value]:,} patients "
        f"({target_percentages[value]:.2f}%)"
    )

# ------------------------------------------------------------
# 3. Readmission summary
# ------------------------------------------------------------
readmitted = target_counts.get(1, 0)
not_readmitted = target_counts.get(0, 0)

readmission_rate = (readmitted / len(df)) * 100

print("\n" + "=" * 70)
print("2. READMISSION SUMMARY")
print("=" * 70)

print(f"Total patients       : {len(df):,}")
print(f"Readmitted           : {readmitted:,}")
print(f"Not readmitted       : {not_readmitted:,}")
print(f"Readmission rate     : {readmission_rate:.2f}%")

# ------------------------------------------------------------
# 4. Imbalance ratio
# ------------------------------------------------------------
imbalance_ratio = not_readmitted / readmitted

print("\n" + "=" * 70)
print("3. CLASS IMBALANCE")
print("=" * 70)

print(
    f"Non-readmitted / Readmitted ratio: "
    f"{imbalance_ratio:.2f}:1"
)

if imbalance_ratio < 1.5:
    print("✓ Classes are approximately balanced.")
elif imbalance_ratio < 3:
    print("⚠ Moderate class imbalance detected.")
else:
    print("⚠ Significant class imbalance detected.")

# ------------------------------------------------------------
# 5. Baseline accuracy
# ------------------------------------------------------------
majority_accuracy = (
    max(target_counts) / len(df) * 100
)

print("\n" + "=" * 70)
print("4. MAJORITY-CLASS BASELINE")
print("=" * 70)

print(
    f"If a model predicted the majority class "
    f"for every patient:"
)

print(f"Baseline accuracy: {majority_accuracy:.2f}%")

print(
    "\nThis is why accuracy alone will NOT be sufficient "
    "for evaluating CarePredict."
)

# ------------------------------------------------------------
# 6. Visualization
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("5. CREATING TARGET DISTRIBUTION PLOT")
print("=" * 70)

labels = [
    "No Readmission",
    "Readmitted\nWithin 30 Days"
]

values = [
    target_counts.get(0, 0),
    target_counts.get(1, 0)
]

plt.figure(figsize=(8, 5))

bars = plt.bar(labels, values)

plt.title("30-Day Hospital Readmission Distribution")
plt.xlabel("Readmission Status")
plt.ylabel("Number of Patients")

# Add values above bars
for bar, value in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:,}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

# Save plot
plt.savefig(
    "../results/target_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\n✓ Plot saved to:")
print("../results/target_distribution.png")

# ------------------------------------------------------------
# 7. Final conclusion
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 7 CONCLUSION")
print("=" * 70)

print(
    f"""
The dataset contains {len(df):,} patients.

{readmitted:,} patients ({readmission_rate:.2f}%) 
were readmitted within 30 days.

{not_readmitted:,} patients 
({100 - readmission_rate:.2f}%) were not readmitted.

The target is moderately imbalanced.

Accuracy alone should not be used as the primary
evaluation metric. Precision, Recall, F1-score,
ROC-AUC and PR-AUC will also be evaluated.
"""
)

print("=" * 70)
