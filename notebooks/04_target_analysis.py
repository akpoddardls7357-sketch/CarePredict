import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CAREPREDICT - TARGET ANALYSIS
# ============================================================

DATA_PATH = "data/processed/cleaned_data.csv"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 70)
print("CAREPREDICT - TARGET ANALYSIS")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD CLEANED DATA
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

target = "readmission_30d"

print(f"\nDataset shape: {df.shape}")

# ------------------------------------------------------------
# 2. TARGET COUNTS
# ------------------------------------------------------------

target_counts = df[target].value_counts().sort_index()

print("\nTarget counts:")
print(target_counts)

# ------------------------------------------------------------
# 3. TARGET PERCENTAGES
# ------------------------------------------------------------

target_percent = (
    df[target]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)

print("\nTarget percentages:")
print(target_percent)

# ------------------------------------------------------------
# 4. READABLE SUMMARY
# ------------------------------------------------------------

not_readmitted = int(
    target_counts.get(0, 0)
)

readmitted = int(
    target_counts.get(1, 0)
)

total = len(df)

print("\nClinical prediction target:")
print(
    f"Total encounters: {total:,}"
)

print(
    f"Not readmitted within 30 days: "
    f"{not_readmitted:,} "
    f"({not_readmitted / total * 100:.2f}%)"
)

print(
    f"Readmitted within 30 days: "
    f"{readmitted:,} "
    f"({readmitted / total * 100:.2f}%)"
)

# ------------------------------------------------------------
# 5. IMBALANCE RATIO
# ------------------------------------------------------------

if readmitted > 0:

    imbalance_ratio = (
        not_readmitted / readmitted
    )

    print(
        f"\nClass imbalance ratio: "
        f"{imbalance_ratio:.2f}:1"
    )

# ------------------------------------------------------------
# 6. PLOT TARGET DISTRIBUTION
# ------------------------------------------------------------

labels = [
    "No 30-Day Readmission",
    "30-Day Readmission"
]

values = [
    not_readmitted,
    readmitted
]

plt.figure(figsize=(8, 6))

bars = plt.bar(
    labels,
    values
)

plt.title(
    "30-Day Hospital Readmission Distribution"
)

plt.ylabel("Number of Encounters")

plt.xticks(
    rotation=10
)

# Add value labels
for bar, value in zip(
    bars,
    values
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:,}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plot_path = os.path.join(
    RESULTS_DIR,
    "target_distribution.png"
)

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"\nTarget distribution plot saved to:\n"
    f"{plot_path}"
)

# ------------------------------------------------------------
# 7. SAVE TARGET SUMMARY
# ------------------------------------------------------------

summary_path = os.path.join(
    RESULTS_DIR,
    "target_analysis.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "CAREPREDICT - 30-DAY READMISSION TARGET ANALYSIS\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Total encounters: {total:,}\n"
    )

    f.write(
        f"Not readmitted within 30 days: "
        f"{not_readmitted:,} "
        f"({not_readmitted / total * 100:.2f}%)\n"
    )

    f.write(
        f"Readmitted within 30 days: "
        f"{readmitted:,} "
        f"({readmitted / total * 100:.2f}%)\n"
    )

    f.write(
        f"Class imbalance ratio: "
        f"{imbalance_ratio:.2f}:1\n"
    )

print(
    f"Target summary saved to:\n"
    f"{summary_path}"
)

print("\n" + "=" * 70)
print("TARGET ANALYSIS COMPLETE")
print("=" * 70)
