import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CAREPREDICT - STEP 9
# Exploratory Data Analysis
# ============================================================

DATA_PATH = "../data/processed/cleaned_data.csv"
TARGET = "readmitted_30_days"

df = pd.read_csv(DATA_PATH)

print("\n" + "=" * 70)
print("CAREPREDICT - EXPLORATORY DATA ANALYSIS")
print("=" * 70)

print(f"\nDataset shape: {df.shape}")

# ------------------------------------------------------------
# 1. Basic statistics
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("1. DESCRIPTIVE STATISTICS")
print("=" * 70)

print(df.describe().round(2))

# ------------------------------------------------------------
# 2. Readmission by age
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("2. AGE ANALYSIS")
print("=" * 70)

age_groups = pd.cut(
    df["age"],
    bins=[17, 30, 45, 60, 75, 90],
    labels=[
        "18-30",
        "31-45",
        "46-60",
        "61-75",
        "76-89"
    ]
)

age_readmission = (
    df.groupby(age_groups, observed=False)[TARGET]
    .mean() * 100
)

print("\nReadmission rate by age group:")
print(age_readmission.round(2))

plt.figure(figsize=(8, 5))

age_readmission.plot(kind="bar")

plt.title("30-Day Readmission Rate by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Readmission Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "../results/readmission_by_age.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ------------------------------------------------------------
# 3. Gender analysis
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("3. GENDER ANALYSIS")
print("=" * 70)

gender_readmission = (
    df.groupby("gender")[TARGET]
    .mean() * 100
)

print("\nReadmission rate by gender:")
print(gender_readmission.round(2))

plt.figure(figsize=(7, 5))

gender_readmission.plot(kind="bar")

plt.title("30-Day Readmission Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Readmission Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "../results/readmission_by_gender.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ------------------------------------------------------------
# 4. Diabetes analysis
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("4. DIABETES ANALYSIS")
print("=" * 70)

diabetes_readmission = (
    df.groupby("diabetes")[TARGET]
    .mean() * 100
)

print("\nReadmission rate:")
print("0 = No diabetes")
print("1 = Diabetes")
print(diabetes_readmission.round(2))

plt.figure(figsize=(7, 5))

diabetes_readmission.plot(kind="bar")

plt.title("30-Day Readmission Rate by Diabetes Status")
plt.xlabel("Diabetes")
plt.ylabel("Readmission Rate (%)")
plt.xticks(
    ticks=[0, 1],
    labels=["No Diabetes", "Diabetes"],
    rotation=0
)

plt.tight_layout()

plt.savefig(
    "../results/readmission_by_diabetes.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ------------------------------------------------------------
# 5. Smoking analysis
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("5. SMOKING ANALYSIS")
print("=" * 70)

smoker_readmission = (
    df.groupby("smoker")[TARGET]
    .mean() * 100
)

print("\nReadmission rate:")
print("0 = Non-smoker")
print("1 = Smoker")
print(smoker_readmission.round(2))

plt.figure(figsize=(7, 5))

smoker_readmission.plot(kind="bar")

plt.title("30-Day Readmission Rate by Smoking Status")
plt.xlabel("Smoking Status")
plt.ylabel("Readmission Rate (%)")
plt.xticks(
    ticks=[0, 1],
    labels=["Non-Smoker", "Smoker"],
    rotation=0
)

plt.tight_layout()

plt.savefig(
    "../results/readmission_by_smoking.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ------------------------------------------------------------
# 6. Pain score analysis
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("6. PAIN SCORE ANALYSIS")
print("=" * 70)

pain_readmission = (
    df.groupby("pain_score")[TARGET]
    .mean() * 100
)

print("\nReadmission rate by pain score:")
print(pain_readmission.round(2))

plt.figure(figsize=(9, 5))

pain_readmission.plot(kind="line", marker="o")

plt.title("30-Day Readmission Rate by Pain Score")
plt.xlabel("Pain Score")
plt.ylabel("Readmission Rate (%)")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "../results/readmission_by_pain_score.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ------------------------------------------------------------
# 7. Glucose analysis
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("7. GLUCOSE ANALYSIS")
print("=" * 70)

glucose_groups = pd.cut(
    df["glucose_fasting"],
    bins=[0, 100, 126, 140, 180, 500],
    labels=[
        "<100",
        "100-126",
        "127-140",
        "141-180",
        ">180"
    ]
)

glucose_readmission = (
    df.groupby(glucose_groups, observed=False)[TARGET]
    .mean() * 100
)

print("\nReadmission rate by fasting glucose:")
print(glucose_readmission.round(2))

plt.figure(figsize=(9, 5))

glucose_readmission.plot(kind="bar")

plt.title("30-Day Readmission Rate by Fasting Glucose")
plt.xlabel("Fasting Glucose")
plt.ylabel("Readmission Rate (%)")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "../results/readmission_by_glucose.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ------------------------------------------------------------
# 8. Vital signs comparison
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("8. VITAL SIGN COMPARISON")
print("=" * 70)

vital_features = [
    "blood_pressure_systolic",
    "blood_pressure_diastolic",
    "heart_rate",
    "respiratory_rate",
    "body_temperature",
    "oxygen_saturation"
]

for feature in vital_features:

    means = df.groupby(TARGET)[feature].mean()

    print(f"\n{feature}")
    print(
        f"Non-readmitted mean : {means.get(0, float('nan')):.2f}"
    )
    print(
        f"Readmitted mean     : {means.get(1, float('nan')):.2f}"
    )

# ------------------------------------------------------------
# 9. Correlation heatmap
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("9. CORRELATION HEATMAP")
print("=" * 70)

correlation = df.corr(numeric_only=True)

plt.figure(figsize=(12, 9))

plt.imshow(
    correlation,
    interpolation="nearest",
    aspect="auto"
)

plt.colorbar()

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=90
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Feature Correlation Matrix")

plt.tight_layout()

plt.savefig(
    "../results/correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ------------------------------------------------------------
# 10. Final EDA summary
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("EDA COMPLETE")
print("=" * 70)

print("""
Generated visualizations:

1. readmission_by_age.png
2. readmission_by_gender.png
3. readmission_by_diabetes.png
4. readmission_by_smoking.png
5. readmission_by_pain_score.png
6. readmission_by_glucose.png
7. correlation_heatmap.png

Vital-sign comparisons were also calculated.
""")

print("Results saved in:")
print("../results/")

print("=" * 70)
