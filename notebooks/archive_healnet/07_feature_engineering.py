import pandas as pd

# ============================================================
# CAREPREDICT - STEP 10
# Feature Engineering
# Dataset: Kaggle HealNet | Vitals & Variables
# ============================================================

INPUT_PATH = "../data/processed/cleaned_data.csv"
OUTPUT_PATH = "../data/processed/engineered_data.csv"

TARGET = "readmitted_30_days"

# ------------------------------------------------------------
# 1. Load cleaned data
# ------------------------------------------------------------
df = pd.read_csv(INPUT_PATH)

print("\n" + "=" * 70)
print("CAREPREDICT - FEATURE ENGINEERING")
print("=" * 70)

print(f"\nOriginal shape: {df.shape}")

# ------------------------------------------------------------
# 2. Blood pressure derived features
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("1. BLOOD PRESSURE FEATURES")
print("=" * 70)

df["pulse_pressure"] = (
    df["blood_pressure_systolic"]
    - df["blood_pressure_diastolic"]
)

df["mean_arterial_pressure"] = (
    df["blood_pressure_systolic"]
    + 2 * df["blood_pressure_diastolic"]
) / 3

print("✓ Created:")
print("  • pulse_pressure")
print("  • mean_arterial_pressure")

# ------------------------------------------------------------
# 3. Heart / respiratory relationship
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("2. VITAL SIGN DERIVED FEATURES")
print("=" * 70)

df["heart_respiratory_ratio"] = (
    df["heart_rate"]
    / df["respiratory_rate"]
)

print("✓ Created:")
print("  • heart_respiratory_ratio")

# ------------------------------------------------------------
# 4. Age groups
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("3. AGE GROUP")
print("=" * 70)

df["age_group"] = pd.cut(
    df["age"],
    bins=[17, 30, 45, 60, 75, 90],
    labels=[
        "18_30",
        "31_45",
        "46_60",
        "61_75",
        "76_89"
    ]
)

print(df["age_group"].value_counts().sort_index())

# ------------------------------------------------------------
# 5. Glucose categories
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("4. GLUCOSE CATEGORY")
print("=" * 70)

df["glucose_category"] = pd.cut(
    df["glucose_fasting"],
    bins=[0, 100, 126, 140, float("inf")],
    labels=[
        "normal",
        "borderline",
        "elevated",
        "high"
    ],
    include_lowest=True
)

print(df["glucose_category"].value_counts().sort_index())

# ------------------------------------------------------------
# 6. Cholesterol categories
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("5. CHOLESTEROL CATEGORY")
print("=" * 70)

df["cholesterol_category"] = pd.cut(
    df["cholesterol_total"],
    bins=[0, 200, 240, float("inf")],
    labels=[
        "desirable",
        "borderline",
        "high"
    ],
    include_lowest=True
)

print(
    df["cholesterol_category"]
    .value_counts()
    .sort_index()
)

# ------------------------------------------------------------
# 7. Pain category
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("6. PAIN CATEGORY")
print("=" * 70)

df["pain_category"] = pd.cut(
    df["pain_score"],
    bins=[-1, 0, 3, 6, 10],
    labels=[
        "none",
        "mild",
        "moderate",
        "severe"
    ],
    include_lowest=True
)

print(
    df["pain_category"]
    .value_counts()
    .sort_index()
)

# ------------------------------------------------------------
# 8. Risk-factor count
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("7. RISK FACTOR COUNT")
print("=" * 70)

df["risk_factor_count"] = (
    df["diabetes"]
    + df["smoker"]
)

print(
    df["risk_factor_count"]
    .value_counts()
    .sort_index()
)

# ------------------------------------------------------------
# 9. Check new features
# ------------------------------------------------------------
new_features = [
    "pulse_pressure",
    "mean_arterial_pressure",
    "heart_respiratory_ratio",
    "age_group",
    "glucose_category",
    "cholesterol_category",
    "pain_category",
    "risk_factor_count"
]

print("\n" + "=" * 70)
print("8. NEW FEATURES")
print("=" * 70)

for feature in new_features:
    print(f"✓ {feature}")

# ------------------------------------------------------------
# 10. Missing-value check
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("9. POST-ENGINEERING QUALITY CHECK")
print("=" * 70)

missing = df.isnull().sum()

print(
    f"Total missing values: "
    f"{missing.sum()}"
)

if missing.sum() == 0:
    print("✓ No missing values created.")
else:
    print("\nColumns containing missing values:")

    print(
        missing[missing > 0]
        .sort_values(ascending=False)
    )

# ------------------------------------------------------------
# 11. Save engineered dataset
# ------------------------------------------------------------
df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 70)
print("10. SAVING ENGINEERED DATA")
print("=" * 70)

print(f"✓ Saved to: {OUTPUT_PATH}")
print(f"Final shape: {df.shape}")

print("\nFeature engineering completed.")
print("=" * 70)
