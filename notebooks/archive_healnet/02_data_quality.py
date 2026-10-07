import pandas as pd

# ============================================================
# CAREPREDICT - STEP 5
# Data Quality Analysis
# Dataset: Kaggle HealNet | Vitals & Variables
# ============================================================

DATA_PATH = "../data/raw/train.csv"
TARGET = "readmitted_30_days"
ID_COLUMN = "patient_id"

# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

print("\n" + "=" * 70)
print("CAREPREDICT - DATA QUALITY ANALYSIS")
print("=" * 70)

print(f"\nDataset shape: {df.shape}")

# ------------------------------------------------------------
# 2. Missing values
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("1. MISSING VALUES")
print("=" * 70)

missing = df.isnull().sum()

missing_table = pd.DataFrame({
    "Missing_Count": missing,
    "Missing_Percentage": (
        missing / len(df) * 100
    ).round(2)
})

print(missing_table)

if missing.sum() == 0:
    print("\n✓ No missing values found.")
else:
    print("\n⚠ Missing values detected.")

# ------------------------------------------------------------
# 3. Duplicate rows
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("2. DUPLICATE ROWS")
print("=" * 70)

duplicate_count = df.duplicated().sum()

print(f"Duplicate rows: {duplicate_count:,}")

if duplicate_count == 0:
    print("✓ No duplicate rows found.")
else:
    print("⚠ Duplicate rows found.")

# ------------------------------------------------------------
# 4. Duplicate patient IDs
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("3. DUPLICATE PATIENT IDs")
print("=" * 70)

duplicate_ids = df[ID_COLUMN].duplicated().sum()

print(f"Duplicate patient IDs: {duplicate_ids:,}")

if duplicate_ids == 0:
    print("✓ Every patient_id is unique.")
else:
    print("⚠ Duplicate patient IDs found.")

# ------------------------------------------------------------
# 5. Target validation
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("4. TARGET VALIDATION")
print("=" * 70)

print(f"Target column: {TARGET}")

print("\nUnique target values:")
print(sorted(df[TARGET].unique()))

invalid_target = ~df[TARGET].isin([0, 1])

print(f"\nInvalid target values: {invalid_target.sum()}")

if invalid_target.sum() == 0:
    print("✓ Target contains only 0 and 1.")
else:
    print("⚠ Invalid target values found.")

# ------------------------------------------------------------
# 6. Numerical range checks
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("5. NUMERICAL RANGE CHECKS")
print("=" * 70)

numeric_columns = [
    "age",
    "blood_pressure_systolic",
    "blood_pressure_diastolic",
    "heart_rate",
    "respiratory_rate",
    "body_temperature",
    "oxygen_saturation",
    "glucose_fasting",
    "cholesterol_total",
    "pain_score",
    "diabetes",
    "smoker"
]

for column in numeric_columns:

    print(f"\n{column}")
    print("-" * 40)

    print(f"Minimum : {df[column].min()}")
    print(f"Maximum : {df[column].max()}")
    print(f"Mean    : {df[column].mean():.2f}")
    print(f"Median  : {df[column].median():.2f}")

# ------------------------------------------------------------
# 7. Binary variable validation
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("6. BINARY VARIABLE VALIDATION")
print("=" * 70)

binary_columns = [
    "diabetes",
    "smoker",
    TARGET
]

for column in binary_columns:

    values = sorted(df[column].unique())

    print(f"{column}: {values}")

    if set(values).issubset({0, 1}):
        print("✓ Valid binary variable")
    else:
        print("⚠ Unexpected values found")

# ------------------------------------------------------------
# 8. Gender validation
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("7. GENDER VALUES")
print("=" * 70)

print(df["gender"].value_counts())

print("\nUnique gender values:")
print(df["gender"].unique())

# ------------------------------------------------------------
# 9. ID format validation
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("8. PATIENT ID VALIDATION")
print("=" * 70)

print("Example patient IDs:")
print(df[ID_COLUMN].head(10).tolist())

print(f"\nUnique patient IDs: {df[ID_COLUMN].nunique():,}")
print(f"Total rows        : {len(df):,}")

if df[ID_COLUMN].nunique() == len(df):
    print("✓ Patient IDs are unique.")
else:
    print("⚠ Patient IDs are not unique.")

# ------------------------------------------------------------
# 10. Final quality summary
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("DATA QUALITY SUMMARY")
print("=" * 70)

print(f"Rows                  : {len(df):,}")
print(f"Columns               : {len(df.columns)}")
print(f"Missing values        : {df.isnull().sum().sum():,}")
print(f"Duplicate rows        : {duplicate_count:,}")
print(f"Duplicate patient IDs : {duplicate_ids:,}")
print(f"Invalid target values : {invalid_target.sum():,}")

print("\nData quality analysis complete.")
print("=" * 70)
