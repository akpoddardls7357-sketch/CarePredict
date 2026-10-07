import pandas as pd

# ============================================================
# CAREPREDICT - STEP 6
# Data Cleaning & Preparation
# Dataset: Kaggle HealNet | Vitals & Variables
# ============================================================

RAW_PATH = "../data/raw/train.csv"
PROCESSED_PATH = "../data/processed/cleaned_data.csv"

TARGET = "readmitted_30_days"
ID_COLUMN = "patient_id"

# ------------------------------------------------------------
# 1. Load raw dataset
# ------------------------------------------------------------
df = pd.read_csv(RAW_PATH)

print("\n" + "=" * 70)
print("CAREPREDICT - DATA CLEANING")
print("=" * 70)

print(f"\nOriginal shape: {df.shape}")

# ------------------------------------------------------------
# 2. Create a copy
# ------------------------------------------------------------
clean_df = df.copy()

# ------------------------------------------------------------
# 3. Remove patient identifier
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("1. REMOVING IDENTIFIER")
print("=" * 70)

if ID_COLUMN in clean_df.columns:
    clean_df = clean_df.drop(columns=[ID_COLUMN])
    print(f"✓ Removed identifier column: {ID_COLUMN}")
else:
    print(f"⚠ {ID_COLUMN} not found.")

# ------------------------------------------------------------
# 4. Verify missing values
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("2. MISSING VALUE CHECK")
print("=" * 70)

missing_count = clean_df.isnull().sum().sum()

print(f"Total missing values: {missing_count}")

if missing_count == 0:
    print("✓ No missing values require imputation.")
else:
    print("⚠ Missing values detected.")

# ------------------------------------------------------------
# 5. Verify target
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("3. TARGET CHECK")
print("=" * 70)

print(clean_df[TARGET].value_counts())

if set(clean_df[TARGET].unique()).issubset({0, 1}):
    print("✓ Target is binary.")
else:
    raise ValueError("Invalid target values detected.")

# ------------------------------------------------------------
# 6. Encode gender
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("4. ENCODING GENDER")
print("=" * 70)

print("Original gender values:")
print(clean_df["gender"].value_counts())

clean_df["gender"] = clean_df["gender"].map({
    "Male": 1,
    "Female": 0
})

if clean_df["gender"].isnull().sum() == 0:
    print("✓ Gender successfully encoded.")
else:
    raise ValueError("Unexpected gender value found.")

print("\nEncoded gender values:")
print(clean_df["gender"].value_counts())

# ------------------------------------------------------------
# 7. Check data types
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("5. DATA TYPES AFTER CLEANING")
print("=" * 70)

print(clean_df.dtypes)

# ------------------------------------------------------------
# 8. Check duplicates again
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("6. DUPLICATE CHECK")
print("=" * 70)

duplicates = clean_df.duplicated().sum()

print(f"Duplicate rows: {duplicates}")

if duplicates == 0:
    print("✓ No duplicate rows.")
else:
    print("⚠ Duplicate rows detected.")

# ------------------------------------------------------------
# 9. Verify final columns
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("7. FINAL COLUMNS")
print("=" * 70)

for i, column in enumerate(clean_df.columns, 1):
    print(f"{i:2}. {column}")

# ------------------------------------------------------------
# 10. Save processed dataset
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("8. SAVING PROCESSED DATA")
print("=" * 70)

clean_df.to_csv(PROCESSED_PATH, index=False)

print(f"✓ Saved cleaned dataset to:")
print(PROCESSED_PATH)

# ------------------------------------------------------------
# 11. Final summary
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("CLEANING SUMMARY")
print("=" * 70)

print(f"Original rows       : {len(df):,}")
print(f"Final rows          : {len(clean_df):,}")
print(f"Original columns    : {len(df.columns)}")
print(f"Final columns       : {len(clean_df.columns)}")
print(f"Missing values      : {clean_df.isnull().sum().sum()}")
print(f"Duplicate rows      : {clean_df.duplicated().sum()}")

print("\nData cleaning completed successfully.")
print("=" * 70)
