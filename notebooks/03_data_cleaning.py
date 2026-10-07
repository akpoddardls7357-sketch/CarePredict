import os
import pandas as pd
import numpy as np

# ============================================================
# CAREPREDICT - DATA CLEANING
# UCI Diabetes 130-US Hospitals
# ============================================================

RAW_PATH = "data/raw/diabetic_data.csv"
OUTPUT_DIR = "data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("CAREPREDICT - DATA CLEANING")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD
# ------------------------------------------------------------

df = pd.read_csv(RAW_PATH)

print(f"\nOriginal shape: {df.shape}")

# ------------------------------------------------------------
# 2. STANDARDIZE MISSING VALUES
# ------------------------------------------------------------

# UCI uses '?' for unknown categorical values.
df = df.replace("?", np.nan)

print("\nConverted '?' values to NaN.")

# ------------------------------------------------------------
# 3. REMOVE DUPLICATES
# ------------------------------------------------------------

before = len(df)

df = df.drop_duplicates().copy()

duplicates_removed = before - len(df)

print(
    f"Duplicate rows removed: {duplicates_removed}"
)

# ------------------------------------------------------------
# 4. VALIDATE TARGET
# ------------------------------------------------------------

valid_targets = {"NO", ">30", "<30"}

invalid_mask = ~df["readmitted"].isin(valid_targets)

invalid_count = invalid_mask.sum()

print(
    f"Invalid target rows: {invalid_count}"
)

if invalid_count > 0:
    df = df.loc[~invalid_mask].copy()

# ------------------------------------------------------------
# 5. CREATE BINARY 30-DAY TARGET
# ------------------------------------------------------------

# <30 = readmitted within 30 days
# NO / >30 = not readmitted within 30 days

df["readmission_30d"] = (
    df["readmitted"] == "<30"
).astype(int)

print("\n30-day target:")
print(
    df["readmission_30d"]
    .value_counts()
    .sort_index()
)

print("\n30-day target percentage:")
print(
    df["readmission_30d"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)

# ------------------------------------------------------------
# 6. REMOVE ENCOUNTERS WITH DEATH/HOSPICE DISPOSITION
# ------------------------------------------------------------

# These discharge destinations represent outcomes where
# ordinary subsequent readmission is not applicable.

death_hospice_ids = [
    11, 13, 14, 19, 20, 21
]

before = len(df)

df = df[
    ~df["discharge_disposition_id"].isin(
        death_hospice_ids
    )
].copy()

removed = before - len(df)

print(
    f"\nDeath/hospice encounters removed: {removed:,}"
)

# ------------------------------------------------------------
# 7. REMOVE IDENTIFIER COLUMNS FROM MODEL DATA
# ------------------------------------------------------------

# IMPORTANT:
# patient_nbr is retained temporarily because it is required
# later for patient-level train/validation/test splitting.
#
# encounter_id and patient_nbr are not predictive features.

# encounter_id will be removed now.
df = df.drop(
    columns=["encounter_id"]
)

# ------------------------------------------------------------
# 8. DROP EXTREMELY HIGH-MISSING FEATURES
# ------------------------------------------------------------

high_missing_columns = [
    "weight",
    "max_glu_serum",
    "A1Cresult",
    "medical_specialty",
    "payer_code"
]

df = df.drop(
    columns=high_missing_columns
)

print("\nDropped high-missing features:")
for col in high_missing_columns:
    print(f"  - {col}")

# ------------------------------------------------------------
# 9. DROP CONSTANT FEATURES
# ------------------------------------------------------------

constant_columns = [
    "examide",
    "citoglipton"
]

df = df.drop(
    columns=constant_columns
)

print("\nDropped constant features:")
for col in constant_columns:
    print(f"  - {col}")

# ------------------------------------------------------------
# 10. CLEAN CATEGORICAL FEATURES
# ------------------------------------------------------------

categorical_columns = [
    "race",
    "gender",
    "age",
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
    "change",
    "diabetesMed",
    "diag_1",
    "diag_2",
    "diag_3"
]

for col in categorical_columns:

    if col in df.columns:
        df[col] = (
            df[col]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

# ------------------------------------------------------------
# 11. NUMERIC FEATURES
# ------------------------------------------------------------

numeric_columns = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses"
]

for col in numeric_columns:

    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

# ------------------------------------------------------------
# 12. TREAT CODED HOSPITAL FIELDS AS CATEGORICAL
# ------------------------------------------------------------

coded_columns = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id"
]

for col in coded_columns:

    df[col] = (
        df[col]
        .fillna(-1)
        .astype(str)
    )

# ------------------------------------------------------------
# 13. HANDLE NUMERIC MISSING VALUES
# ------------------------------------------------------------

numeric_columns_remaining = [
    col
    for col in df.select_dtypes(
        include=["number"]
    ).columns
    if col != "readmission_30d"
]

for col in numeric_columns_remaining:

    median_value = df[col].median()

    df[col] = df[col].fillna(
        median_value
    )

# ------------------------------------------------------------
# 14. AGE AS ORDERED CATEGORY
# ------------------------------------------------------------

age_order = [
    "[0-10)",
    "[10-20)",
    "[20-30)",
    "[30-40)",
    "[40-50)",
    "[50-60)",
    "[60-70)",
    "[70-80)",
    "[80-90)",
    "[90-100)"
]

df["age"] = pd.Categorical(
    df["age"],
    categories=age_order,
    ordered=True
)

# Convert back to string for later sklearn encoding.
df["age"] = df["age"].astype(str)

# ------------------------------------------------------------
# 15. FINAL COLUMN CHECK
# ------------------------------------------------------------

print("\nFinal columns:")
for i, col in enumerate(
    df.columns,
    start=1
):
    print(
        f"{i:02d}. {col}"
    )

# ------------------------------------------------------------
# 16. FINAL MISSING VALUE CHECK
# ------------------------------------------------------------

missing = (
    df.isna()
    .sum()
    .sort_values(
        ascending=False
    )
)

print("\nRemaining missing values:")

remaining_missing = missing[
    missing > 0
]

if len(remaining_missing) == 0:
    print("  None")
else:
    print(remaining_missing)

# ------------------------------------------------------------
# 17. FINAL SHAPE
# ------------------------------------------------------------

print("\nFinal shape:")
print(df.shape)

# ------------------------------------------------------------
# 18. SAVE CLEAN DATA
# ------------------------------------------------------------

output_path = os.path.join(
    OUTPUT_DIR,
    "cleaned_data.csv"
)

df.to_csv(
    output_path,
    index=False
)

print(
    f"\nSaved cleaned dataset to:\n"
    f"{output_path}"
)

# ------------------------------------------------------------
# 19. SAVE CLEANING SUMMARY
# ------------------------------------------------------------

summary_path = os.path.join(
    "results",
    "cleaning_summary.txt"
)

os.makedirs(
    "results",
    exist_ok=True
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "CAREPREDICT - DATA CLEANING SUMMARY\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Original rows: {before:,}\n"
    )

    f.write(
        f"Final rows: {len(df):,}\n"
    )

    f.write(
        f"Final columns: {len(df.columns):,}\n"
    )

    f.write(
        f"Duplicate rows removed: "
        f"{duplicates_removed:,}\n"
    )

    f.write(
        f"Death/hospice encounters removed: "
        f"{removed:,}\n"
    )

    f.write(
        "\nDropped high-missing features:\n"
    )

    for col in high_missing_columns:
        f.write(f"- {col}\n")

    f.write(
        "\nDropped constant features:\n"
    )

    for col in constant_columns:
        f.write(f"- {col}\n")

    f.write(
        "\n30-day target distribution:\n"
    )

    f.write(
        df["readmission_30d"]
        .value_counts()
        .to_string()
    )

print(
    f"Cleaning summary saved to:\n"
    f"{summary_path}"
)

print("\n" + "=" * 70)
print("DATA CLEANING COMPLETE")
print("=" * 70)
