import os
import re
import pandas as pd
import numpy as np

# ============================================================
# CAREPREDICT - CLINICAL FEATURE ENGINEERING
# ============================================================

DATA_PATH = "data/processed/cleaned_data.csv"
OUTPUT_PATH = "data/processed/engineered_data.csv"
RESULTS_DIR = "results"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 70)
print("CAREPREDICT - CLINICAL FEATURE ENGINEERING")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

print(f"\nOriginal shape: {df.shape}")

# ------------------------------------------------------------
# 2. ICD-9 CATEGORY FUNCTION
# ------------------------------------------------------------

def classify_icd9(code):

    if pd.isna(code):
        return "Unknown"

    code = str(code).strip()

    if code in ["?", "", "nan", "None"]:
        return "Unknown"

    # Remove V/E prefix handling separately
    first_char = code[0].upper()

    # V-codes
    if first_char == "V":
        return "Supplementary/Other"

    # E-codes
    if first_char == "E":
        return "Injury/External"

    # Extract numeric ICD-9 portion
    match = re.search(
        r"\d+(\.\d+)?",
        code
    )

    if not match:
        return "Unknown"

    try:
        value = float(match.group())
    except ValueError:
        return "Unknown"

    # --------------------------------------------------------
    # ICD-9 broad clinical categories
    # --------------------------------------------------------

    # Infectious and parasitic diseases
    if 1 <= value < 140:
        return "Infectious"

    # Neoplasms
    if 140 <= value < 240:
        return "Neoplasms"

    # Endocrine, nutritional and metabolic
    if 240 <= value < 280:
        if 250 <= value < 251:
            return "Diabetes"
        return "Endocrine/Metabolic"

    # Diseases of blood
    if 280 <= value < 290:
        return "Blood"

    # Mental disorders
    if 290 <= value < 320:
        return "Mental/Behavioral"

    # Nervous system
    if 320 <= value < 390:
        return "Nervous System"

    # Circulatory system
    if 390 <= value < 460:
        return "Circulatory"

    # Respiratory system
    if 460 <= value < 520:
        return "Respiratory"

    # Digestive system
    if 520 <= value < 580:
        return "Digestive"

    # Genitourinary system
    if 580 <= value < 630:
        return "Genitourinary"

    # Pregnancy
    if 630 <= value < 680:
        return "Pregnancy"

    # Skin and subcutaneous tissue
    if 680 <= value < 710:
        return "Skin"

    # Musculoskeletal
    if 710 <= value < 740:
        return "Musculoskeletal"

    # Congenital anomalies
    if 740 <= value < 760:
        return "Congenital"

    # Perinatal conditions
    if 760 <= value < 780:
        return "Perinatal"

    # Symptoms/signs
    if 780 <= value < 800:
        return "Symptoms/Signs"

    # Injury and poisoning
    if 800 <= value < 1000:
        return "Injury/Poisoning"

    return "Other"


# ------------------------------------------------------------
# 3. DIAGNOSIS CATEGORY FEATURES
# ------------------------------------------------------------

diagnosis_columns = [
    "diag_1",
    "diag_2",
    "diag_3"
]

print("\nCreating diagnosis categories...")

for col in diagnosis_columns:

    new_col = f"{col}_category"

    df[new_col] = df[col].apply(
        classify_icd9
    )

    print(
        f"{col} -> {new_col}"
    )


# ------------------------------------------------------------
# 4. PRIMARY DIAGNOSIS CATEGORY
# ------------------------------------------------------------

df["primary_diagnosis_category"] = (
    df["diag_1_category"]
)


# ------------------------------------------------------------
# 5. DIABETES DIAGNOSIS FLAG
# ------------------------------------------------------------

df["has_diabetes_diagnosis"] = (
    (
        df["diag_1_category"] == "Diabetes"
    )
    |
    (
        df["diag_2_category"] == "Diabetes"
    )
    |
    (
        df["diag_3_category"] == "Diabetes"
    )
).astype(int)


# ------------------------------------------------------------
# 6. CIRCULATORY CONDITION FLAG
# ------------------------------------------------------------

df["has_circulatory_condition"] = (
    (
        df["diag_1_category"] == "Circulatory"
    )
    |
    (
        df["diag_2_category"] == "Circulatory"
    )
    |
    (
        df["diag_3_category"] == "Circulatory"
    )
).astype(int)


# ------------------------------------------------------------
# 7. RESPIRATORY CONDITION FLAG
# ------------------------------------------------------------

df["has_respiratory_condition"] = (
    (
        df["diag_1_category"] == "Respiratory"
    )
    |
    (
        df["diag_2_category"] == "Respiratory"
    )
    |
    (
        df["diag_3_category"] == "Respiratory"
    )
).astype(int)


# ------------------------------------------------------------
# 8. RENAL / GENITOURINARY FLAG
# ------------------------------------------------------------

df["has_genitourinary_condition"] = (
    (
        df["diag_1_category"] == "Genitourinary"
    )
    |
    (
        df["diag_2_category"] == "Genitourinary"
    )
    |
    (
        df["diag_3_category"] == "Genitourinary"
    )
).astype(int)


# ------------------------------------------------------------
# 9. DIAGNOSIS CATEGORY COUNT
# ------------------------------------------------------------

category_columns = [
    "diag_1_category",
    "diag_2_category",
    "diag_3_category"
]

df["diagnosis_category_count"] = (
    df[category_columns]
    .replace("Unknown", np.nan)
    .nunique(axis=1)
)


# ------------------------------------------------------------
# 10. PRIOR HEALTHCARE UTILIZATION
# ------------------------------------------------------------

df["prior_acute_visits"] = (
    df["number_inpatient"]
    +
    df["number_emergency"]
)

df["total_prior_visits"] = (
    df["number_inpatient"]
    +
    df["number_emergency"]
    +
    df["number_outpatient"]
)


# ------------------------------------------------------------
# 11. MEDICATION FEATURES
# ------------------------------------------------------------

medication_columns = [
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
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone"
]

# Keep only columns actually present
medication_columns = [
    col
    for col in medication_columns
    if col in df.columns
]

# Active medication = anything other than "No"
df["active_medication_count"] = (
    df[medication_columns]
    .apply(
        lambda row:
        sum(
            str(value).lower() != "no"
            for value in row
        ),
        axis=1
    )
)

# Medication changed = Up or Down
df["medication_change_count"] = (
    df[medication_columns]
    .apply(
        lambda row:
        sum(
            str(value).lower()
            in ["up", "down"]
            for value in row
        ),
        axis=1
    )
)


# ------------------------------------------------------------
# 12. DIABETES MEDICATION FLAG
# ------------------------------------------------------------

df["on_diabetes_medication"] = (
    df["diabetesMed"]
    .astype(str)
    .str.lower()
    .eq("yes")
    .astype(int)
)


# ------------------------------------------------------------
# 13. AGE NUMERIC FEATURE
# ------------------------------------------------------------

def extract_age_midpoint(age):

    if pd.isna(age):
        return np.nan

    match = re.search(
        r"(\d+)-(\d+)",
        str(age)
    )

    if match:

        lower = int(match.group(1))
        upper = int(match.group(2))

        return (lower + upper) / 2

    return np.nan


df["age_numeric"] = (
    df["age"]
    .apply(extract_age_midpoint)
)


# ------------------------------------------------------------
# 14. SAVE DIAGNOSIS SUMMARY
# ------------------------------------------------------------

diagnosis_summary = {}

for col in [
    "diag_1_category",
    "diag_2_category",
    "diag_3_category"
]:

    diagnosis_summary[col] = (
        df[col]
        .value_counts()
    )

summary_df = pd.DataFrame(
    diagnosis_summary
).fillna(0)

summary_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "diagnosis_category_summary.csv"
    )
)


# ------------------------------------------------------------
# 15. FEATURE SUMMARY
# ------------------------------------------------------------

engineered_features = [
    "diag_1_category",
    "diag_2_category",
    "diag_3_category",
    "primary_diagnosis_category",
    "has_diabetes_diagnosis",
    "has_circulatory_condition",
    "has_respiratory_condition",
    "has_genitourinary_condition",
    "diagnosis_category_count",
    "prior_acute_visits",
    "total_prior_visits",
    "active_medication_count",
    "medication_change_count",
    "on_diabetes_medication",
    "age_numeric"
]

print("\nEngineered features:")

for i, feature in enumerate(
    engineered_features,
    start=1
):

    print(
        f"{i:02d}. {feature}"
    )


# ------------------------------------------------------------
# 16. DATA VALIDATION
# ------------------------------------------------------------

print("\nEngineered feature missing values:")

missing = (
    df[engineered_features]
    .isna()
    .sum()
)

print(
    missing[
        missing > 0
    ]
)


# ------------------------------------------------------------
# 17. FINAL SHAPE
# ------------------------------------------------------------

print(
    f"\nOriginal shape: "
    f"{(len(df), 43)}"
)

print(
    f"Engineered shape: "
    f"{df.shape}"
)


# ------------------------------------------------------------
# 18. SAVE DATASET
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    f"\nEngineered dataset saved to:\n"
    f"{OUTPUT_PATH}"
)


# ------------------------------------------------------------
# 19. SAVE SUMMARY
# ------------------------------------------------------------

summary_path = os.path.join(
    RESULTS_DIR,
    "feature_engineering_summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "CAREPREDICT - FEATURE ENGINEERING SUMMARY\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Original rows: {len(df):,}\n"
    )

    f.write(
        f"Engineered columns: {len(df.columns)}\n\n"
    )

    f.write(
        "Engineered features:\n"
    )

    for feature in engineered_features:

        f.write(
            f"- {feature}\n"
        )

print(
    f"Summary saved to:\n"
    f"{summary_path}"
)

print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 70)
