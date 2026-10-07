import os
import pandas as pd
import numpy as np

DATA_PATH = "data/raw/diabetic_data.csv"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 70)
print("CAREPREDICT - DATA QUALITY ANALYSIS")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

# ------------------------------------------------------------
# 1. Convert '?' to NaN for quality analysis
# ------------------------------------------------------------

df_clean = df.replace("?", np.nan)

print(f"\nDataset shape: {df_clean.shape}")

# ------------------------------------------------------------
# 2. Missing-value analysis
# ------------------------------------------------------------

missing = pd.DataFrame({
    "missing_count": df_clean.isna().sum(),
    "missing_percent": (
        df_clean.isna().mean() * 100
    ).round(2)
})

missing = missing.sort_values(
    "missing_percent",
    ascending=False
)

print("\nMissing-value analysis:")
print(missing.to_string())

# ------------------------------------------------------------
# 3. Duplicate analysis
# ------------------------------------------------------------

duplicate_rows = df_clean.duplicated().sum()

print("\nDuplicate rows:")
print(duplicate_rows)

# ------------------------------------------------------------
# 4. Identifier uniqueness
# ------------------------------------------------------------

print("\nIdentifier analysis:")

print(
    f"Unique encounter IDs: "
    f"{df_clean['encounter_id'].nunique():,}"
)

print(
    f"Unique patient IDs: "
    f"{df_clean['patient_nbr'].nunique():,}"
)

print(
    f"Duplicate encounter IDs: "
    f"{df_clean['encounter_id'].duplicated().sum():,}"
)

# ------------------------------------------------------------
# 5. Target validation
# ------------------------------------------------------------

print("\nTarget validation:")

valid_targets = {"NO", ">30", "<30"}

invalid_targets = set(
    df_clean["readmitted"].dropna().unique()
) - valid_targets

print("Valid target values:", valid_targets)
print("Unexpected target values:", invalid_targets)

# ------------------------------------------------------------
# 6. Numeric range checks
# ------------------------------------------------------------

numeric_columns = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses"
]

print("\nNumeric range analysis:")

range_results = []

for col in numeric_columns:

    values = pd.to_numeric(
        df_clean[col],
        errors="coerce"
    )

    range_results.append({
        "feature": col,
        "min": values.min(),
        "max": values.max(),
        "median": values.median(),
        "missing": values.isna().sum()
    })

range_df = pd.DataFrame(range_results)

print(range_df.to_string(index=False))

# ------------------------------------------------------------
# 7. High-missingness feature identification
# ------------------------------------------------------------

threshold = 50

high_missing = missing[
    missing["missing_percent"] >= threshold
]

print(
    f"\nFeatures with >= {threshold}% missing values:"
)

print(high_missing.to_string())

# ------------------------------------------------------------
# 8. Constant / near-constant features
# ------------------------------------------------------------

unique_counts = df_clean.nunique(
    dropna=False
)

constant_features = unique_counts[
    unique_counts <= 1
]

print("\nConstant features:")
print(constant_features)

# ------------------------------------------------------------
# 9. Save quality report
# ------------------------------------------------------------

report_path = os.path.join(
    RESULTS_DIR,
    "data_quality_report.csv"
)

missing.to_csv(report_path)

print(
    f"\nMissing-value report saved to: "
    f"{report_path}"
)

# ------------------------------------------------------------
# 10. Summary
# ------------------------------------------------------------

summary_path = os.path.join(
    RESULTS_DIR,
    "data_quality_summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write("CAREPREDICT - DATA QUALITY REPORT\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Dataset shape: {df_clean.shape}\n")
    f.write(
        f"Duplicate rows: {duplicate_rows}\n"
    )

    f.write(
        f"Unique patients: "
        f"{df_clean['patient_nbr'].nunique():,}\n"
    )

    f.write(
        f"Unique encounters: "
        f"{df_clean['encounter_id'].nunique():,}\n"
    )

    f.write("\nMissing values:\n")
    f.write(missing.to_string())

    f.write("\n\nHigh-missingness features:\n")
    f.write(high_missing.to_string())

    f.write("\n\nConstant features:\n")
    f.write(constant_features.to_string())

print(
    f"Summary saved to: {summary_path}"
)

print("\n" + "=" * 70)
print("DATA QUALITY ANALYSIS COMPLETE")
print("=" * 70)
