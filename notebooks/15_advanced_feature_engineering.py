import os
import numpy as np
import pandas as pd

print("=" * 70)
print("CAREPREDICT - ADVANCED FEATURE ENGINEERING")
print("=" * 70)

INPUT_PATH = "data/processed/cleaned_data.csv"
OUTPUT_PATH = "data/processed/engineered_data.csv"

df = pd.read_csv(INPUT_PATH)

print("\nOriginal shape:", df.shape)

# ------------------------------------------------------------
# 1. UTILIZATION FEATURES
# ------------------------------------------------------------

print("\nCreating utilization features...")

df["total_prior_visits"] = (
    df["number_outpatient"]
    + df["number_emergency"]
    + df["number_inpatient"]
)

df["acute_visit_count"] = (
    df["number_emergency"]
    + df["number_inpatient"]
)

df["total_prior_encounters"] = (
    df["number_outpatient"]
    + df["number_emergency"]
    + df["number_inpatient"]
)

df["inpatient_visit_ratio"] = (
    df["number_inpatient"]
    / (df["total_prior_encounters"] + 1)
)

df["emergency_visit_ratio"] = (
    df["number_emergency"]
    / (df["total_prior_encounters"] + 1)
)

df["outpatient_visit_ratio"] = (
    df["number_outpatient"]
    / (df["total_prior_encounters"] + 1)
)

# ------------------------------------------------------------
# 2. HOSPITALIZATION INTENSITY
# ------------------------------------------------------------

print("Creating hospitalization intensity features...")

df["hospitalization_intensity"] = (
    df["time_in_hospital"]
    * (df["number_inpatient"] + 1)
)

df["procedures_per_day"] = (
    df["num_procedures"]
    / (df["time_in_hospital"] + 1)
)

df["labs_per_day"] = (
    df["num_lab_procedures"]
    / (df["time_in_hospital"] + 1)
)

df["medications_per_day"] = (
    df["num_medications"]
    / (df["time_in_hospital"] + 1)
)

# ------------------------------------------------------------
# 3. CLINICAL COMPLEXITY
# ------------------------------------------------------------

print("Creating clinical complexity features...")

df["diagnosis_density"] = (
    df["number_diagnoses"]
    / (df["time_in_hospital"] + 1)
)

df["procedure_medication_ratio"] = (
    df["num_procedures"]
    / (df["num_medications"] + 1)
)

df["lab_procedure_ratio"] = (
    df["num_lab_procedures"]
    / (df["num_procedures"] + 1)
)

df["clinical_complexity_score"] = (
    df["number_diagnoses"]
    + df["num_procedures"]
    + df["num_medications"]
)

# ------------------------------------------------------------
# 4. AGE INTERACTIONS
# ------------------------------------------------------------

print("Creating age interaction features...")

age_numeric = df["age"].astype(str).str.extract(
    r"(\d+)"
)[0].astype(float)

df["age_numeric"] = age_numeric

df["age_hospitalization_interaction"] = (
    df["age_numeric"]
    * df["time_in_hospital"]
)

df["age_inpatient_interaction"] = (
    df["age_numeric"]
    * df["number_inpatient"]
)

df["age_diagnosis_interaction"] = (
    df["age_numeric"]
    * df["number_diagnoses"]
)

# ------------------------------------------------------------
# 5. PRIOR UTILIZATION × CLINICAL COMPLEXITY
# ------------------------------------------------------------

print("Creating interaction features...")

df["prior_visits_diagnosis_interaction"] = (
    df["total_prior_visits"]
    * df["number_diagnoses"]
)

df["inpatient_diagnosis_interaction"] = (
    df["number_inpatient"]
    * df["number_diagnoses"]
)

df["emergency_diagnosis_interaction"] = (
    df["number_emergency"]
    * df["number_diagnoses"]
)

df["medication_diagnosis_interaction"] = (
    df["num_medications"]
    * df["number_diagnoses"]
)

# ------------------------------------------------------------
# 6. MEDICATION COMPLEXITY
# ------------------------------------------------------------

print("Creating medication complexity features...")

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
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone"
]

existing_medication_columns = [
    c for c in medication_columns if c in df.columns
]

def medication_changed(x):
    return 0 if str(x).lower() == "no" else 1

for col in existing_medication_columns:
    df[col + "_active"] = (
        df[col]
        .astype(str)
        .apply(medication_changed)
    )

df["medication_complexity"] = sum(
    df[col + "_active"]
    for col in existing_medication_columns
)

# ------------------------------------------------------------
# 7. DIAGNOSIS PRESENCE
# ------------------------------------------------------------

print("Creating diagnosis presence features...")

for col in ["diag_1", "diag_2", "diag_3"]:
    if col in df.columns:
        df[col + "_present"] = (
            df[col].astype(str) != "?"
        ).astype(int)

df["multiple_diagnosis_flag"] = (
    df["number_diagnoses"] >= 3
).astype(int)

df["high_utilization_flag"] = (
    df["total_prior_visits"] >= 3
).astype(int)

df["long_stay_flag"] = (
    df["time_in_hospital"] >= 7
).astype(int)

# ------------------------------------------------------------
# 8. CLEAN NUMERIC FEATURES
# ------------------------------------------------------------

numeric_cols = df.select_dtypes(
    include=[np.number]
).columns

df[numeric_cols] = df[numeric_cols].replace(
    [np.inf, -np.inf],
    np.nan
)

# Fill newly-created numeric missing values
for col in numeric_cols:
    if df[col].isna().any():
        df[col] = df[col].fillna(
            df[col].median()
        )

# ------------------------------------------------------------
# 9. SAVE
# ------------------------------------------------------------

print("\nFinal shape:", df.shape)

new_features = [
    c for c in df.columns
    if c not in pd.read_csv(INPUT_PATH, nrows=1).columns
]

print("\nNew features created:", len(new_features))

for i, feature in enumerate(new_features, 1):
    print(f"{i:02d}. {feature}")

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSaved:")
print(OUTPUT_PATH)

print("\n" + "=" * 70)
print("ADVANCED FEATURE ENGINEERING COMPLETE")
print("=" * 70)
