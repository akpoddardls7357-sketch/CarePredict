import os
import pandas as pd

# ============================================================
# CAREPREDICT - UCI DATA UNDERSTANDING
# Dataset: Diabetes 130-US Hospitals
# ============================================================

DATA_PATH = "data/raw/diabetic_data.csv"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 70)
print("CAREPREDICT - DATA UNDERSTANDING")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

print("\n[1] Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

# ------------------------------------------------------------
# 2. COLUMN INFORMATION
# ------------------------------------------------------------

print("\n[2] Columns")
print("-" * 70)

for i, column in enumerate(df.columns, start=1):
    print(f"{i:02d}. {column}")

# ------------------------------------------------------------
# 3. DATA TYPES
# ------------------------------------------------------------

print("\n[3] Data types")
print("-" * 70)

print(df.dtypes)

# ------------------------------------------------------------
# 4. FIRST 5 ROWS
# ------------------------------------------------------------

print("\n[4] First 5 rows")
print("-" * 70)

print(df.head())

# ------------------------------------------------------------
# 5. BASIC STATISTICS
# ------------------------------------------------------------

print("\n[5] Numeric summary")
print("-" * 70)

print(df.describe(include="all").transpose())

# ------------------------------------------------------------
# 6. TARGET DISTRIBUTION
# ------------------------------------------------------------

print("\n[6] Readmission target")
print("-" * 70)

target_counts = df["readmitted"].value_counts(dropna=False)

print(target_counts)

print("\nTarget percentages:")

target_percentages = (
    df["readmitted"]
    .value_counts(normalize=True, dropna=False)
    .mul(100)
    .round(2)
)

print(target_percentages)

# ------------------------------------------------------------
# 7. MISSING VALUES
# ------------------------------------------------------------

print("\n[7] Missing values")
print("-" * 70)

# Original UCI dataset uses '?' for missing categorical data.
question_mark_counts = (
    df.astype(str)
    .eq("?")
    .sum()
    .sort_values(ascending=False)
)

print("Question-mark values:")
print(question_mark_counts[question_mark_counts > 0])

print("\nNaN values:")

nan_counts = (
    df.isna()
    .sum()
    .sort_values(ascending=False)
)

print(nan_counts[nan_counts > 0])

# ------------------------------------------------------------
# 8. UNIQUE VALUES
# ------------------------------------------------------------

print("\n[8] Unique values")
print("-" * 70)

unique_counts = (
    df.nunique(dropna=False)
    .sort_values(ascending=False)
)

print(unique_counts)

# ------------------------------------------------------------
# 9. DUPLICATES
# ------------------------------------------------------------

print("\n[9] Duplicate rows")
print("-" * 70)

duplicate_rows = df.duplicated().sum()

print(f"Duplicate rows: {duplicate_rows:,}")

# ------------------------------------------------------------
# 10. PATIENT / ENCOUNTER INFORMATION
# ------------------------------------------------------------

print("\n[10] Patient and encounter information")
print("-" * 70)

print(f"Unique patients: {df['patient_nbr'].nunique():,}")
print(f"Unique encounters: {df['encounter_id'].nunique():,}")

encounters_per_patient = df.groupby(
    "patient_nbr"
).size()

print(
    f"Patients with multiple encounters: "
    f"{(encounters_per_patient > 1).sum():,}"
)

print(
    f"Maximum encounters for one patient: "
    f"{encounters_per_patient.max()}"
)

# ------------------------------------------------------------
# 11. DATASET MEMORY
# ------------------------------------------------------------

print("\n[11] Memory usage")
print("-" * 70)

memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

print(f"Memory usage: {memory_mb:.2f} MB")

# ------------------------------------------------------------
# 12. SAVE DATA PROFILE
# ------------------------------------------------------------

profile_path = os.path.join(
    RESULTS_DIR,
    "uci_data_profile.txt"
)

with open(profile_path, "w", encoding="utf-8") as file:

    file.write("CAREPREDICT - UCI DATA PROFILE\n")
    file.write("=" * 70 + "\n\n")

    file.write(f"Shape: {df.shape}\n")
    file.write(f"Rows: {len(df):,}\n")
    file.write(f"Columns: {len(df.columns)}\n\n")

    file.write("Columns:\n")
    file.write("\n".join(df.columns.tolist()))
    file.write("\n\n")

    file.write("Target distribution:\n")
    file.write(target_counts.to_string())
    file.write("\n\n")

    file.write("Target percentages:\n")
    file.write(target_percentages.to_string())
    file.write("\n\n")

    file.write("Missing values:\n")
    file.write(nan_counts.to_string())
    file.write("\n\n")

    file.write(f"Duplicate rows: {duplicate_rows:,}\n")
    file.write(f"Unique patients: {df['patient_nbr'].nunique():,}\n")
    file.write(f"Unique encounters: {df['encounter_id'].nunique():,}\n")

print(f"\nProfile saved to: {profile_path}")

print("\n" + "=" * 70)
print("DATA UNDERSTANDING COMPLETE")
print("=" * 70)
