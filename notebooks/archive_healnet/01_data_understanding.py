import pandas as pd

# ============================================================
# CAREPREDICT - STEP 4
# First Dataset Inspection
# Dataset: Kaggle HealNet | Vitals & Variables
# ============================================================

# ------------------------------------------------------------
# 1. Dataset path
# ------------------------------------------------------------
DATA_PATH = "../data/raw/train.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

# ------------------------------------------------------------
# 2. Dataset shape
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("DATASET SHAPE")
print("=" * 70)

print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]:,}")

# ------------------------------------------------------------
# 3. Column names
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("COLUMNS")
print("=" * 70)

for i, column in enumerate(df.columns, start=1):
    print(f"{i:3}. {column}")

# ------------------------------------------------------------
# 4. Data types
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(df.dtypes)

# ------------------------------------------------------------
# 5. First 5 records
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("FIRST 5 RECORDS")
print("=" * 70)

print(df.head())

# ------------------------------------------------------------
# 6. Last 5 records
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("LAST 5 RECORDS")
print("=" * 70)

print(df.tail())

# ------------------------------------------------------------
# 7. Missing values
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = df.isnull().sum()

missing_table = pd.DataFrame({
    "Missing_Count": missing,
    "Missing_Percentage": (missing / len(df) * 100).round(2)
})

print(missing_table.sort_values(
    by="Missing_Count",
    ascending=False
))

# ------------------------------------------------------------
# 8. Duplicate records
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("DUPLICATE RECORDS")
print("=" * 70)

duplicates = df.duplicated().sum()

print(f"Duplicate rows: {duplicates:,}")

# ------------------------------------------------------------
# 9. Target variable
# ------------------------------------------------------------
TARGET = "readmitted_30_days"

print("\n" + "=" * 70)
print("TARGET VARIABLE")
print("=" * 70)

if TARGET in df.columns:

    print(f"Target column: {TARGET}")

    print("\nTarget values:")
    print(df[TARGET].value_counts(dropna=False))

    print("\nTarget distribution (%):")
    print(
        (df[TARGET].value_counts(normalize=True, dropna=False) * 100)
        .round(2)
    )

else:
    print(f"WARNING: Target column '{TARGET}' was not found!")

# ------------------------------------------------------------
# 10. Unique values
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("NUMBER OF UNIQUE VALUES")
print("=" * 70)

unique_values = df.nunique().sort_values()

print(unique_values)

# ------------------------------------------------------------
# 11. Numerical columns
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("NUMERICAL COLUMNS")
print("=" * 70)

numeric_columns = df.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

for column in numeric_columns:
    print(column)

# ------------------------------------------------------------
# 12. Categorical columns
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("CATEGORICAL / OBJECT COLUMNS")
print("=" * 70)

categorical_columns = df.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

for column in categorical_columns:
    print(column)

# ------------------------------------------------------------
# 13. Statistical summary
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("STATISTICAL SUMMARY")
print("=" * 70)

print(df.describe(include="all").transpose())

# ------------------------------------------------------------
# 14. Final summary
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("DATASET INSPECTION COMPLETE")
print("=" * 70)

print(f"Dataset size       : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Numerical features : {len(numeric_columns)}")
print(f"Categorical fields : {len(categorical_columns)}")
print(f"Duplicate rows     : {duplicates:,}")

if TARGET in df.columns:
    print(f"Target variable    : {TARGET}")

print("=" * 70)
