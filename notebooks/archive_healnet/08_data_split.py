import pandas as pd

from sklearn.model_selection import train_test_split

# ============================================================
# CAREPREDICT - STEP 11
# Train / Validation / Test Split
# ============================================================

INPUT_PATH = "../data/processed/engineered_data.csv"

TRAIN_PATH = "../data/processed/train.csv"
VALIDATION_PATH = "../data/processed/validation.csv"
TEST_PATH = "../data/processed/test.csv"

TARGET = "readmitted_30_days"

# ------------------------------------------------------------
# 1. Load engineered dataset
# ------------------------------------------------------------
df = pd.read_csv(INPUT_PATH)

print("\n" + "=" * 70)
print("CAREPREDICT - DATA SPLITTING")
print("=" * 70)

print(f"\nFull dataset shape: {df.shape}")

# ------------------------------------------------------------
# 2. Separate features and target
# ------------------------------------------------------------
X = df.drop(columns=[TARGET])
y = df[TARGET]

print("\nFeatures shape:", X.shape)
print("Target shape  :", y.shape)

# ------------------------------------------------------------
# 3. First split
#    70% training
#    30% temporary
# ------------------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# ------------------------------------------------------------
# 4. Second split
#    15% validation
#    15% testing
# ------------------------------------------------------------
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# ------------------------------------------------------------
# 5. Reconstruct datasets
# ------------------------------------------------------------
train_df = X_train.copy()
train_df[TARGET] = y_train

validation_df = X_val.copy()
validation_df[TARGET] = y_val

test_df = X_test.copy()
test_df[TARGET] = y_test

# ------------------------------------------------------------
# 6. Print shapes
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("1. SPLIT SIZES")
print("=" * 70)

print(f"Training   : {train_df.shape}")
print(f"Validation : {validation_df.shape}")
print(f"Testing    : {test_df.shape}")

# ------------------------------------------------------------
# 7. Check target distributions
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("2. TARGET DISTRIBUTION")
print("=" * 70)

datasets = {
    "Full Dataset": y,
    "Training": y_train,
    "Validation": y_val,
    "Testing": y_test
}

for name, target_values in datasets.items():

    counts = target_values.value_counts().sort_index()
    percentages = (
        target_values.value_counts(normalize=True)
        .sort_index() * 100
    )

    print(f"\n{name}")

    for cls in counts.index:
        print(
            f"  Class {cls}: "
            f"{counts[cls]:,} "
            f"({percentages[cls]:.2f}%)"
        )

# ------------------------------------------------------------
# 8. Verify no overlap
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("3. SPLIT VALIDATION")
print("=" * 70)

train_indices = set(X_train.index)
val_indices = set(X_val.index)
test_indices = set(X_test.index)

train_val_overlap = train_indices.intersection(val_indices)
train_test_overlap = train_indices.intersection(test_indices)
val_test_overlap = val_indices.intersection(test_indices)

print(f"Train/Validation overlap : {len(train_val_overlap)}")
print(f"Train/Test overlap       : {len(train_test_overlap)}")
print(f"Validation/Test overlap  : {len(val_test_overlap)}")

if (
    len(train_val_overlap) == 0
    and len(train_test_overlap) == 0
    and len(val_test_overlap) == 0
):
    print("\n✓ No data overlap between splits.")
else:
    raise ValueError("Data overlap detected!")

# ------------------------------------------------------------
# 9. Save datasets
# ------------------------------------------------------------
train_df.to_csv(TRAIN_PATH, index=False)
validation_df.to_csv(VALIDATION_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)

print("\n" + "=" * 70)
print("4. FILES SAVED")
print("=" * 70)

print(f"✓ {TRAIN_PATH}")
print(f"✓ {VALIDATION_PATH}")
print(f"✓ {TEST_PATH}")

# ------------------------------------------------------------
# 10. Final summary
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 11 COMPLETE")
print("=" * 70)

print(
    f"""
Full dataset : {len(df):,} rows

Training     : {len(train_df):,} rows
Validation   : {len(validation_df):,} rows
Testing      : {len(test_df):,} rows

Split ratio  : 70% / 15% / 15%

Stratification: ENABLED
Random state  : 42
"""
)

print("=" * 70)
