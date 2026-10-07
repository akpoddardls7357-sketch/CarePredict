import os
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedGroupKFold

# ============================================================
# CAREPREDICT - PATIENT-LEVEL DATA SPLIT
# ============================================================

DATA_PATH = "data/processed/engineered_data.csv"
OUTPUT_DIR = "data/processed/splits"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("CAREPREDICT - PATIENT-LEVEL DATA SPLIT")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

TARGET = "readmission_30d"
GROUP = "patient_nbr"

print(f"\nDataset shape: {df.shape}")

print(
    f"Unique patients: "
    f"{df[GROUP].nunique():,}"
)

print(
    f"Total encounters: "
    f"{len(df):,}"
)

# ------------------------------------------------------------
# 2. CREATE STRATIFIED GROUP FOLDS
# ------------------------------------------------------------

print(
    "\nCreating patient-level stratified folds..."
)

# 20 folds allows us to approximately create:
#
# 14 folds -> 70% train
#  3 folds -> 15% validation
#  3 folds -> 15% test

sgkf = StratifiedGroupKFold(
    n_splits=20,
    shuffle=True,
    random_state=42
)

fold_assignment = np.zeros(
    len(df),
    dtype=int
)

for fold_number, (_, indices) in enumerate(
    sgkf.split(
        df,
        df[TARGET],
        groups=df[GROUP]
    )
):

    fold_assignment[indices] = fold_number


df["_fold"] = fold_assignment

# ------------------------------------------------------------
# 3. ASSIGN TRAIN / VALIDATION / TEST
# ------------------------------------------------------------

# First 14 folds = 70%
# Next 3 folds = 15%
# Last 3 folds = 15%

train_folds = set(range(0, 14))
validation_folds = set(range(14, 17))
test_folds = set(range(17, 20))

train_df = df[
    df["_fold"].isin(train_folds)
].copy()

validation_df = df[
    df["_fold"].isin(validation_folds)
].copy()

test_df = df[
    df["_fold"].isin(test_folds)
].copy()

# Remove internal fold column

train_df.drop(
    columns=["_fold"],
    inplace=True
)

validation_df.drop(
    columns=["_fold"],
    inplace=True
)

test_df.drop(
    columns=["_fold"],
    inplace=True
)

# ------------------------------------------------------------
# 4. SPLIT SIZES
# ------------------------------------------------------------

print("\nSplit sizes:")

print(
    f"Training:   "
    f"{len(train_df):,} "
    f"({len(train_df) / len(df) * 100:.2f}%)"
)

print(
    f"Validation: "
    f"{len(validation_df):,} "
    f"({len(validation_df) / len(df) * 100:.2f}%)"
)

print(
    f"Test:       "
    f"{len(test_df):,} "
    f"({len(test_df) / len(df) * 100:.2f}%)"
)

# ------------------------------------------------------------
# 5. TARGET DISTRIBUTION
# ------------------------------------------------------------

def print_target_distribution(
    name,
    data
):

    counts = (
        data[TARGET]
        .value_counts()
        .sort_index()
    )

    percentages = (
        data[TARGET]
        .value_counts(
            normalize=True
        )
        .sort_index()
        .mul(100)
        .round(2)
    )

    print(
        f"\n{name} target distribution:"
    )

    for value in [0, 1]:

        count = counts.get(
            value,
            0
        )

        percentage = percentages.get(
            value,
            0
        )

        label = (
            "No readmission"
            if value == 0
            else "30-day readmission"
        )

        print(
            f"  {label}: "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )


print_target_distribution(
    "Training",
    train_df
)

print_target_distribution(
    "Validation",
    validation_df
)

print_target_distribution(
    "Test",
    test_df
)

# ------------------------------------------------------------
# 6. PATIENT OVERLAP CHECK
# ------------------------------------------------------------

train_patients = set(
    train_df[GROUP]
)

validation_patients = set(
    validation_df[GROUP]
)

test_patients = set(
    test_df[GROUP]
)

train_val_overlap = (
    train_patients
    &
    validation_patients
)

train_test_overlap = (
    train_patients
    &
    test_patients
)

validation_test_overlap = (
    validation_patients
    &
    test_patients
)

print("\nPatient overlap check:")

print(
    f"Train ∩ Validation: "
    f"{len(train_val_overlap)}"
)

print(
    f"Train ∩ Test: "
    f"{len(train_test_overlap)}"
)

print(
    f"Validation ∩ Test: "
    f"{len(validation_test_overlap)}"
)

# ------------------------------------------------------------
# 7. SAVE SPLITS
# ------------------------------------------------------------

train_path = os.path.join(
    OUTPUT_DIR,
    "train.csv"
)

validation_path = os.path.join(
    OUTPUT_DIR,
    "validation.csv"
)

test_path = os.path.join(
    OUTPUT_DIR,
    "test.csv"
)

train_df.to_csv(
    train_path,
    index=False
)

validation_df.to_csv(
    validation_path,
    index=False
)

test_df.to_csv(
    test_path,
    index=False
)

# ------------------------------------------------------------
# 8. SAVE SPLIT SUMMARY
# ------------------------------------------------------------

summary_path = os.path.join(
    OUTPUT_DIR,
    "split_summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "CAREPREDICT - PATIENT-LEVEL DATA SPLIT\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Total encounters: {len(df):,}\n"
    )

    f.write(
        f"Unique patients: "
        f"{df[GROUP].nunique():,}\n\n"
    )

    f.write(
        f"Training encounters: "
        f"{len(train_df):,}\n"
    )

    f.write(
        f"Validation encounters: "
        f"{len(validation_df):,}\n"
    )

    f.write(
        f"Test encounters: "
        f"{len(test_df):,}\n\n"
    )

    f.write(
        "Patient overlap:\n"
    )

    f.write(
        f"Train/Validation: "
        f"{len(train_val_overlap)}\n"
    )

    f.write(
        f"Train/Test: "
        f"{len(train_test_overlap)}\n"
    )

    f.write(
        f"Validation/Test: "
        f"{len(validation_test_overlap)}\n"
    )

print(
    f"\nSaved training data:\n"
    f"{train_path}"
)

print(
    f"Saved validation data:\n"
    f"{validation_path}"
)

print(
    f"Saved test data:\n"
    f"{test_path}"
)

print(
    f"Saved split summary:\n"
    f"{summary_path}"
)

print("\n" + "=" * 70)
print("PATIENT-LEVEL DATA SPLIT COMPLETE")
print("=" * 70)
