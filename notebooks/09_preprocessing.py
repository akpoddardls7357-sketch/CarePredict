import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

# ============================================================
# CAREPREDICT - PREPROCESSING PIPELINE
# ============================================================

SPLIT_DIR = "data/processed/splits"
MODEL_DIR = "models"
OUTPUT_DIR = "data/processed/preprocessed"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("CAREPREDICT - PREPROCESSING")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD SPLITS
# ------------------------------------------------------------

train = pd.read_csv(
    f"{SPLIT_DIR}/train.csv"
)

validation = pd.read_csv(
    f"{SPLIT_DIR}/validation.csv"
)

test = pd.read_csv(
    f"{SPLIT_DIR}/test.csv"
)

TARGET = "readmission_30d"
PATIENT_ID = "patient_nbr"

print("\nLoaded datasets:")

print(
    f"Training:   {train.shape}"
)

print(
    f"Validation: {validation.shape}"
)

print(
    f"Test:       {test.shape}"
)

# ------------------------------------------------------------
# 2. REMOVE NON-MODEL COLUMNS
# ------------------------------------------------------------

# These columns must never be model features.
#
# patient_nbr:
#   Identifier only.
#
# readmission_30d:
#   Target.
#
# readmitted:
#   Original target from which readmission_30d was created.
#
# Raw diagnosis codes:
#   We use the engineered diagnosis categories instead.
#
# encounter_id:
#   Already removed, but included defensively.

excluded_columns = [
    PATIENT_ID,
    TARGET,
    "readmitted",
    "diag_1",
    "diag_2",
    "diag_3",
    "encounter_id"
]

excluded_columns = [
    col
    for col in excluded_columns
    if col in train.columns
]

print("\nExcluded columns:")

for col in excluded_columns:
    print(
        f"  - {col}"
    )

X_train = train.drop(
    columns=excluded_columns
)

X_validation = validation.drop(
    columns=excluded_columns
)

X_test = test.drop(
    columns=excluded_columns
)

y_train = train[TARGET]
y_validation = validation[TARGET]
y_test = test[TARGET]

# ------------------------------------------------------------
# 3. IDENTIFY DATA TYPES
# ------------------------------------------------------------

categorical_features = (
    X_train
    .select_dtypes(
        include=["object", "category"]
    )
    .columns
    .tolist()
)

numeric_features = (
    X_train
    .select_dtypes(
        include=["number", "bool"]
    )
    .columns
    .tolist()
)

print(
    f"\nNumeric features: "
    f"{len(numeric_features)}"
)

for col in numeric_features:
    print(
        f"  - {col}"
    )

print(
    f"\nCategorical features: "
    f"{len(categorical_features)}"
)

for col in categorical_features:
    print(
        f"  - {col}"
    )

# ------------------------------------------------------------
# 4. PREPROCESSING PIPELINE
# ------------------------------------------------------------

numeric_transformer = StandardScaler()

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ],
    remainder="drop"
)

# ------------------------------------------------------------
# 5. FIT ONLY ON TRAINING DATA
# ------------------------------------------------------------

print(
    "\nFitting preprocessing pipeline "
    "on training data..."
)

X_train_processed = (
    preprocessor.fit_transform(
        X_train
    )
)

print(
    "Transforming validation data..."
)

X_validation_processed = (
    preprocessor.transform(
        X_validation
    )
)

print(
    "Transforming test data..."
)

X_test_processed = (
    preprocessor.transform(
        X_test
    )
)

# ------------------------------------------------------------
# 6. CONVERT TO NUMPY
# ------------------------------------------------------------

X_train_processed = np.asarray(
    X_train_processed,
    dtype=np.float32
)

X_validation_processed = np.asarray(
    X_validation_processed,
    dtype=np.float32
)

X_test_processed = np.asarray(
    X_test_processed,
    dtype=np.float32
)

y_train_array = np.asarray(
    y_train,
    dtype=np.int8
)

y_validation_array = np.asarray(
    y_validation,
    dtype=np.int8
)

y_test_array = np.asarray(
    y_test,
    dtype=np.int8
)

# ------------------------------------------------------------
# 7. SHAPES
# ------------------------------------------------------------

print("\nProcessed shapes:")

print(
    f"Training:   "
    f"{X_train_processed.shape}"
)

print(
    f"Validation: "
    f"{X_validation_processed.shape}"
)

print(
    f"Test:       "
    f"{X_test_processed.shape}"
)

# ------------------------------------------------------------
# 8. GET FEATURE NAMES
# ------------------------------------------------------------

feature_names = (
    preprocessor
    .get_feature_names_out()
)

print(
    f"\nFinal processed feature count: "
    f"{len(feature_names)}"
)

# ------------------------------------------------------------
# 9. SAVE ARRAYS
# ------------------------------------------------------------

np.save(
    f"{OUTPUT_DIR}/X_train.npy",
    X_train_processed
)

np.save(
    f"{OUTPUT_DIR}/X_validation.npy",
    X_validation_processed
)

np.save(
    f"{OUTPUT_DIR}/X_test.npy",
    X_test_processed
)

np.save(
    f"{OUTPUT_DIR}/y_train.npy",
    y_train_array
)

np.save(
    f"{OUTPUT_DIR}/y_validation.npy",
    y_validation_array
)

np.save(
    f"{OUTPUT_DIR}/y_test.npy",
    y_test_array
)

# ------------------------------------------------------------
# 10. SAVE FEATURE NAMES
# ------------------------------------------------------------

feature_names_path = (
    f"{OUTPUT_DIR}/feature_names.csv"
)

pd.DataFrame({
    "feature_name": feature_names
}).to_csv(
    feature_names_path,
    index=False
)

# ------------------------------------------------------------
# 11. SAVE PREPROCESSOR
# ------------------------------------------------------------

preprocessor_path = (
    f"{MODEL_DIR}/preprocessing_pipeline.pkl"
)

joblib.dump(
    preprocessor,
    preprocessor_path
)

# ------------------------------------------------------------
# 12. VALIDATION CHECK
# ------------------------------------------------------------

print("\nValidation checks:")

print(
    f"Training NaNs: "
    f"{np.isnan(X_train_processed).sum()}"
)

print(
    f"Validation NaNs: "
    f"{np.isnan(X_validation_processed).sum()}"
)

print(
    f"Test NaNs: "
    f"{np.isnan(X_test_processed).sum()}"
)

print(
    f"\nTraining positive class: "
    f"{y_train_array.sum():,}"
)

print(
    f"Validation positive class: "
    f"{y_validation_array.sum():,}"
)

print(
    f"Test positive class: "
    f"{y_test_array.sum():,}"
)

print(
    "\nPreprocessing pipeline saved to:"
)

print(
    preprocessor_path
)

print(
    "\nProcessed datasets saved to:"
)

print(
    OUTPUT_DIR
)

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)
