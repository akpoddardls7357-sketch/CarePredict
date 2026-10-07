import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ============================================================
# CAREPREDICT - STEP 12
# Preprocessing Pipeline
# ============================================================

TRAIN_PATH = "../data/processed/train.csv"
VALIDATION_PATH = "../data/processed/validation.csv"
TEST_PATH = "../data/processed/test.csv"

PIPELINE_PATH = "../models/preprocessing_pipeline.pkl"

TARGET = "readmitted_30_days"

# ------------------------------------------------------------
# 1. Load datasets
# ------------------------------------------------------------
train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VALIDATION_PATH)
test_df = pd.read_csv(TEST_PATH)

print("\n" + "=" * 70)
print("CAREPREDICT - PREPROCESSING PIPELINE")
print("=" * 70)

print(f"\nTraining data   : {train_df.shape}")
print(f"Validation data : {val_df.shape}")
print(f"Testing data    : {test_df.shape}")

# ------------------------------------------------------------
# 2. Separate X and y
# ------------------------------------------------------------
X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_val = val_df.drop(columns=[TARGET])
y_val = val_df[TARGET]

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

# ------------------------------------------------------------
# 3. Identify feature types
# ------------------------------------------------------------
categorical_features = [
    "age_group",
    "glucose_category",
    "cholesterol_category",
    "pain_category"
]

numeric_features = [
    column
    for column in X_train.columns
    if column not in categorical_features
]

print("\n" + "=" * 70)
print("1. FEATURE TYPES")
print("=" * 70)

print("\nNumerical features:")
for feature in numeric_features:
    print(f"  • {feature}")

print("\nCategorical features:")
for feature in categorical_features:
    print(f"  • {feature}")

# ------------------------------------------------------------
# 4. Create preprocessing pipeline
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
    ]
)

# ------------------------------------------------------------
# 5. FIT ONLY ON TRAINING DATA
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("2. FITTING PREPROCESSOR")
print("=" * 70)

X_train_processed = preprocessor.fit_transform(X_train)

print("✓ Preprocessor fitted using TRAINING data only.")

# ------------------------------------------------------------
# 6. Transform validation and test
# ------------------------------------------------------------
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

print("✓ Validation data transformed.")
print("✓ Test data transformed.")

# ------------------------------------------------------------
# 7. Check resulting shapes
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("3. PROCESSED DATA SHAPES")
print("=" * 70)

print(f"Original feature count : {X_train.shape[1]}")
print(
    f"Processed train shape  : "
    f"{X_train_processed.shape}"
)
print(
    f"Processed validation   : "
    f"{X_val_processed.shape}"
)
print(
    f"Processed test shape   : "
    f"{X_test_processed.shape}"
)

# ------------------------------------------------------------
# 8. Get transformed feature names
# ------------------------------------------------------------
feature_names = preprocessor.get_feature_names_out()

print("\n" + "=" * 70)
print("4. TRANSFORMED FEATURES")
print("=" * 70)

print(f"Total processed features: {len(feature_names)}")

for i, feature in enumerate(feature_names, 1):
    print(f"{i:3}. {feature}")

# ------------------------------------------------------------
# 9. Verify scaling
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("5. PREPROCESSING VALIDATION")
print("=" * 70)

print(
    "Training processed mean "
    f"(approximately): {X_train_processed.mean():.6f}"
)

print(
    "Training processed std "
    f"(approximately): {X_train_processed.std():.6f}"
)

# ------------------------------------------------------------
# 10. Save preprocessing pipeline
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("6. SAVING PREPROCESSOR")
print("=" * 70)

joblib.dump(
    preprocessor,
    PIPELINE_PATH
)

print(f"✓ Saved preprocessing pipeline:")
print(PIPELINE_PATH)

# ------------------------------------------------------------
# 11. Save processed arrays
# ------------------------------------------------------------
import numpy as np

np.save(
    "../data/processed/X_train.npy",
    X_train_processed
)

np.save(
    "../data/processed/X_validation.npy",
    X_val_processed
)

np.save(
    "../data/processed/X_test.npy",
    X_test_processed
)

np.save(
    "../data/processed/y_train.npy",
    y_train.to_numpy()
)

np.save(
    "../data/processed/y_validation.npy",
    y_val.to_numpy()
)

np.save(
    "../data/processed/y_test.npy",
    y_test.to_numpy()
)

print("\n✓ Processed arrays saved.")

# ------------------------------------------------------------
# 12. Final summary
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 12 COMPLETE")
print("=" * 70)

print(
    """
Preprocessing pipeline:

Numerical features
        ↓
StandardScaler

Categorical features
        ↓
OneHotEncoder

        ↓
ColumnTransformer
        ↓
Processed ML features

The preprocessor was FIT only on training data.
Validation and test data were TRANSFORMED only.
"""
)

print("=" * 70)
