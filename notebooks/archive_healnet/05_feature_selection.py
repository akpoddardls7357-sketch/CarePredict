import pandas as pd
import numpy as np

# ============================================================
# CAREPREDICT - STEP 8
# Feature Selection
# Dataset: Kaggle HealNet | Vitals & Variables
# ============================================================

DATA_PATH = "../data/processed/cleaned_data.csv"

TARGET = "readmitted_30_days"

# ------------------------------------------------------------
# 1. Load processed dataset
# ------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

print("\n" + "=" * 70)
print("CAREPREDICT - FEATURE SELECTION")
print("=" * 70)

print(f"\nDataset shape: {df.shape}")

# ------------------------------------------------------------
# 2. Separate features and target
# ------------------------------------------------------------
X = df.drop(columns=[TARGET])
y = df[TARGET]

print("\n" + "=" * 70)
print("1. FEATURE / TARGET SEPARATION")
print("=" * 70)

print(f"Number of features : {X.shape[1]}")
print(f"Number of samples  : {X.shape[0]}")
print(f"Target             : {TARGET}")

# ------------------------------------------------------------
# 3. List features
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("2. AVAILABLE FEATURES")
print("=" * 70)

for i, column in enumerate(X.columns, 1):
    print(f"{i:2}. {column}")

# ------------------------------------------------------------
# 4. Identify feature types
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("3. FEATURE TYPES")
print("=" * 70)

numeric_features = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\nNumerical features:")
for feature in numeric_features:
    print(f"  • {feature}")

print("\nCategorical features:")
for feature in categorical_features:
    print(f"  • {feature}")

# ------------------------------------------------------------
# 5. Check constant features
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("4. CONSTANT FEATURE CHECK")
print("=" * 70)

constant_features = [
    column
    for column in X.columns
    if X[column].nunique() <= 1
]

if len(constant_features) == 0:
    print("✓ No constant features found.")
else:
    print("⚠ Constant features:")
    for feature in constant_features:
        print(f"  • {feature}")

# ------------------------------------------------------------
# 6. Check highly correlated numerical features
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("5. CORRELATION CHECK")
print("=" * 70)

if len(numeric_features) > 1:

    correlation_matrix = X[numeric_features].corr()

    high_correlations = []

    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):

            feature_1 = correlation_matrix.columns[i]
            feature_2 = correlation_matrix.columns[j]

            correlation = correlation_matrix.iloc[i, j]

            if abs(correlation) >= 0.80:

                high_correlations.append(
                    (
                        feature_1,
                        feature_2,
                        correlation
                    )
                )

    if high_correlations:

        print(
            "\nHighly correlated feature pairs "
            "(|correlation| >= 0.80):"
        )

        for feature_1, feature_2, correlation in high_correlations:

            print(
                f"  {feature_1} ↔ {feature_2}: "
                f"{correlation:.3f}"
            )

    else:

        print(
            "✓ No feature pairs have "
            "|correlation| >= 0.80."
        )

# ------------------------------------------------------------
# 7. Feature-target correlation
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("6. NUMERICAL FEATURE / TARGET CORRELATION")
print("=" * 70)

target_correlations = []

for feature in numeric_features:

    correlation = X[feature].corr(y)

    target_correlations.append(
        (feature, correlation)
    )

target_correlations.sort(
    key=lambda x: abs(x[1]),
    reverse=True
)

for feature, correlation in target_correlations:

    print(
        f"{feature:30} "
        f"{correlation: .4f}"
    )

# ------------------------------------------------------------
# 8. Final feature list
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("7. FINAL FEATURE SET")
print("=" * 70)

final_features = X.columns.tolist()

print(
    f"\nTotal model features: "
    f"{len(final_features)}"
)

for feature in final_features:
    print(f"  ✓ {feature}")

# ------------------------------------------------------------
# 9. Final validation
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("8. FEATURE SELECTION VALIDATION")
print("=" * 70)

if TARGET not in final_features:
    print("✓ Target successfully excluded from features.")
else:
    print("⚠ TARGET LEAKAGE: target is present in features!")

if "patient_id" not in final_features:
    print("✓ patient_id successfully excluded.")
else:
    print("⚠ patient_id is still present.")

if len(constant_features) == 0:
    print("✓ No constant features need removal.")

print("\nFeature selection analysis complete.")
print("=" * 70)
