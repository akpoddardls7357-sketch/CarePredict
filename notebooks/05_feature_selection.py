import os
import pandas as pd
import numpy as np

# ============================================================
# CAREPREDICT - FEATURE SELECTION
# ============================================================

DATA_PATH = "data/processed/cleaned_data.csv"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 70)
print("CAREPREDICT - FEATURE SELECTION")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

target = "readmission_30d"

print(f"\nDataset shape: {df.shape}")

# ------------------------------------------------------------
# 2. IDENTIFY IDENTIFIER COLUMNS
# ------------------------------------------------------------

identifier_columns = [
    "patient_nbr"
]

print("\nIdentifier columns excluded from modeling:")

for col in identifier_columns:
    print(f"  - {col}")

# ------------------------------------------------------------
# 3. INITIAL FEATURE SET
# ------------------------------------------------------------

excluded_columns = identifier_columns + [
    target,
    "readmitted"
]

feature_columns = [
    col
    for col in df.columns
    if col not in excluded_columns
]

print(
    f"\nInitial model features: "
    f"{len(feature_columns)}"
)

# ------------------------------------------------------------
# 4. CONSTANT FEATURES
# ------------------------------------------------------------

unique_counts = (
    df[feature_columns]
    .nunique(dropna=False)
)

constant_features = unique_counts[
    unique_counts <= 1
].index.tolist()

print("\nConstant features:")

if constant_features:
    for col in constant_features:
        print(f"  - {col}")
else:
    print("  None")

# Remove them
feature_columns = [
    col
    for col in feature_columns
    if col not in constant_features
]

# ------------------------------------------------------------
# 5. HIGH-CARDINALITY CATEGORICAL FEATURES
# ------------------------------------------------------------

categorical_features = df[
    feature_columns
].select_dtypes(
    include=["object","category"]
).columns.tolist()

print(
    "\nCategorical features and cardinality:"
)

cardinality = []

for col in categorical_features:

    n_unique = df[col].nunique(
        dropna=False
    )

    cardinality.append({
        "feature": col,
        "unique_values": n_unique
    })

    print(
        f"  {col}: {n_unique}"
    )

cardinality_df = pd.DataFrame(
    cardinality
).sort_values(
    "unique_values",
    ascending=False
)

# ------------------------------------------------------------
# 6. NUMERIC FEATURES
# ------------------------------------------------------------

numeric_features = df[
    feature_columns
].select_dtypes(
    include=["number"]
).columns.tolist()

print(
    f"\nNumeric features: "
    f"{len(numeric_features)}"
)

for col in numeric_features:
    print(
        f"  - {col}"
    )

# ------------------------------------------------------------
# 7. TARGET ASSOCIATION FOR NUMERIC FEATURES
# ------------------------------------------------------------

print(
    "\nNumeric feature correlation with "
    "30-day readmission:"
)

correlations = []

for col in numeric_features:

    corr = df[
        [col, target]
    ].corr(
        numeric_only=True
    ).iloc[0, 1]

    correlations.append({
        "feature": col,
        "correlation": corr
    })

correlation_df = pd.DataFrame(
    correlations
)

correlation_df["abs_correlation"] = (
    correlation_df["correlation"]
    .abs()
)

correlation_df = correlation_df.sort_values(
    "abs_correlation",
    ascending=False
)

print(
    correlation_df[
        [
            "feature",
            "correlation"
        ]
    ].to_string(index=False)
)

# ------------------------------------------------------------
# 8. HIGHLY CORRELATED NUMERIC FEATURES
# ------------------------------------------------------------

if len(numeric_features) > 1:

    numeric_corr = df[
        numeric_features
    ].corr()

    high_corr_pairs = []

    for i in range(
        len(numeric_corr.columns)
    ):

        for j in range(
            i + 1,
            len(numeric_corr.columns)
        ):

            corr_value = numeric_corr.iloc[
                i,
                j
            ]

            if abs(corr_value) >= 0.80:

                high_corr_pairs.append({
                    "feature_1":
                        numeric_corr.columns[i],

                    "feature_2":
                        numeric_corr.columns[j],

                    "correlation":
                        corr_value
                })

    high_corr_df = pd.DataFrame(
        high_corr_pairs
    )

else:

    high_corr_df = pd.DataFrame()

print(
    "\nHighly correlated numeric pairs "
    "(absolute correlation >= 0.80):"
)

if len(high_corr_df) > 0:
    print(
        high_corr_df.to_string(
            index=False
        )
    )
else:
    print("  None")

# ------------------------------------------------------------
# 9. FINAL FEATURE LIST
# ------------------------------------------------------------

print(
    "\n========================================"
)

print(
    f"Final candidate features: "
    f"{len(feature_columns)}"
)

print(
    "========================================"
)

for i, col in enumerate(
    feature_columns,
    start=1
):

    print(
        f"{i:02d}. {col}"
    )

# ------------------------------------------------------------
# 10. SAVE REPORT
# ------------------------------------------------------------

report_path = os.path.join(
    RESULTS_DIR,
    "feature_selection_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "CAREPREDICT - FEATURE SELECTION REPORT\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Dataset shape: {df.shape}\n"
    )

    f.write(
        f"Initial features: "
        f"{len(df.columns) - 2}\n"
    )

    f.write(
        f"Final candidate features: "
        f"{len(feature_columns)}\n\n"
    )

    f.write(
        "Excluded identifier columns:\n"
    )

    for col in identifier_columns:
        f.write(
            f"- {col}\n"
        )

    f.write(
        "\nConstant features:\n"
    )

    for col in constant_features:
        f.write(
            f"- {col}\n"
        )

    f.write(
        "\nCategorical cardinality:\n"
    )

    f.write(
        cardinality_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\nNumeric-target correlations:\n"
    )

    f.write(
        correlation_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\nHighly correlated pairs:\n"
    )

    if len(high_corr_df) > 0:
        f.write(
            high_corr_df.to_string(
                index=False
            )
        )
    else:
        f.write(
            "None"
        )

    f.write(
        "\n\nFinal candidate features:\n"
    )

    for col in feature_columns:
        f.write(
            f"- {col}\n"
        )

print(
    f"\nFeature-selection report saved to:\n"
    f"{report_path}"
)

print("\n" + "=" * 70)
print("FEATURE SELECTION COMPLETE")
print("=" * 70)
