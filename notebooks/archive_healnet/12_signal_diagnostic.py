import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# ============================================================
# CAREPREDICT - MODEL SIGNAL DIAGNOSTIC
# ============================================================

TRAIN_PATH = "../data/processed/train.csv"

TARGET = "readmitted_30_days"

df = pd.read_csv(TRAIN_PATH)

print("\n" + "=" * 70)
print("CAREPREDICT - MODEL SIGNAL DIAGNOSTIC")
print("=" * 70)

# ------------------------------------------------------------
# 1. Target relationship with numerical features
# ------------------------------------------------------------

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

numeric_columns.remove(TARGET)

print("\n" + "=" * 70)
print("1. FEATURE / TARGET CORRELATION")
print("=" * 70)

correlations = []

for feature in numeric_columns:

    correlation = df[feature].corr(df[TARGET])

    correlations.append(
        (feature, correlation)
    )

correlations.sort(
    key=lambda x: abs(x[1]),
    reverse=True
)

for feature, correlation in correlations:

    print(
        f"{feature:30} "
        f"{correlation: .5f}"
    )

# ------------------------------------------------------------
# 2. Train an unweighted Random Forest
# ------------------------------------------------------------

X = df.drop(columns=[TARGET])
y = df[TARGET]

# Only numerical columns for this diagnostic
X = X[numeric_columns]

print("\n" + "=" * 70)
print("2. RANDOM FOREST SIGNAL TEST")
print("=" * 70)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1,
    class_weight=None
)

model.fit(X, y)

probabilities = model.predict_proba(X)[:, 1]

auc = roc_auc_score(
    y,
    probabilities
)

print(f"\nTraining ROC-AUC: {auc:.4f}")

# ------------------------------------------------------------
# 3. Feature importance
# ------------------------------------------------------------

importance = pd.Series(
    model.feature_importances_,
    index=numeric_columns
).sort_values(
    ascending=False
)

print("\n" + "=" * 70)
print("3. RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 70)

for feature, value in importance.items():

    print(
        f"{feature:30} "
        f"{value:.5f}"
    )

# ------------------------------------------------------------
# 4. Final interpretation
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("4. DIAGNOSTIC INTERPRETATION")
print("=" * 70)

if auc < 0.55:

    print("""
⚠ Very weak predictive signal detected.

The current features may contain little information
about the readmission target.

This does NOT mean the ML pipeline is broken.

It may indicate that the dataset's target is weakly
related to the available variables.
""")

elif auc < 0.70:

    print("""
Moderate predictive signal detected.
The nonlinear Random Forest model is finding
some useful relationships.
""")

else:

    print("""
Strong predictive signal detected.
Nonlinear relationships appear to exist.
""")

print("=" * 70)
