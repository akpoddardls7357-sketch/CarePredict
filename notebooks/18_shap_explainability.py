import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data/processed/preprocessed"
MODEL_PATH = "models/xgboost_focused_tuned.pkl"
PIPELINE_PATH = "models/preprocessing_pipeline.pkl"
RESULTS_DIR = "results"

THRESHOLD = 0.51

# Number of validation patients used for global explanation
SHAP_SAMPLE_SIZE = 2000

# Patient index used for example explanation
PATIENT_INDEX = 0


# ============================================================
# SETUP
# ============================================================

os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 70)
print("CAREPREDICT - SHAP EXPLAINABILITY")
print("=" * 70)

print("\nModel is LOCKED.")
print(f"Model: {MODEL_PATH}")
print(f"Classification threshold: {THRESHOLD}")

print("\nSHAP explanations will use VALIDATION data.")
print("The untouched TEST set will not be used.")


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("\n[1/7] Loading validation data...")

X_validation = np.load(
    os.path.join(DATA_DIR, "X_validation.npy")
)

y_validation = np.load(
    os.path.join(DATA_DIR, "y_validation.npy")
)

print(f"Validation shape: {X_validation.shape}")
print(f"Validation targets: {y_validation.shape}")


# ============================================================
# LOAD MODEL
# ============================================================

print("\n[2/7] Loading locked XGBoost model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# GET FEATURE NAMES
# ============================================================

print("\n[3/7] Loading feature names...")

feature_names = None

try:
    preprocessing_pipeline = joblib.load(PIPELINE_PATH)

    if hasattr(
        preprocessing_pipeline,
        "get_feature_names_out"
    ):
        feature_names = (
            preprocessing_pipeline
            .get_feature_names_out()
        )

except Exception as e:
    print(
        "Could not extract feature names from preprocessing "
        f"pipeline: {e}"
    )


# Fallback
if feature_names is None:
    feature_names = np.array(
        [f"feature_{i}" for i in range(X_validation.shape[1])]
    )

feature_names = np.asarray(feature_names).astype(str)

# Clean common sklearn prefixes
feature_names = np.array([
    name.replace("num__", "")
        .replace("cat__", "")
    for name in feature_names
])

if len(feature_names) != X_validation.shape[1]:

    print(
        "\nWARNING: Feature-name count does not match "
        "feature count."
    )

    feature_names = np.array([
        f"feature_{i}"
        for i in range(X_validation.shape[1])
    ])

print(f"Number of features: {len(feature_names)}")


# ============================================================
# PREPARE SHAP SAMPLE
# ============================================================

print("\n[4/7] Preparing SHAP sample...")

sample_size = min(
    SHAP_SAMPLE_SIZE,
    len(X_validation)
)

# Fixed random state for reproducibility
rng = np.random.default_rng(42)

sample_indices = rng.choice(
    len(X_validation),
    size=sample_size,
    replace=False
)

X_sample = X_validation[sample_indices]

print(f"SHAP sample size: {len(X_sample)}")


# ============================================================
# CREATE SHAP EXPLAINER
# ============================================================

print("\n[5/7] Creating SHAP TreeExplainer...")

explainer = shap.TreeExplainer(model)

print("TreeExplainer created successfully.")


# ============================================================
# CALCULATE SHAP VALUES
# ============================================================

print("\nCalculating SHAP values...")

shap_values = explainer.shap_values(
    X_sample
)

# XGBoost binary classifiers may return:
# - ndarray
# - list of arrays depending on SHAP version

if isinstance(shap_values, list):

    if len(shap_values) == 2:
        shap_values = shap_values[1]
    else:
        shap_values = shap_values[0]

shap_values = np.asarray(shap_values)

print(f"SHAP values shape: {shap_values.shape}")


# ============================================================
# GLOBAL FEATURE IMPORTANCE
# ============================================================

print("\n[6/7] Calculating global feature importance...")

mean_abs_shap = np.mean(
    np.abs(shap_values),
    axis=0
)

global_importance = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": mean_abs_shap
})

global_importance = (
    global_importance
    .sort_values(
        "mean_abs_shap",
        ascending=False
    )
    .reset_index(drop=True)
)

global_importance["rank"] = (
    np.arange(len(global_importance)) + 1
)

# Reorder columns
global_importance = global_importance[
    [
        "rank",
        "feature",
        "mean_abs_shap"
    ]
]


# ============================================================
# SAVE GLOBAL IMPORTANCE
# ============================================================

global_importance.to_csv(
    os.path.join(
        RESULTS_DIR,
        "shap_global_feature_importance.csv"
    ),
    index=False
)


# ============================================================
# TOP 20 FEATURES
# ============================================================

top_n = min(20, len(global_importance))

top_features = global_importance.head(top_n)

print("\n" + "=" * 70)
print("TOP SHAP FEATURES")
print("=" * 70)

print(
    top_features[
        [
            "rank",
            "feature",
            "mean_abs_shap"
        ]
    ].to_string(index=False)
)


# ============================================================
# GLOBAL IMPORTANCE PLOT
# ============================================================

plt.figure(figsize=(10, 8))

plot_data = top_features.iloc[::-1]

plt.barh(
    plot_data["feature"],
    plot_data["mean_abs_shap"]
)

plt.xlabel("Mean Absolute SHAP Value")
plt.ylabel("Feature")
plt.title(
    "CarePredict - Top 20 Global Feature Importance"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "shap_global_feature_importance.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PATIENT-LEVEL EXPLANATION
# ============================================================

print("\n[7/7] Creating patient-level explanation...")

patient_index = min(
    PATIENT_INDEX,
    len(X_validation) - 1
)

X_patient = X_validation[
    patient_index:patient_index + 1
]

y_patient = int(
    y_validation[patient_index]
)

patient_probability = float(
    model.predict_proba(X_patient)[0, 1]
)

patient_prediction = int(
    patient_probability >= THRESHOLD
)

patient_shap = explainer.shap_values(
    X_patient
)

if isinstance(patient_shap, list):

    if len(patient_shap) == 2:
        patient_shap = patient_shap[1]
    else:
        patient_shap = patient_shap[0]

patient_shap = np.asarray(
    patient_shap
).reshape(-1)

patient_explanation = pd.DataFrame({
    "feature": feature_names,
    "feature_value": X_patient[0],
    "shap_value": patient_shap,
    "absolute_shap": np.abs(patient_shap)
})

patient_explanation["impact"] = np.where(
    patient_explanation["shap_value"] > 0,
    "increases_risk",
    "decreases_risk"
)

patient_explanation = (
    patient_explanation
    .sort_values(
        "absolute_shap",
        ascending=False
    )
    .reset_index(drop=True)
)

patient_explanation["rank"] = (
    np.arange(len(patient_explanation)) + 1
)

patient_explanation = patient_explanation[
    [
        "rank",
        "feature",
        "feature_value",
        "shap_value",
        "absolute_shap",
        "impact"
    ]
]


# ============================================================
# SAVE PATIENT EXPLANATION
# ============================================================

patient_explanation.to_csv(
    os.path.join(
        RESULTS_DIR,
        "shap_patient_example.csv"
    ),
    index=False
)


# ============================================================
# SAVE PATIENT SUMMARY
# ============================================================

top_positive = (
    patient_explanation[
        patient_explanation["shap_value"] > 0
    ]
    .head(10)
)

top_negative = (
    patient_explanation[
        patient_explanation["shap_value"] < 0
    ]
    .head(10)
)

patient_summary = {
    "patient_index": int(patient_index),
    "actual_outcome": y_patient,
    "predicted_probability": patient_probability,
    "classification_threshold": THRESHOLD,
    "predicted_class": patient_prediction,
    "predicted_label": (
        "High Risk"
        if patient_prediction == 1
        else "Low Risk"
    ),
    "top_risk_increasing_features": (
        top_positive["feature"].tolist()
    ),
    "top_risk_decreasing_features": (
        top_negative["feature"].tolist()
    )
}

with open(
    os.path.join(
        RESULTS_DIR,
        "shap_patient_example.json"
    ),
    "w"
) as f:
    json.dump(
        patient_summary,
        f,
        indent=4
    )


# ============================================================
# PATIENT SHAP PLOT
# ============================================================

patient_plot = patient_explanation.head(15).copy()

patient_plot = patient_plot.iloc[::-1]

plt.figure(figsize=(10, 7))

plt.barh(
    patient_plot["feature"],
    patient_plot["shap_value"]
)

plt.axvline(
    0,
    linewidth=1
)

plt.xlabel("SHAP Value")
plt.ylabel("Feature")

plt.title(
    f"Patient-Level Readmission Risk Explanation "
    f"(Risk Score: {patient_probability:.2%})"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "shap_patient_example.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SHAP EXPLAINABILITY COMPLETE")
print("=" * 70)

print(
    f"\nExample patient risk score: "
    f"{patient_probability:.2%}"
)

print(
    f"Classification: "
    f"{'HIGH RISK' if patient_prediction == 1 else 'LOW RISK'}"
)

print(
    f"Actual outcome: {y_patient}"
)

print("\nSaved artifacts:")

print(
    "  results/shap_global_feature_importance.csv"
)

print(
    "  results/shap_global_feature_importance.png"
)

print(
    "  results/shap_patient_example.csv"
)

print(
    "  results/shap_patient_example.json"
)

print(
    "  results/shap_patient_example.png"
)

print("\nModel remains locked.")
print("Test set remains untouched.")
