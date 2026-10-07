import os
import joblib
import numpy as np
import pandas as pd
import shap


# ============================================================
# CAREPREDICT - PATIENT EXPLAINABILITY ENGINE
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "xgboost_focused_tuned.pkl"
)

PIPELINE_PATH = os.path.join(
    BASE_DIR,
    "models",
    "preprocessing_pipeline.pkl"
)

THRESHOLD = 0.51


# ============================================================
# LOAD MODEL + PREPROCESSOR
# ============================================================

MODEL = joblib.load(MODEL_PATH)
PREPROCESSOR = joblib.load(PIPELINE_PATH)

EXPLAINER = shap.TreeExplainer(MODEL)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

MEDICATION_COLUMNS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone"
]


def engineer_features(df):

    df = df.copy()
    
    # --------------------------------------------------------
    # ENSURE ALL TRAINING-TIME MEDICATION COLUMNS EXIST
    # --------------------------------------------------------

    for medication in MEDICATION_COLUMNS:

        if medication not in df.columns:
            df[medication] = "No"

    # --------------------------------------------------------
    # Utilization
    # --------------------------------------------------------

    df["total_prior_visits"] = (
        df["number_outpatient"]
        + df["number_emergency"]
        + df["number_inpatient"]
    )

    df["acute_visit_count"] = (
        df["number_emergency"]
        + df["number_inpatient"]
    )

    df["total_prior_encounters"] = (
        df["number_outpatient"]
        + df["number_emergency"]
        + df["number_inpatient"]
    )

    df["inpatient_visit_ratio"] = (
        df["number_inpatient"]
        / (df["total_prior_encounters"] + 1)
    )

    df["emergency_visit_ratio"] = (
        df["number_emergency"]
        / (df["total_prior_encounters"] + 1)
    )

    df["outpatient_visit_ratio"] = (
        df["number_outpatient"]
        / (df["total_prior_encounters"] + 1)
    )

    # --------------------------------------------------------
    # Hospitalization
    # --------------------------------------------------------

    df["hospitalization_intensity"] = (
        df["time_in_hospital"]
        * (df["number_inpatient"] + 1)
    )

    df["procedures_per_day"] = (
        df["num_procedures"]
        / (df["time_in_hospital"] + 1)
    )

    df["labs_per_day"] = (
        df["num_lab_procedures"]
        / (df["time_in_hospital"] + 1)
    )

    df["medications_per_day"] = (
        df["num_medications"]
        / (df["time_in_hospital"] + 1)
    )

    # --------------------------------------------------------
    # Clinical complexity
    # --------------------------------------------------------

    df["diagnosis_density"] = (
        df["number_diagnoses"]
        / (df["time_in_hospital"] + 1)
    )

    df["procedure_medication_ratio"] = (
        df["num_procedures"]
        / (df["num_medications"] + 1)
    )

    df["lab_procedure_ratio"] = (
        df["num_lab_procedures"]
        / (df["num_procedures"] + 1)
    )

    df["clinical_complexity_score"] = (
        df["number_diagnoses"]
        + df["num_procedures"]
        + df["num_medications"]
    )

    # --------------------------------------------------------
    # Age
    # --------------------------------------------------------

    df["age_numeric"] = (
        df["age"]
        .astype(str)
        .str.extract(r"(\d+)")[0]
        .astype(float)
    )

    df["age_hospitalization_interaction"] = (
        df["age_numeric"]
        * df["time_in_hospital"]
    )

    df["age_inpatient_interaction"] = (
        df["age_numeric"]
        * df["number_inpatient"]
    )

    df["age_diagnosis_interaction"] = (
        df["age_numeric"]
        * df["number_diagnoses"]
    )

    # --------------------------------------------------------
    # Interactions
    # --------------------------------------------------------

    df["prior_visits_diagnosis_interaction"] = (
        df["total_prior_visits"]
        * df["number_diagnoses"]
    )

    df["inpatient_diagnosis_interaction"] = (
        df["number_inpatient"]
        * df["number_diagnoses"]
    )

    df["emergency_diagnosis_interaction"] = (
        df["number_emergency"]
        * df["number_diagnoses"]
    )

    df["medication_diagnosis_interaction"] = (
        df["num_medications"]
        * df["number_diagnoses"]
    )

    # --------------------------------------------------------
    # Medication complexity
    # --------------------------------------------------------

    existing_medications = [
        col
        for col in MEDICATION_COLUMNS
        if col in df.columns
    ]

    for col in existing_medications:

        df[col + "_active"] = (
            df[col]
            .astype(str)
            .apply(
                lambda x:
                0 if x.lower() == "no" else 1
            )
        )

    df["medication_complexity"] = sum(
        df[col + "_active"]
        for col in existing_medications
    )

    # --------------------------------------------------------
    # Diagnosis presence
    # --------------------------------------------------------

    for col in [
        "diag_1",
        "diag_2",
        "diag_3"
    ]:

        if col not in df.columns:
            df[col] = "?"

        df[col + "_present"] = (
            df[col]
            .astype(str)
            != "?"
        ).astype(int)

    df["multiple_diagnosis_flag"] = (
        df["number_diagnoses"] >= 3
    ).astype(int)

    df["high_utilization_flag"] = (
        df["total_prior_visits"] >= 3
    ).astype(int)

    df["long_stay_flag"] = (
        df["time_in_hospital"] >= 7
    ).astype(int)

    # --------------------------------------------------------
    # Numeric cleanup
    # --------------------------------------------------------

    numeric_cols = df.select_dtypes(
        include=[np.number]
    ).columns

    df[numeric_cols] = df[numeric_cols].replace(
        [np.inf, -np.inf],
        np.nan
    )

    for col in numeric_cols:

        if df[col].isna().any():

            df[col] = df[col].fillna(
                df[col].median()
            )

    return df


# ============================================================
# EXPLANATION FUNCTION
# ============================================================

def explain_patient(patient_data, top_n=10):

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    if isinstance(patient_data, dict):

        df = pd.DataFrame(
            [patient_data]
        )

    elif isinstance(patient_data, pd.DataFrame):

        df = patient_data.copy()

    else:

        raise TypeError(
            "patient_data must be a dictionary "
            "or pandas DataFrame."
        )
        
    # --------------------------------------------------------
    # ENSURE ALL TRAINING-TIME MEDICATION COLUMNS EXIST
    # --------------------------------------------------------

    # The saved preprocessing pipeline was trained with
    # all 21 medication columns. Reports may only mention
    # a few medications, so missing ones must be represented
    # as "No".

    for medication in MEDICATION_COLUMNS:

        if medication not in df.columns:
            df[medication] = "No"

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    engineered = engineer_features(df)

    # --------------------------------------------------------
    # Remove non-model columns
    # --------------------------------------------------------

    excluded = [
        "patient_nbr",
        "readmission_30d",
        "readmitted",
        "diag_1",
        "diag_2",
        "diag_3",
        "encounter_id"
    ]

    excluded = [
        col
        for col in excluded
        if col in engineered.columns
    ]

    X = engineered.drop(
        columns=excluded
    )

    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    X_processed = PREPROCESSOR.transform(X)

    X_processed = np.asarray(
        X_processed,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    probability = float(
        MODEL.predict_proba(
            X_processed
        )[0, 1]
    )

    prediction = int(
        probability >= THRESHOLD
    )

    # --------------------------------------------------------
    # Feature names
    # --------------------------------------------------------

    feature_names = (
        PREPROCESSOR
        .get_feature_names_out()
    )

    feature_names = np.array([
        name.replace("num__", "")
            .replace("cat__", "")
        for name in feature_names
    ])

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    shap_values = EXPLAINER.shap_values(
        X_processed
    )

    if isinstance(shap_values, list):

        if len(shap_values) == 2:
            shap_values = shap_values[1]
        else:
            shap_values = shap_values[0]

    shap_values = np.asarray(
        shap_values
    ).reshape(-1)

    # --------------------------------------------------------
    # Explanation dataframe
    # --------------------------------------------------------

    explanation = pd.DataFrame({
        "feature": feature_names,
        "value": X_processed[0],
        "shap_value": shap_values,
        "absolute_shap": np.abs(shap_values)
    })

    explanation["impact"] = np.where(
        explanation["shap_value"] > 0,
        "Increases Risk",
        "Decreases Risk"
    )

    explanation = (
        explanation
        .sort_values(
            "absolute_shap",
            ascending=False
        )
        .reset_index(drop=True)
    )

    explanation["rank"] = (
        np.arange(len(explanation)) + 1
    )

    # --------------------------------------------------------
    # Top risk factors
    # --------------------------------------------------------

    risk_factors = (
        explanation[
            explanation["shap_value"] > 0
        ]
        .head(top_n)
        .copy()
    )

    protective_factors = (
        explanation[
            explanation["shap_value"] < 0
        ]
        .head(top_n)
        .copy()
    )

    return {
        "risk_probability": probability,
        "risk_percentage": probability * 100,
        "prediction": prediction,
        "risk_level": (
            "High Risk"
            if prediction == 1
            else "Low Risk"
        ),
        "threshold": THRESHOLD,
        "all_features": explanation,
        "risk_factors": risk_factors,
        "protective_factors": protective_factors
    }
