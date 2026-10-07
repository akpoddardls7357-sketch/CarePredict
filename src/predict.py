import os
import joblib
import numpy as np
import pandas as pd


# ============================================================
# CAREPREDICT - PRODUCTION PREDICTION ENGINE
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
# MEDICATION FEATURES
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


# ============================================================
# LOAD ARTIFACTS
# ============================================================

print("Loading CarePredict model...")

MODEL = joblib.load(
    MODEL_PATH
)

PREPROCESSOR = joblib.load(
    PIPELINE_PATH
)

print("Model and preprocessing pipeline loaded.")


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(df):
    """
    Reproduce the exact feature engineering used during training.
    """

    df = df.copy()
    
    # ============================================================
    # ENSURE ALL MEDICATION FEATURES EXIST
    # ============================================================

    medication_columns = [
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
        "metformin-pioglitazone",
    ]

    # Reports may not contain every medication.
    # Missing medication information is represented as "No".
    for medication in medication_columns:
        if medication not in df.columns:
            df[medication] = "No"

    # ============================================================
    # MEDICATION ACTIVE FLAGS
    # ============================================================

    for medication in medication_columns:
        active_column = f"{medication}_active"

        df[active_column] = (
            df[medication]
            .astype(str)
            .str.strip()
            .str.lower()
            .ne("no")
            .astype(int)
        )

    # Total medication complexity
    active_columns = [
        f"{medication}_active"
        for medication in medication_columns
    ]

    df["medication_complexity"] = (
        df[active_columns].sum(axis=1)
    )

    # --------------------------------------------------------
    # 1. UTILIZATION FEATURES
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
    # 2. HOSPITALIZATION INTENSITY
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
    # 3. CLINICAL COMPLEXITY
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
    # 4. AGE FEATURES
    # --------------------------------------------------------

    age_numeric = (
        df["age"]
        .astype(str)
        .str.extract(r"(\d+)")[0]
        .astype(float)
    )

    df["age_numeric"] = age_numeric

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
    # 5. INTERACTION FEATURES
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
    # 6. MEDICATION COMPLEXITY
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
                0
                if x.lower() == "no"
                else 1
            )
        )

    df["medication_complexity"] = sum(
        df[col + "_active"]
        for col in existing_medications
    )

    # --------------------------------------------------------
    # 7. DIAGNOSIS PRESENCE
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
    # 8. CLEAN NUMERIC VALUES
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
# PREDICTION FUNCTION
# ============================================================

def predict_readmission(patient_data):
    """
    Predict 30-day readmission risk.

    Parameters
    ----------
    patient_data : dict or pandas.DataFrame
        Raw patient information.

    Returns
    -------
    dict
        Risk probability, classification,
        and processed feature information.
    """

    # --------------------------------------------------------
    # Convert input to DataFrame
    # --------------------------------------------------------

    if isinstance(patient_data, dict):

        df = pd.DataFrame(
            [patient_data]
        )
        
        # ============================================================
        # ENSURE ALL TRAINING-TIME MEDICATION COLUMNS EXIST
        # ============================================================

        medication_columns = [
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
            "metformin-pioglitazone",
        ]

        # Reports may only mention some medications.
        # Missing medication fields are treated as "No".
        for medication in medication_columns:
            if medication not in df.columns:
                df[medication] = "No"

    elif isinstance(
        patient_data,
        pd.DataFrame
    ):

        df = patient_data.copy()

    else:

        raise TypeError(
            "patient_data must be a dictionary "
            "or pandas DataFrame."
        )

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    df_engineered = engineer_features(
        df
    )

    # --------------------------------------------------------
    # Remove non-model columns
    # --------------------------------------------------------

    excluded_columns = [
        "patient_nbr",
        "readmission_30d",
        "readmitted",
        "diag_1",
        "diag_2",
        "diag_3",
        "encounter_id"
    ]

    excluded_columns = [
        col
        for col in excluded_columns
        if col in df_engineered.columns
    ]

    X = df_engineered.drop(
        columns=excluded_columns
    )

    # ========================================================
    # FINAL MODEL INPUT COMPATIBILITY CHECK
    # ========================================================

    # The saved preprocessing pipeline was trained with all
    # medication columns. Uploaded reports may contain only
    # some of them, so guarantee they exist here immediately
    # before preprocessing.

    for medication in MEDICATION_COLUMNS:

        if medication not in X.columns:
            X[medication] = "No"

        active_column = f"{medication}_active"

        X[active_column] = (
            X[medication]
            .astype(str)
            .str.strip()
            .str.lower()
            .ne("no")
            .astype(int)
        )

        # Recalculate medication complexity
    active_columns = [
        f"{medication}_active"
        for medication in MEDICATION_COLUMNS
    ]

    X["medication_complexity"] = X[active_columns].sum(axis=1)

    # --------------------------------------------------------
    # Transform using SAVED pipeline
    # --------------------------------------------------------

    X_processed = PREPROCESSOR.transform(
        X
    )

    X_processed = np.asarray(
        X_processed,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Predict probability
    # --------------------------------------------------------

    probability = float(
        MODEL.predict_proba(
            X_processed
        )[0, 1]
    )

    prediction = int(
        probability >= THRESHOLD
    )

    if prediction == 1:
        risk_level = "High Risk"
    else:
        risk_level = "Low Risk"

    return {
        "risk_probability": probability,
        "risk_percentage": probability * 100,
        "prediction": prediction,
        "risk_level": risk_level,
        "threshold": THRESHOLD
    }
