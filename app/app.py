import os
import sys

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


from src.predict import predict_readmission
from src.explain import explain_patient


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CarePredict",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #64748b;
        margin-bottom: 25px;
    }

    .risk-card {
        padding: 25px;
        border-radius: 15px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        text-align: center;
    }

    .metric-card {
        padding: 18px;
        border-radius: 12px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
    }

    .high-risk {
        color: #dc2626;
        font-size: 32px;
        font-weight: 700;
    }

    .low-risk {
        color: #16a34a;
        font-size: 32px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def pretty_feature_name(name):
    """Convert model feature names into readable dashboard labels."""

    name = str(name)

    replacements = {
        "numeric__": "",
        "categorical__": "",
        "_interaction": " interaction",
        "_ratio": " ratio",
        "_count": " count",
        "_complexity": " complexity",
        "_intensity": " intensity",
        "_per_day": " per day",
        "_active": " active",
        "_present": " present",
        "_flag": "",
        "_": " "
    }

    for old, new in replacements.items():
        name = name.replace(old, new)

    return name.strip().title()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🏥 CarePredict</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Predictive Analytics Dashboard for Value-Based Healthcare
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("CarePredict")

    st.markdown(
        """
        **Purpose**

        Estimate the probability of 30-day hospital
        readmission and provide model-based explanations
        for the prediction.
        """
    )

    st.divider()

    st.subheader("Locked Model")

    st.write("**Algorithm:** XGBoost")
    st.write("**Threshold:** 0.51")
    st.write("**Test ROC-AUC:** 0.6676")
    st.write("**Test PR-AUC:** 0.2255")

    st.divider()

    st.caption(
        "Decision-support prototype. Predictions should "
        "not replace clinical judgment."
    )


# ============================================================
# NAVIGATION
# ============================================================

page = st.radio(
    "Navigation",
    [
        "🏠 Overview",
        "👤 Patient Risk Prediction",
        "📊 Model Performance",
        "🔎 Explainability"
    ],
    horizontal=True
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.header("Overview")

    st.write(
        """
        CarePredict is a machine-learning based risk
        stratification system designed to estimate the
        probability of 30-day hospital readmission.
        """
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Test ROC-AUC",
            "0.6676"
        )

    with col2:
        st.metric(
            "Test PR-AUC",
            "0.2255"
        )

    with col3:
        st.metric(
            "Recall",
            "47.08%"
        )

    with col4:
        st.metric(
            "F1 Score",
            "27.40%"
        )

    st.divider()

    st.subheader("How CarePredict Works")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("### 1️⃣ Patient Data")
        st.write(
            "Demographic, admission, utilization, "
            "hospitalization and medication information."
        )

    with c2:
        st.markdown("### 2️⃣ Feature Engineering")
        st.write(
            "Clinical complexity, utilization and "
            "interaction features are generated."
        )

    with c3:
        st.markdown("### 3️⃣ XGBoost")
        st.write(
            "The locked XGBoost model estimates "
            "30-day readmission probability."
        )

    with c4:
        st.markdown("### 4️⃣ SHAP")
        st.write(
            "SHAP explains the factors influencing "
            "each individual prediction."
        )

    st.divider()

    st.info(
        "CarePredict is designed as a risk-stratification "
        "and decision-support prototype, not as a standalone "
        "clinical decision system."
    )


# ============================================================
# PATIENT RISK PREDICTION
# ============================================================

elif page == "👤 Patient Risk Prediction":

    st.header("Patient Risk Prediction")

    st.write(
        "Enter patient information to estimate 30-day "
        "readmission risk."
    )

    with st.form("patient_form"):

        # ----------------------------------------------------
        # DEMOGRAPHICS
        # ----------------------------------------------------

        st.subheader("Demographics")

        col1, col2, col3 = st.columns(3)

        with col1:
            race = st.selectbox(
                "Race",
                [
                    "Caucasian",
                    "AfricanAmerican",
                    "Asian",
                    "Hispanic",
                    "Other",
                    "?"
                ]
            )

        with col2:
            gender = st.selectbox(
                "Gender",
                [
                    "Male",
                    "Female",
                    "Unknown/Invalid"
                ]
            )

        with col3:
            age = st.selectbox(
                "Age Group",
                [
                    "[0-10)",
                    "[10-20)",
                    "[20-30)",
                    "[30-40)",
                    "[40-50)",
                    "[50-60)",
                    "[60-70)",
                    "[70-80)",
                    "[80-90)",
                    "[90-100)"
                ],
                index=7
            )

        # ----------------------------------------------------
        # ADMISSION
        # ----------------------------------------------------

        st.subheader("Admission Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            admission_type_id = st.number_input(
                "Admission Type ID",
                min_value=1,
                max_value=8,
                value=1,
                step=1
            )

        with col2:
            discharge_disposition_id = st.number_input(
                "Discharge Disposition ID",
                min_value=1,
                max_value=30,
                value=1,
                step=1
            )

        with col3:
            admission_source_id = st.number_input(
                "Admission Source ID",
                min_value=1,
                max_value=25,
                value=7,
                step=1
            )

        # ----------------------------------------------------
        # HOSPITALIZATION
        # ----------------------------------------------------

        st.subheader("Hospitalization & Utilization")

        col1, col2, col3 = st.columns(3)

        with col1:
            time_in_hospital = st.number_input(
                "Time in Hospital (days)",
                min_value=1,
                max_value=20,
                value=4,
                step=1
            )

        with col2:
            num_lab_procedures = st.number_input(
                "Number of Lab Procedures",
                min_value=0,
                max_value=150,
                value=45,
                step=1
            )

        with col3:
            num_procedures = st.number_input(
                "Number of Procedures",
                min_value=0,
                max_value=10,
                value=1,
                step=1
            )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            num_medications = st.number_input(
                "Number of Medications",
                min_value=0,
                max_value=100,
                value=12,
                step=1
            )

        with col2:
            number_outpatient = st.number_input(
                "Outpatient Visits",
                min_value=0,
                max_value=100,
                value=0,
                step=1
            )

        with col3:
            number_emergency = st.number_input(
                "Emergency Visits",
                min_value=0,
                max_value=100,
                value=1,
                step=1
            )

        with col4:
            number_inpatient = st.number_input(
                "Prior Inpatient Visits",
                min_value=0,
                max_value=100,
                value=2,
                step=1
            )

        number_diagnoses = st.number_input(
            "Number of Diagnoses",
            min_value=1,
            max_value=20,
            value=7,
            step=1
        )

        # ----------------------------------------------------
        # DIAGNOSIS INFORMATION
        # ----------------------------------------------------

        st.subheader("Diagnosis Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            diag_1 = st.text_input(
                "Primary Diagnosis",
                value="?"
            )

        with col2:
            diag_2 = st.text_input(
                "Secondary Diagnosis",
                value="?"
            )

        with col3:
            diag_3 = st.text_input(
                "Tertiary Diagnosis",
                value="?"
            )

        # ----------------------------------------------------
        # MEDICATIONS
        # ----------------------------------------------------

        st.subheader("Medication Status")

        medication_options = [
            "No",
            "Steady",
            "Up",
            "Down"
        ]

        medications = {}

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
            "metformin-pioglitazone"
        ]

        for start in range(
            0,
            len(medication_columns),
            3
        ):

            cols = st.columns(3)

            for col, medication in zip(
                cols,
                medication_columns[start:start + 3]
            ):

                with col:

                    medications[medication] = (
                        st.selectbox(
                            medication.replace(
                                "_",
                                " "
                            ).title(),
                            medication_options,
                            key=f"med_{medication}"
                        )
                    )

        # ----------------------------------------------------
        # DIABETES
        # ----------------------------------------------------

        st.subheader("Diabetes Medication")

        col1, col2 = st.columns(2)

        with col1:
            change = st.selectbox(
                "Medication Change",
                [
                    "No",
                    "Ch",
                    "?"
                ]
            )

        with col2:
            diabetesMed = st.selectbox(
                "Diabetes Medication",
                [
                    "Yes",
                    "No"
                ]
            )

        st.divider()

        submitted = st.form_submit_button(
            "🔍 Predict Readmission Risk",
            use_container_width=True
        )

    # --------------------------------------------------------
    # RUN PREDICTION
    # --------------------------------------------------------

    if submitted:

        patient = {
            "race": race,
            "gender": gender,
            "age": age,
            "admission_type_id": admission_type_id,
            "discharge_disposition_id":
                discharge_disposition_id,
            "admission_source_id":
                admission_source_id,
            "time_in_hospital":
                time_in_hospital,
            "num_lab_procedures":
                num_lab_procedures,
            "num_procedures":
                num_procedures,
            "num_medications":
                num_medications,
            "number_outpatient":
                number_outpatient,
            "number_emergency":
                number_emergency,
            "number_inpatient":
                number_inpatient,
            "number_diagnoses":
                number_diagnoses,
            "diag_1": diag_1,
            "diag_2": diag_2,
            "diag_3": diag_3,
            "change": change,
            "diabetesMed": diabetesMed
        }

        patient.update(
            medications
        )

        try:

            prediction = predict_readmission(
                patient
            )

            explanation = explain_patient(
                patient
            )

            st.divider()

            # ------------------------------------------------
            # RISK RESULT
            # ------------------------------------------------

            st.subheader("Readmission Risk")

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Risk Probability",
                    f"{prediction['risk_percentage']:.2f}%"
                )

            with col2:

                if prediction["prediction"] == 1:
                    st.error(
                        "HIGH RISK"
                    )
                else:
                    st.success(
                        "LOW RISK"
                    )

            with col3:

                st.metric(
                    "Decision Threshold",
                    "51%"
                )

            st.progress(
                min(
                    prediction["risk_probability"],
                    1.0
                )
            )

            # ------------------------------------------------
            # EXPLANATION
            # ------------------------------------------------

            st.subheader(
                "🔎 Why did the model make this prediction?"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    "### Factors Increasing Risk"
                )

                risk_df = (
                    explanation[
                        "risk_factors"
                    ][
                        [
                            "feature",
                            "shap_value"
                        ]
                    ]
                    .head(8)
                    .copy()
                )

                risk_df["feature"] = (
                    risk_df["feature"]
                    .apply(
                        pretty_feature_name
                    )
                )

                risk_df.columns = [
                    "Factor",
                    "SHAP Impact"
                ]

                st.dataframe(
                    risk_df,
                    hide_index=True,
                    use_container_width=True
                )

            with col2:

                st.markdown(
                    "### Factors Decreasing Risk"
                )

                protective_df = (
                    explanation[
                        "protective_factors"
                    ][
                        [
                            "feature",
                            "shap_value"
                        ]
                    ]
                    .head(8)
                    .copy()
                )

                protective_df["feature"] = (
                    protective_df["feature"]
                    .apply(
                        pretty_feature_name
                    )
                )

                protective_df.columns = [
                    "Factor",
                    "SHAP Impact"
                ]

                st.dataframe(
                    protective_df,
                    hide_index=True,
                    use_container_width=True
                )

        except Exception as e:

            st.error(
                "Prediction failed."
            )

            st.exception(e)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📊 Model Performance":

    st.header("Model Performance")

    st.write(
        "Final performance on the untouched test set."
    )

    metrics = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC",
            "PR-AUC"
        ],
        "Score": [
            0.7158,
            0.1932,
            0.4708,
            0.2740,
            0.6676,
            0.2255
        ]
    })

    st.dataframe(
        metrics,
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    st.subheader("Confusion Matrix")

    cm_path = os.path.join(
        BASE_DIR,
        "results",
        "final_test_confusion_matrix.csv"
    )

    if os.path.exists(cm_path):

        cm = pd.read_csv(
            cm_path,
            index_col=0
        )

        st.dataframe(
            cm,
            use_container_width=True
        )

    else:

        st.info(
            "Confusion matrix artifact not found."
        )

    st.divider()

    st.subheader(
        "Global SHAP Feature Importance"
    )

    shap_path = os.path.join(
        BASE_DIR,
        "results",
        "shap_global_feature_importance.csv"
    )

    if os.path.exists(shap_path):

        shap_df = pd.read_csv(
            shap_path
        ).head(20)

        shap_df["feature"] = (
            shap_df["feature"]
            .apply(
                pretty_feature_name
            )
        )

        st.bar_chart(
            shap_df.set_index(
                "feature"
            )["mean_abs_shap"]
        )

    else:

        st.warning(
            "SHAP global importance file not found."
        )


# ============================================================
# EXPLAINABILITY
# ============================================================

elif page == "🔎 Explainability":

    st.header("Model Explainability")

    st.write(
        """
        CarePredict uses SHAP (SHapley Additive exPlanations)
        to identify the factors contributing to individual
        predictions and overall model behavior.
        """
    )

    shap_path = os.path.join(
        BASE_DIR,
        "results",
        "shap_global_feature_importance.csv"
    )

    if os.path.exists(shap_path):

        shap_df = pd.read_csv(
            shap_path
        ).head(20)

        shap_df["feature"] = (
            shap_df["feature"]
            .apply(
                pretty_feature_name
            )
        )

        st.subheader(
            "Top Global Risk Drivers"
        )

        st.dataframe(
            shap_df[
                [
                    "rank",
                    "feature",
                    "mean_abs_shap"
                ]
            ],
            hide_index=True,
            use_container_width=True
        )

    else:

        st.warning(
            "SHAP results are unavailable."
        )

    st.divider()

    st.subheader(
        "Interpretation"
    )

    st.write(
        """
        A positive SHAP contribution indicates that a feature
        pushes the prediction toward higher readmission risk,
        while a negative contribution pushes it toward lower
        predicted risk.

        SHAP values describe model behavior; they should not
        be interpreted as causal effects.
        """
    )
