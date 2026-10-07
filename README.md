# CarePredict

## Predictive Analytics Dashboard for Value-Based Healthcare

CarePredict is a machine-learning-based healthcare analytics project designed to estimate the risk of **30-day hospital readmission** from structured EHR-like healthcare data.

The project combines:

- Healthcare data preprocessing
- Feature engineering
- Patient-level data splitting
- Comparative machine learning
- XGBoost-based risk prediction
- SHAP-based model interpretability
- Patient-level risk assessment
- Interactive dashboard visualizations

> **Important:** CarePredict is an academic decision-support prototype. Its predictions are statistical estimates from historical data and are **not medical diagnoses, treatment recommendations, or a replacement for clinical judgment**.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Objectives](#objectives)
4. [Key Features](#key-features)
5. [Project Architecture](#project-architecture)
6. [Machine Learning Workflow](#machine-learning-workflow)
7. [Dataset](#dataset)
8. [Data Preprocessing](#data-preprocessing)
9. [Feature Engineering](#feature-engineering)
10. [Machine Learning Models](#machine-learning-models)
11. [Model Evaluation](#model-evaluation)
12. [Final Model](#final-model)
13. [Explainability with SHAP](#explainability-with-shap)
14. [Dashboard](#dashboard)
15. [Project Structure](#project-structure)
16. [Technology Stack](#technology-stack)
17. [Installation](#installation)
18. [Running the Backend](#running-the-backend)
19. [Running the Frontend](#running-the-frontend)
20. [API Endpoints](#api-endpoints)
21. [Example Prediction Workflow](#example-prediction-workflow)
22. [Model Limitations](#model-limitations)
23. [Ethical Considerations](#ethical-considerations)
24. [Future Scope](#future-scope)
25. [Expected Outcome](#expected-outcome)
26. [Contributors](#contributors)
27. [License](#license)

---

# Project Overview

Healthcare organizations generate large amounts of patient and encounter data. However, raw healthcare data by itself does not directly provide an easy way to identify patterns associated with future outcomes.

CarePredict applies machine-learning techniques to structured EHR-like data to estimate whether a patient encounter belongs to a **30-day readmission risk class**.

The system transforms healthcare data through a complete analytical pipeline:

```text
Raw Healthcare Data
        ↓
Data Cleaning & Validation
        ↓
Feature Engineering
        ↓
Patient-Level Data Split
        ↓
Preprocessing
        ↓
Model Training
        ↓
Model Comparison
        ↓
XGBoost Model
        ↓
Risk Probability
        ↓
SHAP Explainability
        ↓
Dashboard
```

The dashboard is intended to make the model output easier to understand by displaying:

- Estimated readmission probability
- Risk classification
- Important contributing factors
- Protective/decreasing factors
- Model performance metrics
- Population-level analytics

---

# Problem Statement

Hospital readmission is an important healthcare outcome because repeated hospitalization can be associated with increased healthcare utilization and complex patient-care requirements.

The problem addressed by CarePredict is:

> **How can structured healthcare data be used to build a predictive analytics solution that estimates 30-day hospital readmission risk and presents the result in an interpretable dashboard for data-driven healthcare decision support?**

The project addresses several challenges:

1. Healthcare datasets contain missing and inconsistent values.
2. Numerical and categorical variables require different preprocessing.
3. The target outcome is imbalanced.
4. Healthcare data can contain many interacting variables.
5. A prediction alone is difficult to interpret without an explanation layer.
6. Model performance must be evaluated using more than accuracy.
7. Patient-level data leakage must be avoided during model evaluation.

---

# Objectives

The major objectives of CarePredict are:

1. Identify and prepare a structured EHR-like healthcare dataset.
2. Clean and validate healthcare records.
3. Engineer meaningful utilization and clinical-complexity features.
4. Build a reproducible preprocessing pipeline.
5. Split data at the patient level to reduce information leakage.
6. Train multiple machine-learning models.
7. Compare Logistic Regression, Random Forest and XGBoost.
8. Select and tune a suitable predictive model.
9. Evaluate the model using multiple classification metrics.
10. Generate patient-level readmission risk predictions.
11. Explain predictions using SHAP.
12. Present the results through a dashboard.
13. Demonstrate how predictive analytics can support value-based healthcare workflows.

---

# Key Features

## 1. Healthcare Data Processing

The project processes structured EHR-like healthcare records containing information such as:

- Demographics
- Admission information
- Discharge information
- Hospital stay duration
- Previous healthcare utilization
- Laboratory procedures
- Procedures
- Medication information
- Diagnosis information

## 2. Feature Engineering

Raw variables are transformed into additional analytical features representing:

- Prior healthcare utilization
- Acute visits
- Hospitalization intensity
- Diagnosis density
- Medication complexity
- Clinical complexity
- Age-related interactions
- Diagnosis-related interactions

## 3. Multiple Model Comparison

Three classification approaches are evaluated:

- Logistic Regression
- Random Forest
- XGBoost

## 4. Risk Prediction

The final model generates:

- Readmission probability
- Risk percentage
- Binary prediction
- Risk level

## 5. Explainable AI

SHAP is used to identify features that contribute most strongly to model predictions.

## 6. Dashboard

The dashboard provides a user-oriented interface for:

- Patient assessment
- Prediction results
- Risk factors
- Model analytics
- Model performance

---

# Project Architecture

The high-level application architecture is:

```text
                    ┌─────────────────────┐
                    │   Healthcare Data   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Data Preprocessing  │
                    │ & Feature Engineering│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Machine Learning    │
                    │      Models         │
                    └──────────┬──────────┘
                               │
                       ┌───────┴────────┐
                       ▼                ▼
                ┌─────────────┐  ┌─────────────┐
                │ Prediction  │  │    SHAP     │
                │   Engine    │  │ Explanation │
                └──────┬──────┘  └──────┬──────┘
                       │                │
                       └───────┬────────┘
                               ▼
                    ┌─────────────────────┐
                    │ CarePredict         │
                    │ Dashboard           │
                    └─────────────────────┘
```

---

# Machine Learning Workflow

The machine-learning workflow follows these stages:

### Step 1 — Data Collection

A structured healthcare dataset containing patient encounter information is used.

### Step 2 — Data Validation

The dataset is checked for:

- Missing values
- Invalid values
- Duplicate records
- Incorrect data types
- Target distribution
- Potential leakage variables

### Step 3 — Feature Engineering

Additional features are generated from the original healthcare variables.

### Step 4 — Patient-Level Splitting

The dataset is divided into training, validation and test partitions while maintaining patient-level separation.

### Step 5 — Preprocessing

Numerical and categorical features are transformed using a reproducible preprocessing pipeline.

### Step 6 — Model Training

Multiple classification models are trained.

### Step 7 — Model Comparison

The models are compared using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion matrix

### Step 8 — Model Selection

XGBoost is selected for focused tuning based on the comparative modelling results.

### Step 9 — Explainability

SHAP is used to understand global and patient-level model behaviour.

### Step 10 — Dashboard

The analytical results are presented through the CarePredict dashboard.

---

# Dataset

The project uses a structured EHR-like healthcare dataset containing encounter-level information.

Representative feature groups include:

| Feature Group | Examples |
|---|---|
| Demographic | Race, gender, age |
| Admission | Admission type, admission source |
| Discharge | Discharge disposition |
| Hospitalization | Time in hospital |
| Utilization | Outpatient, emergency and inpatient visits |
| Procedures | Number of procedures |
| Laboratory | Number of laboratory procedures |
| Medication | Number and types of medications |
| Diagnosis | Number of diagnoses and diagnosis indicators |
| Target | 30-day readmission |

The target represents the 30-day readmission outcome used for binary classification.

> The dataset should be appropriately de-identified or publicly available for academic use.

---

# Data Preprocessing

The preprocessing pipeline is designed to provide consistent transformations between training and prediction.

## Main preprocessing operations

### 1. Identifier removal

Identifiers and fields that should not be used as predictive features are excluded.

Examples include:

```text
patient_nbr
encounter_id
```

### 2. Target separation

The target variable is separated from the feature matrix.

### 3. Missing-value handling

Missing and placeholder values are handled according to their variable type.

### 4. Numerical preprocessing

Numerical variables are transformed using appropriate imputation and scaling procedures.

### 5. Categorical preprocessing

Categorical variables are encoded using one-hot encoding.

Unknown categories are handled so that new records do not break the preprocessing pipeline.

### 6. Pipeline preservation

The preprocessing pipeline is saved so the same transformations can be applied consistently to future patient records.

---

# Feature Engineering

Feature engineering is one of the main analytical components of CarePredict.

The project creates features such as:

## Utilization Features

```text
total_prior_visits
acute_visit_count
total_prior_encounters
inpatient_visit_ratio
emergency_visit_ratio
```

These features summarize previous healthcare utilization.

## Hospitalization Features

```text
hospitalization_intensity
procedures_per_day
labs_per_day
medications_per_day
diagnosis_density
```

These features normalize healthcare activity relative to hospital stay.

## Complexity Features

```text
clinical_complexity_score
medication_complexity
multiple_diagnosis_flag
```

These features attempt to represent the overall complexity of an encounter.

## Interaction Features

Examples include:

```text
age_diagnosis_interaction
age_inpatient_interaction
inpatient_diagnosis_interaction
prior_visits_diagnosis_interaction
medication_diagnosis_interaction
emergency_diagnosis_interaction
```

Interaction features allow the model to represent relationships between different patient and encounter characteristics.

---

# Machine Learning Models

## Logistic Regression

Logistic Regression is used as a baseline classification model.

Advantages:

- Simple
- Fast
- Easy to interpret
- Useful as a baseline

Its limitation is that it may not capture complex nonlinear relationships as effectively as tree-based ensemble models.

---

## Random Forest

Random Forest is an ensemble of decision trees.

Advantages:

- Captures nonlinear relationships
- Handles feature interactions
- Robust for structured/tabular data
- Provides a useful comparison against linear models

---

## XGBoost

XGBoost is a gradient-boosted decision-tree algorithm.

It was selected as the main model because it is effective for structured/tabular datasets and can represent nonlinear feature relationships and interactions.

The project uses a focused/tuned XGBoost configuration rather than relying only on default parameters.

---

# Model Evaluation

Because the readmission class is imbalanced, several metrics are used.

| Metric | Description |
|---|---|
| Accuracy | Overall percentage of correct predictions |
| Precision | Percentage of predicted positive cases that are actually positive |
| Recall | Percentage of actual positive cases identified |
| F1-score | Harmonic mean of precision and recall |
| ROC-AUC | Overall class-ranking/discrimination performance |
| PR-AUC | Precision-recall performance, useful for imbalanced outcomes |
| Confusion Matrix | Detailed distribution of prediction errors |

Accuracy alone is not sufficient because a model can achieve reasonable accuracy while performing poorly on the minority class.

---

# Model Results

The measured model results are:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6315 | 0.1724 | 0.5881 | 0.2666 | 0.6547 | 0.2082 |
| Random Forest | 0.7179 | 0.1920 | 0.4602 | 0.2709 | 0.6575 | 0.2020 |
| XGBoost Baseline | 0.6518 | 0.1798 | 0.5775 | 0.2742 | 0.6659 | 0.2166 |
| Tuned XGBoost — Validation | 0.6923 | 0.1917 | 0.5292 | 0.2815 | 0.6715 | 0.2200 |
| Final XGBoost — Test | **0.7158** | **0.1932** | **0.4708** | **0.2740** | **0.6676** | **0.2255** |

The final test set was kept separate from model tuning.

Final test-set confusion matrix:

```text
                 Predicted
                 0       1
Actual 0       9866    3337
Actual 1        898     799
```

The results demonstrate moderate discrimination while also showing the difficulty of predicting the minority readmission class.

---

# Final Model

The focused XGBoost model uses the following configuration:

```text
n_estimators       = 350
max_depth          = 5
learning_rate      = 0.02
min_child_weight   = 10
subsample          = 0.9
colsample_bytree   = 0.9
reg_alpha          = 0.3
reg_lambda         = 8
gamma              = 0
scale_pos_weight   = 7.0
```

The model is stored as:

```text
models/xgboost_focused_tuned.pkl
```

The preprocessing pipeline is stored as:

```text
models/preprocessing_pipeline.pkl
```

A validation-based decision threshold of **0.51** was selected during the analytical workflow.

The threshold controls the conversion of predicted probability into the binary classification.

---

# Explainability with SHAP

CarePredict uses SHAP to explain model predictions.

SHAP helps answer:

> "Which features contributed most to this model's prediction?"

It is used at two levels.

## Global Explainability

Global SHAP analysis identifies features that have the greatest overall influence on model predictions.

Top features include:

1. `discharge_disposition_id`
2. `inpatient_diagnosis_interaction`
3. `acute_visit_count`
4. `hospitalization_intensity`
5. `number_inpatient`
6. `medication_complexity`
7. `age_diagnosis_interaction`
8. `inpatient_visit_ratio`
9. `age_inpatient_interaction`
10. `medication_diagnosis_interaction`

## Patient-Level Explainability

For an individual prediction, SHAP can identify:

- Factors increasing the model output
- Factors decreasing the model output
- Relative contribution of individual features

This allows the dashboard to show more than a single risk percentage.

> SHAP describes the behaviour of the trained model. A high SHAP contribution does not mean that the feature independently causes readmission.

---

# Dashboard

The dashboard is designed as the visualization and analytics layer of CarePredict.

## Main Dashboard Components

### 1. Patient Assessment

Allows patient/encounter information to be reviewed before analysis.

### 2. Risk Result

Displays:

- Risk probability
- Risk percentage
- Risk classification
- Decision threshold

### 3. Risk Factors

Displays important model factors contributing to the prediction.

### 4. Protective / Decreasing Factors

Displays factors that decrease the model's predicted output for the selected patient.

### 5. Model Analytics

Shows:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion matrix

### 6. Population Analytics

Can be used to visualize:

- Readmission distribution
- Utilization patterns
- Patient demographics
- Hospitalization patterns
- Model prediction distribution

---

# Project Structure

A representative project structure is:

```text
CarePredict/
│
├── backend/
│   └── main.py
│
├── frontend/
│   ├── app/
│   │   ├── page.tsx
│   │   └── ...
│   ├── public/
│   ├── package.json
│   └── ...
│
├── src/
│   ├── predict.py
│   ├── explain.py
│   └── ...
│
├── notebooks/
│   ├── 08_data_split.py
│   ├── 09_preprocessing.py
│   └── ...
│
├── models/
│   ├── xgboost_focused_tuned.pkl
│   ├── preprocessing_pipeline.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── xgboost_tuned.pkl
│
├── data/
│   ├── raw/
│   └── processed/
│
├── results/
│
├── requirements.txt
├── README.md
├── Dockerfile
└── .gitignore
```

Some files/directories may differ depending on the current repository version.

---

# Technology Stack

## Machine Learning

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Joblib

## Backend

- Python
- FastAPI
- Uvicorn

## Frontend

- Next.js
- React
- TypeScript

## Data Processing

- Pandas
- NumPy
- Scikit-learn preprocessing pipelines

## Visualization

- Interactive dashboard components
- Model performance charts
- Risk indicators
- SHAP-based explanation views

## Development

- Git
- GitHub
- VS Code
- Python virtual environment

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/akpoddardls7357-sketch/CarePredict.git
cd CarePredict
```

## 2. Create a Python virtual environment

Linux / WSL:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

## 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Backend

Move into the backend directory if required:

```bash
cd backend
```

Start the FastAPI application:

```bash
uvicorn main:app --reload --port 8000
```

The backend will be available locally at:

```text
http://127.0.0.1:8000
```

FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

---

# Running the Frontend

Open another terminal and move into the frontend directory:

```bash
cd frontend
```

Install JavaScript dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

The frontend will normally be available at:

```text
http://localhost:3000
```

The frontend communicates with the backend for prediction, explanation and file-processing operations.

---

# API Endpoints

The backend exposes the following main endpoints.

## Health Check

```http
GET /health
```

Used to check whether the backend is running.

---

## Prediction

```http
POST /predict
```

Receives patient information and returns the model prediction.

Example response structure:

```json
{
  "risk_probability": 0.5983,
  "risk_percentage": 59.83,
  "prediction": 1,
  "risk_level": "High Risk",
  "threshold": 0.51
}
```

---

## Explanation

```http
POST /explain
```

Returns the model prediction together with SHAP-based risk and protective factors.

---

## File Upload

```http
POST /upload
```

Used to process supported patient-report input and extract structured patient information for review.

---

# Example Prediction Workflow

The intended analytical workflow is:

```text
1. Open CarePredict
        ↓
2. Provide patient/encounter information
        ↓
3. Review extracted or entered information
        ↓
4. Submit the patient data
        ↓
5. Preprocess the input
        ↓
6. Generate model probability
        ↓
7. Apply classification threshold
        ↓
8. Generate SHAP explanation
        ↓
9. Display risk and contributing factors
```

For demonstration purposes, synthetic or appropriately de-identified data should be used.

---

# Model Limitations

CarePredict has several important limitations.

## 1. Historical Data Dependence

The model learns patterns from historical data. Changes in healthcare practices, populations or data collection can affect performance.

## 2. Class Imbalance

The positive readmission class is substantially smaller than the negative class. This makes precision-recall trade-offs important.

## 3. Generalization

Performance on one dataset does not guarantee the same performance in another hospital or population.

## 4. Missing Information

Structured EHR-like datasets may not contain all factors relevant to readmission.

## 5. No Causal Inference

The model identifies statistical patterns. It does not establish causal relationships.

## 6. Discharge-Time Information

The feature set includes discharge-related information. Therefore, the analytical framing is **post-encounter/discharge-time readmission risk assessment**, not an early-admission prediction system.

## 7. Clinical Validation

The project has not established clinical effectiveness through prospective hospital validation.

---

# Ethical Considerations

Healthcare machine learning requires careful consideration of privacy, fairness and responsible interpretation.

CarePredict should follow these principles:

- Use appropriately de-identified data.
- Do not expose personal patient identifiers.
- Do not treat predictions as medical diagnoses.
- Keep human oversight in any consequential decision.
- Evaluate model performance across relevant demographic groups.
- Clearly communicate model uncertainty and limitations.
- Avoid using model explanations as evidence of causality.
- Do not use the system to automatically determine patient treatment.

---

# Value-Based Healthcare Relevance

CarePredict aligns with the value-based healthcare theme by demonstrating how predictive analytics can support:

### Risk Stratification

Patients/encounters can be grouped according to estimated risk.

### Resource Prioritization

Higher-risk cases can be identified for additional review.

### Utilization Analysis

Historical healthcare utilization can be analyzed to identify patterns associated with readmission.

### Data-Driven Decision Support

Healthcare teams can use model output as an additional analytical input.

### Performance Analysis

Organizations can examine model performance and readmission patterns to support continuous analytical improvement.

The system is intended to **support**, not replace, professional judgement.

---

# Future Scope

Possible future improvements include:

1. External validation using independent healthcare datasets.
2. Probability calibration.
3. Fairness and subgroup analysis.
4. Additional machine-learning algorithms.
5. Longitudinal patient modelling.
6. Temporal sequence modelling.
7. Model drift monitoring.
8. Improved dashboard analytics.
9. More detailed cohort analysis.
10. Prospective evaluation in a controlled healthcare workflow.
11. Improved uncertainty estimation.
12. Investigation of different prediction horizons.
13. Integration with additional healthcare data sources where legally and ethically appropriate.

---

# Expected Outcome

The expected outcome of CarePredict is:

> **A working predictive model with an interpretable dashboard demonstrating data-driven healthcare decision support.**

The project demonstrates the complete process of converting structured healthcare data into:

```text
Healthcare Data
      ↓
Information
      ↓
Features
      ↓
Machine Learning
      ↓
Risk Prediction
      ↓
Model Explanation
      ↓
Healthcare Analytics
```

---

# Contributors

**Project:** CarePredict  
**Domain:** Big Data & Machine Learning in Healthcare

| Role | Name |
|---|---|
| Machine Learning / AI | __________________ |
| Backend | __________________ |
| Frontend / Dashboard | __________________ |
| Project Lead | __________________ |

---

# License

This project is intended primarily for **academic and educational purposes**.

Before using CarePredict with real patient data or in a real healthcare environment, appropriate legal, ethical, privacy, security and clinical validation requirements must be satisfied.

---

# Acknowledgement

This project was developed as an academic demonstration of predictive analytics, machine learning and explainable AI concepts in healthcare.

The project follows the objective of the provided academic template:

> **Predictive Analytics Dashboard for Value-Based Healthcare**

and focuses on demonstrating how structured healthcare data can be transformed into interpretable predictive insights.
