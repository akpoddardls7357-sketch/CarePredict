import os
import numpy as np
import pandas as pd
import joblib

from xgboost import XGBClassifier

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from sklearn.model_selection import ParameterSampler


# ============================================================
# CAREPREDICT - XGBOOST FOCUSED HYPERPARAMETER TUNING
# ============================================================

DATA_DIR = "data/processed/preprocessed"
MODEL_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 70)
print("CAREPREDICT - XGBOOST FOCUSED HYPERPARAMETER TUNING")
print("=" * 70)


# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

X_train = np.load(
    f"{DATA_DIR}/X_train.npy"
)

y_train = np.load(
    f"{DATA_DIR}/y_train.npy"
)

X_validation = np.load(
    f"{DATA_DIR}/X_validation.npy"
)

y_validation = np.load(
    f"{DATA_DIR}/y_validation.npy"
)

print("\nData loaded:")
print(f"Training:   {X_train.shape}")
print(f"Validation: {X_validation.shape}")


# ------------------------------------------------------------
# 2. FOCUSED SEARCH SPACE
# ------------------------------------------------------------

# Previous best region:
#
# n_estimators      = 350
# max_depth         = 5
# learning_rate     = 0.02
# min_child_weight  = 10
# subsample         = 0.90
# colsample_bytree  = 0.90
# reg_alpha         = 0.30
# reg_lambda        = 8
# gamma             = 0
# scale_pos_weight  = 7.0
#
# We now search around this region instead of starting from zero.

param_grid = {

    "n_estimators": [
        300,
        350,
        400,
        450,
        500,
        550
    ],

    "max_depth": [
        4,
        5,
        6
    ],

    "learning_rate": [
        0.01,
        0.015,
        0.02,
        0.025,
        0.03
    ],

    "min_child_weight": [
        5,
        7,
        8,
        10,
        12,
        15
    ],

    "subsample": [
        0.80,
        0.85,
        0.90,
        0.95,
        1.00
    ],

    "colsample_bytree": [
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
        1.00
    ],

    "gamma": [
        0,
        0.01,
        0.03,
        0.05,
        0.10
    ],

    "reg_alpha": [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50
    ],

    "reg_lambda": [
        5,
        6,
        7,
        8,
        9,
        10,
        12
    ],

    "scale_pos_weight": [
        6.0,
        6.5,
        7.0,
        7.5,
        7.8,
        8.0,
        8.5
    ]
}


# ------------------------------------------------------------
# 3. GENERATE FOCUSED RANDOM CONFIGURATIONS
# ------------------------------------------------------------

N_TRIALS = 40

parameter_samples = list(
    ParameterSampler(
        param_grid,
        n_iter=N_TRIALS,
        random_state=123
    )
)

print(
    f"\nNumber of focused tuning trials: "
    f"{len(parameter_samples)}"
)


# ------------------------------------------------------------
# 4. BASELINE
# ------------------------------------------------------------

baseline_params = {
    "n_estimators": 350,
    "max_depth": 5,
    "learning_rate": 0.02,
    "min_child_weight": 10,
    "subsample": 0.90,
    "colsample_bytree": 0.90,
    "reg_alpha": 0.30,
    "reg_lambda": 8,
    "gamma": 0,
    "scale_pos_weight": 7.0
}

print("\nEvaluating previous best configuration...")

baseline_model = XGBClassifier(
    **baseline_params,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=-1,
    random_state=42
)

baseline_model.fit(
    X_train,
    y_train
)

baseline_probability = (
    baseline_model.predict_proba(X_validation)[:, 1]
)

baseline_prediction = (
    baseline_probability >= 0.50
).astype(int)

baseline_pr_auc = average_precision_score(
    y_validation,
    baseline_probability
)

baseline_roc_auc = roc_auc_score(
    y_validation,
    baseline_probability
)

baseline_accuracy = accuracy_score(
    y_validation,
    baseline_prediction
)

baseline_precision = precision_score(
    y_validation,
    baseline_prediction,
    zero_division=0
)

baseline_recall = recall_score(
    y_validation,
    baseline_prediction,
    zero_division=0
)

baseline_f1 = f1_score(
    y_validation,
    baseline_prediction,
    zero_division=0
)

print("\nPrevious best baseline:")
print(f"Accuracy:  {baseline_accuracy:.4f}")
print(f"Precision: {baseline_precision:.4f}")
print(f"Recall:    {baseline_recall:.4f}")
print(f"F1:        {baseline_f1:.4f}")
print(f"ROC-AUC:   {baseline_roc_auc:.4f}")
print(f"PR-AUC:    {baseline_pr_auc:.4f}")


# ------------------------------------------------------------
# 5. FOCUSED TUNING
# ------------------------------------------------------------

results = []

best_score = baseline_pr_auc
best_model = baseline_model
best_params = baseline_params.copy()

print("\nStarting focused tuning...\n")


for trial, params in enumerate(
    parameter_samples,
    start=1
):

    print(
        f"[Trial {trial:02d}/{N_TRIALS}] Training..."
    )

    model = XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    probability = (
        model.predict_proba(X_validation)[:, 1]
    )

    prediction = (
        probability >= 0.50
    ).astype(int)

    pr_auc = average_precision_score(
        y_validation,
        probability
    )

    roc_auc = roc_auc_score(
        y_validation,
        probability
    )

    accuracy = accuracy_score(
        y_validation,
        prediction
    )

    precision = precision_score(
        y_validation,
        prediction,
        zero_division=0
    )

    recall = recall_score(
        y_validation,
        prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_validation,
        prediction,
        zero_division=0
    )

    results.append({
        "trial": trial,
        "PR-AUC": pr_auc,
        "ROC-AUC": roc_auc,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        **params
    })

    print(
        f"    PR-AUC: {pr_auc:.4f} | "
        f"ROC-AUC: {roc_auc:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1: {f1:.4f}"
    )

    if pr_auc > best_score:

        best_score = pr_auc
        best_model = model
        best_params = params.copy()

        print(
            "    >>> NEW BEST MODEL <<<"
        )


# ------------------------------------------------------------
# 6. SAVE SEARCH RESULTS
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "PR-AUC",
    ascending=False
)

results_df.to_csv(
    f"{RESULTS_DIR}/xgboost_focused_tuning_results.csv",
    index=False
)


# ------------------------------------------------------------
# 7. BEST MODEL PERFORMANCE
# ------------------------------------------------------------

best_probability = (
    best_model.predict_proba(X_validation)[:, 1]
)

best_prediction = (
    best_probability >= 0.50
).astype(int)

best_accuracy = accuracy_score(
    y_validation,
    best_prediction
)

best_precision = precision_score(
    y_validation,
    best_prediction,
    zero_division=0
)

best_recall = recall_score(
    y_validation,
    best_prediction,
    zero_division=0
)

best_f1 = f1_score(
    y_validation,
    best_prediction,
    zero_division=0
)

best_roc_auc = roc_auc_score(
    y_validation,
    best_probability
)

best_pr_auc = average_precision_score(
    y_validation,
    best_probability
)

cm = confusion_matrix(
    y_validation,
    best_prediction
)


# ------------------------------------------------------------
# 8. PRINT RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FOCUSED TUNING COMPLETE")
print("=" * 70)

print("\nPrevious Best Baseline:")
print(f"PR-AUC:   {baseline_pr_auc:.4f}")
print(f"ROC-AUC:  {baseline_roc_auc:.4f}")

print("\nBest Focused Model:")
print(f"Accuracy:  {best_accuracy:.4f}")
print(f"Precision: {best_precision:.4f}")
print(f"Recall:    {best_recall:.4f}")
print(f"F1:        {best_f1:.4f}")
print(f"ROC-AUC:   {best_roc_auc:.4f}")
print(f"PR-AUC:    {best_pr_auc:.4f}")

print("\nImprovement:")
print(
    f"PR-AUC improvement: "
    f"{best_pr_auc - baseline_pr_auc:+.4f}"
)

print(
    f"ROC-AUC improvement: "
    f"{best_roc_auc - baseline_roc_auc:+.4f}"
)

print("\nBest Parameters:")

for key, value in best_params.items():
    print(f"  {key}: {value}")

print("\nConfusion Matrix:")
print(cm)


# ------------------------------------------------------------
# 9. SAVE BEST MODEL
# ------------------------------------------------------------

tuned_model_path = (
    f"{MODEL_DIR}/xgboost_focused_tuned.pkl"
)

joblib.dump(
    best_model,
    tuned_model_path
)


# ------------------------------------------------------------
# 10. SAVE METRICS
# ------------------------------------------------------------

metrics_path = (
    f"{RESULTS_DIR}/xgboost_focused_tuned_metrics.txt"
)

with open(
    metrics_path,
    "w"
) as f:

    f.write(
        "CAREPREDICT - FOCUSED TUNED XGBOOST\n"
    )

    f.write("=" * 60 + "\n\n")

    f.write(
        f"Baseline PR-AUC: "
        f"{baseline_pr_auc:.6f}\n"
    )

    f.write(
        f"Focused Tuned PR-AUC: "
        f"{best_pr_auc:.6f}\n\n"
    )

    f.write(
        f"Accuracy: "
        f"{best_accuracy:.6f}\n"
    )

    f.write(
        f"Precision: "
        f"{best_precision:.6f}\n"
    )

    f.write(
        f"Recall: "
        f"{best_recall:.6f}\n"
    )

    f.write(
        f"F1: "
        f"{best_f1:.6f}\n"
    )

    f.write(
        f"ROC-AUC: "
        f"{best_roc_auc:.6f}\n"
    )

    f.write(
        f"PR-AUC: "
        f"{best_pr_auc:.6f}\n\n"
    )

    f.write(
        "Best Parameters:\n"
    )

    for key, value in best_params.items():

        f.write(
            f"{key}: {value}\n"
        )


print(
    "\nFocused tuned model saved to:"
)

print(tuned_model_path)

print(
    "\nSearch results saved to:"
)

print(
    f"{RESULTS_DIR}/xgboost_focused_tuning_results.csv"
)

print(
    "\nMetrics saved to:"
)

print(metrics_path)

print("\n" + "=" * 70)
print("FOCUSED XGBOOST TUNING COMPLETE")
print("=" * 70)
