import numpy as np
import joblib

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

# ============================================================
# CAREPREDICT - STEP 16
# XGBoost Classifier
# ============================================================

# ------------------------------------------------------------
# 1. Load preprocessed data
# ------------------------------------------------------------

X_train = np.load(
    "../data/processed/X_train.npy"
)

X_val = np.load(
    "../data/processed/X_validation.npy"
)

y_train = np.load(
    "../data/processed/y_train.npy"
)

y_val = np.load(
    "../data/processed/y_validation.npy"
)

print("\n" + "=" * 70)
print("CAREPREDICT - XGBOOST")
print("=" * 70)

print(f"\nTraining features   : {X_train.shape}")
print(f"Validation features : {X_val.shape}")

# ------------------------------------------------------------
# 2. Calculate class imbalance
# ------------------------------------------------------------

negative_count = np.sum(y_train == 0)
positive_count = np.sum(y_train == 1)

scale_pos_weight = (
    negative_count / positive_count
)

# ------------------------------------------------------------
# 3. Model configuration
# ------------------------------------------------------------

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.03,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.2,
    reg_lambda=2.0,
    gamma=0.1,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

print("\n" + "=" * 70)
print("1. MODEL CONFIGURATION")
print("=" * 70)

print("Model             : XGBoost")
print("Estimators        : 300")
print("Max depth         : 4")
print("Learning rate     : 0.03")
print("Min child weight  : 5")
print("Subsample         : 0.80")
print("Column sampling   : 0.80")
print("Regularization    : Enabled")
print(f"scale_pos_weight  : {scale_pos_weight:.4f}")

# ------------------------------------------------------------
# 4. Train
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("2. TRAINING")
print("=" * 70)

model.fit(
    X_train,
    y_train
)

print("✓ XGBoost training complete.")

# ------------------------------------------------------------
# 5. Validation predictions
# ------------------------------------------------------------

y_pred = model.predict(
    X_val
)

y_prob = model.predict_proba(
    X_val
)[:, 1]

# ------------------------------------------------------------
# 6. Metrics
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_val,
    y_pred
)

precision = precision_score(
    y_val,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_val,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_val,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_val,
    y_prob
)

pr_auc = average_precision_score(
    y_val,
    y_prob
)

cm = confusion_matrix(
    y_val,
    y_pred
)

# ------------------------------------------------------------
# 7. Validation results
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("3. VALIDATION PERFORMANCE")
print("=" * 70)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print(f"PR-AUC    : {pr_auc:.4f}")

# ------------------------------------------------------------
# 8. Confusion matrix
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("4. CONFUSION MATRIX")
print("=" * 70)

print(
    "\n                 Predicted"
)

print(
    "              0          1"
)

print(
    f"Actual 0   {cm[0,0]:8d}   {cm[0,1]:8d}"
)

print(
    f"Actual 1   {cm[1,0]:8d}   {cm[1,1]:8d}"
)

# ------------------------------------------------------------
# 9. Feature importance
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("5. FEATURE IMPORTANCE")
print("=" * 70)

pipeline = joblib.load(
    "../models/preprocessing_pipeline.pkl"
)

feature_names = (
    pipeline.get_feature_names_out()
)

importances = model.feature_importances_

importance_pairs = sorted(
    zip(feature_names, importances),
    key=lambda x: x[1],
    reverse=True
)

for feature, importance in importance_pairs[:15]:

    print(
        f"{feature:45} "
        f"{importance:.5f}"
    )

# ------------------------------------------------------------
# 10. Save model
# ------------------------------------------------------------

model_path = (
    "../models/xgboost.pkl"
)

joblib.dump(
    model,
    model_path
)

print("\n" + "=" * 70)
print("6. MODEL SAVED")
print("=" * 70)

print(
    f"✓ {model_path}"
)

# ------------------------------------------------------------
# 11. Save predictions
# ------------------------------------------------------------

np.save(
    "../results/xgboost_val_predictions.npy",
    y_pred
)

np.save(
    "../results/xgboost_val_probabilities.npy",
    y_prob
)

print("✓ Validation predictions saved.")

# ------------------------------------------------------------
# 12. Final summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 16 COMPLETE")
print("=" * 70)

print(
    f"""
XGBoost validation results:

Accuracy  : {accuracy:.4f}
Precision : {precision:.4f}
Recall    : {recall:.4f}
F1-Score  : {f1:.4f}
ROC-AUC   : {roc_auc:.4f}
PR-AUC    : {pr_auc:.4f}
"""
)

print("=" * 70)
