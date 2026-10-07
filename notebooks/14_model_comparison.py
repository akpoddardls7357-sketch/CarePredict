import os
import pandas as pd

print("=" * 70)
print("CAREPREDICT - MODEL COMPARISON")
print("=" * 70)

MODEL_FILES = {
    "Logistic Regression": "results/logistic_regression_metrics.txt",
    "Random Forest": "results/random_forest_metrics.txt",
    "XGBoost": "results/xgboost_metrics.txt"
}


def load_metrics(filepath):
    metrics = {}

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            if ":" not in line:
                continue

            key, value = line.split(":", 1)

            key = key.strip()
            value = value.strip()

            try:
                metrics[key] = float(value)
            except ValueError:
                continue

    return metrics


results = []

for model_name, filepath in MODEL_FILES.items():

    if not os.path.exists(filepath):
        print(f"WARNING: Missing file: {filepath}")
        continue

    metrics = load_metrics(filepath)

    results.append({
        "Model": model_name,
        "Accuracy": metrics.get("Accuracy", 0),
        "Precision": metrics.get("Precision", 0),
        "Recall": metrics.get("Recall", 0),
        "F1": metrics.get("F1", 0),
        "ROC-AUC": metrics.get("ROC-AUC", 0),
        "PR-AUC": metrics.get("PR-AUC", 0)
    })


comparison = pd.DataFrame(results)

print("\nValidation Model Comparison:\n")

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# Rank models by PR-AUC
ranked = comparison.sort_values(
    by="PR-AUC",
    ascending=False
).reset_index(drop=True)

print("\nModels ranked by PR-AUC:")

for i, row in ranked.iterrows():
    print(
        f"{i + 1}. {row['Model']} "
        f"(PR-AUC = {row['PR-AUC']:.4f})"
    )

# Save comparison
output_path = "results/model_comparison.csv"
comparison.to_csv(output_path, index=False)

print(f"\nComparison saved to: {output_path}")

print("\n" + "=" * 70)
print("MODEL COMPARISON COMPLETE")
print("=" * 70)
