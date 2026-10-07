import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CAREPREDICT - EXPLORATORY DATA ANALYSIS
# ============================================================

DATA_PATH = "data/processed/cleaned_data.csv"
RESULTS_DIR = "results/eda"

os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 70)
print("CAREPREDICT - EXPLORATORY DATA ANALYSIS")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

TARGET = "readmission_30d"

# ------------------------------------------------------------
# Helper function
# ------------------------------------------------------------

def save_plot(filename):
    path = os.path.join(RESULTS_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# 1. READMISSION BY AGE
# ============================================================

age_order = [
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
]

age_rate = (
    df.groupby("age", observed=False)[TARGET]
    .mean()
    .reindex(age_order)
    .mul(100)
)

plt.figure(figsize=(10, 6))

plt.bar(
    age_rate.index.astype(str),
    age_rate.values
)

plt.title("30-Day Readmission Rate by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Readmission Rate (%)")
plt.xticks(rotation=45)

save_plot("readmission_by_age.png")


# ============================================================
# 2. READMISSION BY GENDER
# ============================================================

gender_rate = (
    df.groupby("gender")[TARGET]
    .mean()
    .mul(100)
)

plt.figure(figsize=(7, 5))

plt.bar(
    gender_rate.index,
    gender_rate.values
)

plt.title("30-Day Readmission Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Readmission Rate (%)")

save_plot("readmission_by_gender.png")


# ============================================================
# 3. READMISSION BY RACE
# ============================================================

race_rate = (
    df.groupby("race")[TARGET]
    .mean()
    .sort_values(ascending=False)
    .mul(100)
)

plt.figure(figsize=(10, 6))

plt.bar(
    race_rate.index,
    race_rate.values
)

plt.title("30-Day Readmission Rate by Race")
plt.xlabel("Race")
plt.ylabel("Readmission Rate (%)")
plt.xticks(rotation=35)

save_plot("readmission_by_race.png")


# ============================================================
# 4. PRIOR INPATIENT VISITS
# ============================================================

inpatient_rate = (
    df.groupby("number_inpatient")[TARGET]
    .mean()
    .mul(100)
)

plt.figure(figsize=(10, 6))

plt.plot(
    inpatient_rate.index,
    inpatient_rate.values,
    marker="o"
)

plt.title("30-Day Readmission Rate vs Prior Inpatient Visits")
plt.xlabel("Number of Prior Inpatient Visits")
plt.ylabel("Readmission Rate (%)")
plt.grid(alpha=0.3)

save_plot("readmission_by_prior_inpatient.png")


# ============================================================
# 5. EMERGENCY VISITS
# ============================================================

emergency_rate = (
    df.groupby("number_emergency")[TARGET]
    .mean()
    .mul(100)
)

plt.figure(figsize=(10, 6))

plt.plot(
    emergency_rate.index,
    emergency_rate.values,
    marker="o"
)

plt.title("30-Day Readmission Rate vs Prior Emergency Visits")
plt.xlabel("Number of Prior Emergency Visits")
plt.ylabel("Readmission Rate (%)")
plt.grid(alpha=0.3)

save_plot("readmission_by_emergency.png")


# ============================================================
# 6. OUTPATIENT VISITS
# ============================================================

outpatient_rate = (
    df.groupby("number_outpatient")[TARGET]
    .mean()
    .mul(100)
)

plt.figure(figsize=(10, 6))

plt.plot(
    outpatient_rate.index,
    outpatient_rate.values,
    marker="o"
)

plt.title("30-Day Readmission Rate vs Prior Outpatient Visits")
plt.xlabel("Number of Prior Outpatient Visits")
plt.ylabel("Readmission Rate (%)")
plt.grid(alpha=0.3)

save_plot("readmission_by_outpatient.png")


# ============================================================
# 7. LENGTH OF STAY
# ============================================================

los_rate = (
    df.groupby("time_in_hospital")[TARGET]
    .mean()
    .mul(100)
)

plt.figure(figsize=(10, 6))

plt.plot(
    los_rate.index,
    los_rate.values,
    marker="o"
)

plt.title("30-Day Readmission Rate vs Length of Stay")
plt.xlabel("Time in Hospital (days)")
plt.ylabel("Readmission Rate (%)")
plt.grid(alpha=0.3)

save_plot("readmission_by_length_of_stay.png")


# ============================================================
# 8. NUMBER OF DIAGNOSES
# ============================================================

diagnosis_rate = (
    df.groupby("number_diagnoses")[TARGET]
    .mean()
    .mul(100)
)

plt.figure(figsize=(10, 6))

plt.plot(
    diagnosis_rate.index,
    diagnosis_rate.values,
    marker="o"
)

plt.title("30-Day Readmission Rate vs Number of Diagnoses")
plt.xlabel("Number of Diagnoses")
plt.ylabel("Readmission Rate (%)")
plt.grid(alpha=0.3)

save_plot("readmission_by_diagnoses.png")


# ============================================================
# 9. NUMBER OF MEDICATIONS
# ============================================================

medication_rate = (
    df.groupby("num_medications")[TARGET]
    .mean()
    .mul(100)
)

plt.figure(figsize=(10, 6))

plt.plot(
    medication_rate.index,
    medication_rate.values
)

plt.title("30-Day Readmission Rate vs Number of Medications")
plt.xlabel("Number of Medications")
plt.ylabel("Readmission Rate (%)")
plt.grid(alpha=0.3)

save_plot("readmission_by_medications.png")


# ============================================================
# 10. DIABETES MEDICATION STATUS
# ============================================================

diabetes_med_rate = (
    df.groupby("diabetesMed")[TARGET]
    .mean()
    .mul(100)
)

plt.figure(figsize=(7, 5))

plt.bar(
    diabetes_med_rate.index,
    diabetes_med_rate.values
)

plt.title("30-Day Readmission Rate by Diabetes Medication Status")
plt.xlabel("Diabetes Medication")
plt.ylabel("Readmission Rate (%)")

save_plot("readmission_by_diabetes_medication.png")


# ============================================================
# 11. MEDICATION CHANGE
# ============================================================

change_rate = (
    df.groupby("change")[TARGET]
    .mean()
    .mul(100)
)

plt.figure(figsize=(7, 5))

plt.bar(
    change_rate.index,
    change_rate.values
)

plt.title("30-Day Readmission Rate by Medication Change")
plt.xlabel("Medication Change")
plt.ylabel("Readmission Rate (%)")

save_plot("readmission_by_medication_change.png")


# ============================================================
# 12. ADMISSION TYPE
# ============================================================

admission_rate = (
    df.groupby("admission_type_id")[TARGET]
    .mean()
    .mul(100)
)

plt.figure(figsize=(9, 5))

plt.bar(
    admission_rate.index.astype(str),
    admission_rate.values
)

plt.title("30-Day Readmission Rate by Admission Type")
plt.xlabel("Admission Type ID")
plt.ylabel("Readmission Rate (%)")

save_plot("readmission_by_admission_type.png")


# ============================================================
# 13. DISCHARGE DISPOSITION
# ============================================================

discharge_rate = (
    df.groupby("discharge_disposition_id")[TARGET]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(12, 6))

plt.bar(
    discharge_rate.index.astype(str),
    discharge_rate.values * 100
)

plt.title("30-Day Readmission Rate by Discharge Disposition")
plt.xlabel("Discharge Disposition ID")
plt.ylabel("Readmission Rate (%)")
plt.xticks(rotation=45)

save_plot("readmission_by_discharge_disposition.png")


# ============================================================
# 14. SUMMARY TABLE
# ============================================================

summary = pd.DataFrame({
    "feature": [
        "number_inpatient",
        "number_emergency",
        "number_outpatient",
        "time_in_hospital",
        "number_diagnoses",
        "num_medications"
    ],
    "correlation_with_readmission": [
        df[
            ["number_inpatient", TARGET]
        ].corr().iloc[0, 1],

        df[
            ["number_emergency", TARGET]
        ].corr().iloc[0, 1],

        df[
            ["number_outpatient", TARGET]
        ].corr().iloc[0, 1],

        df[
            ["time_in_hospital", TARGET]
        ].corr().iloc[0, 1],

        df[
            ["number_diagnoses", TARGET]
        ].corr().iloc[0, 1],

        df[
            ["num_medications", TARGET]
        ].corr().iloc[0, 1]
    ]
})

summary = summary.sort_values(
    "correlation_with_readmission",
    ascending=False
)

summary_path = os.path.join(
    RESULTS_DIR,
    "eda_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)

print(f"\nEDA summary saved: {summary_path}")

print("\n" + "=" * 70)
print("EDA COMPLETE")
print("=" * 70)
