const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
"use client";

import { ChangeEvent, DragEvent, useRef, useState } from "react";

function getDecisionSupport(
  riskPercentage: number,
  riskLevel: string,
  riskFactors: Array<{
    feature: string;
    shap_value: number;
    impact: string;
  }>,
) {
  const topFactors = riskFactors
    .filter((factor) => factor.impact === "Increases Risk")
    .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
    .slice(0, 4);

  const factorNames = topFactors.map((factor) =>
    factor.feature
      .replace("numeric__", "")
      .replace("categorical__", "")
      .replace(/_/g, " "),
  );

  if (riskLevel === "High Risk" || riskPercentage >= 51) {
    return {
      priority: "Elevated",
      title: "Prioritize additional review",
      description:
        "The model estimates an elevated probability of 30-day readmission. The result can be used to prioritize this case for additional review and appropriate follow-up planning.",
      actions: [
        "Review discharge readiness and transition-of-care information.",
        "Review the patient's prior utilization pattern.",
        "Consider appropriate follow-up planning based on the patient's circumstances.",
      ],
      factors: factorNames,
    };
  }

  if (
    riskLevel === "Moderate Risk" ||
    (riskPercentage >= 30 && riskPercentage < 51)
  ) {
    return {
      priority: "Moderate",
      title: "Consider focused review",
      description:
        "The model estimates a moderate probability of 30-day readmission. This result can support focused review of the factors contributing to the prediction.",
      actions: [
        "Review the main factors contributing to the estimated risk.",
        "Consider whether additional follow-up planning is appropriate.",
        "Use the prediction alongside other available patient information.",
      ],
      factors: factorNames,
    };
  }

  return {
    priority: "Standard",
    title: "Routine review",
    description:
      "The model estimates a lower probability of 30-day readmission. The result can be considered alongside other patient information during routine care planning.",
    actions: [
      "Continue routine review of available patient information.",
      "Consider standard follow-up planning where appropriate.",
      "Use the prediction as supplementary analytical information.",
    ],
    factors: factorNames,
  };
}

const API_URL = "http://127.0.0.1:8000";

type PatientData = {
  race: string;
  gender: string;
  age: string;
  admission_type_id: number;
  discharge_disposition_id: number;
  admission_source_id: number;
  time_in_hospital: number;
  num_lab_procedures: number;
  num_procedures: number;
  num_medications: number;
  number_outpatient: number;
  number_emergency: number;
  number_inpatient: number;
  number_diagnoses: number;
  insulin: string;
  change: string;
  diabetesMed: string;
};

type Factor = {
  feature: string;
  value: number;
  shap_value: number;
  absolute_shap: number;
  impact: string;
  rank: number;
};

type PredictionResult = {
  risk_probability: number;
  risk_percentage: number;
  prediction: number;
  risk_level: string;
  threshold: number;
};

type ExplanationResult = {
  risk_probability: number;
  risk_percentage: number;
  risk_level: string;
  threshold: number;
  risk_factors: Factor[];
  protective_factors: Factor[];
};

type UploadResult = {
  filename: string;
  extraction_method: string;
  text: string;
  text_length: number;
  patient_data?: Partial<PatientData>;
};

const EMPTY_PATIENT: PatientData = {
  race: "Caucasian",
  gender: "Male",
  age: "[70-80]",
  admission_type_id: 1,
  discharge_disposition_id: 1,
  admission_source_id: 7,
  time_in_hospital: 4,
  num_lab_procedures: 45,
  num_procedures: 1,
  num_medications: 12,
  number_outpatient: 0,
  number_emergency: 1,
  number_inpatient: 2,
  number_diagnoses: 7,
  insulin: "Steady",
  change: "No",
  diabetesMed: "Yes",
};

const FEATURE_LABELS: Record<string, string> = {
  inpatient_diagnosis_interaction: "Inpatient Visits × Diagnoses",
  acute_visit_count: "Acute Care Visits",
  hospitalization_intensity: "Hospitalization Intensity",
  number_inpatient: "Prior Inpatient Visits",
  medication_complexity: "Medication Complexity",
  inpatient_visit_ratio: "Inpatient Visit Ratio",
  age_inpatient_interaction: "Age × Inpatient Visits",
  prior_visits_diagnosis_interaction: "Prior Visits × Diagnoses",
  num_lab_procedures: "Laboratory Procedures",
  lab_procedure_ratio: "Lab Procedure Ratio",
  discharge_disposition_id: "Discharge Disposition",
  medication_diagnosis_interaction: "Medications × Diagnoses",
  number_emergency: "Prior Emergency Visits",
  emergency_diagnosis_interaction: "Emergency Visits × Diagnoses",
  num_medications: "Number of Medications",
  medications_per_day: "Medications per Hospital Day",
  admission_source_id: "Admission Source",
};

function cleanFeatureName(feature: string) {
  const cleaned = feature
    .replace(/^numeric__/, "")
    .replace(/^categorical__/, "");

  if (FEATURE_LABELS[cleaned]) {
    return FEATURE_LABELS[cleaned];
  }

  return cleaned
    .replace(/_interaction/g, "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export default function Home() {
  const [assessmentStarted, setAssessmentStarted] = useState(false);

  const [reviewing, setReviewing] = useState(false);

  const [results, setResults] = useState(false);

  const [patient, setPatient] = useState<PatientData>(EMPTY_PATIENT);

  const [uploading, setUploading] = useState(false);

  const [analyzing, setAnalyzing] = useState(false);

  const [uploadError, setUploadError] = useState("");

  const [analysisError, setAnalysisError] = useState("");

  const [uploadResult, setUploadResult] = useState<UploadResult | null>(null);

  const [prediction, setPrediction] = useState<PredictionResult | null>(null);

  const [explanation, setExplanation] = useState<ExplanationResult | null>(
    null,
  );

  const fileInputRef = useRef<HTMLInputElement>(null);

  function scrollToSection(id: string) {
    setResults(false);
    setAssessmentStarted(false);
    setReviewing(false);

    setTimeout(() => {
      document.getElementById(id)?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }, 50);
  }

  function startAssessment() {
    setAssessmentStarted(true);
    setReviewing(false);
    setResults(false);
    setUploadError("");
    setAnalysisError("");
  }

  function updatePatient(field: keyof PatientData, value: string) {
    setPatient((current) => {
      const numericFields: (keyof PatientData)[] = [
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
      ];

      if (numericFields.includes(field)) {
        return {
          ...current,
          [field]: Number(value),
        };
      }

      return {
        ...current,
        [field]: value,
      };
    });
  }

  async function uploadReport(file: File) {
    setUploading(true);
    setUploadError("");
    setUploadResult(null);

    try {
      const formData = new FormData();

      formData.append("file", file);

      const response = await fetch(`${API_URL}/upload`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const error = await response.text();
        throw new Error(error || "Upload failed");
      }

      const data: UploadResult = await response.json();

      setUploadResult(data);

      if (data.patient_data) {
        setPatient({
          ...EMPTY_PATIENT,
          ...data.patient_data,
        });
      }

      setReviewing(true);
    } catch (error) {
      console.error(error);

      setUploadError(
        "We couldn't process this report. Please check the file and try again.",
      );
    } finally {
      setUploading(false);
    }
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];

    if (file) {
      uploadReport(file);
    }
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();

    const file = event.dataTransfer.files?.[0];

    if (file) {
      uploadReport(file);
    }
  }

  async function analyzeRisk() {
    setAnalyzing(true);
    setAnalysisError("");

    try {
      const payload = {
        patient,
      };

      const [predictionResponse, explanationResponse] = await Promise.all([
        fetch(`${API_URL}/predict`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
        }),

        fetch(`${API_URL}/explain`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
        }),
      ]);

      if (!predictionResponse.ok) {
        throw new Error("Prediction request failed");
      }

      if (!explanationResponse.ok) {
        throw new Error("Explanation request failed");
      }

      const predictionData: PredictionResult = await predictionResponse.json();

      const explanationData: ExplanationResult =
        await explanationResponse.json();

      setPrediction(predictionData);
      setExplanation(explanationData);
      setResults(true);
      setAssessmentStarted(false);
    } catch (error) {
      console.error(error);

      setAnalysisError(
        "Unable to complete the analysis. Make sure the CarePredict API is running on port 8000.",
      );
    } finally {
      setAnalyzing(false);
    }
  }

  function resetAssessment() {
    setAssessmentStarted(false);
    setReviewing(false);
    setResults(false);
    setUploadResult(null);
    setPrediction(null);
    setExplanation(null);
    setPatient(EMPTY_PATIENT);
    setUploadError("");
    setAnalysisError("");
  }

  return (
    <main className="min-h-screen bg-[#f5f8f7] text-slate-900">
      {/* ================================================= */}
      {/* NAVIGATION */}
      {/* ================================================= */}

      <nav className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-5 md:px-8">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-600 text-lg font-bold text-white">
              C
            </div>

            <div>
              <h1 className="text-xl font-bold tracking-tight">CarePredict</h1>

              <p className="text-xs text-slate-500">
                Predictive Healthcare Analytics
              </p>
            </div>
          </div>

          <div className="hidden items-center gap-8 text-sm font-medium text-slate-600 md:flex">
            <button
              type="button"
              onClick={() => scrollToSection("dashboard")}
              className="text-emerald-700 transition hover:text-emerald-800"
            >
              Dashboard
            </button>

            <button
              type="button"
              onClick={() => scrollToSection("analytics")}
              className="transition hover:text-emerald-700"
            >
              Analytics
            </button>

            <button
              type="button"
              onClick={() => scrollToSection("about")}
              className="transition hover:text-emerald-700"
            >
              About Model
            </button>
          </div>

          <button
            onClick={startAssessment}
            className="rounded-xl bg-slate-900 px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-emerald-700"
          >
            New Assessment
          </button>
        </div>
      </nav>

      {/* ================================================= */}
      {/* HERO */}
      {/* ================================================= */}

      {!results && (
        <>
          <section
            id="dashboard"
            className="mx-auto max-w-7xl px-6 pb-12 pt-16 md:px-8"
          >
            <div className="grid items-center gap-12 lg:grid-cols-[1.3fr_0.7fr]">
              <div>
                <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-emerald-200 bg-emerald-50 px-4 py-2 text-sm font-medium text-emerald-700">
                  <span className="h-2 w-2 rounded-full bg-emerald-500" />
                  XGBoost Risk Intelligence
                </div>

                <h2 className="max-w-3xl text-5xl font-bold leading-tight tracking-tight text-slate-950 md:text-6xl">
                  Predictive intelligence
                  <span className="block text-emerald-600">
                    for better patient outcomes.
                  </span>
                </h2>

                <p className="mt-6 max-w-2xl text-lg leading-8 text-slate-600">
                  Transform patient information into an interpretable 30-day
                  hospital readmission risk assessment using machine learning
                  and explainable AI.
                </p>

                <div className="mt-8 flex flex-wrap gap-4">
                  <button
                    onClick={startAssessment}
                    className="rounded-xl bg-emerald-600 px-7 py-3.5 font-semibold text-white shadow-lg shadow-emerald-600/20 transition hover:bg-emerald-700"
                  >
                    Start Assessment →
                  </button>

                  <button
                    onClick={() => scrollToSection("analytics")}
                    className="rounded-xl border border-slate-300 bg-white px-7 py-3.5 font-semibold text-slate-700 transition hover:border-emerald-400 hover:text-emerald-700"
                  >
                    View Model Analytics
                  </button>
                </div>
              </div>

              {/* Risk preview */}

              <div className="rounded-3xl border border-slate-200 bg-white p-7 shadow-xl shadow-slate-200/60">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-slate-500">
                      Example Risk Assessment
                    </p>

                    <h3 className="mt-1 text-xl font-bold">
                      30-Day Readmission
                    </h3>
                  </div>

                  <div className="rounded-full bg-red-50 px-3 py-1 text-xs font-bold text-red-600">
                    HIGH RISK
                  </div>
                </div>

                <div className="my-8 flex justify-center">
                  <div className="relative flex h-48 w-48 items-center justify-center rounded-full border-[18px] border-red-100">
                    <div className="absolute inset-0 rounded-full border-[18px] border-transparent border-t-red-500 border-r-red-500 rotate-[-35deg]" />

                    <div className="text-center">
                      <p className="text-5xl font-bold text-slate-900">59.8%</p>

                      <p className="mt-1 text-sm text-slate-500">
                        predicted probability
                      </p>
                    </div>
                  </div>
                </div>

                <div className="rounded-2xl bg-slate-50 p-4">
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-500">Decision threshold</span>

                    <span className="font-semibold">51%</span>
                  </div>

                  <div className="mt-3 h-2 overflow-hidden rounded-full bg-slate-200">
                    <div className="h-full w-[60%] rounded-full bg-red-500" />
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* ================================================= */}
          {/* MODEL SNAPSHOT */}
          {/* ================================================= */}

          <section
            id="analytics"
            className="border-y border-slate-200 bg-white"
          >
            <div className="mx-auto max-w-7xl px-6 py-12 md:px-8">
              <p className="text-sm font-semibold uppercase tracking-wider text-emerald-600">
                Model Snapshot
              </p>

              <h3 className="mt-2 text-2xl font-bold">
                Transparent performance metrics
              </h3>

              <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                <Metric title="ROC-AUC" value="66.76%" />

                <Metric title="PR-AUC" value="22.55%" />

                <Metric title="Recall" value="47.08%" />

                <Metric title="F1 Score" value="27.40%" />
              </div>
            </div>
          </section>

          {/* ================================================= */}
          {/* WORKFLOW */}
          {/* ================================================= */}

          <section id="about" className="mx-auto max-w-7xl px-6 py-16 md:px-8">
            <div className="mb-10 max-w-2xl">
              <p className="text-sm font-semibold uppercase tracking-wider text-emerald-600">
                Simple workflow
              </p>

              <h3 className="mt-2 text-3xl font-bold">
                From patient report to explainable insight.
              </h3>

              <p className="mt-3 text-slate-600">
                Upload, review, analyze, and understand the factors behind the
                model&apos;s prediction.
              </p>
            </div>

            <div className="grid gap-5 md:grid-cols-3">
              <Workflow
                number="01"
                title="Upload Report"
                description="Upload a supported PDF or image containing patient information."
              />

              <Workflow
                number="02"
                title="Review Patient Data"
                description="Review and correct extracted information before running the model."
              />

              <Workflow
                number="03"
                title="Understand Risk"
                description="View the predicted risk and the factors contributing to the model prediction."
              />
            </div>
          </section>
        </>
      )}

      {/* ================================================= */}
      {/* RESULTS DASHBOARD */}
      {/* ================================================= */}

      {results && prediction && explanation && (
        <section className="mx-auto max-w-7xl px-6 py-10 md:px-8">
          <div className="mb-8 flex flex-wrap items-center justify-between gap-4">
            <div>
              <p className="text-sm font-semibold uppercase tracking-wider text-emerald-600">
                Assessment Complete
              </p>

              <h2 className="mt-1 text-3xl font-bold">
                Readmission Risk Analysis
              </h2>

              <p className="mt-2 text-slate-500">
                Explainable prediction generated from the reviewed patient data.
              </p>
            </div>

            <button
              onClick={resetAssessment}
              className="rounded-xl border border-slate-300 bg-white px-5 py-3 font-semibold text-slate-700 hover:border-emerald-400 hover:text-emerald-700"
            >
              ← New Assessment
            </button>
          </div>

          {/* Top cards */}

          <div className="grid gap-6 lg:grid-cols-[1fr_1.4fr]">
            {/* Risk card */}

            <div className="rounded-3xl bg-slate-950 p-8 text-white shadow-xl">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-400">
                    30-Day Readmission Risk
                  </p>

                  <p className="mt-1 text-lg font-semibold">Model Prediction</p>
                </div>

                <span className="rounded-full bg-red-500/15 px-3 py-1 text-xs font-bold text-red-300">
                  {prediction.risk_level.toUpperCase()}
                </span>
              </div>

              <div className="my-10 flex justify-center">
                <div
                  className="relative flex h-64 w-64 items-center justify-center rounded-full"
                  style={{
                    background: `conic-gradient(#ef4444 ${Math.min(
                      prediction.risk_percentage,
                      100,
                    )}%, #334155 0)`,
                  }}
                >
                  <div className="flex h-52 w-52 flex-col items-center justify-center rounded-full bg-slate-950">
                    <span className="text-6xl font-bold">
                      {prediction.risk_percentage.toFixed(1)}%
                    </span>

                    <span className="mt-2 text-sm text-slate-400">
                      predicted probability
                    </span>
                  </div>
                </div>
              </div>

              <div className="rounded-2xl bg-white/5 p-5">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-slate-400">
                    Decision threshold
                  </span>

                  <span className="font-bold">
                    {(prediction.threshold * 100).toFixed(0)}%
                  </span>
                </div>

                <div className="relative mt-4 h-3 rounded-full bg-slate-700">
                  <div
                    className="h-3 rounded-full bg-red-500"
                    style={{
                      width: `${Math.min(prediction.risk_percentage, 100)}%`,
                    }}
                  />

                  <div
                    className="absolute top-[-5px] h-5 w-0.5 bg-white"
                    style={{
                      left: `${prediction.threshold * 100}%`,
                    }}
                  />
                </div>

                <p className="mt-3 text-xs text-slate-500">
                  The probability is above the model&apos;s classification
                  threshold.
                </p>
              </div>
            </div>

            {/* Patient summary */}

            <div className="rounded-3xl border border-slate-200 bg-white p-8 shadow-sm">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-semibold text-emerald-600">
                    REVIEWED PATIENT
                  </p>

                  <h3 className="mt-1 text-2xl font-bold">Patient Summary</h3>
                </div>

                <div className="rounded-xl bg-emerald-50 px-3 py-2 text-sm font-semibold text-emerald-700">
                  Verified Input
                </div>
              </div>

              <div className="mt-7 grid gap-4 sm:grid-cols-2">
                <Summary label="Age" value={patient.age} />

                <Summary label="Gender" value={patient.gender} />

                <Summary label="Race" value={patient.race} />

                <Summary
                  label="Hospital Stay"
                  value={`${patient.time_in_hospital} days`}
                />

                <Summary
                  label="Prior Inpatient Visits"
                  value={String(patient.number_inpatient)}
                />

                <Summary
                  label="Prior Emergency Visits"
                  value={String(patient.number_emergency)}
                />

                <Summary
                  label="Diagnoses"
                  value={String(patient.number_diagnoses)}
                />

                <Summary
                  label="Medications"
                  value={String(patient.num_medications)}
                />
              </div>

              <div className="mt-6 rounded-2xl bg-amber-50 p-4 text-sm leading-6 text-amber-800">
                This result is a machine-learning risk estimate, not a diagnosis
                or a substitute for professional clinical judgment.
              </div>
            </div>
          </div>

          {/* Explanation */}

          <div className="mt-6 grid gap-6 lg:grid-cols-2">
            <FactorCard
              title="Factors Increasing Model Risk"
              subtitle="Largest positive SHAP contributions"
              factors={explanation.risk_factors}
              positive
            />

            <FactorCard
              title="Factors Decreasing Model Risk"
              subtitle="Largest negative SHAP contributions"
              factors={explanation.protective_factors}
              positive={false}
            />
          </div>

          <div className="mt-6 rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div>
                <p className="text-sm font-semibold text-emerald-600">
                  MODEL TRANSPARENCY
                </p>

                <h3 className="mt-1 text-xl font-bold">
                  How to interpret these factors
                </h3>
              </div>

              <span className="rounded-full bg-slate-100 px-4 py-2 text-xs font-semibold text-slate-600">
                SHAP Explanation
              </span>
            </div>

            <p className="mt-4 max-w-4xl text-sm leading-7 text-slate-600">
              SHAP values describe how individual model inputs contributed to
              this particular prediction relative to the model&apos;s baseline.
              They represent model behavior and should not be interpreted as
              causal medical effects.
            </p>
          </div>

          {/* ================================================= */}
          {/* DATA-DRIVEN DECISION SUPPORT */}
          {/* ================================================= */}

          {(() => {
            const decisionSupport = getDecisionSupport(
              prediction.risk_percentage,
              prediction.risk_level,
              explanation.risk_factors,
            );

            return (
              <div className="mt-6 rounded-3xl border border-slate-200 bg-white p-7 shadow-sm">
                {/* Header */}

                <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
                  <div>
                    <p className="text-sm font-semibold uppercase tracking-wider text-emerald-600">
                      DATA-DRIVEN DECISION SUPPORT
                    </p>

                    <h3 className="mt-2 text-2xl font-bold text-slate-950">
                      {decisionSupport.title}
                    </h3>

                    <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">
                      {decisionSupport.description}
                    </p>
                  </div>

                  {/* Dynamic Priority */}

                  <div className="rounded-2xl bg-slate-50 px-6 py-4 text-center">
                    <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                      Decision Priority
                    </p>

                    <p className="mt-1 text-lg font-bold text-slate-900">
                      {decisionSupport.priority}
                    </p>
                  </div>
                </div>

                {/* Main Decision Support */}

                <div className="mt-7 grid gap-6 lg:grid-cols-2">
                  {/* Contributing Factors */}

                  <div className="rounded-2xl border border-slate-200 p-6">
                    <p className="text-sm font-semibold uppercase tracking-wider text-slate-500">
                      Why this result?
                    </p>

                    <h4 className="mt-2 text-lg font-bold text-slate-900">
                      Patient-specific contributing factors
                    </h4>

                    <div className="mt-5 space-y-3">
                      {decisionSupport.factors.length > 0 ? (
                        decisionSupport.factors.map((factor, index) => (
                          <div
                            key={`${factor}-${index}`}
                            className="flex items-center gap-3 rounded-xl bg-red-50 px-4 py-3"
                          >
                            <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-red-100 text-xs font-bold text-red-700">
                              {index + 1}
                            </div>

                            <p className="text-sm font-medium capitalize text-slate-700">
                              {factor}
                            </p>
                          </div>
                        ))
                      ) : (
                        <p className="text-sm text-slate-500">
                          No major risk-increasing factors were identified.
                        </p>
                      )}
                    </div>
                  </div>

                  {/* Dynamic Care Planning */}

                  <div className="rounded-2xl border border-slate-200 p-6">
                    <p className="text-sm font-semibold uppercase tracking-wider text-slate-500">
                      Suggested Review Focus
                    </p>

                    <h4 className="mt-2 text-lg font-bold text-slate-900">
                      Areas for further consideration
                    </h4>

                    <div className="mt-5 space-y-3">
                      {decisionSupport.actions.map((action, index) => (
                        <div
                          key={index}
                          className="flex gap-3 rounded-xl bg-emerald-50 px-4 py-3"
                        >
                          <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-emerald-100 text-xs font-bold text-emerald-700">
                            {index + 1}
                          </div>

                          <p className="text-sm leading-6 text-slate-600">
                            {action}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Data-driven explanation */}

                <div className="mt-6 rounded-2xl bg-slate-50 p-5">
                  <div className="flex items-start gap-3">
                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-emerald-100 text-sm font-bold text-emerald-700">
                      AI
                    </div>

                    <div>
                      <p className="font-semibold text-slate-900">
                        How this decision support is generated
                      </p>

                      <p className="mt-1 text-sm leading-6 text-slate-600">
                        The displayed priorities are generated from this
                        patient&apos;s predicted readmission probability and the
                        SHAP factors contributing to the individual prediction.
                        Therefore, the areas shown here change when the patient
                        data and model output change.
                      </p>
                    </div>
                  </div>
                </div>

                {/* Safety / Transparency */}

                <div className="mt-5 rounded-xl border border-amber-200 bg-amber-50 px-5 py-4">
                  <p className="text-sm font-semibold text-amber-900">
                    Analytical decision support only
                  </p>

                  <p className="mt-1 text-xs leading-5 text-amber-800">
                    These insights summarize model behavior and are intended to
                    support analytical review. They are not autonomous medical
                    or treatment recommendations.
                  </p>
                </div>
              </div>
            );
          })()}
        </section>
      )}

      {/* ================================================= */}
      {/* ASSESSMENT MODAL */}
      {/* ================================================= */}

      {assessmentStarted && (
        <div className="fixed inset-0 z-50 overflow-y-auto bg-slate-950/50 p-4 backdrop-blur-sm md:p-8">
          <div className="mx-auto my-4 w-full max-w-5xl rounded-3xl bg-white p-6 shadow-2xl md:p-8">
            {/* Header */}

            <div className="flex items-start justify-between">
              <div>
                <p className="text-sm font-semibold text-emerald-600">
                  NEW ASSESSMENT
                </p>

                <h3 className="mt-2 text-2xl font-bold">
                  {reviewing
                    ? "Review Patient Information"
                    : "Upload Patient Report"}
                </h3>

                <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-500">
                  {reviewing
                    ? "Review the extracted information and correct anything that is inaccurate before analysis."
                    : "Upload a PDF, JPG, JPEG, PNG, or WEBP patient report. Extracted information will be presented for review before analysis."}
                </p>
              </div>

              <button
                onClick={resetAssessment}
                className="text-2xl text-slate-400 hover:text-slate-700"
              >
                ×
              </button>
            </div>

            {!reviewing && (
              <>
                {/* Upload area */}

                <div
                  onDragOver={(event) => event.preventDefault()}
                  onDrop={handleDrop}
                  className="mt-8 rounded-3xl border-2 border-dashed border-slate-300 bg-slate-50 p-12 text-center transition hover:border-emerald-400 hover:bg-emerald-50/30"
                >
                  <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl bg-emerald-100 text-3xl text-emerald-700">
                    ↑
                  </div>

                  <p className="mt-5 text-lg font-bold">
                    {uploading
                      ? "Processing report..."
                      : "Drop patient report here"}
                  </p>

                  <p className="mt-2 text-sm text-slate-500">
                    PDF, JPG, JPEG, PNG or WEBP
                  </p>

                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".pdf,.jpg,.jpeg,.png,.webp"
                    onChange={handleFileChange}
                    className="hidden"
                  />

                  <button
                    type="button"
                    disabled={uploading}
                    onClick={() => fileInputRef.current?.click()}
                    className="mt-6 rounded-xl bg-slate-900 px-6 py-3 font-semibold text-white transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    {uploading ? "Processing..." : "Choose File"}
                  </button>
                </div>

                {uploadError && (
                  <div className="mt-5 rounded-2xl bg-red-50 p-4 text-sm text-red-700">
                    {uploadError}
                  </div>
                )}

                <p className="mt-5 text-center text-xs text-slate-400">
                  Patient information should be reviewed before prediction.
                </p>
              </>
            )}

            {reviewing && (
              <>
                {/* Upload success */}

                {uploadResult && (
                  <div className="mt-6 rounded-2xl bg-emerald-50 p-4">
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <div>
                        <p className="font-bold text-emerald-800">
                          Report processed successfully ✓
                        </p>

                        <p className="mt-1 text-sm text-emerald-700">
                          {uploadResult.filename}
                        </p>
                      </div>

                      <span className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-emerald-700">
                        {uploadResult.extraction_method}
                      </span>
                    </div>
                  </div>
                )}

                {/* Patient fields */}

                <div className="mt-7">
                  <div className="mb-5">
                    <h4 className="text-lg font-bold">Patient Information</h4>

                    <p className="text-sm text-slate-500">
                      Verify the values before sending them to the model.
                    </p>
                  </div>

                  <div className="grid gap-5 md:grid-cols-2">
                    <Field
                      label="Age"
                      value={patient.age}
                      onChange={(value) => updatePatient("age", value)}
                    />

                    <Field
                      label="Gender"
                      value={patient.gender}
                      onChange={(value) => updatePatient("gender", value)}
                    />

                    <Field
                      label="Race"
                      value={patient.race}
                      onChange={(value) => updatePatient("race", value)}
                    />

                    <Field
                      label="Hospital Stay (days)"
                      type="number"
                      value={patient.time_in_hospital}
                      onChange={(value) =>
                        updatePatient("time_in_hospital", value)
                      }
                    />

                    <Field
                      label="Lab Procedures"
                      type="number"
                      value={patient.num_lab_procedures}
                      onChange={(value) =>
                        updatePatient("num_lab_procedures", value)
                      }
                    />

                    <Field
                      label="Procedures"
                      type="number"
                      value={patient.num_procedures}
                      onChange={(value) =>
                        updatePatient("num_procedures", value)
                      }
                    />

                    <Field
                      label="Medications"
                      type="number"
                      value={patient.num_medications}
                      onChange={(value) =>
                        updatePatient("num_medications", value)
                      }
                    />

                    <Field
                      label="Number of Diagnoses"
                      type="number"
                      value={patient.number_diagnoses}
                      onChange={(value) =>
                        updatePatient("number_diagnoses", value)
                      }
                    />

                    <Field
                      label="Prior Outpatient Visits"
                      type="number"
                      value={patient.number_outpatient}
                      onChange={(value) =>
                        updatePatient("number_outpatient", value)
                      }
                    />

                    <Field
                      label="Prior Emergency Visits"
                      type="number"
                      value={patient.number_emergency}
                      onChange={(value) =>
                        updatePatient("number_emergency", value)
                      }
                    />

                    <Field
                      label="Prior Inpatient Visits"
                      type="number"
                      value={patient.number_inpatient}
                      onChange={(value) =>
                        updatePatient("number_inpatient", value)
                      }
                    />

                    <Field
                      label="Admission Type ID"
                      type="number"
                      value={patient.admission_type_id}
                      onChange={(value) =>
                        updatePatient("admission_type_id", value)
                      }
                    />

                    <Field
                      label="Discharge Disposition ID"
                      type="number"
                      value={patient.discharge_disposition_id}
                      onChange={(value) =>
                        updatePatient("discharge_disposition_id", value)
                      }
                    />

                    <Field
                      label="Admission Source ID"
                      type="number"
                      value={patient.admission_source_id}
                      onChange={(value) =>
                        updatePatient("admission_source_id", value)
                      }
                    />

                    <Field
                      label="Insulin"
                      value={patient.insulin}
                      onChange={(value) => updatePatient("insulin", value)}
                    />

                    <Field
                      label="Medication Change"
                      value={patient.change}
                      onChange={(value) => updatePatient("change", value)}
                    />

                    <Field
                      label="Diabetes Medication"
                      value={patient.diabetesMed}
                      onChange={(value) => updatePatient("diabetesMed", value)}
                    />
                  </div>
                </div>

                {analysisError && (
                  <div className="mt-6 rounded-2xl bg-red-50 p-4 text-sm text-red-700">
                    {analysisError}
                  </div>
                )}

                {/* Buttons */}

                <div className="mt-8 flex flex-wrap items-center justify-between gap-4 border-t border-slate-200 pt-6">
                  <button
                    onClick={() => {
                      setReviewing(false);
                      setUploadResult(null);
                    }}
                    className="rounded-xl border border-slate-300 bg-white px-5 py-3 font-semibold text-slate-700 hover:border-slate-400"
                  >
                    ← Back
                  </button>

                  <button
                    onClick={analyzeRisk}
                    disabled={analyzing}
                    className="rounded-xl bg-emerald-600 px-7 py-3.5 font-semibold text-white shadow-lg shadow-emerald-600/20 transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    {analyzing ? "Analyzing..." : "Analyze Readmission Risk →"}
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </main>
  );
}

/* ========================================================= */
/* COMPONENTS */
/* ========================================================= */

function Metric({ title, value }: { title: string; value: string }) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-slate-50 p-6">
      <p className="text-sm font-medium text-slate-500">{title}</p>

      <p className="mt-2 text-3xl font-bold text-slate-950">{value}</p>
    </div>
  );
}

function Workflow({
  number,
  title,
  description,
}: {
  number: string;
  title: string;
  description: string;
}) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
      <span className="text-sm font-bold text-emerald-600">{number}</span>

      <h4 className="mt-5 text-xl font-bold">{title}</h4>

      <p className="mt-2 leading-7 text-slate-600">{description}</p>
    </div>
  );
}

function Field({
  label,
  value,
  onChange,
  type = "text",
}: {
  label: string;
  value: string | number;
  onChange: (value: string) => void;
  type?: string;
}) {
  return (
    <label className="block">
      <span className="mb-2 block text-sm font-semibold text-slate-700">
        {label}
      </span>

      <input
        type={type}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="w-full rounded-xl border border-slate-300 bg-white px-4 py-3 text-slate-900 outline-none transition focus:border-emerald-500 focus:ring-4 focus:ring-emerald-500/10"
      />
    </label>
  );
}

function Summary({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl bg-slate-50 p-4">
      <p className="text-xs font-medium uppercase tracking-wide text-slate-400">
        {label}
      </p>

      <p className="mt-1 font-semibold text-slate-900">{value}</p>
    </div>
  );
}

function FactorCard({
  title,
  subtitle,
  factors,
  positive,
}: {
  title: string;
  subtitle: string;
  factors: Factor[];
  positive: boolean;
}) {
  const maxValue = Math.max(
    ...factors.map((factor) => Math.abs(factor.shap_value)),
    0.001,
  );

  return (
    <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h3 className="text-xl font-bold">{title}</h3>

          <p className="mt-1 text-sm text-slate-500">{subtitle}</p>
        </div>

        <div
          className={`rounded-full px-3 py-1 text-xs font-bold ${
            positive
              ? "bg-red-50 text-red-600"
              : "bg-emerald-50 text-emerald-700"
          }`}
        >
          {positive ? "Risk ↑" : "Risk ↓"}
        </div>
      </div>

      <div className="mt-6 space-y-5">
        {factors.slice(0, 6).map((factor) => {
          const width = (Math.abs(factor.shap_value) / maxValue) * 100;

          return (
            <div key={`${factor.feature}-${factor.rank}`}>
              <div className="flex items-center justify-between gap-4">
                <span className="text-sm font-semibold text-slate-700">
                  {cleanFeatureName(factor.feature)}
                </span>

                <span
                  className={`text-sm font-bold ${
                    positive ? "text-red-600" : "text-emerald-600"
                  }`}
                >
                  {factor.shap_value > 0 ? "+" : ""}
                  {factor.shap_value.toFixed(3)}
                </span>
              </div>

              <div className="mt-2 h-2 overflow-hidden rounded-full bg-slate-100">
                <div
                  className={`h-full rounded-full ${
                    positive ? "bg-red-400" : "bg-emerald-500"
                  }`}
                  style={{
                    width: `${Math.max(width, 4)}%`,
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
