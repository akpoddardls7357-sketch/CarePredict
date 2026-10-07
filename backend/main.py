from backend.extractor import (
    extract_report,
    parse_patient_fields
)
from pathlib import Path
import sys

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


from src.predict import predict_readmission
from src.explain import explain_patient
from backend.extractor import extract_report


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="CarePredict API",
    description="Predictive analytics API for 30-day hospital readmission risk.",
    version="1.0.0"
)

origins = [
    "https://care-predict-ten.vercel.app",
    "https://care-predict-dy0zt1i95-ankit-kumars-projects-d7f976e0.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://Your-FRONTEND.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class PatientRequest(BaseModel):
    patient: dict


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "CarePredict API",
        "model": "XGBoost",
        "threshold": 0.51
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
def predict(request: PatientRequest):

    result = predict_readmission(
        request.patient
    )

    return result


# ============================================================
# EXPLANATION
# ============================================================

@app.post("/explain")
def explain(request: PatientRequest):

    result = explain_patient(
        request.patient
    )

    return {
        "risk_probability": result["risk_probability"],
        "risk_percentage": result["risk_percentage"],
        "risk_level": result["risk_level"],
        "threshold": result["threshold"],
        "risk_factors": (
            result["risk_factors"]
            .to_dict(orient="records")
        ),
        "protective_factors": (
            result["protective_factors"]
            .to_dict(orient="records")
        )
    }

# ============================================================
# REPORT UPLOAD
# ============================================================

@app.post("/upload")
async def upload_report(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided."
        )

    allowed_extensions = {
        ".pdf",
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }

    suffix = Path(file.filename).suffix.lower()

    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload PDF, JPG, PNG, or WEBP."
        )

    content = await file.read()

    # Keep uploads reasonably sized for this local application.
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="File is too large. Maximum size is 10 MB."
        )

    try:
        result = extract_report(
            filename=file.filename,
            content_type=file.content_type or "",
            content=content,
        )
        
        patient_data = parse_patient_fields(
            result["text"]
        )

        result["patient_data"] = patient_data

        return result

    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not extract report content: {exc}"
        )
