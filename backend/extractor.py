from pathlib import Path
import io

import fitz
from PIL import Image
import pytesseract


ALLOWED_TYPES = {
    "application/pdf",
    "image/jpeg",
    "image/png",
    "image/webp",
}


def extract_from_pdf(content: bytes) -> str:
    text_parts = []

    document = fitz.open(stream=content, filetype="pdf")

    for page in document:
        text_parts.append(page.get_text())

    document.close()

    return "\n".join(text_parts).strip()


def extract_from_image(content: bytes) -> str:
    image = Image.open(io.BytesIO(content))
    text = pytesseract.image_to_string(image)

    return text.strip()


def extract_report(filename: str, content_type: str, content: bytes) -> dict:
    suffix = Path(filename).suffix.lower()

    if content_type == "application/pdf" or suffix == ".pdf":
        text = extract_from_pdf(content)
        extraction_method = "PDF text extraction"

    elif content_type.startswith("image/") or suffix in {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }:
        text = extract_from_image(content)
        extraction_method = "OCR"

    else:
        raise ValueError(
            "Unsupported file type. Please upload a PDF, JPG, PNG, or WEBP."
        )

    return {
        "filename": filename,
        "content_type": content_type,
        "extraction_method": extraction_method,
        "text": text,
        "text_length": len(text),
    }
    
import re


def _find_value(text: str, label: str):
    """
    Find a value appearing on the line immediately after a label.
    """
    pattern = rf"{re.escape(label)}\s*\n([^\n]+)"
    match = re.search(pattern, text, re.IGNORECASE)

    if match:
        return match.group(1).strip()

    return None


def parse_patient_fields(text: str) -> dict:
    """
    Extract the fields required by the CarePredict model
    from the structured report text.

    Missing values are returned as None rather than guessed.
    """

    patient = {
        "race": _find_value(text, "Race"),
        "gender": _find_value(text, "Sex"),
        "age": None,

        "admission_type_id": None,
        "discharge_disposition_id": None,
        "admission_source_id": None,

        "time_in_hospital": None,
        "num_lab_procedures": None,
        "num_procedures": None,
        "num_medications": None,

        "number_outpatient": None,
        "number_emergency": None,
        "number_inpatient": None,
        "number_diagnoses": None,

        "insulin": None,
        "change": None,
        "diabetesMed": None,
    }

    # -------------------------
    # Age
    # -------------------------

    age_text = _find_value(text, "Age")

    if age_text:
        age_match = re.search(r"\d+", age_text)

        if age_match:
            age = int(age_match.group())

            if age < 10:
                patient["age"] = "[0-10]"
            elif age < 20:
                patient["age"] = "[10-20]"
            elif age < 30:
                patient["age"] = "[20-30]"
            elif age < 40:
                patient["age"] = "[30-40]"
            elif age < 50:
                patient["age"] = "[40-50]"
            elif age < 60:
                patient["age"] = "[50-60]"
            elif age < 70:
                patient["age"] = "[60-70]"
            elif age < 80:
                patient["age"] = "[70-80]"
            elif age < 90:
                patient["age"] = "[80-90]"
            else:
                patient["age"] = "[90-100]"

    # -------------------------
    # Numeric fields
    # -------------------------

    numeric_fields = {
        "Admission Type ID": "admission_type_id",
        "Discharge Disposition ID": "discharge_disposition_id",
        "Admission Source ID": "admission_source_id",
        "Time in Hospital": "time_in_hospital",
        "Lab Procedures": "num_lab_procedures",
        "Procedures": "num_procedures",
        "Medications": "num_medications",
        "Prior Outpatient Visits": "number_outpatient",
        "Prior Emergency Visits": "number_emergency",
        "Prior Inpatient Visits": "number_inpatient",
        "Number of Diagnoses": "number_diagnoses",
    }

    for label, field in numeric_fields.items():

        value = _find_value(text, label)

        if value:
            number = re.search(r"\d+", value)

            if number:
                patient[field] = int(number.group())

    # -------------------------
    # Medication information
    # -------------------------

    patient["insulin"] = _find_value(text, "Insulin")
    patient["change"] = _find_value(text, "Change in medication")
    patient["diabetesMed"] = _find_value(
        text,
        "Diabetes medication"
    )

    return patient
