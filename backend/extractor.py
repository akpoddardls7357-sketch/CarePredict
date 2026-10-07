from pathlib import Path
import io
import re

import fitz
from PIL import Image
import pytesseract


ALLOWED_TYPES = {
    "application/pdf",
    "image/jpeg",
    "image/png",
    "image/webp",
}


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_from_pdf(content: bytes) -> str:
    text_parts = []

    document = fitz.open(
        stream=content,
        filetype="pdf"
    )

    for page in document:
        text_parts.append(page.get_text())

    document.close()

    return "\n".join(text_parts).strip()


# ============================================================
# IMAGE / OCR EXTRACTION
# ============================================================

def extract_from_image(content: bytes) -> str:
    image = Image.open(
        io.BytesIO(content)
    )

    text = pytesseract.image_to_string(
        image
    )

    return text.strip()


# ============================================================
# FLEXIBLE FIELD PARSER
# ============================================================

def _extract_field(
    text: str,
    labels: list[str]
):
    """
    Extract a value from OCR text.

    Supports formats such as:

        Race: AfricanAmerican
        Race = AfricanAmerican
        Race - AfricanAmerican
        Race    AfricanAmerican

    The parser is intentionally tolerant of OCR formatting.
    """

    for label in labels:

        escaped_label = re.escape(label)

        pattern = rf"""
            (?im)
            ^\s*
            {escaped_label}
            \s*
            (?::|=|-)?\s*
            (.+?)
            \s*$
        """

        match = re.search(
            pattern,
            text,
            re.VERBOSE
        )

        if match:

            value = match.group(1).strip()

            if value:
                return value

    return None


def _extract_numeric_field(
    text: str,
    labels: list[str]
):
    """
    Extract an integer-valued field.
    """

    value = _extract_field(
        text,
        labels
    )

    if value is None:
        return None

    match = re.search(
        r"-?\d+",
        value
    )

    if match:
        return int(
            match.group()
        )

    return None


def parse_patient_fields(text: str) -> dict:
    """
    Convert extracted report/OCR text into the patient
    dictionary expected by the CarePredict prediction engine.

    This parser supports multiple label variations so that
    PDF and OCR-generated reports can use slightly different
    wording without breaking extraction.
    """

    patient_data = {

        # ----------------------------------------------------
        # DEMOGRAPHICS
        # ----------------------------------------------------

        "race": _extract_field(
            text,
            [
                "Race",
            ]
        ),

        "gender": _extract_field(
            text,
            [
                "Gender",
                "Sex",
            ]
        ),

        "age": _extract_field(
            text,
            [
                "Age",
            ]
        ),

        # ----------------------------------------------------
        # ADMISSION / DISCHARGE
        # ----------------------------------------------------

        "admission_type_id": _extract_numeric_field(
            text,
            [
                "Admission Type ID",
                "Admission Type",
            ]
        ),

        "discharge_disposition_id": _extract_numeric_field(
            text,
            [
                "Discharge Disposition ID",
                "Discharge Disposition",
            ]
        ),

        "admission_source_id": _extract_numeric_field(
            text,
            [
                "Admission Source ID",
                "Admission Source",
            ]
        ),

        # ----------------------------------------------------
        # HOSPITALIZATION
        # ----------------------------------------------------

        "time_in_hospital": _extract_numeric_field(
            text,
            [
                "Time in Hospital",
                "Hospital Stay",
                "Hospital Stay (days)",
                "Length of Stay",
            ]
        ),

        # ----------------------------------------------------
        # PROCEDURES / MEDICATIONS
        # ----------------------------------------------------

        "num_lab_procedures": _extract_numeric_field(
            text,
            [
                "Number of Lab Procedures",
                "Lab Procedures",
                "Laboratory Procedures",
                "Number of Laboratory Procedures",
            ]
        ),

        "num_procedures": _extract_numeric_field(
            text,
            [
                "Number of Procedures",
                "Procedures",
                "Procedure Count",
            ]
        ),

        "num_medications": _extract_numeric_field(
            text,
            [
                "Number of Medications",
                "Medications",
                "Medication Count",
            ]
        ),

        # ----------------------------------------------------
        # PRIOR UTILIZATION
        # ----------------------------------------------------

        "number_outpatient": _extract_numeric_field(
            text,
            [
                "Number Outpatient",
                "Number of Outpatient Visits",
                "Outpatient Visits",
                "Prior Outpatient Visits",
            ]
        ),

        "number_emergency": _extract_numeric_field(
            text,
            [
                "Number Emergency",
                "Number of Emergency Visits",
                "Emergency Visits",
                "Prior Emergency Visits",
            ]
        ),

        "number_inpatient": _extract_numeric_field(
            text,
            [
                "Number Inpatient",
                "Number of Inpatient Visits",
                "Inpatient Visits",
                "Prior Inpatient Visits",
            ]
        ),

        "number_diagnoses": _extract_numeric_field(
            text,
            [
                "Number of Diagnoses",
                "Number Diagnoses",
                "Diagnoses",
                "Diagnosis Count",
            ]
        ),

        # ----------------------------------------------------
        # MEDICATION INFORMATION
        # ----------------------------------------------------

        "insulin": _extract_field(
            text,
            [
                "Insulin",
            ]
        ),

        "change": _extract_field(
            text,
            [
                "Change",
                "Medication Change",
            ]
        ),

        "diabetesMed": _extract_field(
            text,
            [
                "Diabetes Medication",
                "Diabetes Med",
                "DiabetesMed",
            ]
        ),
    }

    return patient_data


# ============================================================
# MAIN REPORT EXTRACTION
# ============================================================

def extract_report(
    filename: str,
    content_type: str,
    content: bytes
) -> dict:

    suffix = Path(
        filename
    ).suffix.lower()

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    if (
        content_type == "application/pdf"
        or suffix == ".pdf"
    ):

        text = extract_from_pdf(
            content
        )

        extraction_method = (
            "PDF text extraction"
        )

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    elif (
        content_type.startswith("image/")
        or suffix in {
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
        }
    ):

        text = extract_from_image(
            content
        )

        extraction_method = "OCR"

    else:

        raise ValueError(
            "Unsupported file type. "
            "Please upload a PDF, JPG, PNG, or WEBP."
        )

    return {
        "filename": filename,
        "content_type": content_type,
        "extraction_method": extraction_method,
        "text": text,
        "text_length": len(text),
    }