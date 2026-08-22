"""
Medical Report Categorizer Module.
Parses extracted clinical document Markdown into categorized organ system & health domain tables:
- Heart & Cardiovascular
- Liver & Hepatic
- Pancreas & Endocrine
- Nutrients & Vitamins
- Overall Health & Hematology/CBC
- Active Medications
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from llm_integrations import create_llm_manager, get_available_models

logger = logging.getLogger(__name__)


class CategorizedLabItem(BaseModel):
    marker: str = Field(..., description="Name of lab test or marker (e.g. ALT, LDL, HbA1c, Vitamin D, WBC)")
    value: str = Field(..., description="Measured result value with units (e.g. '45 U/L', '126 mg/dL')")
    reference_range: Optional[str] = Field(None, description="Normal reference range if stated (e.g. '7 - 56 U/L')")
    status: str = Field("Normal", description="Clinical status: Normal, High, Low, or Critical")
    notes: Optional[str] = Field(None, description="Short clinical assessment or organ impact note")


class CategorizedPatientReport(BaseModel):
    heart: List[CategorizedLabItem] = Field(default_factory=list, description="Cardiovascular, Lipids & Heart markers")
    liver: List[CategorizedLabItem] = Field(default_factory=list, description="Hepatic & Liver function markers")
    pancreas: List[CategorizedLabItem] = Field(default_factory=list, description="Pancreatic, Endocrine & Glucose markers")
    nutrients: List[CategorizedLabItem] = Field(default_factory=list, description="Vitamins, Minerals, Iron & Electrolytes")
    overall_health: List[CategorizedLabItem] = Field(default_factory=list, description="CBC, Hematology, Inflammatory & Kidney markers")
    medications: List[CategorizedLabItem] = Field(default_factory=list, description="Active medications, dosages & frequency")
    summary: Optional[str] = Field(None, description="Executive clinical summary of findings")


def parse_json_safely(text: str) -> Dict[str, Any]:
    """Extract JSON object from string robustly."""
    if not text:
        return {}
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        import re
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return {}


def categorize_medical_markdown(
    markdown_text: str,
    model_name: str = "grok-4.5"
) -> Dict[str, Any]:
    """Parse medical markdown text into structured organ-system categories using LLM."""
    if not markdown_text or not markdown_text.strip():
        return CategorizedPatientReport().model_dump()

    available_models = get_available_models()
    provider_name = available_models.get(model_name, model_name)
    
    try:
        llm_manager = create_llm_manager(provider_name)
    except Exception as e:
        logger.warning(f"Could not initialize provider '{provider_name}', falling back to default model: {e}")
        llm_manager = create_llm_manager("grok-4.5")

    system_prompt = """You are an expert clinical pathologist and medical intelligence extractor.
Your task is to analyze raw medical document text (labs, vitals, clinical reports) and extract all measurements into structured JSON categories.

Classify every finding into one of the following 6 categories:
1. `heart`: Blood pressure, Heart rate, Lipids (Cholesterol, LDL, HDL, Triglycerides), Troponin, Cardiac markers.
2. `liver`: ALT (SGPT), AST (SGOT), Bilirubin (Total/Direct), Alkaline Phosphatase (ALP), Albumin, GGT.
3. `pancreas`: Fasting Glucose, HbA1c, Insulin, Amylase, Lipase.
4. `nutrients`: Vitamin D, B12, Folate, Iron, Ferritin, Calcium, Magnesium, Electrolytes (Sodium, Potassium, Chloride).
5. `overall_health`: Complete Blood Count (WBC, RBC, Hemoglobin, Hematocrit, Platelets), Inflammatory (CRP, ESR), Kidney Function (Creatinine, BUN, eGFR).
6. `medications`: Active prescriptions, dosages, frequency.

Return ONLY a valid JSON object matching this structure:
{
  "heart": [{"marker": "LDL Cholesterol", "value": "130 mg/dL", "reference_range": "<100 mg/dL", "status": "High", "notes": "Borderline elevated"}],
  "liver": [{"marker": "ALT (SGPT)", "value": "45 U/L", "reference_range": "7-56 U/L", "status": "Normal", "notes": "Normal"}],
  "pancreas": [{"marker": "HbA1c", "value": "5.8%", "reference_range": "<5.7%", "status": "High", "notes": "Prediabetes range"}],
  "nutrients": [{"marker": "Vitamin D (25-OH)", "value": "18 ng/mL", "reference_range": "30-100 ng/mL", "status": "Low", "notes": "Deficiency"}],
  "overall_health": [{"marker": "WBC", "value": "6.5 x10^3/uL", "reference_range": "4.5-11.0", "status": "Normal", "notes": "Normal"}],
  "medications": [{"marker": "Metformin", "value": "500 mg", "reference_range": "Twice daily", "status": "Normal", "notes": "Oral antihyperglycemic"}],
  "summary": "Clinical summary of key findings..."
}
Do NOT include markdown wrapping or extra text outside JSON."""

    user_prompt = f"Extract and categorize medical data from this clinical document:\n\n{markdown_text[:12000]}"

    try:
        response, _ = llm_manager.generate_response(prompt=user_prompt, system_prompt=system_prompt)
        parsed_dict = parse_json_safely(response)
        validated = CategorizedPatientReport.model_validate(parsed_dict)
        return validated.model_dump()
    except Exception as e:
        logger.error(f"Error categorizing medical report with LLM: {e}")
        return CategorizedPatientReport().model_dump()


class PatientExtractedDemographics(BaseModel):
    name: Optional[str] = Field(None, description="Extracted patient name if mentioned")
    age: Optional[int] = Field(None, description="Patient age if mentioned")
    gender: Optional[str] = Field(None, description="Patient gender (Male, Female, Other) if mentioned")
    primary_condition: Optional[str] = Field(None, description="Main condition or primary clinical complaint")


class PatientClassifiedProfile(BaseModel):
    demographics: PatientExtractedDemographics = Field(default_factory=PatientExtractedDemographics)
    metadata_tags: Dict[str, str] = Field(default_factory=dict, description="Key-value attributes (e.g. Allergies, Diet, Comorbidities, Vitals)")
    categorized_data: CategorizedPatientReport = Field(default_factory=CategorizedPatientReport)
    summary: Optional[str] = Field(None, description="Concise clinical summary")


def classify_patient_description(
    description_text: str,
    model_name: str = "grok-4.5"
) -> Dict[str, Any]:
    """
    Analyze free-form patient description / clinical narrative and automatically classify:
    1. Demographics (name, age, gender, primary condition)
    2. Key-value metadata tags (Allergies, Diet, Comorbidities, Family History, Vitals)
    3. Categorized Organ System measurements (Heart, Liver, Pancreas, Nutrients, CBC/Overall, Meds)
    """
    if not description_text or not description_text.strip():
        return PatientClassifiedProfile().model_dump()

    available_models = get_available_models()
    provider_name = available_models.get(model_name, model_name)

    try:
        llm_manager = create_llm_manager(provider_name)
    except Exception as e:
        logger.warning(f"Could not initialize provider '{provider_name}', falling back to default: {e}")
        llm_manager = create_llm_manager("grok-4.5")

    system_prompt = """You are an expert clinical intake specialist and medical AI classifier.
Your task is to analyze a free-form patient description, clinical history, or medical case summary and extract structured data.

Extract and classify:
1. `demographics`:
   - `name`: Patient's name if stated (otherwise null)
   - `age`: Patient's age in years if stated (as integer, otherwise null)
   - `gender`: "Male", "Female", or "Other" if identifiable (otherwise null)
   - `primary_condition`: Primary diagnosis or chief clinical presentation (e.g. "Type 2 Diabetes Mellitus", "Essential Hypertension", "Chest Pain")

2. `metadata_tags`: A key-value dictionary of clinical tags and context:
   - e.g. "Allergies": "Penicillin"
   - e.g. "Diet": "High-carbohydrate, ultra-processed"
   - e.g. "Comorbidities": "Hypertension, Dyslipidemia"
   - e.g. "Smoking Status": "Non-smoker"
   - e.g. "Family History": "Father had T2D"

3. `categorized_data`: Group all clinical measurements, findings, and medications into the 6 organ system domains:
   - `heart`: Cardiovascular, Blood Pressure, Pulse, Lipids (LDL, HDL, Triglycerides), Cardiac markers
   - `liver`: ALT, AST, Bilirubin, ALP, Albumin, Hepatic notes
   - `pancreas`: Fasting Glucose, HbA1c, Insulin, Endocrine findings
   - `nutrients`: Vitamins, Minerals (Iron, Calcium, Potassium, Sodium, etc.)
   - `overall_health`: CBC, WBC, Platelets, Inflammatory (CRP, ESR), Renal (Creatinine, eGFR), Weight changes
   - `medications`: Active drugs, OTC supplements, dosages, frequency

Each item in categorized_data must be: {"marker": str, "value": str, "reference_range": str, "status": "Normal"|"High"|"Low"|"Critical", "notes": str}

Return ONLY valid JSON matching:
{
  "demographics": {
    "name": "Robert",
    "age": 50,
    "gender": "Male",
    "primary_condition": "Type 2 Diabetes Mellitus"
  },
  "metadata_tags": {
    "Diet": "High-carbohydrate, ultra-processed",
    "Symptoms": "Polydipsia, polyuria, nocturia, fatigue, blurred vision",
    "Weight Change": "+5 kg in 6 months",
    "Comorbidity": "Hypertension"
  },
  "categorized_data": {
    "heart": [{"marker": "Blood Pressure / History", "value": "Hypertension", "reference_range": "<120/80 mmHg", "status": "High", "notes": "Managed on Lisinopril"}],
    "liver": [],
    "pancreas": [{"marker": "Fasting Blood Glucose", "value": "165 mg/dL", "reference_range": "70-99 mg/dL", "status": "High", "notes": "Above diabetes diagnostic threshold (>=126 mg/dL)"}],
    "nutrients": [],
    "overall_health": [{"marker": "Weight Change", "value": "+5 kg", "reference_range": "Stable", "status": "High", "notes": "Over 6 months"}],
    "medications": [{"marker": "Lisinopril", "value": "10 mg daily", "reference_range": "Prescription", "status": "Normal", "notes": "Antihypertensive"}]
  },
  "summary": "50yo male with symptomatic hyperglycemia and metabolic risk factors."
}
Do NOT include markdown formatting or commentary outside JSON."""

    user_prompt = f"Analyze and classify this patient clinical description:\n\n{description_text[:12000]}"

    try:
        response, _ = llm_manager.generate_response(prompt=user_prompt, system_prompt=system_prompt)
        parsed_dict = parse_json_safely(response)
        validated = PatientClassifiedProfile.model_validate(parsed_dict)
        return validated.model_dump()
    except Exception as e:
        logger.error(f"Error classifying patient description with LLM: {e}")
        return PatientClassifiedProfile().model_dump()
