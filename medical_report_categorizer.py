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
