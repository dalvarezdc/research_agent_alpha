#!/usr/bin/env python3
"""
Organ Analysis Module
Handles identification and analysis of organs affected by medical procedures.
"""

from typing import Dict, List, Optional
import logging
import json
import re
from dataclasses import dataclass

from medical_reasoning_agent import MedicalInput


@dataclass
class OrganInfo:
    """Information about an organ and its involvement in a procedure"""
    name: str
    affected: bool
    risk_level: str  # low, moderate, high
    pathways: List[str]
    mechanisms: List[str]


class OrganAnalyzer:
    """Focused class for organ identification and analysis"""
    
    def __init__(self, llm_manager=None):
        self.llm_manager = llm_manager
        self.logger = logging.getLogger(__name__)
        
        # Simplified organ mapping
        self.organ_map = {
            "MRI Scanner": {
                "with_contrast": ["kidneys", "brain"],
                "without_contrast": ["brain"]
            },
            "CT Scan": {
                "with_contrast": ["kidneys", "thyroid"],
                "without_contrast": []
            },
            "Cardiac Catheterization": {
                "with_contrast": ["heart", "kidneys", "blood_vessels"],
                "without_contrast": ["heart", "blood_vessels"]
            }
        }
    
    def identify_affected_organs(self, medical_input: MedicalInput) -> List[str]:
        """Identify organs affected by the procedure"""
        self.logger.info(f"Identifying organs for: {medical_input.procedure}")
        
        # Try LLM first
        if self.llm_manager:
            try:
                return self._llm_identify_organs(medical_input)
            except Exception as e:
                self.logger.warning(f"LLM organ identification failed: {e}")
        
        # Fallback to mapping
        return self._fallback_identify_organs(medical_input)
    
    def _llm_identify_organs(self, medical_input: MedicalInput) -> List[str]:
        """Use LLM to identify organs"""
        prompt = f"""
        Identify organs affected by: {medical_input.procedure}
        Details: {medical_input.details}
        
        Return only a JSON list: ["organ1", "organ2"]
        """
        
        response = self.llm_manager.medical_analysis_with_fallback(
            {"procedure": medical_input.procedure, "details": medical_input.details},
            "organ_identification"
        )
        
        return self._parse_organ_response(response.get("analysis", ""))
    
    def _fallback_identify_organs(self, medical_input: MedicalInput) -> List[str]:
        """Fallback organ identification using mapping"""
        procedure = medical_input.procedure.strip()
        has_contrast = "contrast" in medical_input.details.lower()
        
        detail_key = "with_contrast" if has_contrast else "without_contrast"
        organs = self.organ_map.get(procedure, {}).get(detail_key, ["kidneys"])
        
        self.logger.info(f"Fallback identified organs: {organs}")
        return organs
    
    def _parse_organ_response(self, response: str) -> List[str]:
        """Parse organ list from LLM response"""
        # Try JSON first
        json_match = re.search(r'\[(.*?)\]', response)
        if json_match:
            try:
                organs_str = '[' + json_match.group(1) + ']'
                organs = json.loads(organs_str)
                return [organ.lower().strip() for organ in organs if isinstance(organ, str)]
            except (json.JSONDecodeError, ValueError):
                pass
        
        # Fallback to text parsing
        common_organs = ["kidney", "brain", "liver", "heart", "lung", "thyroid"]
        found = []
        text_lower = response.lower()
        
        for organ in common_organs:
            if organ in text_lower:
                if organ == "kidney":
                    found.append("kidneys")
                else:
                    found.append(organ)
        
        return list(set(found)) if found else ["kidneys"]
    
    def analyze_organ_involvement(self, organ: str, medical_input: MedicalInput) -> OrganInfo:
        """Analyze how an organ is involved in the procedure"""
        # Simplified analysis - could be expanded
        risk_levels = {
            "kidneys": "moderate",
            "brain": "low", 
            "heart": "high",
            "liver": "low",
            "thyroid": "moderate"
        }
        
        pathways = {
            "kidneys": ["glomerular_filtration", "tubular_secretion"],
            "brain": ["blood_brain_barrier"],
            "heart": ["coronary_circulation"],
            "liver": ["hepatic_metabolism"],
            "thyroid": ["iodine_uptake"]
        }
        
        return OrganInfo(
            name=organ,
            affected=True,
            risk_level=risk_levels.get(organ, "low"),
            pathways=pathways.get(organ, ["unknown"]),
            mechanisms=[f"{medical_input.procedure} affects {organ}"]
        )