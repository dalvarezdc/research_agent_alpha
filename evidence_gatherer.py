#!/usr/bin/env python3
"""
Evidence Gathering Module
Handles collection and processing of medical evidence for procedures.
"""

from typing import Dict, List, Optional, Any
import logging
from functools import lru_cache

from medical_reasoning_agent import MedicalInput


class EvidenceGatherer:
    """Simplified evidence collection focused on essential data"""
    
    def __init__(self, llm_manager=None):
        self.llm_manager = llm_manager
        self.logger = logging.getLogger(__name__)
        
        # Core evidence database - simplified
        self.evidence_db = {
            "kidneys": {
                "pathway": "glomerular_filtration",
                "risks": ["dehydration", "pre_existing_ckd", "age_over_65"],
                "protection": ["adequate_hydration", "normal_kidney_function"],
                "quality": "strong"
            },
            "brain": {
                "pathway": "blood_brain_barrier",
                "risks": ["repeated_exposure", "linear_contrast_agents"],
                "protection": ["macrocyclic_agents", "normal_kidney_function"],
                "quality": "moderate"
            },
            "heart": {
                "pathway": "coronary_circulation", 
                "risks": ["cardiac_disease", "arrhythmias"],
                "protection": ["cardiac_monitoring", "appropriate_timing"],
                "quality": "strong"
            },
            "liver": {
                "pathway": "minimal_hepatic_metabolism",
                "risks": ["severe_liver_disease"],
                "protection": ["normal_liver_function"],
                "quality": "limited"
            },
            "thyroid": {
                "pathway": "iodine_uptake",
                "risks": ["hyperthyroidism", "iodine_sensitivity"],
                "protection": ["thyroid_function_check"],
                "quality": "moderate"
            }
        }
    
    @lru_cache(maxsize=32)
    def gather_evidence(self, organ: str, procedure: str) -> Dict[str, Any]:
        """Gather evidence for organ-procedure combination"""
        self.logger.info(f"Gathering evidence for {organ} in {procedure}")
        
        # Try LLM first if available
        if self.llm_manager:
            try:
                return self._llm_gather_evidence(organ, procedure)
            except Exception as e:
                self.logger.warning(f"LLM evidence gathering failed: {e}")
        
        # Fallback to database
        return self._fallback_evidence(organ)
    
    def _llm_gather_evidence(self, organ: str, procedure: str) -> Dict[str, Any]:
        """Use LLM to gather evidence"""
        prompt = f"""
        Analyze evidence for {organ} in {procedure}:
        
        Provide:
        1. Primary pathway involved
        2. Main risk factors (3-5 key ones)
        3. Protective factors (3-5 key ones)
        4. Evidence quality (strong/moderate/limited)
        
        Be concise and evidence-based.
        """
        
        response = self.llm_manager.medical_analysis_with_fallback(
            {"organ": organ, "procedure": procedure},
            "evidence_gathering"
        )
        
        return self._parse_evidence_response(response.get("analysis", ""), organ)
    
    def _fallback_evidence(self, organ: str) -> Dict[str, Any]:
        """Get evidence from local database"""
        return self.evidence_db.get(organ, {
            "pathway": "unknown",
            "risks": ["consult_physician"],
            "protection": ["medical_supervision"],
            "quality": "limited"
        })
    
    def _parse_evidence_response(self, response: str, organ: str) -> Dict[str, Any]:
        """Parse evidence from LLM response - simplified"""
        evidence = {
            "pathway": "pathway_not_specified",
            "risks": [],
            "protection": [],
            "quality": "limited"
        }
        
        # Simple text parsing
        lines = response.lower().split('\n')
        for line in lines:
            if "pathway" in line or "mechanism" in line:
                if "kidney" in line or "renal" in line:
                    evidence["pathway"] = "renal_elimination"
                elif "liver" in line or "hepatic" in line:
                    evidence["pathway"] = "hepatic_metabolism"
            
            if "risk" in line:
                if "dehydration" in line:
                    evidence["risks"].append("dehydration")
                if "kidney" in line or "renal" in line:
                    evidence["risks"].append("kidney_impairment")
            
            if "protect" in line or "prevent" in line:
                if "hydration" in line:
                    evidence["protection"].append("adequate_hydration")
                if "monitor" in line:
                    evidence["protection"].append("monitoring")
        
        # Set defaults if empty
        if not evidence["risks"]:
            evidence["risks"] = self.evidence_db.get(organ, {}).get("risks", ["unknown"])
        if not evidence["protection"]:
            evidence["protection"] = self.evidence_db.get(organ, {}).get("protection", ["medical_supervision"])
        
        return evidence
    
    def get_evidence_summary(self, organs: List[str], procedure: str) -> Dict[str, Dict[str, Any]]:
        """Get evidence summary for multiple organs"""
        summary = {}
        for organ in organs:
            summary[organ] = self.gather_evidence(organ, procedure)
        return summary