#!/usr/bin/env python3
"""
Recommendation Synthesis Module
Handles creation of medical recommendations based on evidence.
"""

from typing import Dict, List, Any
import logging

from medical_reasoning_agent import MedicalInput


class RecommendationSynthesizer:
    """Simplified recommendation synthesis"""
    
    def __init__(self, llm_manager=None):
        self.llm_manager = llm_manager
        self.logger = logging.getLogger(__name__)
        
        # Core recommendations database
        self.recommendations_db = {
            "kidneys": {
                "known": [
                    "Ensure adequate hydration before and after procedure",
                    "Monitor kidney function with creatinine levels",
                    "Avoid nephrotoxic medications 48 hours before/after"
                ],
                "potential": [
                    "Consider N-acetylcysteine for high-risk patients",
                    "Sodium bicarbonate hydration may provide additional protection"
                ],
                "debunked": [
                    "Furosemide does not prevent contrast nephropathy",
                    "Dopamine is not protective and may be harmful"
                ]
            },
            "brain": {
                "known": [
                    "Use macrocyclic gadolinium agents when possible",
                    "Avoid unnecessary repeat MRI with contrast"
                ],
                "potential": [
                    "Consider alternative imaging if multiple prior exposures",
                    "Monitor for neurological symptoms in high-exposure patients"
                ],
                "debunked": [
                    "Chelation therapy is not recommended for gadolinium retention",
                    "Linear agents are not safer than macrocyclic agents"
                ]
            },
            "heart": {
                "known": [
                    "Continuous cardiac monitoring during procedure",
                    "Have emergency cardiac medications available",
                    "Monitor for arrhythmias post-procedure"
                ],
                "potential": [
                    "Pre-procedure cardiac risk stratification",
                    "Consider prophylactic medications for high-risk patients"
                ],
                "debunked": [
                    "Routine beta-blockers are not recommended for all patients",
                    "Prophylactic antiarrhythmics are not routinely indicated"
                ]
            }
        }
    
    def synthesize_recommendations(self, organ: str, evidence: Dict[str, Any], 
                                 medical_input: MedicalInput) -> Dict[str, List[str]]:
        """Generate recommendations for an organ based on evidence"""
        self.logger.info(f"Synthesizing recommendations for {organ}")
        
        # Try LLM first
        if self.llm_manager:
            try:
                return self._llm_synthesize(organ, evidence, medical_input)
            except Exception as e:
                self.logger.warning(f"LLM synthesis failed: {e}")
        
        # Fallback to database
        return self._fallback_recommendations(organ)
    
    def _llm_synthesize(self, organ: str, evidence: Dict[str, Any], 
                       medical_input: MedicalInput) -> Dict[str, List[str]]:
        """Use LLM to synthesize recommendations"""
        prompt = f"""
        Based on this evidence for {organ} in {medical_input.procedure}:
        - Pathway: {evidence.get('pathway', 'unknown')}
        - Risks: {evidence.get('risks', [])}
        - Protection: {evidence.get('protection', [])}
        
        Provide 3 categories:
        1. KNOWN (evidence-based): 2-3 proven recommendations
        2. POTENTIAL (investigational): 1-2 promising approaches  
        3. DEBUNKED (harmful): 1-2 disproven treatments
        
        Be concise and specific.
        """
        
        response = self.llm_manager.medical_analysis_with_fallback(
            {
                "organ": organ,
                "evidence": evidence,
                "procedure": medical_input.procedure
            },
            "recommendation_synthesis"
        )
        
        return self._parse_recommendations(response.get("analysis", ""), organ)
    
    def _fallback_recommendations(self, organ: str) -> Dict[str, List[str]]:
        """Get recommendations from database"""
        return self.recommendations_db.get(organ, {
            "known": ["Consult healthcare provider before procedure"],
            "potential": ["Follow standard medical protocols"],
            "debunked": ["Avoid unproven treatments"]
        })
    
    def _parse_recommendations(self, response: str, organ: str) -> Dict[str, List[str]]:
        """Parse recommendations from LLM response"""
        recommendations = {
            "known": [],
            "potential": [],
            "debunked": []
        }
        
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect sections
            if "known" in line.lower() or "evidence" in line.lower():
                current_section = "known"
            elif "potential" in line.lower() or "investigational" in line.lower():
                current_section = "potential"
            elif "debunked" in line.lower() or "harmful" in line.lower():
                current_section = "debunked"
            elif line.startswith('-') or line.startswith('•') or line.startswith('*'):
                if current_section:
                    rec = line[1:].strip()
                    if len(rec) > 10:  # Filter out short items
                        recommendations[current_section].append(rec)
        
        # Fallback if parsing failed
        for section in recommendations:
            if not recommendations[section]:
                recommendations[section] = self.recommendations_db.get(organ, {}).get(section, [f"Standard {section} recommendations"])
        
        return recommendations
    
    def synthesize_all_recommendations(self, organs_evidence: Dict[str, Dict[str, Any]], 
                                     medical_input: MedicalInput) -> Dict[str, Dict[str, List[str]]]:
        """Generate recommendations for all organs"""
        all_recommendations = {}
        
        for organ, evidence in organs_evidence.items():
            all_recommendations[organ] = self.synthesize_recommendations(
                organ, evidence, medical_input
            )
        
        return all_recommendations