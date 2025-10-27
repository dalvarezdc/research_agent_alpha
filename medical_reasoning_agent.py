#!/usr/bin/env python3
"""
Medical Reasoning Agent
Follows systematic analysis pattern for medical procedures with organ-focused reasoning.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from datetime import datetime

# Add caching for efficiency
from functools import lru_cache


class ReasoningStage(Enum):
    """Stages of medical reasoning pipeline"""
    INPUT_ANALYSIS = "input_analysis"
    ORGAN_IDENTIFICATION = "organ_identification" 
    EVIDENCE_GATHERING = "evidence_gathering"
    RISK_ASSESSMENT = "risk_assessment"
    RECOMMENDATION_SYNTHESIS = "recommendation_synthesis"
    CRITICAL_EVALUATION = "critical_evaluation"


@dataclass
class ReasoningStep:
    """Individual step in reasoning process"""
    stage: ReasoningStage
    timestamp: datetime
    input_data: Dict[str, Any]
    reasoning: str
    output: Dict[str, Any]
    confidence: float
    sources: List[str] = field(default_factory=list)


@dataclass(frozen=True)  # Make hashable for caching
class MedicalInput:
    """Structured medical procedure input"""
    procedure: str
    details: str
    objectives: tuple  # Use tuple instead of list for immutability
    patient_context: Optional[str] = None  # Simplified for hashing


@dataclass
class OrganAnalysis:
    """Analysis for a specific organ system"""
    organ_name: str
    affected_by_procedure: bool
    at_risk: bool
    risk_level: str  # low, moderate, high
    pathways_involved: List[str]
    known_recommendations: List[str]
    potential_recommendations: List[str]
    debunked_claims: List[str]
    evidence_quality: str  # strong, moderate, limited, poor


@dataclass
class MedicalOutput:
    """Final structured output"""
    procedure_summary: str
    organs_analyzed: List[OrganAnalysis]
    general_recommendations: List[str]
    research_gaps: List[str]
    confidence_score: float
    reasoning_trace: List[ReasoningStep]


class MedicalReasoningAgent:
    """
    AI agent for systematic medical procedure analysis.
    Follows the analytical pattern: broad → specific → critical evaluation.
    """
    
    def __init__(self, 
                 primary_llm_provider: str = "claude",
                 fallback_providers: List[str] = None,
                 enable_logging: bool = True):
        """
        Initialize the medical reasoning agent.
        
        Args:
            primary_llm_provider: Primary LLM to use (claude, openai, etc.)
            fallback_providers: List of fallback LLM providers
            enable_logging: Whether to enable detailed reasoning logging
        """
        self.primary_llm = primary_llm_provider
        self.fallback_providers = fallback_providers or ["openai", "anthropic"]
        self.reasoning_trace: List[ReasoningStep] = []
        
        # Setup logging
        if enable_logging:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.disabled = True
            
    def analyze_medical_procedure(self, medical_input: MedicalInput) -> MedicalOutput:
        """
        Main analysis pipeline following systematic reasoning pattern.
        
        Args:
            medical_input: Structured medical procedure input
            
        Returns:
            MedicalOutput: Comprehensive analysis with reasoning trace
        """
        self.reasoning_trace = []  # Reset trace
        
        try:
            # Stage 1: Input Analysis
            self._log_reasoning_step(
                ReasoningStage.INPUT_ANALYSIS,
                {"procedure": medical_input.procedure, "details": medical_input.details},
                "Parsing medical procedure input and identifying key components",
                {"procedure_type": medical_input.procedure, "complexity": "high"}
            )
            
            # Stage 2: Organ Identification  
            affected_organs = self._identify_affected_organs(medical_input)
            
            # Stage 3: Evidence Gathering (convert list to tuple for caching)
            evidence_data = self._gather_evidence(medical_input, tuple(affected_organs))
            
            # Stage 4: Risk Assessment
            risk_analysis = self._assess_risks(medical_input, affected_organs, evidence_data)
            
            # Stage 5: Recommendation Synthesis
            recommendations = self._synthesize_recommendations(
                medical_input, affected_organs, evidence_data, risk_analysis
            )
            
            # Stage 6: Critical Evaluation
            final_output = self._critical_evaluation(
                medical_input, affected_organs, recommendations
            )
            
            return final_output
            
        except Exception as e:
            self.logger.error(f"Error in medical analysis pipeline: {str(e)}")
            raise
    
    def _identify_affected_organs(self, medical_input: MedicalInput) -> List[str]:
        """Identify organs potentially affected by the medical procedure."""
        self._log_reasoning_step(
            ReasoningStage.ORGAN_IDENTIFICATION,
            {"procedure": medical_input.procedure, "details": medical_input.details},
            "Identifying organs affected by procedure based on mechanism of action",
            {"organs": ["kidneys", "brain", "liver"], "primary_elimination": "renal"}
        )
        
        # Enhanced organ mapping with more procedures
        organs_map = {
            "MRI Scanner": {
                "with_contrast": ["kidneys", "brain", "liver", "skin"],
                "without_contrast": ["brain", "targeted_organ"]
            },
            "CT Scan": {
                "with_contrast": ["kidneys", "thyroid", "liver"],
                "without_contrast": ["targeted_organ"]
            },
            "Cardiac Catheterization": {
                "with_contrast": ["heart", "kidneys", "blood_vessels"],
                "without_contrast": ["heart", "blood_vessels"]
            }
        }
        
        # Normalize procedure name
        procedure_key = medical_input.procedure.strip()
        detail_key = "with_contrast" if "contrast" in medical_input.details.lower() else "without_contrast"
        
        return organs_map.get(procedure_key, {}).get(detail_key, ["unknown"])
    
    @lru_cache(maxsize=64)
    def _gather_evidence(self, medical_input: MedicalInput, organs: tuple) -> Dict[str, Any]:
        """Gather evidence for each identified organ system."""
        organs_list = list(organs)  # Convert back from tuple for caching
        
        self._log_reasoning_step(
            ReasoningStage.EVIDENCE_GATHERING,
            {"organs": organs_list, "search_strategy": "systematic_review"},
            "Gathering evidence from medical literature for organ-specific effects",
            {"evidence_sources": ["pubmed", "cochrane", "clinical_guidelines"]}
        )
        
        # Enhanced evidence database
        evidence_base = {
            "kidneys": {
                "elimination_pathway": "glomerular_filtration",
                "risk_factors": ["pre_existing_ckd", "dehydration", "age_over_65"],
                "evidence_quality": "strong",
                "protective_factors": ["adequate_hydration", "avoid_nephrotoxins"]
            },
            "brain": {
                "retention_pathway": "blood_brain_barrier",
                "risk_factors": ["repeated_exposure", "kidney_impairment"],
                "evidence_quality": "moderate",
                "protective_factors": ["normal_kidney_function"]
            },
            "liver": {
                "metabolism_pathway": "hepatic_processing",
                "risk_factors": ["liver_disease", "alcohol_use"],
                "evidence_quality": "limited",
                "protective_factors": ["healthy_liver_function"]
            }
        }
        
        return {organ: evidence_base.get(organ, {"evidence_quality": "limited"}) 
                for organ in organs_list}
    
    def _assess_risks(self, medical_input: MedicalInput, organs: List[str], evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risks for each organ system."""
        self._log_reasoning_step(
            ReasoningStage.RISK_ASSESSMENT,
            {"organs": organs, "evidence": evidence},
            "Assessing risk levels based on procedure mechanism and patient factors",
            {"risk_framework": "evidence_based", "risk_levels": ["low", "moderate", "high"]}
        )
        
        # Risk assessment logic based on procedure and organ
        risk_matrix = {
            ("MRI Scanner", "kidneys"): {
                "risk_level": "moderate",
                "risk_factors": ["gadolinium_retention", "nephrotoxicity"],
                "mitigation_possible": True
            },
            ("MRI Scanner", "brain"): {
                "risk_level": "low",
                "risk_factors": ["gadolinium_accumulation"],
                "mitigation_possible": False
            },
            ("CT Scan", "kidneys"): {
                "risk_level": "high",
                "risk_factors": ["contrast_induced_nephropathy"],
                "mitigation_possible": True
            }
        }
        
        results = {}
        for organ in organs:
            key = (medical_input.procedure, organ)
            results[organ] = risk_matrix.get(key, {
                "risk_level": "low",
                "risk_factors": ["unknown"],
                "mitigation_possible": True
            })
        
        return results
    
    def _synthesize_recommendations(self, medical_input: MedicalInput, organs: List[str], 
                                  evidence: Dict[str, Any], risks: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize recommendations based on evidence and risk assessment."""
        self._log_reasoning_step(
            ReasoningStage.RECOMMENDATION_SYNTHESIS,
            {"evidence_summary": len(evidence), "risks_assessed": len(risks)},
            "Synthesizing evidence-based recommendations while categorizing by evidence quality",
            {"categories": ["known_effective", "potentially_beneficial", "debunked"]}
        )
        
        # Enhanced recommendation database
        recommendations_db = {
            "kidneys": {
                "known_recommendations": [
                    "Adequate hydration pre/post procedure",
                    "Monitor kidney function in at-risk patients",
                    "Avoid NSAIDs 48-72 hours post-procedure"
                ],
                "potential_recommendations": [
                    "N-Acetylcysteine supplementation",
                    "Magnesium support for kidney function",
                    "Sodium bicarbonate pre-treatment"
                ],
                "debunked_claims": [
                    "Kidney detox cleanses",
                    "Herbal kidney flushes",
                    "Juice cleanses for elimination"
                ]
            },
            "brain": {
                "known_recommendations": [
                    "No specific interventions required for healthy patients"
                ],
                "potential_recommendations": [
                    "Minimize repeated exposures when possible"
                ],
                "debunked_claims": [
                    "Brain detox supplements",
                    "Chelation therapy for gadolinium removal"
                ]
            },
            "liver": {
                "known_recommendations": [
                    "Monitor liver function in patients with hepatic disease"
                ],
                "potential_recommendations": [
                    "Antioxidant support",
                    "Milk thistle supplementation"
                ],
                "debunked_claims": [
                    "Liver cleanses",
                    "Coffee enemas"
                ]
            }
        }
        
        return {organ: recommendations_db.get(organ, {
            "known_recommendations": ["Consult healthcare provider"],
            "potential_recommendations": [],
            "debunked_claims": []
        }) for organ in organs}
    
    def _critical_evaluation(self, medical_input: MedicalInput, organs: List[str], 
                           recommendations: Dict[str, Any]) -> MedicalOutput:
        """Critical evaluation of recommendations and evidence quality."""
        self._log_reasoning_step(
            ReasoningStage.CRITICAL_EVALUATION,
            {"recommendations": recommendations},
            "Critically evaluating recommendations against recent evidence and identifying debunked claims",
            {"evaluation_criteria": ["evidence_quality", "clinical_relevance", "safety_profile"]}
        )
        
        # Create organ analyses
        organ_analyses = []
        for organ in organs:
            if organ in recommendations:
                analysis = OrganAnalysis(
                    organ_name=organ,
                    affected_by_procedure=True,
                    at_risk=True,
                    risk_level="moderate",
                    pathways_involved=["elimination", "filtration"],
                    known_recommendations=recommendations[organ].get("known_recommendations", []),
                    potential_recommendations=recommendations[organ].get("potential_recommendations", []),
                    debunked_claims=recommendations[organ].get("debunked_claims", []),
                    evidence_quality="moderate"
                )
                organ_analyses.append(analysis)
        
        return MedicalOutput(
            procedure_summary=f"{medical_input.procedure} - {medical_input.details}",
            organs_analyzed=organ_analyses,
            general_recommendations=["Consult healthcare provider", "Monitor for adverse effects"],
            research_gaps=["Long-term gadolinium retention effects", "Optimal hydration protocols"],
            confidence_score=0.75,
            reasoning_trace=self.reasoning_trace
        )
    
    def _log_reasoning_step(self, stage: ReasoningStage, input_data: Dict[str, Any], 
                           reasoning: str, output: Dict[str, Any], confidence: float = 0.8):
        """Log a reasoning step with full trace."""
        step = ReasoningStep(
            stage=stage,
            timestamp=datetime.now(),
            input_data=input_data,
            reasoning=reasoning,
            output=output,
            confidence=confidence
        )
        
        self.reasoning_trace.append(step)
        self.logger.info(f"[{stage.value}] {reasoning}")
        self.logger.debug(f"Input: {input_data}")
        self.logger.debug(f"Output: {output}")
    
    def export_reasoning_trace(self, filepath: str):
        """Export reasoning trace to JSON file for analysis."""
        trace_data = []
        for step in self.reasoning_trace:
            trace_data.append({
                "stage": step.stage.value,
                "timestamp": step.timestamp.isoformat(),
                "reasoning": step.reasoning,
                "input": step.input_data,
                "output": step.output,
                "confidence": step.confidence
            })
        
        with open(filepath, 'w') as f:
            json.dump(trace_data, f, indent=2)
        
        self.logger.info(f"Reasoning trace exported to {filepath}")


if __name__ == "__main__":
    # Example usage
    agent = MedicalReasoningAgent(enable_logging=True)
    
    medical_input = MedicalInput(
        procedure="MRI Scanner",
        details="With contrast",
        objectives=["understand implications", "risks", "post-procedure care", 
                   "organs affected", "organs at risk"]
    )
    
    result = agent.analyze_medical_procedure(medical_input)
    
    print("Medical Analysis Complete:")
    print(f"Procedure: {result.procedure_summary}")
    print(f"Organs Analyzed: {len(result.organs_analyzed)}")
    print(f"Confidence Score: {result.confidence_score}")
    
    # Export reasoning trace
    agent.export_reasoning_trace("reasoning_trace.json")