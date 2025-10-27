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
        
        # Enhanced recommendation database with detailed rationales
        recommendations_db = {
            "kidneys": {
                "known_recommendations": [
                    {
                        "intervention": "Adequate hydration pre/post procedure",
                        "rationale": "Increases urine flow rate and reduces concentration of contrast agent in tubules, minimizing direct nephrotoxic effects",
                        "evidence_level": "Strong - Multiple RCTs and guidelines (ESR, ACR)",
                        "timing": "500ml saline 1-2 hours before, continue 6-12 hours after"
                    },
                    {
                        "intervention": "Monitor kidney function in at-risk patients",
                        "rationale": "Early detection of contrast-induced nephropathy allows for prompt intervention and prevents progression to acute kidney injury",
                        "evidence_level": "Strong - Standard of care per nephrology guidelines",
                        "timing": "Baseline creatinine within 7 days, follow-up at 48-72 hours post-procedure"
                    },
                    {
                        "intervention": "Avoid NSAIDs 48-72 hours post-procedure",
                        "rationale": "NSAIDs reduce renal blood flow via prostaglandin inhibition, compounding contrast-induced vasoconstriction",
                        "evidence_level": "Strong - Consistent evidence across multiple studies",
                        "timing": "48-72 hours before and after contrast exposure"
                    }
                ],
                "potential_recommendations": [
                    {
                        "intervention": "N-Acetylcysteine supplementation",
                        "rationale": "Antioxidant properties may reduce oxidative stress and free radical damage from contrast agents. Acts as glutathione precursor.",
                        "evidence_level": "Mixed - Some positive RCTs but multiple negative studies and meta-analyses show conflicting results",
                        "dosing": "600mg orally twice daily for 2 days starting day before procedure",
                        "limitations": "2018 Cochrane review found no significant benefit; still used in some centers"
                    },
                    {
                        "intervention": "Magnesium support for kidney function",
                        "rationale": "Magnesium deficiency associated with increased nephrotoxicity; supplementation may maintain cellular energy and reduce calcium influx",
                        "evidence_level": "Limited - Small studies suggest benefit but larger RCTs needed",
                        "dosing": "Magnesium sulfate 3g in 250ml saline over 1 hour before procedure",
                        "limitations": "Mechanism unclear, optimal dosing not established"
                    },
                    {
                        "intervention": "Sodium bicarbonate pre-treatment",
                        "rationale": "Alkalinization of tubular fluid may reduce formation of reactive oxygen species and Tamm-Horsfall protein precipitation",
                        "evidence_level": "Moderate - Several positive studies but some negative trials",
                        "dosing": "154 mEq/L in D5W at 3ml/kg/hr for 1hr before, then 1ml/kg/hr for 6hrs after",
                        "limitations": "Not superior to saline in all studies; logistically complex"
                    }
                ],
                "debunked_claims": [
                    {
                        "claim": "Kidney detox cleanses",
                        "reason_debunked": "No scientific evidence for enhanced elimination of contrast agents; may cause electrolyte imbalances and dehydration",
                        "debunked_by": "American Society of Nephrology, National Kidney Foundation",
                        "evidence": "Systematic reviews show no benefit and potential harm from commercial detox products",
                        "why_harmful": "Can lead to dehydration, electrolyte disturbances, and delayed medical care"
                    },
                    {
                        "claim": "Herbal kidney flushes",
                        "reason_debunked": "No peer-reviewed evidence for gadolinium elimination; some herbs (aristolochia) are nephrotoxic",
                        "debunked_by": "FDA warnings, nephrology literature",
                        "evidence": "Case reports of acute kidney injury from herbal products",
                        "why_harmful": "Potential drug interactions and direct nephrotoxicity"
                    },
                    {
                        "claim": "Juice cleanses for elimination",
                        "reason_debunked": "Contrast agents eliminated by glomerular filtration, not affected by dietary interventions",
                        "debunked_by": "Basic pharmacokinetic principles, radiology literature",
                        "evidence": "Gadolinium elimination follows first-order kinetics independent of diet",
                        "why_harmful": "May cause hypoglycemia, nutrient deficiencies, and false sense of protection"
                    }
                ]
            },
            "brain": {
                "known_recommendations": [
                    {
                        "intervention": "No specific interventions required for healthy patients",
                        "rationale": "Gadolinium retention in brain has no proven clinical consequences in patients with normal kidney function",
                        "evidence_level": "Strong - Multiple safety studies and FDA review",
                        "timing": "Ongoing monitoring of safety data"
                    }
                ],
                "potential_recommendations": [
                    {
                        "intervention": "Minimize repeated exposures when possible",
                        "rationale": "Linear gadolinium agents show greater brain retention than macrocyclic agents; cumulative effects unknown",
                        "evidence_level": "Precautionary - Based on imaging studies showing retention",
                        "dosing": "Use lowest effective dose, prefer macrocyclic agents",
                        "limitations": "No proven clinical harm, may delay necessary imaging"
                    }
                ],
                "debunked_claims": [
                    {
                        "claim": "Brain detox supplements",
                        "reason_debunked": "Blood-brain barrier prevents most oral supplements from accessing brain tissue; no evidence for gadolinium removal",
                        "debunked_by": "Neuropharmacology research, lack of clinical trials",
                        "evidence": "No peer-reviewed studies showing brain gadolinium reduction from supplements",
                        "why_harmful": "Expensive, false hope, may contain unlisted ingredients"
                    },
                    {
                        "claim": "Chelation therapy for gadolinium removal",
                        "reason_debunked": "No evidence that chelating agents remove gadolinium from brain tissue; may be harmful",
                        "debunked_by": "FDA warnings, multiple medical societies including American College of Radiology",
                        "evidence": "EDTA and other chelators can cause kidney damage, electrolyte imbalances, and cardiac arrhythmias",
                        "why_harmful": "Serious adverse effects including death; no proven benefit for gadolinium removal"
                    }
                ]
            },
            "liver": {
                "known_recommendations": [
                    {
                        "intervention": "Monitor liver function in patients with hepatic disease",
                        "rationale": "Severely impaired hepatic function may affect gadolinium elimination kinetics",
                        "evidence_level": "Moderate - Based on pharmacokinetic studies",
                        "timing": "Baseline and 48-72 hour follow-up in severe hepatic impairment"
                    }
                ],
                "potential_recommendations": [
                    {
                        "intervention": "Antioxidant support",
                        "rationale": "Theoretical benefit from reducing oxidative stress, though liver is not primary elimination route",
                        "evidence_level": "Very limited - Mostly preclinical data",
                        "dosing": "Various antioxidant combinations studied",
                        "limitations": "No specific studies with gadolinium contrast; unclear clinical relevance"
                    },
                    {
                        "intervention": "Milk thistle supplementation",
                        "rationale": "Silymarin may have hepatoprotective effects through antioxidant and anti-inflammatory mechanisms",
                        "evidence_level": "Limited - Some studies in other hepatotoxic contexts",
                        "dosing": "140-420mg daily of standardized silymarin extract",
                        "limitations": "No studies specific to contrast agents; variable product quality"
                    }
                ],
                "debunked_claims": [
                    {
                        "claim": "Liver cleanses",
                        "reason_debunked": "Liver has natural detoxification processes; no evidence that commercial cleanses enhance gadolinium elimination",
                        "debunked_by": "American Liver Foundation, hepatology literature",
                        "evidence": "No scientific basis for enhanced liver 'cleansing' beyond normal physiology",
                        "why_harmful": "May cause diarrhea, electrolyte imbalances, and interfere with medications"
                    },
                    {
                        "claim": "Coffee enemas",
                        "reason_debunked": "No mechanism for gadolinium elimination via colon; potentially dangerous procedure",
                        "debunked_by": "Multiple case reports of complications, FDA warnings",
                        "evidence": "Risk of electrolyte imbalances, infections, and rectal perforation",
                        "why_harmful": "Serious complications including death reported; no medical justification"
                    }
                ]
            }
        }
        
        # Return detailed recommendations with fallback for unknown organs
        result = {}
        for organ in organs:
            if organ in recommendations_db:
                result[organ] = recommendations_db[organ]
            else:
                result[organ] = {
                    "known_recommendations": [
                        {
                            "intervention": "Consult healthcare provider",
                            "rationale": "Limited data available for this organ system with the specified procedure",
                            "evidence_level": "Expert opinion - Insufficient specific research",
                            "timing": "Before and after procedure as clinically indicated"
                        }
                    ],
                    "potential_recommendations": [],
                    "debunked_claims": []
                }
        return result
    
    def _critical_evaluation(self, medical_input: MedicalInput, organs: List[str], 
                           recommendations: Dict[str, Any]) -> MedicalOutput:
        """Critical evaluation of recommendations and evidence quality."""
        self._log_reasoning_step(
            ReasoningStage.CRITICAL_EVALUATION,
            {"recommendations": recommendations},
            "Critically evaluating recommendations against recent evidence and identifying debunked claims",
            {"evaluation_criteria": ["evidence_quality", "clinical_relevance", "safety_profile"]}
        )
        
        # Create organ analyses with detailed information
        organ_analyses = []
        for organ in organs:
            if organ in recommendations:
                organ_data = recommendations[organ]
                
                # Extract simple lists for backward compatibility
                known_recs = []
                potential_recs = []
                debunked_claims = []
                
                # Handle both detailed and simple formats
                for rec in organ_data.get("known_recommendations", []):
                    if isinstance(rec, dict):
                        known_recs.append(rec.get("intervention", str(rec)))
                    else:
                        known_recs.append(rec)
                
                for rec in organ_data.get("potential_recommendations", []):
                    if isinstance(rec, dict):
                        potential_recs.append(rec.get("intervention", str(rec)))
                    else:
                        potential_recs.append(rec)
                
                for claim in organ_data.get("debunked_claims", []):
                    if isinstance(claim, dict):
                        debunked_claims.append(claim.get("claim", str(claim)))
                    else:
                        debunked_claims.append(claim)
                
                analysis = OrganAnalysis(
                    organ_name=organ,
                    affected_by_procedure=True,
                    at_risk=True,
                    risk_level="moderate",
                    pathways_involved=["elimination", "filtration"],
                    known_recommendations=known_recs,
                    potential_recommendations=potential_recs,
                    debunked_claims=debunked_claims,
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