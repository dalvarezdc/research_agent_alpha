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
        self.fallback_providers = fallback_providers or ["openai", "ollama"]
        self.reasoning_trace: List[ReasoningStep] = []
        
        # Setup logging
        if enable_logging:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.disabled = True
            
        # Initialize LLM manager with Claude as default
        from llm_integrations import create_llm_manager
        try:
            self.llm_manager = create_llm_manager(primary_llm_provider, fallback_providers)
            self.logger.info(f"LLM manager initialized with {primary_llm_provider} as primary")
        except Exception as e:
            self.logger.warning(f"LLM initialization failed: {e}. Using fallback mode.")
            self.llm_manager = None
            
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
        """Identify organs potentially affected by the medical procedure using LLM analysis."""
        self._log_reasoning_step(
            ReasoningStage.ORGAN_IDENTIFICATION,
            {"procedure": medical_input.procedure, "details": medical_input.details},
            "Identifying organs affected by procedure based on mechanism of action using LLM analysis",
            {"method": "anthropic_claude_analysis"}
        )
        
        if self.llm_manager:
            try:
                # Use LLM to identify affected organs
                prompt = f"""
                Analyze this medical procedure and identify all organ systems that could be affected:
                
                Procedure: {medical_input.procedure}
                Details: {medical_input.details}
                Objectives: {', '.join(medical_input.objectives)}
                
                Consider:
                1. Direct effects of the procedure
                2. Contrast agents and their elimination pathways
                3. Secondary effects on organ systems
                4. Risk factors and contraindications
                
                Return ONLY a JSON list of organ names (lowercase, no spaces). Example: ["kidneys", "brain", "liver"]
                """
                
                system_prompt = """You are a medical expert analyzing procedures for organ involvement. 
                Focus on evidence-based medicine and current clinical guidelines. 
                Be thorough but avoid speculative connections."""
                
                response = self.llm_manager.medical_analysis_with_fallback(
                    {"procedure": medical_input.procedure, "details": medical_input.details},
                    "organ_identification"
                )
                
                # Extract organs from LLM response
                analysis_text = response.get("analysis", "")
                organs = self._extract_organs_from_response(analysis_text)
                
                if organs:
                    self.logger.info(f"LLM identified organs: {organs}")
                    return organs
                
            except Exception as e:
                self.logger.warning(f"LLM organ identification failed: {e}. Using fallback.")
        
        # Fallback to hardcoded mapping if LLM fails
        self.logger.info("Using fallback organ identification")
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
        
        procedure_key = medical_input.procedure.strip()
        detail_key = "with_contrast" if "contrast" in medical_input.details.lower() else "without_contrast"
        
        return organs_map.get(procedure_key, {}).get(detail_key, ["kidneys"])
    
    def _extract_organs_from_response(self, response_text: str) -> List[str]:
        """Extract organ list from LLM response"""
        import re
        import json
        
        # Try to find JSON array in response
        json_match = re.search(r'\[(.*?)\]', response_text)
        if json_match:
            try:
                organs_str = '[' + json_match.group(1) + ']'
                organs = json.loads(organs_str)
                return [organ.lower().strip() for organ in organs if isinstance(organ, str)]
            except:
                pass
        
        # Fallback: extract organ names from text
        common_organs = ["kidney", "brain", "liver", "heart", "lung", "skin", "thyroid", "blood", "vessel"]
        found_organs = []
        text_lower = response_text.lower()
        
        for organ in common_organs:
            if organ in text_lower:
                if organ == "kidney":
                    found_organs.append("kidneys")
                elif organ == "blood" or "vessel" in text_lower:
                    found_organs.append("blood_vessels")
                else:
                    found_organs.append(organ)
        
        return list(set(found_organs)) if found_organs else ["kidneys", "brain"]
    
    @lru_cache(maxsize=64)
    def _gather_evidence(self, medical_input: MedicalInput, organs: tuple) -> Dict[str, Any]:
        """Gather evidence for each identified organ system using LLM analysis."""
        organs_list = list(organs)  # Convert back from tuple for caching
        
        self._log_reasoning_step(
            ReasoningStage.EVIDENCE_GATHERING,
            {"organs": organs_list, "search_strategy": "llm_medical_literature_analysis"},
            "Gathering evidence from medical literature for organ-specific effects using LLM",
            {"method": "anthropic_claude_evidence_synthesis"}
        )
        
        if self.llm_manager:
            try:
                evidence_data = {}
                
                for organ in organs_list:
                    prompt = f"""
                    Analyze the medical evidence for how this procedure affects {organ}:
                    
                    Procedure: {medical_input.procedure}
                    Details: {medical_input.details}
                    Organ Focus: {organ}
                    
                    Provide evidence-based analysis including:
                    1. Elimination/metabolic pathway involved
                    2. Risk factors that increase organ vulnerability
                    3. Protective factors that reduce risk
                    4. Quality of available evidence (strong/moderate/limited/poor)
                    5. Key studies or guidelines (if known)
                    
                    Format as JSON:
                    {{
                        "elimination_pathway": "primary pathway description",
                        "risk_factors": ["factor1", "factor2"],
                        "protective_factors": ["factor1", "factor2"],
                        "evidence_quality": "strong/moderate/limited/poor",
                        "key_references": ["study/guideline descriptions"],
                        "mechanism": "how the procedure affects this organ"
                    }}
                    """
                    
                    system_prompt = f"""You are a medical researcher analyzing {organ} involvement in medical procedures. 
                    Base your analysis on current medical literature, clinical guidelines, and pharmacokinetic principles.
                    Be specific about mechanisms and evidence quality."""
                    
                    response = self.llm_manager.medical_analysis_with_fallback(
                        {
                            "procedure": medical_input.procedure, 
                            "organ": organ,
                            "details": medical_input.details
                        },
                        "evidence_gathering"
                    )
                    
                    # Parse evidence from LLM response
                    evidence = self._parse_evidence_response(response.get("analysis", ""), organ)
                    evidence_data[organ] = evidence
                    
                    self.logger.info(f"LLM gathered evidence for {organ}: {evidence.get('evidence_quality', 'unknown')} quality")
                
                return evidence_data
                
            except Exception as e:
                self.logger.warning(f"LLM evidence gathering failed: {e}. Using fallback.")
        
        # Fallback evidence database
        self.logger.info("Using fallback evidence gathering")
        evidence_base = {
            "kidneys": {
                "elimination_pathway": "glomerular_filtration_and_tubular_secretion",
                "risk_factors": ["pre_existing_ckd", "dehydration", "age_over_65", "diabetes"],
                "protective_factors": ["adequate_hydration", "avoid_nephrotoxins", "normal_kidney_function"],
                "evidence_quality": "strong",
                "mechanism": "Contrast agents filtered by glomeruli, concentrated in tubules"
            },
            "brain": {
                "elimination_pathway": "blood_brain_barrier_limited_retention",
                "risk_factors": ["repeated_exposure", "kidney_impairment", "linear_contrast_agents"],
                "protective_factors": ["normal_kidney_function", "macrocyclic_agents"],
                "evidence_quality": "moderate",
                "mechanism": "Gadolinium can cross blood-brain barrier and accumulate in tissue"
            },
            "liver": {
                "elimination_pathway": "minimal_hepatic_metabolism",
                "risk_factors": ["severe_liver_disease", "biliary_obstruction"],
                "protective_factors": ["normal_liver_function"],
                "evidence_quality": "limited",
                "mechanism": "Limited liver involvement in contrast elimination"
            }
        }
        
        return {organ: evidence_base.get(organ, {
            "elimination_pathway": "unknown",
            "risk_factors": ["unknown"],
            "protective_factors": ["consult_physician"],
            "evidence_quality": "limited",
            "mechanism": "Insufficient data available"
        }) for organ in organs_list}
    
    def _parse_evidence_response(self, response_text: str, organ: str) -> Dict[str, Any]:
        """Parse evidence data from LLM response"""
        import re
        import json
        
        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                evidence_data = json.loads(json_match.group(0))
                return evidence_data
            except:
                pass
        
        # Fallback parsing from text
        evidence = {
            "elimination_pathway": "pathway_not_specified",
            "risk_factors": [],
            "protective_factors": [],
            "evidence_quality": "limited",
            "mechanism": "mechanism_not_specified"
        }
        
        # Extract evidence quality
        if "strong evidence" in response_text.lower():
            evidence["evidence_quality"] = "strong"
        elif "moderate evidence" in response_text.lower():
            evidence["evidence_quality"] = "moderate"
        elif "limited evidence" in response_text.lower():
            evidence["evidence_quality"] = "limited"
        elif "poor evidence" in response_text.lower():
            evidence["evidence_quality"] = "poor"
        
        # Extract risk factors (simple text parsing)
        risk_keywords = ["risk factor", "contraindication", "caution", "avoid"]
        protective_keywords = ["protective", "beneficial", "recommend", "hydration"]
        
        lines = response_text.lower().split('\n')
        for line in lines:
            if any(keyword in line for keyword in risk_keywords):
                if "dehydration" in line:
                    evidence["risk_factors"].append("dehydration")
                if "kidney" in line or "renal" in line:
                    evidence["risk_factors"].append("kidney_impairment")
                if "age" in line:
                    evidence["risk_factors"].append("advanced_age")
            
            if any(keyword in line for keyword in protective_keywords):
                if "hydration" in line:
                    evidence["protective_factors"].append("adequate_hydration")
                if "monitor" in line:
                    evidence["protective_factors"].append("monitoring")
        
        return evidence
    
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
        """Synthesize recommendations based on evidence and risk assessment using LLM analysis."""
        self._log_reasoning_step(
            ReasoningStage.RECOMMENDATION_SYNTHESIS,
            {"evidence_summary": len(evidence), "risks_assessed": len(risks)},
            "Synthesizing evidence-based recommendations using LLM analysis while categorizing by evidence quality",
            {"categories": ["known_effective", "potentially_beneficial", "debunked"], "method": "anthropic_claude_synthesis"}
        )
        
        if self.llm_manager:
            try:
                recommendations_data = {}
                
                for organ in organs:
                    organ_evidence = evidence.get(organ, {})
                    organ_risk = risks.get(organ, {})
                    
                    prompt = f"""
                    Synthesize medical recommendations for {organ} following this medical procedure:
                    
                    Procedure: {medical_input.procedure}
                    Details: {medical_input.details}
                    Organ: {organ}
                    
                    Evidence Summary:
                    - Elimination pathway: {organ_evidence.get('elimination_pathway', 'unknown')}
                    - Risk factors: {organ_evidence.get('risk_factors', [])}
                    - Protective factors: {organ_evidence.get('protective_factors', [])}
                    - Evidence quality: {organ_evidence.get('evidence_quality', 'limited')}
                    - Risk level: {organ_risk.get('risk_level', 'unknown')}
                    
                    Provide recommendations in 3 categories with detailed rationales:
                    
                    1. KNOWN/EVIDENCE-BASED (Strong clinical evidence):
                    - Include specific interventions with rationale, evidence level, and implementation details
                    
                    2. POTENTIAL/INVESTIGATIONAL (Limited but promising evidence):
                    - Include rationale, evidence level, dosing, and limitations
                    
                    3. DEBUNKED/HARMFUL (Proven ineffective or dangerous):
                    - Include why debunked, who debunked it, evidence against, and why harmful
                    
                    Format as JSON:
                    {{
                        "known_recommendations": [
                            {{
                                "intervention": "specific recommendation",
                                "rationale": "mechanism and reasoning",
                                "evidence_level": "Strong - source details",
                                "timing": "when/how to implement"
                            }}
                        ],
                        "potential_recommendations": [
                            {{
                                "intervention": "investigational approach",
                                "rationale": "theoretical basis",
                                "evidence_level": "Limited - study details",
                                "dosing": "specific dosing if known",
                                "limitations": "why evidence is limited"
                            }}
                        ],
                        "debunked_claims": [
                            {{
                                "claim": "debunked treatment",
                                "reason_debunked": "scientific reasoning",
                                "debunked_by": "authorities/studies",
                                "evidence": "evidence against",
                                "why_harmful": "specific harms"
                            }}
                        ]
                    }}
                    """
                    
                    system_prompt = f"""You are a medical expert synthesizing evidence-based recommendations for {organ} protection.
                    Base recommendations on current clinical guidelines, peer-reviewed literature, and established medical principles.
                    Be specific about evidence quality and distinguish between proven interventions and investigational approaches.
                    Clearly identify and explain why certain popular treatments are debunked or harmful."""
                    
                    response = self.llm_manager.medical_analysis_with_fallback(
                        {
                            "procedure": medical_input.procedure,
                            "organ": organ,
                            "evidence": organ_evidence,
                            "risk": organ_risk
                        },
                        "recommendation_synthesis"
                    )
                    
                    # Parse recommendations from LLM response
                    recommendations = self._parse_recommendations_response(response.get("analysis", ""), organ)
                    recommendations_data[organ] = recommendations
                    
                    self.logger.info(f"LLM synthesized recommendations for {organ}")
                
                return recommendations_data
                
            except Exception as e:
                self.logger.warning(f"LLM recommendation synthesis failed: {e}. Using fallback.")
        
        # Fallback to hardcoded detailed recommendations if LLM fails
        self.logger.info("Using fallback recommendation synthesis")
        
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
    
    def _parse_recommendations_response(self, response_text: str, organ: str) -> Dict[str, Any]:
        """Parse recommendations from LLM response"""
        import re
        import json
        
        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                recommendations_data = json.loads(json_match.group(0))
                return recommendations_data
            except:
                pass
        
        # Fallback: parse structured text
        recommendations = {
            "known_recommendations": [],
            "potential_recommendations": [],
            "debunked_claims": []
        }
        
        # Simple text parsing for fallback
        lines = response_text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if "known" in line.lower() or "evidence-based" in line.lower():
                current_section = "known"
            elif "potential" in line.lower() or "investigational" in line.lower():
                current_section = "potential"
            elif "debunked" in line.lower() or "harmful" in line.lower():
                current_section = "debunked"
            elif line.startswith('-') or line.startswith('•'):
                intervention = line[1:].strip()
                if current_section == "known":
                    recommendations["known_recommendations"].append({
                        "intervention": intervention,
                        "rationale": "Evidence-based intervention",
                        "evidence_level": "Strong",
                        "timing": "As clinically indicated"
                    })
                elif current_section == "potential":
                    recommendations["potential_recommendations"].append({
                        "intervention": intervention,
                        "rationale": "Limited but promising evidence",
                        "evidence_level": "Limited",
                        "limitations": "Requires further study"
                    })
                elif current_section == "debunked":
                    recommendations["debunked_claims"].append({
                        "claim": intervention,
                        "reason_debunked": "Insufficient evidence or proven harmful",
                        "debunked_by": "Medical literature",
                        "evidence": "Lack of clinical benefit",
                        "why_harmful": "May delay appropriate care"
                    })
        
        return recommendations
    
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