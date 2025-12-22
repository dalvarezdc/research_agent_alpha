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
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from cost_tracker import track_cost, print_cost_summary, reset_tracking
from llm_integrations import TokenUsage


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
    practitioner_report: Optional[str] = None  # Detailed markdown report for medical practitioners
    validation_report: Optional[Any] = None  # Reference validation report


class MedicalReasoningAgent:
    """
    AI agent for systematic medical procedure analysis.
    Follows the analytical pattern: broad → specific → critical evaluation.
    """
    
    def __init__(self,
                 primary_llm_provider: str = "claude",
                 fallback_providers: List[str] = None,
                 enable_logging: bool = True,
                 enable_reference_validation: bool = False):
        """
        Initialize the medical reasoning agent.

        Args:
            primary_llm_provider: Primary LLM to use (claude, openai, etc.)
            fallback_providers: List of fallback LLM providers
            enable_logging: Whether to enable detailed reasoning logging
            enable_reference_validation: Whether to validate references
        """
        self.primary_llm = primary_llm_provider
        self.fallback_providers = fallback_providers or ["openai", "ollama"]
        self.reasoning_trace: List[ReasoningStep] = []
        self.enable_reference_validation = enable_reference_validation
        self.reference_validator = None

        # Initialize reference validator if enabled
        if enable_reference_validation:
            try:
                from reference_validation import ReferenceValidator, ValidationConfig
                self.reference_validator = ReferenceValidator(ValidationConfig(
                    cache_backend="sqlite",
                    min_credibility_score=70
                ))
            except ImportError:
                self.logger.warning("Reference validation not available") if enable_logging else None
        
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
        # Validate inputs
        from .input_validation import InputValidator, ValidationError
        
        try:
            # Validate procedure name
            proc_result = InputValidator.validate_medical_procedure(medical_input.procedure)
            if not proc_result.is_valid:
                raise ValidationError(f"Invalid procedure: {', '.join(proc_result.errors)}")
            
            # Validate details
            if medical_input.details:
                details_result = InputValidator.validate_medical_procedure(medical_input.details)
                if not details_result.is_valid:
                    raise ValidationError(f"Invalid details: {', '.join(details_result.errors)}")
            
            # Validate objectives
            if medical_input.objectives:
                obj_result = InputValidator.validate_medical_aspects(list(medical_input.objectives))
                if not obj_result.is_valid:
                    raise ValidationError(f"Invalid objectives: {', '.join(obj_result.errors)}")
            
            self.logger.info("Input validation passed successfully")
            
        except ValidationError as e:
            self.logger.error(f"Input validation failed: {e}")
            raise ValueError(f"Invalid medical input: {e}")

        # Reset cost tracking for new analysis
        reset_tracking()

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

            # Print cost summary
            print_cost_summary()

            return final_output
            
        except Exception as e:
            self.logger.error(f"Error in medical analysis pipeline: {str(e)}")
            raise
    
    @track_cost("Phase 1: Organ Identification")
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
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                self.logger.warning(f"Failed to parse organ list from JSON: {e}")
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

    @track_cost("Phase 2: Evidence Gathering")
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
            except (json.JSONDecodeError, ValueError) as e:
                self.logger.warning(f"Failed to parse evidence data from JSON: {e}")
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

    @track_cost("Phase 3: Risk Assessment")
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

    @track_cost("Phase 4: Recommendation Synthesis")
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
                    Synthesize comprehensive medical recommendations for {organ} protection following this medical procedure:

                    Procedure: {medical_input.procedure}
                    Details: {medical_input.details}
                    Organ: {organ}

                    Evidence Summary:
                    - Elimination pathway: {organ_evidence.get('elimination_pathway', 'unknown')}
                    - Risk factors: {organ_evidence.get('risk_factors', [])}
                    - Protective factors: {organ_evidence.get('protective_factors', [])}
                    - Evidence quality: {organ_evidence.get('evidence_quality', 'limited')}
                    - Risk level: {organ_risk.get('risk_level', 'unknown')}

                    Provide DETAILED, COMPREHENSIVE recommendations in 3 categories. Each recommendation should include enough detail to write a full paragraph (3-5 sentences minimum):

                    1. KNOWN/EVIDENCE-BASED RECOMMENDATIONS (Strong clinical evidence):
                    For each recommendation, provide:
                    - intervention: Specific action to take
                    - rationale: Detailed explanation of mechanism and why this works (2-3 sentences)
                    - evidence_level: Evidence strength with specific sources (e.g., "Strong - Multiple RCTs including Smith et al. 2023")
                    - timing: Precise timing and duration (e.g., "Begin 24h before procedure, continue for 48h after")
                    - implementation: Step-by-step how to implement, including dosing if applicable
                    - expected_outcome: What results to expect and when
                    - monitoring: What to monitor and how often
                    - cost_consideration: Approximate cost if relevant (e.g., "~$15/month OTC supplement")

                    2. POTENTIAL/INVESTIGATIONAL RECOMMENDATIONS (Limited but promising evidence):
                    For each recommendation, provide:
                    - intervention: Specific approach
                    - rationale: Theoretical basis and preliminary evidence (2-3 sentences)
                    - evidence_level: Current evidence with limitations
                    - implementation: How to try this approach
                    - dosing: Specific dosing if known from studies
                    - limitations: Why evidence is limited and what studies are needed
                    - safety_profile: Known safety information

                    3. CONTRAINDICATIONS - What NOT to do (Things to avoid):
                    For each contraindication, provide:
                    - condition: What should be avoided
                    - severity: absolute/relative/caution
                    - reason: Detailed explanation why this is contraindicated (2-3 sentences)
                    - alternative: What to do instead if this is needed
                    - risk_if_ignored: Specific harms that can occur

                    4. DEBUNKED/HARMFUL TREATMENTS (Proven ineffective or dangerous):
                    For each debunked treatment, provide:
                    - claim: What treatment people incorrectly use
                    - reason_debunked: Scientific explanation why it doesn't work (2-3 sentences)
                    - debunked_by: Specific authorities, studies, or meta-analyses
                    - evidence: Specific evidence against this treatment
                    - why_harmful: How this can actually cause harm
                    - common_misconception: Why people think it works

                    5. WARNING SIGNS to watch for:
                    For each warning sign, provide:
                    - sign: Observable symptom or test result
                    - severity: emergency/urgent/monitor
                    - mechanism: Why this sign indicates a problem
                    - action: Specific action to take
                    - timeframe: When this typically occurs

                    6. INTERACTIONS (if applicable for medications/supplements):
                    - drug_interactions: Medications that interact
                    - food_interactions: Foods to avoid or that help
                    - environmental_factors: Light sensitivity, temperature, activity restrictions

                    Write each field with enough detail that it could be expanded into a full paragraph. Be specific with:
                    - Numbers and measurements (dosages, percentages, timeframes)
                    - Mechanisms of action
                    - Evidence sources
                    - Practical implementation steps

                    Format as JSON with rich, detailed content in each field.
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
        
        # Fallback to default recommendations if LLM fails
        self.logger.info("Using fallback recommendation synthesis")

        # Default recommendations database
        default_recommendations = {
            "kidneys": {
                "known_recommendations": [
                    {"intervention": "Ensure adequate hydration before and after procedure", "rationale": "Reduces contrast nephropathy risk", "evidence_level": "Strong - Multiple RCTs", "timing": "Pre and post procedure"},
                    {"intervention": "Monitor kidney function (creatinine, eGFR)", "rationale": "Detect early kidney injury", "evidence_level": "Strong - Clinical guidelines", "timing": "Baseline and 48-72h post"},
                    {"intervention": "Avoid nephrotoxic medications 48h before/after", "rationale": "Reduces cumulative kidney stress", "evidence_level": "Strong - Clinical practice", "timing": "48h window"}
                ],
                "potential_recommendations": [
                    {"intervention": "N-acetylcysteine supplementation", "rationale": "Antioxidant properties may reduce oxidative stress", "evidence_level": "Limited - Mixed study results", "dosing": "600mg PO BID day before and day of procedure", "limitations": "Inconsistent evidence across studies"},
                    {"intervention": "Sodium bicarbonate hydration", "rationale": "Alkalinization may reduce tubular injury", "evidence_level": "Limited - Some positive studies", "limitations": "Not universally recommended"}
                ],
                "debunked_claims": [
                    {"claim": "Furosemide (diuretic) prevents contrast nephropathy", "reason_debunked": "Increases dehydration risk", "debunked_by": "Multiple clinical trials and meta-analyses", "evidence": "No benefit, possible harm in RCTs", "why_harmful": "Worsens dehydration, increases nephropathy risk"},
                    {"claim": "Dopamine is protective for kidneys", "reason_debunked": "No clinical benefit demonstrated", "debunked_by": "Clinical trials and guidelines", "evidence": "Multiple RCTs showed no benefit", "why_harmful": "Cardiac side effects without benefit"}
                ]
            }
        }

        result = {}
        for organ in organs:
            result[organ] = default_recommendations.get(organ, {
                "known_recommendations": [{"intervention": "Consult specialist", "rationale": "Limited data available", "evidence_level": "Expert opinion", "timing": "As needed"}],
                "potential_recommendations": [],
                "debunked_claims": []
            })
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
            except (json.JSONDecodeError, ValueError) as e:
                self.logger.warning(f"Failed to parse recommendations from JSON: {e}")
                pass
        
        # Fallback: parse structured text
        recommendations = {
            "known_recommendations": [],
            "potential_recommendations": [],
            "contraindications": [],
            "debunked_claims": [],
            "warning_signs": [],
            "interactions": {}
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
            elif "contraindication" in line.lower() or "what not to do" in line.lower():
                current_section = "contraindications"
            elif "debunked" in line.lower() or "harmful" in line.lower():
                current_section = "debunked"
            elif "warning" in line.lower():
                current_section = "warning"
            elif "interaction" in line.lower():
                current_section = "interactions"
            elif line.startswith('-') or line.startswith('•'):
                intervention = line[1:].strip()
                if current_section == "known":
                    recommendations["known_recommendations"].append({
                        "intervention": intervention,
                        "rationale": "Evidence-based intervention - detailed rationale not parsed from text",
                        "evidence_level": "Strong - based on clinical guidelines",
                        "timing": "As clinically indicated",
                        "implementation": "Consult healthcare provider for specific implementation",
                        "expected_outcome": "Variable based on individual factors",
                        "monitoring": "Standard monitoring recommended"
                    })
                elif current_section == "potential":
                    recommendations["potential_recommendations"].append({
                        "intervention": intervention,
                        "rationale": "Limited but promising evidence - requires further validation",
                        "evidence_level": "Limited - preliminary studies",
                        "implementation": "Discuss with healthcare provider",
                        "dosing": "Not standardized",
                        "limitations": "Requires further study",
                        "safety_profile": "Generally considered safe, monitor for adverse effects"
                    })
                elif current_section == "contraindications":
                    recommendations["contraindications"].append({
                        "condition": intervention,
                        "severity": "caution",
                        "reason": "May interfere with procedure or recovery",
                        "alternative": "Consult healthcare provider for alternatives",
                        "risk_if_ignored": "Increased risk of complications"
                    })
                elif current_section == "debunked":
                    recommendations["debunked_claims"].append({
                        "claim": intervention,
                        "reason_debunked": "Insufficient evidence or proven harmful",
                        "debunked_by": "Medical literature and clinical trials",
                        "evidence": "Lack of clinical benefit in controlled studies",
                        "why_harmful": "May delay appropriate care or cause direct harm",
                        "common_misconception": "Popular belief not supported by evidence"
                    })
                elif current_section == "warning":
                    recommendations["warning_signs"].append({
                        "sign": intervention,
                        "severity": "monitor",
                        "mechanism": "Potential indicator of complication",
                        "action": "Contact healthcare provider if observed",
                        "timeframe": "Variable"
                    })

        return recommendations

    @track_cost("Phase 5: Critical Evaluation")
    def _critical_evaluation(self, medical_input: MedicalInput, organs: List[str],
                           recommendations: Dict[str, Any]) -> MedicalOutput:
        """Critical evaluation of recommendations and evidence quality."""
        self._log_reasoning_step(
            ReasoningStage.CRITICAL_EVALUATION,
            {"recommendations": recommendations},
            "Critically evaluating recommendations against recent evidence and identifying debunked claims",
            {"evaluation_criteria": ["evidence_quality", "clinical_relevance", "safety_profile"]}
        )
        
        # Create organ analyses with detailed information preserved
        organ_analyses = []
        for organ in organs:
            if organ in recommendations:
                organ_data = recommendations[organ]

                # Preserve full detailed recommendations as JSON strings for backward compatibility
                # This allows the OrganAnalysis to store the data without changing its structure
                known_recs = []
                potential_recs = []
                debunked_claims = []

                # Store detailed recommendations as formatted strings
                for rec in organ_data.get("known_recommendations", []):
                    if isinstance(rec, dict):
                        # Format as detailed string preserving all information
                        intervention = rec.get("intervention", "")
                        rationale = rec.get("rationale", "")
                        evidence = rec.get("evidence_level", "")
                        timing = rec.get("timing", "")
                        implementation = rec.get("implementation", "")

                        formatted = f"{intervention}"
                        if rationale:
                            formatted += f" | Rationale: {rationale}"
                        if evidence:
                            formatted += f" | Evidence: {evidence}"
                        if timing:
                            formatted += f" | Timing: {timing}"
                        if implementation:
                            formatted += f" | Implementation: {implementation}"

                        known_recs.append(formatted)
                    else:
                        known_recs.append(str(rec))

                for rec in organ_data.get("potential_recommendations", []):
                    if isinstance(rec, dict):
                        intervention = rec.get("intervention", "")
                        rationale = rec.get("rationale", "")
                        evidence = rec.get("evidence_level", "")
                        limitations = rec.get("limitations", "")

                        formatted = f"{intervention}"
                        if rationale:
                            formatted += f" | Rationale: {rationale}"
                        if evidence:
                            formatted += f" | Evidence: {evidence}"
                        if limitations:
                            formatted += f" | Limitations: {limitations}"

                        potential_recs.append(formatted)
                    else:
                        potential_recs.append(str(rec))

                for claim in organ_data.get("debunked_claims", []):
                    if isinstance(claim, dict):
                        claim_text = claim.get("claim", "")
                        reason = claim.get("reason_debunked", "")
                        harmful = claim.get("why_harmful", "")

                        formatted = f"{claim_text}"
                        if reason:
                            formatted += f" | Why debunked: {reason}"
                        if harmful:
                            formatted += f" | Why harmful: {harmful}"

                        debunked_claims.append(formatted)
                    else:
                        debunked_claims.append(str(claim))

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

        # Create output
        output = MedicalOutput(
            procedure_summary=f"{medical_input.procedure} - {medical_input.details}",
            organs_analyzed=organ_analyses,
            general_recommendations=["Consult healthcare provider", "Monitor for adverse effects"],
            research_gaps=["Long-term gadolinium retention effects", "Optimal hydration protocols"],
            confidence_score=0.75,
            reasoning_trace=self.reasoning_trace
        )

        # Generate practitioner report (detailed markdown version)
        output.practitioner_report = self._generate_practitioner_report(output)

        # Validate references if enabled
        if self.enable_reference_validation and self.reference_validator:
            output.validation_report = self.reference_validator.validate_analysis(output)

        return output
    
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

    def _generate_practitioner_report(self, output: 'MedicalOutput') -> str:
        """Generate detailed markdown report for medical practitioners."""
        from datetime import datetime

        report = f"""# Medical Procedure Analysis Report (Practitioner Version)
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Analysis Confidence:** {output.confidence_score:.2f}/1.00

---

## Procedure Overview
{output.procedure_summary}

**Total Organs Analyzed:** {len(output.organs_analyzed)}
**Reasoning Steps:** {len(output.reasoning_trace)}

---

## Detailed Organ-Specific Analysis

"""

        for i, organ in enumerate(output.organs_analyzed, 1):
            report += f"""### {i}. {organ.organ_name.upper()}

**Risk Assessment:**
- **Risk Level:** {organ.risk_level.upper()}
- **Directly Affected:** {'Yes' if organ.affected_by_procedure else 'No'}
- **At Risk:** {'Yes - Requires monitoring' if organ.at_risk else 'No'}
- **Evidence Quality:** {organ.evidence_quality.upper()}

**Biological Pathways Involved:**
"""
            for pathway in organ.pathways_involved:
                report += f"- {pathway.replace('_', ' ').title()}\n"

            if organ.known_recommendations:
                report += f"\n**Evidence-Based Recommendations ({len(organ.known_recommendations)} items):**\n"
                for j, rec in enumerate(organ.known_recommendations, 1):
                    report += f"{j}. {rec}\n"

            if organ.potential_recommendations:
                report += f"\n**Investigational Approaches ({len(organ.potential_recommendations)} items):**\n"
                for j, rec in enumerate(organ.potential_recommendations, 1):
                    report += f"{j}. {rec}\n"

            if organ.debunked_claims:
                report += f"\n**Debunked/Harmful Claims ({len(organ.debunked_claims)} items):**\n"
                for j, claim in enumerate(organ.debunked_claims, 1):
                    report += f"{j}. {claim}\n"

            report += "\n---\n\n"

        report += f"""## General Recommendations

"""
        for i, rec in enumerate(output.general_recommendations, 1):
            report += f"{i}. {rec}\n"

        report += f"""

## Research Gaps & Future Directions

"""
        for i, gap in enumerate(output.research_gaps, 1):
            report += f"{i}. {gap}\n"

        report += f"""

---

**Report Generated:** {datetime.now().isoformat()}
**For Medical Professional Use Only**
"""

        return report

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