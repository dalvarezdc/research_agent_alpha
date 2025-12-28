#!/usr/bin/env python3
"""
Medication Analyzer
Analyzes medications with comprehensive interaction analysis and detailed recommendations.
Extends the medical_reasoning_agent framework with medication-specific capabilities.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import sys
import json
from pathlib import Path

# Add parent directory to path for cost_tracker import
sys.path.insert(0, str(Path(__file__).parent.parent))
from cost_tracker import track_cost, print_cost_summary, reset_tracking

from .medical_reasoning_agent import (
    MedicalReasoningAgent, TokenUsage, ReasoningStage, ReasoningStep
)

# Import DSPy for structured output
import dspy
from pydantic import BaseModel

# Import Pydantic schemas for structured outputs
from .dspy_schemas import (
    PharmacologyData,
    DrugInteractionsData,
    FoodInteractionsData,
    SafetyProfileData,
    RecommendationsData,
    MonitoringData
)


class InteractionSeverity(Enum):
    """Severity levels for drug interactions"""
    SEVERE = "severe"  # Contraindicated or requires immediate intervention
    MAJOR = "major"  # Same as severe (LLM sometimes uses this)
    MODERATE = "moderate"  # Requires monitoring or dose adjustment
    MINOR = "minor"  # Usually manageable

    @classmethod
    def from_string(cls, value: str):
        """Convert string to enum, handling variations"""
        value_lower = value.lower().strip()
        if value_lower in ("severe", "major", "critical"):
            return cls.SEVERE
        elif value_lower == "moderate":
            return cls.MODERATE
        elif value_lower in ("minor", "low"):
            return cls.MINOR
        else:
            return cls.MODERATE  # Default to moderate if unknown


class InteractionType(Enum):
    """Types of medication interactions"""
    DRUG_DRUG = "drug-drug"
    DRUG_FOOD = "drug-food"
    DRUG_SUPPLEMENT = "drug-supplement"
    DRUG_ENVIRONMENTAL = "drug-environmental"  # Light, temperature, etc.


@dataclass
class Interaction:
    """Detailed interaction information"""
    interaction_type: InteractionType
    interacting_agent: str
    severity: InteractionSeverity
    mechanism: str  # How the interaction occurs
    clinical_effect: str  # What happens clinically
    management: str  # How to manage the interaction
    time_separation: Optional[str] = None  # If timing can prevent interaction
    evidence_level: str = "moderate"
    references: List[str] = field(default_factory=list)


@dataclass
class MedicationInput:
    """Input for medication analysis"""
    medication_name: str
    indication: Optional[str] = None  # Primary use
    patient_medications: List[str] = field(default_factory=list)  # Other meds
    patient_conditions: List[str] = field(default_factory=list)  # Medical conditions
    patient_context: Optional[Dict[str, Any]] = None  # Age, pregnancy, etc.


@dataclass
class MedicationOutput:
    """Comprehensive medication analysis output"""
    medication_name: str
    drug_class: str
    mechanism_of_action: str

    # Pharmacokinetics
    absorption: str
    metabolism: str
    elimination: str
    half_life: str

    # Clinical use
    approved_indications: List[str] = field(default_factory=list)
    off_label_uses: List[str] = field(default_factory=list)

    # Dosing
    standard_dosing: str = ""
    dose_adjustments: Dict[str, str] = field(default_factory=dict)  # renal, hepatic, etc.

    # Safety
    common_adverse_effects: List[str] = field(default_factory=list)
    serious_adverse_effects: List[str] = field(default_factory=list)
    contraindications: List[Dict[str, str]] = field(default_factory=list)
    black_box_warnings: List[str] = field(default_factory=list)

    # Interactions
    drug_interactions: List[Interaction] = field(default_factory=list)
    food_interactions: List[Interaction] = field(default_factory=list)
    environmental_considerations: List[str] = field(default_factory=list)

    # Recommendations (using same structure as medical_reasoning_agent)
    evidence_based_recommendations: List[Dict[str, str]] = field(default_factory=list)
    what_not_to_do: List[Dict[str, str]] = field(default_factory=list)
    debunked_claims: List[Dict[str, str]] = field(default_factory=list)

    # Monitoring
    monitoring_requirements: List[str] = field(default_factory=list)
    warning_signs: List[Dict[str, str]] = field(default_factory=list)

    # Metadata
    evidence_quality: str = "moderate"
    analysis_confidence: float = 0.75
    reasoning_trace: List[ReasoningStep] = field(default_factory=list)
    total_token_usage: TokenUsage = field(default_factory=TokenUsage)
    practitioner_report: Optional[str] = None  # Detailed markdown report for medical practitioners
    validation_report: Optional[Any] = None  # Reference validation report


class MedicationAnalyzer(MedicalReasoningAgent):
    """
    Analyzes medications with focus on interactions and comprehensive guidance.
    Extends MedicalReasoningAgent to leverage existing LLM infrastructure.
    """

    def __init__(self,
                 primary_llm_provider: str = "claude",
                 fallback_providers: List[str] = None,
                 enable_logging: bool = True,
                 enable_reference_validation: bool = False,
                 enable_web_research: bool = False):
        """
        Initialize medication analyzer.

        Args:
            primary_llm_provider: Primary LLM to use
            fallback_providers: Fallback LLM providers
            enable_logging: Enable detailed logging
            enable_reference_validation: Validate drug interaction references
        """
        super().__init__(
            primary_llm_provider,
            fallback_providers,
            enable_logging,
            enable_reference_validation,
            enable_web_research,
        )

        # Setup DSPy for structured output
        try:
            self.llm_manager.setup_dspy_integration()
            self.use_dspy = True
            self.logger.info("DSPy structured output enabled")
        except Exception as e:
            self.logger.warning(f"DSPy setup failed: {e}. Falling back to manual parsing.")
            self.use_dspy = False

    def analyze_medication(self, medication_input: MedicationInput) -> MedicationOutput:
        """
        Comprehensive medication analysis with interactions.

        Args:
            medication_input: Structured medication input

        Returns:
            MedicationOutput: Comprehensive analysis results
        """
        self.logger.info(f"Starting medication analysis for: {medication_input.medication_name}")
        self.reasoning_trace = []  # Reset trace
        reset_tracking()  # Reset cost tracking for this analysis

        # Initialize token usage tracker for this analysis
        self.total_token_usage = TokenUsage()

        try:
            # Phase 1: Pharmacology basics
            pharmacology = self._analyze_pharmacology(medication_input)

            # Phase 2: Interaction analysis
            interactions = self._analyze_interactions(medication_input, pharmacology)

            # Phase 3: Safety profile
            safety_profile = self._analyze_safety_profile(medication_input, pharmacology)

            # Phase 4: Clinical recommendations
            recommendations = self._synthesize_medication_recommendations(
                medication_input,
                pharmacology,
                interactions,
                safety_profile
            )

            # Phase 5: Monitoring requirements
            monitoring = self._determine_monitoring_requirements(
                medication_input,
                pharmacology,
                safety_profile
            )

            # Synthesize final output
            output = self._synthesize_medication_output(
                medication_input,
                pharmacology,
                interactions,
                safety_profile,
                recommendations,
                monitoring
            )

            # Print cost summary
            print_cost_summary()

            return output

        except Exception as e:
            self.logger.error(f"Medication analysis failed: {e}")
            raise

    @track_cost("Phase 1: Pharmacology Analysis")
    def _analyze_pharmacology(self, medication_input: MedicationInput) -> Dict[str, Any]:
        """Analyze medication pharmacology using DSPy structured output"""
        self._log_reasoning_step(
            ReasoningStage.INPUT_ANALYSIS,
            {"medication": medication_input.medication_name},
            "Analyzing medication pharmacology and mechanism of action",
            {"method": "dspy_structured_generation"}
        )

        prompt = f"""
        Provide comprehensive pharmacology information for {medication_input.medication_name}:

        1. DRUG CLASS: Pharmacologic and therapeutic class
        2. MECHANISM: Detailed mechanism of action at molecular level
        3. PHARMACOKINETICS: Absorption, distribution, metabolism, elimination, half-life with specific values
        4. CLINICAL USE: FDA-approved indications and common off-label uses
        5. DOSING: Standard dosing and adjustments for renal/hepatic impairment

        Return complete, detailed information for all fields.
        """

        try:
            # Try DSPy structured generation first
            if self.use_dspy:
                pharmacology = self._generate_with_dspy(prompt, PharmacologyData)
                if pharmacology:
                    return pharmacology.model_dump()

            # Fallback to manual parsing with schema
            schema = PharmacologyData.model_json_schema()
            structured_prompt = f"""
            {prompt}

            CRITICAL: Respond with ONLY valid JSON matching this schema:
            {json.dumps(schema, indent=2)}

            Start with {{ and end with }}. No other text.
            """

            system_prompt = """You are a clinical pharmacologist. Respond ONLY with valid JSON matching the schema."""

            response, token_usage = self.llm_manager.get_available_provider().generate_response(
                structured_prompt, system_prompt
            )

            # Accumulate token usage for cost tracking
            if token_usage:
                self.total_token_usage.add(token_usage)

            pharmacology = self._parse_pharmacology_response(response)
            return pharmacology

        except Exception as e:
            self.logger.error(f"Pharmacology analysis failed: {e}")
            return self._get_fallback_pharmacology(medication_input.medication_name)

    @track_cost("Phase 2: Interaction Analysis")
    def _analyze_interactions(self,
                            medication_input: MedicationInput,
                            pharmacology: Dict[str, Any]) -> Dict[str, List[Interaction]]:
        """Comprehensive interaction analysis"""
        self._log_reasoning_step(
            ReasoningStage.EVIDENCE_GATHERING,
            {"medication": medication_input.medication_name,
             "concurrent_meds": len(medication_input.patient_medications)},
            "Analyzing drug interactions including drug-drug, drug-food, and environmental factors",
            {"analysis_types": ["drug-drug", "drug-food", "drug-environmental"]}
        )

        interactions = {
            "drug_drug": [],
            "drug_food": [],
            "drug_supplement": [],
            "drug_environmental": []
        }

        # Analyze each interaction type
        interactions["drug_drug"] = self._analyze_drug_drug_interactions(
            medication_input, pharmacology
        )

        interactions["drug_food"] = self._analyze_drug_food_interactions(
            medication_input, pharmacology
        )

        interactions["drug_supplement"] = self._analyze_drug_supplement_interactions(
            medication_input, pharmacology
        )

        interactions["drug_environmental"] = self._analyze_environmental_factors(
            medication_input, pharmacology
        )

        return interactions

    def _analyze_drug_drug_interactions(self,
                                       medication_input: MedicationInput,
                                       pharmacology: Dict[str, Any]) -> List[Interaction]:
        """Analyze drug-drug interactions with structured output"""

        context_meds = ""
        if medication_input.patient_medications:
            context_meds = f"\n\nPatient is currently taking: {', '.join(medication_input.patient_medications)}"

        # Get JSON schema
        schema = DrugInteractionsData.model_json_schema()

        prompt = f"""
        Analyze drug-drug interactions for {medication_input.medication_name}.{context_meds}

        CRITICAL: Respond with ONLY valid JSON matching this exact schema:

        {json.dumps(schema, indent=2)}

        Categorize interactions by severity:
        - SEVERE: Contraindicated or requiring immediate intervention
        - MODERATE: Require monitoring or dose adjustment
        - MINOR: Usually manageable

        For each interaction include:
        - interacting_agent: Specific medication name
        - severity: "severe", "moderate", or "minor"
        - mechanism: Pharmacologic mechanism (PK or PD)
        - clinical_effect: Clinical consequences
        - management: Management strategy
        - time_separation: Timing if applicable
        - evidence_level: Quality of evidence

        Respond with ONLY the JSON object.
        """

        system_prompt = """You are a clinical pharmacist. Respond ONLY with valid JSON matching the schema.
        No explanatory text, just raw JSON."""

        try:
            response, token_usage = self.llm_manager.get_available_provider().generate_response(
                prompt, system_prompt
            )

            # Accumulate token usage
            if token_usage:
                self.total_token_usage.add(token_usage)

            interactions_data = self._parse_with_pydantic(
                response,
                DrugInteractionsData,
                fallback_value=None
            )

            if not interactions_data:
                return []

            # Convert to Interaction objects
            interactions = []
            for interaction_detail in (interactions_data.severe_interactions +
                                     interactions_data.moderate_interactions +
                                     interactions_data.minor_interactions):
                interactions.append(Interaction(
                    interaction_type=InteractionType.DRUG_DRUG,
                    interacting_agent=interaction_detail.interacting_agent,
                    severity=InteractionSeverity.from_string(interaction_detail.severity),
                    mechanism=interaction_detail.mechanism,
                    clinical_effect=interaction_detail.clinical_effect,
                    management=interaction_detail.management,
                    time_separation=interaction_detail.time_separation,
                    evidence_level=interaction_detail.evidence_level
                ))

            return interactions

        except Exception as e:
            self.logger.error(f"Drug-drug interaction analysis failed: {e}")
            return []

    def _analyze_drug_food_interactions(self,
                                       medication_input: MedicationInput,
                                       pharmacology: Dict[str, Any]) -> List[Interaction]:
        """Analyze drug-food interactions"""

        prompt = f"""
        Analyze drug-food interactions for {medication_input.medication_name}:

        Provide detailed analysis of:

        1. FOODS TO AVOID:
           - Specific foods/beverages that significantly affect the medication
           - Mechanism (absorption, metabolism, effect)
           - Clinical impact (e.g., "reduces absorption by 50%")
           - Management strategy

        2. FOODS THAT ENHANCE EFFICACY OR REDUCE SIDE EFFECTS:
           - Foods that improve absorption or tolerability
           - Mechanism and magnitude of effect
           - Timing recommendations

        3. ALCOHOL INTERACTIONS:
           - Severity and clinical effects
           - Mechanism
           - Specific recommendations (avoid, limit, timing)

        4. GRAPEFRUIT/CITRUS INTERACTIONS:
           - If applicable, detailed mechanism (CYP3A4 inhibition)
           - Duration of effect
           - Alternatives

        5. TIMING WITH MEALS:
           - Should medication be taken with food or on empty stomach?
           - Rationale (absorption, GI irritation, etc.)
           - Specific timing instructions

        Format as JSON array of interaction objects with rich detail.
        """

        system_prompt = """You are a clinical pharmacist providing food-drug interaction guidance.
        Be specific about mechanisms, timing, and clinical significance."""

        try:
            response, token_usage = self.llm_manager.get_available_provider().generate_response(
                prompt, system_prompt
            )

            # Accumulate token usage
            if token_usage:
                self.total_token_usage.add(token_usage)

            return self._parse_interaction_response(response, InteractionType.DRUG_FOOD)

        except Exception as e:
            self.logger.error(f"Drug-food interaction analysis failed: {e}")
            return []

    def _analyze_drug_supplement_interactions(self,
                                             medication_input: MedicationInput,
                                             pharmacology: Dict[str, Any]) -> List[Interaction]:
        """Analyze interactions with supplements and herbal products"""

        prompt = f"""
        Analyze interactions between {medication_input.medication_name} and common supplements/herbs:

        Focus on:
        1. St. John's Wort (CYP inducer)
        2. Vitamin K (for anticoagulants)
        3. Calcium, magnesium, iron (absorption)
        4. Herbal products with similar effects
        5. Common vitamin/mineral supplements

        For each significant interaction provide:
        - Supplement name
        - Severity and mechanism
        - Clinical effect
        - Management (avoid, separate timing, monitor)

        Format as JSON array.
        """

        system_prompt = """You are a clinical pharmacist specializing in herb-drug interactions.
        Focus on evidence-based, clinically significant interactions."""

        try:
            response, token_usage = self.llm_manager.get_available_provider().generate_response(
                prompt, system_prompt
            )

            # Accumulate token usage
            if token_usage:
                self.total_token_usage.add(token_usage)

            return self._parse_interaction_response(response, InteractionType.DRUG_SUPPLEMENT)

        except Exception as e:
            self.logger.error(f"Drug-supplement interaction analysis failed: {e}")
            return []

    def _analyze_environmental_factors(self,
                                      medication_input: MedicationInput,
                                      pharmacology: Dict[str, Any]) -> List[Dict[str, str]]:
        """Analyze environmental and lifestyle factors"""

        prompt = f"""
        Analyze environmental and lifestyle considerations for {medication_input.medication_name}:

        1. PHOTOSENSITIVITY:
           - Does this medication cause sun sensitivity?
           - Mechanism and severity
           - Protective measures needed

        2. TEMPERATURE SENSITIVITY:
           - Storage requirements
           - Effect of temperature on patient (heat/cold sensitivity)
           - Travel considerations

        3. ACTIVITY RESTRICTIONS:
           - Driving/operating machinery
           - Exercise limitations
           - Altitude considerations

        4. LIGHT EXPOSURE:
           - Should medication be protected from light?
           - Any concerns with bright lights or screens?

        5. SOUND SENSITIVITY:
           - Any ototoxicity concerns?
           - Noise-related considerations?

        Format as list of environmental considerations with detailed explanations.
        """

        system_prompt = """You are a clinical pharmacist providing comprehensive patient counseling.
        Include practical, actionable environmental and lifestyle guidance."""

        try:
            response, token_usage = self.llm_manager.get_available_provider().generate_response(
                prompt, system_prompt
            )

            # Accumulate token usage
            if token_usage:
                self.total_token_usage.add(token_usage)

            return self._parse_environmental_response(response)

        except Exception as e:
            self.logger.error(f"Environmental factors analysis failed: {e}")
            return []

    @track_cost("Phase 3: Safety Profile Assessment")
    def _analyze_safety_profile(self,
                                medication_input: MedicationInput,
                                pharmacology: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze medication safety profile"""
        self._log_reasoning_step(
            ReasoningStage.RISK_ASSESSMENT,
            {"medication": medication_input.medication_name},
            "Analyzing safety profile including adverse effects and contraindications",
            {"focus_areas": ["adverse_effects", "contraindications", "warnings"]}
        )

        prompt = f"""
        Provide comprehensive safety profile for {medication_input.medication_name}:

        1. ADVERSE EFFECTS:
           - Common (>10%): List with approximate frequencies
           - Frequent (1-10%): Clinically significant effects
           - Serious (any frequency): Effects requiring medical attention
           - Rare but important (<1%): Including idiosyncratic reactions

        2. BLACK BOX WARNINGS:
           - All FDA black box warnings with detailed explanations
           - Risk mitigation strategies

        3. CONTRAINDICATIONS:
           For each contraindication provide:
           - condition: Specific contraindication
           - severity: "absolute" or "relative"
           - reason: Detailed pathophysiologic explanation
           - alternative: What to use instead
           - risk_if_ignored: Specific consequences

        4. WARNING SIGNS:
           - Early signs of serious adverse effects
           - When to seek immediate medical attention
           - Routine monitoring signs

        5. SPECIAL POPULATIONS:
           - Pregnancy category/risk summary
           - Breastfeeding compatibility
           - Pediatric considerations
           - Geriatric considerations
           - Renal impairment
           - Hepatic impairment

        Format as JSON with comprehensive detail in each section.
        """

        system_prompt = """You are a drug safety expert providing evidence-based safety information.
        Be specific about risks, frequencies, and management strategies.
        Base information on FDA labeling and post-market surveillance data."""

        try:
            response, token_usage = self.llm_manager.get_available_provider().generate_response(
                prompt, system_prompt
            )

            # Accumulate token usage
            if token_usage:
                self.total_token_usage.add(token_usage)

            return self._parse_safety_profile_response(response)

        except Exception as e:
            self.logger.error(f"Safety profile analysis failed: {e}")
            return {}

    @track_cost("Phase 4: Recommendation Synthesis")
    def _synthesize_medication_recommendations(self,
                                              medication_input: MedicationInput,
                                              pharmacology: Dict[str, Any],
                                              interactions: Dict[str, Any],
                                              safety_profile: Dict[str, Any]) -> Dict[str, List[Dict]]:
        """Synthesize evidence-based medication recommendations"""
        self._log_reasoning_step(
            ReasoningStage.RECOMMENDATION_SYNTHESIS,
            {"medication": medication_input.medication_name},
            "Synthesizing evidence-based recommendations for optimal medication use",
            {"categories": ["evidence_based", "what_not_to_do", "debunked"]}
        )

        prompt = f"""
        Synthesize comprehensive recommendations for {medication_input.medication_name}.
        All recommendations must be evidence-based.

        WHAT TO DO:
        Provide detailed guidance on:
        1. How to maximize medication effectiveness
        2. How to minimize adverse effects
        3. Supportive measures that help
        4. Timing optimization
        5. Lifestyle modifications that enhance outcomes

        For each recommendation provide:
        - intervention: Specific action
        - rationale: Why this works (mechanism and evidence)
        - evidence_level: Quality of supporting evidence
        - implementation: Step-by-step how to do this
        - expected_outcome: What to expect and when
        - monitoring: What to track

        WHAT NOT TO DO:
        Unsafe practices or misuse to avoid:
        - action: What to avoid
        - rationale: Why it is unsafe or ineffective
        - evidence_level: Quality of supporting evidence
        - risk_if_ignored: Specific consequences
        - safer_alternative: Evidence-based alternative
        - exceptions: Any narrow cases where avoidance does not apply

        DEBUNKED CLAIMS:
        Common misconceptions about this medication. A debunked claim must be:
        1. A specific, commonly repeated statement,
        2. Contradicted by labeling/guidelines/trials or large reviews,
        3. Distinct from behavior advice (avoid overlap with WHAT NOT TO DO).

        For each debunked claim provide:
        - claim: What people incorrectly believe
        - reason_debunked: Why it's wrong
        - evidence: Studies/reviews debunking
        - why_harmful: How this misconception causes harm
        - debunked_by: Source type (e.g., FDA label, RCTs, meta-analyses)
        - common_misconception: Why people believe it

        Format as JSON with paragraph-level detail in each field.
        """

        system_prompt = """You are a clinical pharmacist synthesizing evidence-based medication guidance.
        Focus on practical, actionable recommendations supported by clinical evidence."""

        try:
            response, token_usage = self.llm_manager.get_available_provider().generate_response(
                prompt, system_prompt
            )

            # Accumulate token usage
            if token_usage:
                self.total_token_usage.add(token_usage)

            return self._parse_recommendations_response(response)

        except Exception as e:
            self.logger.error(f"Recommendation synthesis failed: {e}")
            return {
                "evidence_based": [],
                "what_not_to_do": [],
                "debunked": []
            }

    @track_cost("Phase 5: Monitoring Requirements")
    def _determine_monitoring_requirements(self,
                                          medication_input: MedicationInput,
                                          pharmacology: Dict[str, Any],
                                          safety_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Determine monitoring requirements"""

        prompt = f"""
        Determine monitoring requirements for {medication_input.medication_name}:

        1. BASELINE ASSESSMENTS:
           - What tests/evaluations before starting?
           - Why each is needed

        2. ROUTINE MONITORING:
           - What parameters to monitor
           - Frequency (e.g., "weekly for first month, then monthly")
           - Target ranges or concerning values
           - Rationale for monitoring

        3. SYMPTOM MONITORING:
           - What symptoms to watch for
           - How to assess
           - When to report to provider

        Format as structured JSON.
        """

        system_prompt = """You are a clinical pharmacist providing monitoring guidance.
        Be specific about what, when, and why to monitor."""

        try:
            response, token_usage = self.llm_manager.get_available_provider().generate_response(
                prompt, system_prompt
            )

            # Accumulate token usage
            if token_usage:
                self.total_token_usage.add(token_usage)

            return self._parse_monitoring_response(response)

        except Exception as e:
            self.logger.error(f"Monitoring requirements analysis failed: {e}")
            return {}

    # ========== DSPy Structured Output Methods ==========

    def _generate_with_dspy(self, prompt: str, output_model: type[BaseModel]) -> Optional[BaseModel]:
        """
        Generate structured output using DSPy TypedPredictor.

        Args:
            prompt: Instruction prompt
            output_model: Pydantic model class for output structure

        Returns:
            Validated Pydantic model instance or None
        """
        if not self.use_dspy:
            return None

        try:
            # Create a DSPy signature dynamically
            class GenerationSignature(dspy.Signature):
                """Generate structured medical information"""
                instruction = dspy.InputField(desc="Task instructions")
                output = dspy.OutputField(desc="Structured output")

            # Use TypedPredictor for structured generation
            predictor = dspy.TypedPredictor(GenerationSignature)
            result = predictor(instruction=prompt)

            # Validate with Pydantic
            validated = output_model.model_validate_json(result.output)
            self.logger.info(f"DSPy successfully generated {output_model.__name__}")
            return validated

        except Exception as e:
            self.logger.warning(f"DSPy generation failed for {output_model.__name__}: {e}")
            return None

    # ========== Parsing Helper Methods ==========

    def _parse_with_pydantic(self, response: str, model_class: type, fallback_value: Any = None) -> Any:
        """
        Parse LLM response using Pydantic model validation.

        Args:
            response: Raw LLM response
            model_class: Pydantic model class to validate against
            fallback_value: Value to return if parsing fails

        Returns:
            Validated Pydantic model instance or fallback value
        """
        import re

        # Extract JSON from response
        json_str = None

        # Try code block first
        code_block_match = re.search(r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```', response, re.DOTALL)
        if code_block_match:
            json_str = code_block_match.group(1)
        else:
            # Try to find JSON object/array
            json_match = re.search(r'(\{.*\}|\[.*\])', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)

        if not json_str:
            self.logger.warning(f"No JSON found in response for {model_class.__name__}")
            return fallback_value

        # Clean JSON
        json_str = re.sub(r'//.*?\n', '\n', json_str)  # Remove comments
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        json_str = re.sub(r',\s*}', '}', json_str)  # Trailing commas
        json_str = re.sub(r',\s*]', ']', json_str)

        # Try to parse and validate with Pydantic
        try:
            data = json.loads(json_str)
            validated = model_class.model_validate(data)
            self.logger.info(f"Successfully parsed {model_class.__name__}")
            return validated
        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON decode error for {model_class.__name__}: {e}")
            return fallback_value
        except Exception as e:
            self.logger.warning(f"Pydantic validation error for {model_class.__name__}: {e}")
            return fallback_value

    def _parse_pharmacology_response(self, response: str) -> Dict[str, Any]:
        """Parse pharmacology data using Pydantic validation"""
        pharmacology = self._parse_with_pydantic(
            response,
            PharmacologyData,
            fallback_value=None
        )

        if pharmacology:
            return pharmacology.model_dump()

        self.logger.warning("Could not parse pharmacology response, using fallback")
        return {}

    def _parse_interaction_response(self, response: str, interaction_type: InteractionType) -> List[Interaction]:
        """Parse interactions using Pydantic validation"""
        # Try to parse based on interaction type
        if interaction_type == InteractionType.DRUG_FOOD:
            interactions_data = self._parse_with_pydantic(
                response,
                FoodInteractionsData,
                fallback_value=None
            )
            if not interactions_data:
                return []

            # Convert food interactions to Interaction objects
            interactions = []
            for food_detail in interactions_data.foods_to_avoid + interactions_data.foods_that_help:
                interactions.append(Interaction(
                    interaction_type=InteractionType.DRUG_FOOD,
                    interacting_agent=food_detail.food_or_beverage,
                    severity=InteractionSeverity.from_string(food_detail.interaction_type),
                    mechanism=food_detail.mechanism,
                    clinical_effect=food_detail.clinical_impact,
                    management=food_detail.management,
                    time_separation=food_detail.timing_guidance,
                    evidence_level="moderate"
                ))

            if interactions_data.alcohol_interaction:
                ai = interactions_data.alcohol_interaction
                interactions.append(Interaction(
                    interaction_type=InteractionType.DRUG_FOOD,
                    interacting_agent="Alcohol",
                    severity=InteractionSeverity.from_string(ai.interaction_type),
                    mechanism=ai.mechanism,
                    clinical_effect=ai.clinical_impact,
                    management=ai.management,
                    time_separation=ai.timing_guidance,
                    evidence_level="moderate"
                ))

            return interactions
        else:
            # For other interaction types, use simple JSON parsing
            import re
            import json

            json_match = re.search(r'(\{.*\}|\[.*\])', response, re.DOTALL)
            if not json_match:
                return []

            try:
                json_str = json_match.group(1)
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                data = json.loads(json_str)

                interactions = []
                items = data if isinstance(data, list) else [data]

                for item in items:
                    if isinstance(item, dict):
                        try:
                            interactions.append(Interaction(
                                interaction_type=interaction_type,
                                interacting_agent=item.get('interacting_agent', item.get('agent', 'Unknown')),
                                severity=InteractionSeverity.from_string(item.get('severity', 'moderate')),
                                mechanism=item.get('mechanism', ''),
                                clinical_effect=item.get('clinical_effect', ''),
                                management=item.get('management', ''),
                                time_separation=item.get('time_separation'),
                                evidence_level=item.get('evidence_level', 'moderate')
                            ))
                        except Exception as e:
                            self.logger.warning(f"Failed to parse interaction item: {e}")
                            continue

                return interactions
            except Exception as e:
                self.logger.warning(f"Failed to parse interactions: {e}")
                return []

    def _parse_environmental_response(self, response: str) -> List[Dict[str, str]]:
        """Parse environmental considerations"""
        # Simple text parsing as fallback
        considerations = []
        # Would parse the response and extract environmental factors
        return considerations

    def _parse_safety_profile_response(self, response: str) -> Dict[str, Any]:
        """Parse safety profile using Pydantic validation"""
        safety_profile = self._parse_with_pydantic(
            response,
            SafetyProfileData,
            fallback_value=None
        )

        if safety_profile:
            return safety_profile.model_dump()

        self.logger.warning("Could not parse safety profile, using empty data")
        return {}

    def _parse_recommendations_response(self, response: str) -> Dict[str, List[Dict]]:
        """Parse recommendations using Pydantic validation"""
        recommendations = self._parse_with_pydantic(
            response,
            RecommendationsData,
            fallback_value=None
        )

        if recommendations:
            return {
                "evidence_based": [rec.model_dump() for rec in recommendations.evidence_based],
                "what_not_to_do": [rec.model_dump() for rec in recommendations.what_not_to_do],
                "debunked": [rec.model_dump() for rec in recommendations.debunked]
            }

        self.logger.warning("Could not parse recommendations, using empty data")
        return {
            "evidence_based": [],
            "what_not_to_do": [],
            "debunked": []
        }

    def _parse_monitoring_response(self, response: str) -> Dict[str, Any]:
        """Parse monitoring requirements using Pydantic validation"""
        monitoring = self._parse_with_pydantic(
            response,
            MonitoringData,
            fallback_value=None
        )

        if monitoring:
            return monitoring.model_dump()

        self.logger.warning("Could not parse monitoring response, using empty data")
        return {}

    def _synthesize_medication_output(self,
                                     medication_input: MedicationInput,
                                     pharmacology: Dict[str, Any],
                                     interactions: Dict[str, Any],
                                     safety_profile: Dict[str, Any],
                                     recommendations: Dict[str, Any],
                                     monitoring: Dict[str, Any]) -> MedicationOutput:
        """Synthesize all components into final output"""

        # Combine all interaction types
        all_interactions = []
        for interaction_list in interactions.values():
            all_interactions.extend(interaction_list)

        output = MedicationOutput(
            medication_name=medication_input.medication_name,
            drug_class=pharmacology.get('drug_class', ''),
            mechanism_of_action=pharmacology.get('mechanism_of_action', ''),
            absorption=pharmacology.get('absorption', ''),
            metabolism=pharmacology.get('metabolism', ''),
            elimination=pharmacology.get('elimination', ''),
            half_life=pharmacology.get('half_life', ''),
            approved_indications=pharmacology.get('approved_indications', []),
            off_label_uses=pharmacology.get('off_label_uses', []),
            standard_dosing=pharmacology.get('standard_dosing', ''),
            dose_adjustments=pharmacology.get('dose_adjustments', {}),
            common_adverse_effects=safety_profile.get('common_adverse_effects', []),
            serious_adverse_effects=safety_profile.get('serious_adverse_effects', []),
            contraindications=safety_profile.get('contraindications', []),
            black_box_warnings=safety_profile.get('black_box_warnings', []),
            drug_interactions=[i for i in all_interactions if i.interaction_type == InteractionType.DRUG_DRUG],
            food_interactions=[i for i in all_interactions if i.interaction_type == InteractionType.DRUG_FOOD],
            environmental_considerations=interactions.get('drug_environmental', []),
            evidence_based_recommendations=recommendations.get('evidence_based', []),
            what_not_to_do=recommendations.get('what_not_to_do', []),
            debunked_claims=recommendations.get('debunked', []),
            monitoring_requirements=monitoring.get('routine_monitoring', []),
            warning_signs=safety_profile.get('warning_signs', []),
            evidence_quality='moderate',
            analysis_confidence=0.75,
            reasoning_trace=self.reasoning_trace
        )

        # Generate practitioner report (detailed markdown version)
        output.practitioner_report = self._generate_medication_practitioner_report(output)

        # Validate references if enabled
        if self.enable_reference_validation and self.reference_validator:
            output.validation_report = self.reference_validator.validate_analysis(output)

        return output

    def _get_fallback_pharmacology(self, medication_name: str) -> Dict[str, Any]:
        """Fallback pharmacology data when LLM fails"""
        return {
            "drug_class": "Not available",
            "mechanism_of_action": "Requires detailed analysis",
            "absorption": "See prescribing information",
            "metabolism": "See prescribing information",
            "elimination": "See prescribing information",
            "half_life": "See prescribing information"
        }

    def _generate_medication_practitioner_report(self, output: 'MedicationOutput') -> str:
        """Generate detailed markdown report for medical practitioners."""
        from datetime import datetime

        report = f"""# 💊 Medication Analysis Report (Practitioner Version)
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Medication:** {output.medication_name}
**Drug Class:** {output.drug_class}
**Analysis Confidence:** {output.analysis_confidence:.2f}/1.00

---

## 🧬 Pharmacology

### 🎯 Mechanism of Action
{output.mechanism_of_action}

### ⚗️ Pharmacokinetics
- **Absorption:** {output.absorption}
- **Metabolism:** {output.metabolism}
- **Elimination:** {output.elimination}
- **Half-Life:** {output.half_life}

---

## 💉 Clinical Use

### ✅ Approved Indications
"""
        for i, indication in enumerate(output.approved_indications, 1):
            report += f"{i}. {indication}\n"

        if output.off_label_uses:
            report += "\n### 🔬 Off-Label Uses\n"
            for i, use in enumerate(output.off_label_uses, 1):
                report += f"{i}. {use}\n"

        report += f"""

### 💊 Dosing
**Standard Dosing:** {output.standard_dosing}

"""
        if output.dose_adjustments:
            report += "**Dose Adjustments:**\n"
            for adjustment_type, adjustment_info in output.dose_adjustments.items():
                report += f"- **{adjustment_type.replace('_', ' ').title()}:** {adjustment_info}\n"

        report += """

---

## ⚠️ Safety Profile

"""
        if output.black_box_warnings:
            report += "### 🚨 BLACK BOX WARNINGS\n\n"
            for i, warning in enumerate(output.black_box_warnings, 1):
                report += f"{i}. {warning}\n\n"

        if output.contraindications:
            report += f"### ❌ Contraindications ({len(output.contraindications)} identified)\n\n"
            for contra in output.contraindications:
                if isinstance(contra, dict):
                    report += f"- **{contra.get('condition', 'N/A')}** ({contra.get('severity', 'N/A')})\n"
                    report += f"  - Reason: {contra.get('reason', 'N/A')}\n"
                    if contra.get('alternative'):
                        report += f"  - Alternative: {contra.get('alternative')}\n"
                else:
                    report += f"- {contra}\n"
            report += "\n"

        report += "### 🔴 Adverse Effects\n\n"
        if output.common_adverse_effects:
            report += "**Common (>10%):**\n"
            for effect in output.common_adverse_effects:
                report += f"- {effect}\n"
            report += "\n"

        if output.serious_adverse_effects:
            report += "**Serious (Any Frequency):**\n"
            for effect in output.serious_adverse_effects:
                report += f"- {effect}\n"
            report += "\n"

        report += """

---

## 🔗 Drug-Drug Interactions

"""
        if output.drug_interactions:
            severe = [i for i in output.drug_interactions if i.severity == InteractionSeverity.SEVERE]
            moderate = [i for i in output.drug_interactions if i.severity == InteractionSeverity.MODERATE]
            minor = [i for i in output.drug_interactions if i.severity == InteractionSeverity.MINOR]

            if severe:
                report += f"### 🔴 SEVERE Interactions ({len(severe)})\n\n"
                for interaction in severe:
                    report += f"#### {interaction.interacting_agent}\n"
                    report += f"**Mechanism:** {interaction.mechanism}\n\n"
                    report += f"**Clinical Effect:** {interaction.clinical_effect}\n\n"
                    report += f"**Management:** {interaction.management}\n\n"
                    if interaction.time_separation:
                        report += f"**Time Separation:** {interaction.time_separation}\n\n"
                    report += f"**Evidence Level:** {interaction.evidence_level}\n\n"

            if moderate:
                report += f"### 🟡 Moderate Interactions ({len(moderate)})\n\n"
                for interaction in moderate:
                    report += f"**{interaction.interacting_agent}:** {interaction.clinical_effect}\n\n"

            if minor:
                report += f"### 🟢 Minor Interactions ({len(minor)})\n\n"
                for interaction in minor:
                    report += f"- {interaction.interacting_agent}: {interaction.clinical_effect}\n"
        else:
            report += "No significant drug-drug interactions identified.\n"

        report += """

---

## 🍎 Food & Lifestyle Interactions

"""
        if output.food_interactions:
            for interaction in output.food_interactions:
                severity_emoji = "🔴" if interaction.severity == InteractionSeverity.SEVERE else "🟡" if interaction.severity == InteractionSeverity.MODERATE else "🟢"
                report += f"{severity_emoji} **{interaction.interacting_agent}** ({interaction.severity.value.upper()})\n"
                report += f"- **Mechanism:** {interaction.mechanism}\n"
                report += f"- **Clinical Effect:** {interaction.clinical_effect}\n"
                report += f"- **Management:** {interaction.management}\n\n"
        else:
            report += "No significant food interactions identified.\n"

        report += """

---

## 🌍 Environmental Considerations

"""
        if output.environmental_considerations:
            for i, consideration in enumerate(output.environmental_considerations, 1):
                if isinstance(consideration, dict):
                    report += f"{i}. **{consideration.get('type', 'N/A')}:** {consideration.get('description', 'N/A')}\n"
                else:
                    report += f"{i}. {consideration}\n"
        else:
            report += "No significant environmental considerations identified.\n"

        report += """

---

## 💡 Recommendations

"""
        if output.evidence_based_recommendations:
            report += "### ✅ What TO DO:\n\n"
            for i, rec in enumerate(output.evidence_based_recommendations, 1):
                if isinstance(rec, dict):
                    report += f"#### {i}. {rec.get('intervention', 'N/A')}\n\n"
                    report += f"**Rationale:** {rec.get('rationale', 'N/A')}\n\n"
                    report += f"**Evidence Level:** {rec.get('evidence_level', 'N/A')}\n\n"
                    report += f"**Implementation:** {rec.get('implementation', 'N/A')}\n\n"
                else:
                    report += f"{i}. {rec}\n"

        if output.what_not_to_do:
            report += "### ❌ What NOT TO DO:\n\n"
            for i, rec in enumerate(output.what_not_to_do, 1):
                if isinstance(rec, dict):
                    report += f"#### {i}. {rec.get('action', 'N/A')}\n\n"
                    report += f"**Rationale:** {rec.get('rationale', 'N/A')}\n\n"
                    report += f"**Evidence Level:** {rec.get('evidence_level', 'N/A')}\n\n"
                    report += f"**Risk If Ignored:** {rec.get('risk_if_ignored', 'N/A')}\n\n"
                    if rec.get('safer_alternative'):
                        report += f"**Safer Alternative:** {rec.get('safer_alternative')}\n\n"
                    if rec.get('exceptions'):
                        report += f"**Exceptions:** {rec.get('exceptions')}\n\n"
                else:
                    report += f"{i}. {rec}\n"

        if output.debunked_claims:
            report += "### 🧯 Debunked Claims:\n\n"
            for i, claim in enumerate(output.debunked_claims, 1):
                if isinstance(claim, dict):
                    report += f"#### {i}. {claim.get('claim', 'N/A')}\n\n"
                    report += f"**Why Debunked:** {claim.get('reason_debunked', 'N/A')}\n\n"
                    report += f"**Evidence Against:** {claim.get('evidence', 'N/A')}\n\n"
                    report += f"**Why Harmful:** {claim.get('why_harmful', 'N/A')}\n\n"
                else:
                    report += f"{i}. {claim}\n"

        report += """

---

## 📊 Monitoring Requirements

"""
        if output.monitoring_requirements:
            for i, req in enumerate(output.monitoring_requirements, 1):
                if isinstance(req, dict):
                    report += f"{i}. **{req.get('parameter', 'N/A')}**\n"
                    report += f"   - Frequency: {req.get('frequency', 'N/A')}\n"
                    report += f"   - Rationale: {req.get('rationale', 'N/A')}\n\n"
                else:
                    report += f"{i}. {req}\n"

        if output.warning_signs:
            report += "\n### ⚠️ Warning Signs\n\n"
            for sign in output.warning_signs:
                if isinstance(sign, dict):
                    report += f"**{sign.get('sign', 'N/A')}** ({sign.get('severity', 'N/A')})\n"
                    report += f"- Action: {sign.get('action', 'N/A')}\n\n"

        report += f"""

---

**Report Generated:** {datetime.now().isoformat()}
**For Medical Professional Use Only**
**Evidence Quality:** {output.evidence_quality.upper()}
"""

        return report

    def export_medication_analysis(self, output: MedicationOutput, filepath: str):
        """Export medication analysis to JSON"""
        import json
        from dataclasses import asdict

        # Convert to dict (handling nested dataclasses and enums)
        def convert(obj):
            if hasattr(obj, '__dict__'):
                return {k: convert(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            elif isinstance(obj, Enum):
                return obj.value
            else:
                return obj

        output_dict = convert(output)

        with open(filepath, 'w') as f:
            json.dump(output_dict, f, indent=2)

        self.logger.info(f"Medication analysis exported to {filepath}")


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Medication Analyzer")
    parser.add_argument('medication', help='Medication name')
    parser.add_argument('--indication', help='Primary indication')
    parser.add_argument('--other-meds', nargs='*', help='Other medications patient is taking')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    parser.add_argument('--llm', default='claude', help='LLM provider')

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = MedicationAnalyzer(primary_llm_provider=args.llm)

    # Create input
    med_input = MedicationInput(
        medication_name=args.medication,
        indication=args.indication,
        patient_medications=args.other_meds or []
    )

    # Analyze
    print(f"Analyzing {args.medication}...")
    result = analyzer.analyze_medication(med_input)

    # Export
    analyzer.export_medication_analysis(result, args.output)
    print(f"\nAnalysis complete: {args.output}")

    # Display summary
    print(f"\nDrug Class: {result.drug_class}")
    print(f"Mechanism: {result.mechanism_of_action[:100]}...")
    print(f"Drug-Drug Interactions: {len(result.drug_interactions)}")
    print(f"Food Interactions: {len(result.food_interactions)}")
    print(f"What TO DO: {len(result.evidence_based_recommendations)}")
    print(f"What NOT TO DO: {len(result.what_not_to_do)}")
    print(f"Debunked Claims: {len(result.debunked_claims)}")


if __name__ == "__main__":
    main()
