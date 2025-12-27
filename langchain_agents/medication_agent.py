"""
LangChain-based medication analyzer.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from cost_tracker import print_cost_summary, reset_tracking, track_cost
from medical_procedure_analyzer.medical_reasoning_agent import ReasoningStage, ReasoningStep
from medical_procedure_analyzer.medication_analyzer import (
    Interaction,
    InteractionSeverity,
    InteractionType,
    MedicationInput,
    MedicationOutput,
)

from .base import LangChainAgentBase, LangChainAgentConfig


class _InteractionModel(BaseModel):
    interaction_type: str
    interacting_agent: str
    severity: str
    mechanism: str
    clinical_effect: str
    management: str
    time_separation: Optional[str] = None
    evidence_level: str = "moderate"


class _MedicationOutputModel(BaseModel):
    medication_name: str
    drug_class: str
    mechanism_of_action: str
    absorption: str
    metabolism: str
    elimination: str
    half_life: str
    approved_indications: List[str] = Field(default_factory=list)
    off_label_uses: List[str] = Field(default_factory=list)
    standard_dosing: str = ""
    dose_adjustments: Dict[str, str] = Field(default_factory=dict)
    common_adverse_effects: List[str] = Field(default_factory=list)
    serious_adverse_effects: List[str] = Field(default_factory=list)
    contraindications: List[Dict[str, str]] = Field(default_factory=list)
    black_box_warnings: List[str] = Field(default_factory=list)
    drug_interactions: List[_InteractionModel] = Field(default_factory=list)
    food_interactions: List[_InteractionModel] = Field(default_factory=list)
    environmental_considerations: List[str] = Field(default_factory=list)
    evidence_based_recommendations: List[Dict[str, str]] = Field(default_factory=list)
    investigational_approaches: List[Dict[str, str]] = Field(default_factory=list)
    debunked_claims: List[Dict[str, str]] = Field(default_factory=list)
    monitoring_requirements: List[str] = Field(default_factory=list)
    warning_signs: List[Dict[str, str]] = Field(default_factory=list)
    evidence_quality: str = "moderate"
    analysis_confidence: float = 0.75


class LangChainMedicationAnalyzer(LangChainAgentBase):
    """
    LangChain-based medication analyzer with structured JSON outputs.
    """

    def __init__(
        self,
        primary_llm_provider: str = "claude-sonnet",
        fallback_providers: Optional[List[str]] = None,
        enable_logging: bool = True,
        enable_reference_validation: bool = False,
    ):
        config = LangChainAgentConfig(
            primary_llm_provider=primary_llm_provider,
            fallback_providers=fallback_providers or ["openai", "ollama"],
            enable_logging=enable_logging,
            enable_reference_validation=enable_reference_validation,
        )
        super().__init__(config)
        self.reasoning_trace: List[ReasoningStep] = []

    def analyze_medication(self, medication_input: MedicationInput) -> MedicationOutput:
        reset_tracking()
        self.reasoning_trace = []

        output_model = self._generate_medication_output(medication_input)
        output = self._to_dataclass(output_model)
        output.reasoning_trace = self.reasoning_trace

        if self.enable_reference_validation and self.reference_validator:
            output.validation_report = self.reference_validator.validate_analysis(output)

        print_cost_summary()
        return output

    @track_cost("Medication Analysis (LangChain)")
    def _generate_medication_output(
        self, medication_input: MedicationInput
    ) -> _MedicationOutputModel:
        system_prompt = (
            "You are a clinical pharmacist producing structured medication analysis. "
            "Return ONLY valid JSON."
        )
        user_prompt = """
Analyze the medication below and return a comprehensive structured report.

Medication: {medication}
Indication: {indication}
Other medications: {other_meds}

Return JSON matching this schema:
{schema}
"""
        response = self._call_llm(
            system_prompt,
            user_prompt,
            medication=medication_input.medication_name,
            indication=medication_input.indication or "N/A",
            other_meds=", ".join(medication_input.patient_medications) or "None",
            schema=_MedicationOutputModel.model_json_schema(),
        )
        parsed = self._parse_json(response)
        model: Optional[_MedicationOutputModel] = None

        if isinstance(parsed, dict):
            try:
                model = _MedicationOutputModel.model_validate(parsed)
            except Exception:
                model = None

        if model is None:
            model = _MedicationOutputModel(
                medication_name=medication_input.medication_name,
                drug_class="unknown",
                mechanism_of_action="",
                absorption="",
                metabolism="",
                elimination="",
                half_life="",
                analysis_confidence=0.5,
            )

        self._log_step(
            ReasoningStage.INPUT_ANALYSIS,
            {"medication": medication_input.medication_name},
            "Generated structured medication analysis using LangChain prompts",
            {"confidence": model.analysis_confidence},
        )
        return model

    def _to_dataclass(self, model: _MedicationOutputModel) -> MedicationOutput:
        return MedicationOutput(
            medication_name=model.medication_name,
            drug_class=model.drug_class,
            mechanism_of_action=model.mechanism_of_action,
            absorption=model.absorption,
            metabolism=model.metabolism,
            elimination=model.elimination,
            half_life=model.half_life,
            approved_indications=model.approved_indications,
            off_label_uses=model.off_label_uses,
            standard_dosing=model.standard_dosing,
            dose_adjustments=model.dose_adjustments,
            common_adverse_effects=model.common_adverse_effects,
            serious_adverse_effects=model.serious_adverse_effects,
            contraindications=model.contraindications,
            black_box_warnings=model.black_box_warnings,
            drug_interactions=[
                self._interaction_from_model(item)
                for item in model.drug_interactions
            ],
            food_interactions=[
                self._interaction_from_model(item)
                for item in model.food_interactions
            ],
            environmental_considerations=model.environmental_considerations,
            evidence_based_recommendations=model.evidence_based_recommendations,
            investigational_approaches=model.investigational_approaches,
            debunked_claims=model.debunked_claims,
            monitoring_requirements=model.monitoring_requirements,
            warning_signs=model.warning_signs,
            evidence_quality=model.evidence_quality,
            analysis_confidence=model.analysis_confidence,
            reasoning_trace=self.reasoning_trace,
        )

    def _interaction_from_model(self, model: _InteractionModel) -> Interaction:
        return Interaction(
            interaction_type=self._normalize_interaction_type(model.interaction_type),
            interacting_agent=model.interacting_agent,
            severity=InteractionSeverity.from_string(model.severity),
            mechanism=model.mechanism,
            clinical_effect=model.clinical_effect,
            management=model.management,
            time_separation=model.time_separation,
            evidence_level=model.evidence_level,
        )

    def _normalize_interaction_type(self, raw_type: str) -> InteractionType:
        normalized = raw_type.strip().lower().replace("_", "-")
        for candidate in InteractionType:
            if candidate.value == normalized:
                return candidate
        return InteractionType.DRUG_DRUG

    def _log_step(
        self,
        stage: ReasoningStage,
        input_data: dict[str, Any],
        reasoning: str,
        output: dict[str, Any],
        confidence: float = 0.8,
    ) -> None:
        self.reasoning_trace.append(
            ReasoningStep(
                stage=stage,
                timestamp=datetime.now(),
                input_data=input_data,
                reasoning=reasoning,
                output=output,
                confidence=confidence,
            )
        )
