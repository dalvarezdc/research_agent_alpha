"""
LangChain-based medication analyzer.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from cost_tracker import print_cost_summary, reset_tracking, track_cost, CostTracker
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
    what_not_to_do: List[Dict[str, str]] = Field(default_factory=list)
    debunked_claims: List[Dict[str, str]] = Field(default_factory=list)
    monitoring_requirements: List[str] = Field(default_factory=list)
    warning_signs: List[Dict[str, str]] = Field(default_factory=list)
    evidence_quality: str = "moderate"
    analysis_confidence: float = 0.75
    references: List[str] = Field(default_factory=list)


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
        enable_web_research: bool = False,
    ):
        config = LangChainAgentConfig(
            primary_llm_provider=primary_llm_provider,
            fallback_providers=fallback_providers
            or ["claude-sonnet", "grok-4.3", "openai", "ollama"],
            enable_logging=enable_logging,
            enable_reference_validation=enable_reference_validation,
            enable_web_research=enable_web_research,
        )
        super().__init__(config)
        self.reasoning_trace: List[ReasoningStep] = []
        self.cost_tracker = CostTracker()

    def analyze_medication(self, medication_input: MedicationInput) -> MedicationOutput:
        reset_tracking()
        self.cost_tracker.reset()
        self.reasoning_trace = []
        self.web_context = self._build_web_context(medication_input.medication_name)

        output_model = self._generate_medication_output(medication_input)
        output = self._to_dataclass(output_model)
        output.reasoning_trace = self.reasoning_trace

        # Build layered patient + practitioner reports (Conclusions → Reasoning →
        # Statistical Appendix) using the shared base helpers.
        self._build_layered_medication_reports(output)

        if self.enable_reference_validation and self.reference_validator:
            output.validation_report = self.reference_validator.validate_analysis(output)

        from cost_tracker import get_cost_summary as _module_summary
        self.cost_tracker._phase_costs = _module_summary()["phases"][:]
        self.cost_tracker.print_summary()
        return output

    @track_cost("Medication Layered Report (LangChain)")
    def _build_layered_medication_reports(self, output: MedicationOutput) -> None:
        """
        Populate ``output.patient_report`` and ``output.practitioner_report`` with
        layered documents (Conclusions → Reasoning → Statistical Appendix).

        The plain-language Conclusions+Reasoning layers are produced by one LLM
        call; the Statistical Appendix (evidence grading, per-interaction evidence
        levels, confidence) is assembled deterministically so numbers are never
        dropped.
        """
        # Deterministic source content passed to the plain-language layering call.
        def _fmt_recs(items: list) -> str:
            out = []
            for it in items:
                if isinstance(it, dict):
                    label = it.get("intervention") or it.get("action") or it.get("claim") or ""
                    detail = it.get("rationale") or it.get("reason_debunked") or it.get("risks") or ""
                    out.append(f"- {label}. {detail}".strip())
                else:
                    out.append(f"- {it}")
            return "\n".join(out)

        source = (
            f"Medication: {output.medication_name} ({output.drug_class})\n"
            f"Mechanism of action: {output.mechanism_of_action}\n"
            f"Key indications: {', '.join(output.approved_indications) or 'not established'}\n"
            f"Black box warnings: {', '.join(output.black_box_warnings) or 'none'}\n"
            f"Standard dosing: {output.standard_dosing or 'not established'}\n"
            f"Pharmacokinetics: absorption {output.absorption}; metabolism "
            f"{output.metabolism}; elimination {output.elimination}; half-life {output.half_life}\n"
            f"Serious adverse effects: {', '.join(output.serious_adverse_effects) or 'not established'}\n"
            f"Common adverse effects: {', '.join(output.common_adverse_effects) or 'not established'}\n"
            f"What to do:\n{_fmt_recs(output.evidence_based_recommendations)}\n"
            f"What not to do:\n{_fmt_recs(output.what_not_to_do)}\n"
            f"Debunked claims:\n{_fmt_recs(output.debunked_claims)}\n"
            f"Monitoring: {', '.join(output.monitoring_requirements) or 'not established'}\n"
        )

        framing = (
            "clinical, evidence-graded tone. Emphasize safety and what the patient "
            "should do, in plain words."
        )
        plain_layers = self._layer_plain_language(source, framing=framing,
                                                  audit_step="medication_layering")

        # Statistical Appendix (deterministic). Critical safety info is included
        # here verbatim so it is guaranteed present regardless of LLM phrasing.
        appendix_sections: dict[str, list[str]] = {}
        if output.black_box_warnings:
            appendix_sections["⚠️ Black Box Warnings"] = list(output.black_box_warnings)
        if output.serious_adverse_effects:
            appendix_sections["Serious Adverse Effects"] = list(output.serious_adverse_effects)
        interaction_lines = []
        for label, items in (
            ("Drug interactions", output.drug_interactions),
            ("Food interactions", output.food_interactions),
        ):
            for inter in items:
                sev = getattr(getattr(inter, "severity", None), "value", "") or ""
                ev = getattr(inter, "evidence_level", "") or ""
                agent_name = getattr(inter, "interacting_agent", "") or ""
                effect = getattr(inter, "clinical_effect", "") or ""
                entry = f"{label}: {agent_name} — {effect}"
                if sev:
                    entry += f" (severity: {sev})"
                if ev:
                    entry += f" [evidence: {ev}]"
                interaction_lines.append(entry)
        if interaction_lines:
            appendix_sections["Interaction Evidence"] = interaction_lines

        grading = [
            f"Overall evidence quality: {output.evidence_quality}",
            f"Analysis confidence: {output.analysis_confidence:.2f}/1.00",
        ]
        if output.dose_adjustments:
            for k, v in output.dose_adjustments.items():
                grading.append(f"Dose adjustment ({k}): {v}")
        appendix_sections["Evidence Quality & Grading"] = grading

        appendix = self._build_statistical_appendix(appendix_sections)

        patient, practitioner = self._build_layered_report(
            conclusions_and_reasoning=plain_layers,
            appendix=appendix,
        )

        # Verification guard: black box warnings and serious adverse effects must
        # survive into the practitioner layer.
        must_survive = list(output.black_box_warnings) + list(output.serious_adverse_effects)
        self._verify_no_silent_loss(
            practitioner, must_survive, audit_step="medication_layering_loss_check"
        )

        output.patient_report = patient
        output.practitioner_report = practitioner

    @track_cost("Medication Analysis (LangChain)")
    def _generate_medication_output(
        self, medication_input: MedicationInput
    ) -> _MedicationOutputModel:
        system_prompt = (
            "You are a clinical pharmacist producing structured medication analysis. "
            "Return ONLY valid JSON."
        )
        _doc_ctx_block = (
            "Document context (from an attached file):\n{document_context}\n"
            if self.document_context
            else ""
        )
        user_prompt = """
Analyze the medication below and return a comprehensive structured report.

Medication: {medication}
Indication: {indication}
Other medications: {other_meds}
Web research context:
{web_context}
""" + _doc_ctx_block + """
Return JSON matching this schema:
{schema}
"""
        user_prompt += """

Recommendations guidance:
- "evidence_based_recommendations": evidence-based WHAT TO DO actions with rationale, evidence_level,
  implementation, expected_outcome, and monitoring when possible.
- "what_not_to_do": evidence-based WHAT NOT TO DO actions with risks and safer alternatives.
- "debunked_claims": false or misleading public beliefs about the medicine. A debunked claim must be:
  1) a specific, commonly repeated statement,
  2) contradicted by labeling/guidelines/trials/large reviews,
  3) distinct from behavior advice (avoid overlap with WHAT NOT TO DO).
  Provide claim, reason_debunked, evidence, why_harmful, and debunked_by when possible.
Requirements:
- Do not leave "intervention" or "action" blank. Use a short imperative sentence.
- Avoid "N/A". If unknown, write "not established" with a brief rationale.
- "references": provide 3-8 APA 7 citations supporting the analysis, each MUST
  include a DOI, PMID, or direct URL.
"""
        if self._is_grok():
            user_prompt += """

Grok-specific requirements:
- Do not omit any section in the schema.
- Provide ≥3 items for list fields when applicable (interactions, adverse effects, recommendations).
- Ensure contraindications and warning signs are populated with realistic clinical content.
- Use evidence qualifiers (e.g., strong/moderate/limited) instead of leaving empty.
"""
        _call_kwargs: dict = dict(
            audit_step="medication_analysis",
            medication=medication_input.medication_name,
            indication=medication_input.indication or "not specified",
            other_meds=", ".join(medication_input.patient_medications) or "None",
            web_context=self.web_context or "None",
            schema=_MedicationOutputModel.model_json_schema(),
        )
        if self.document_context:
            _call_kwargs["document_context"] = self.document_context
        response = self._call_llm(
            system_prompt,
            user_prompt,
            **_call_kwargs,
        )
        parsed = self._parse_json(response)

        if not isinstance(parsed, dict):
            raise RuntimeError(
                f"Medication analysis failed: LLM response could not be parsed as JSON. "
                f"Response length: {len(response)} chars. "
                f"This usually means the response was truncated (max_tokens too low) "
                f"or the model returned plain text instead of JSON. "
                f"Response preview: {response[:200]!r}"
            )

        try:
            model = _MedicationOutputModel.model_validate(parsed)
        except Exception as exc:
            raise RuntimeError(
                f"Medication analysis failed: LLM returned valid JSON but it did not match "
                f"the expected schema for {medication_input.medication_name}. "
                f"Validation error: {exc}. "
                f"JSON keys returned: {list(parsed.keys())}"
            ) from exc

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
            what_not_to_do=model.what_not_to_do,
            debunked_claims=model.debunked_claims,
            monitoring_requirements=model.monitoring_requirements,
            warning_signs=model.warning_signs,
            evidence_quality=model.evidence_quality,
            analysis_confidence=model.analysis_confidence,
            reasoning_trace=self.reasoning_trace,
            references=[
                {"raw_citation": c.strip()}
                for c in model.references
                if c and c.strip()
            ],
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
