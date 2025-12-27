"""
LangChain-based medical procedure analyzer.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, Field

from cost_tracker import print_cost_summary, reset_tracking, track_cost
from medical_procedure_analyzer.input_validation import InputValidator, ValidationError
from medical_procedure_analyzer.medical_reasoning_agent import (
    MedicalInput,
    MedicalOutput,
    OrganAnalysis,
    ReasoningStage,
    ReasoningStep,
)

from .base import LangChainAgentBase, LangChainAgentConfig


class _OrganList(BaseModel):
    organs: List[str] = Field(default_factory=list)


class _OrganAnalysisModel(BaseModel):
    organ_name: str
    affected_by_procedure: bool
    at_risk: bool
    risk_level: str
    pathways_involved: List[str] = Field(default_factory=list)
    known_recommendations: List[str] = Field(default_factory=list)
    potential_recommendations: List[str] = Field(default_factory=list)
    debunked_claims: List[str] = Field(default_factory=list)
    evidence_quality: str


class _ProcedureSummary(BaseModel):
    procedure_summary: str
    confidence_score: float
    general_recommendations: List[str] = Field(default_factory=list)
    research_gaps: List[str] = Field(default_factory=list)


class LangChainMedicalReasoningAgent(LangChainAgentBase):
    """
    LangChain-based agent for medical procedure analysis.
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

    def analyze_medical_procedure(self, medical_input: MedicalInput) -> MedicalOutput:
        self._validate_input(medical_input)
        reset_tracking()
        self.reasoning_trace = []

        organs = self._identify_organs(medical_input)
        organ_analyses = self._analyze_organs(medical_input, organs)
        summary = self._summarize_procedure(medical_input, organ_analyses)

        output = MedicalOutput(
            procedure_summary=summary.procedure_summary,
            organs_analyzed=organ_analyses,
            general_recommendations=summary.general_recommendations,
            research_gaps=summary.research_gaps,
            confidence_score=summary.confidence_score,
            reasoning_trace=self.reasoning_trace,
        )

        output.practitioner_report = self._generate_practitioner_report(output)

        if self.enable_reference_validation and self.reference_validator:
            output.validation_report = self.reference_validator.validate_analysis(output)

        print_cost_summary()
        return output

    def _validate_input(self, medical_input: MedicalInput) -> None:
        try:
            proc_result = InputValidator.validate_medical_procedure(
                medical_input.procedure
            )
            if not proc_result.is_valid:
                raise ValidationError(f"Invalid procedure: {', '.join(proc_result.errors)}")

            if medical_input.details:
                details_result = InputValidator.validate_medical_procedure(
                    medical_input.details
                )
                if not details_result.is_valid:
                    raise ValidationError(
                        f"Invalid details: {', '.join(details_result.errors)}"
                    )

            if medical_input.objectives:
                obj_result = InputValidator.validate_medical_aspects(
                    list(medical_input.objectives)
                )
                if not obj_result.is_valid:
                    raise ValidationError(
                        f"Invalid objectives: {', '.join(obj_result.errors)}"
                    )
        except ValidationError as exc:
            raise ValueError(f"Invalid medical input: {exc}") from exc

    @track_cost("Phase 1: Organ Identification (LangChain)")
    def _identify_organs(self, medical_input: MedicalInput) -> List[str]:
        system_prompt = (
            "You are a medical expert identifying organs affected by a procedure. "
            "Return ONLY valid JSON."
        )
        user_prompt = """
Identify organ systems affected by this procedure.

Procedure: {procedure}
Details: {details}
Objectives: {objectives}

Return JSON matching this schema:
{schema}
"""
        response = self._call_llm(
            system_prompt,
            user_prompt,
            procedure=medical_input.procedure,
            details=medical_input.details,
            objectives=", ".join(medical_input.objectives),
            schema=_OrganList.model_json_schema(),
        )
        parsed = self._parse_json(response)
        organs: List[str] = []

        if isinstance(parsed, dict):
            try:
                organs = _OrganList.model_validate(parsed).organs
            except Exception:
                organs = []

        if not organs:
            organs = self._fallback_organs(medical_input)

        self._log_step(
            ReasoningStage.ORGAN_IDENTIFICATION,
            {"procedure": medical_input.procedure, "details": medical_input.details},
            "Identified affected organs using LangChain prompts",
            {"organs": organs},
        )
        return [o.strip().lower() for o in organs if isinstance(o, str)]

    @track_cost("Phase 2: Organ Analysis (LangChain)")
    def _analyze_organs(
        self, medical_input: MedicalInput, organs: List[str]
    ) -> List[OrganAnalysis]:
        system_prompt = (
            "You are a medical expert producing structured organ-specific analysis. "
            "Return ONLY valid JSON."
        )
        user_prompt = """
Analyze organ-specific impact for the procedure below.

Procedure: {procedure}
Details: {details}
Organs: {organs}

Return a JSON list of objects that match this schema:
{schema}
"""
        response = self._call_llm(
            system_prompt,
            user_prompt,
            procedure=medical_input.procedure,
            details=medical_input.details,
            organs=", ".join(organs),
            schema=_OrganAnalysisModel.model_json_schema(),
        )
        parsed = self._parse_json(response)
        analyses: List[OrganAnalysis] = []

        if isinstance(parsed, list):
            for item in parsed:
                try:
                    model = _OrganAnalysisModel.model_validate(item)
                    analyses.append(
                        OrganAnalysis(
                            organ_name=model.organ_name,
                            affected_by_procedure=model.affected_by_procedure,
                            at_risk=model.at_risk,
                            risk_level=model.risk_level,
                            pathways_involved=model.pathways_involved,
                            known_recommendations=model.known_recommendations,
                            potential_recommendations=model.potential_recommendations,
                            debunked_claims=model.debunked_claims,
                            evidence_quality=model.evidence_quality,
                        )
                    )
                except Exception:
                    continue

        if not analyses:
            analyses = [
                OrganAnalysis(
                    organ_name=organ,
                    affected_by_procedure=True,
                    at_risk=True,
                    risk_level="moderate",
                    pathways_involved=[],
                    known_recommendations=[],
                    potential_recommendations=[],
                    debunked_claims=[],
                    evidence_quality="limited",
                )
                for organ in organs
            ]

        self._log_step(
            ReasoningStage.EVIDENCE_GATHERING,
            {"organs": organs},
            "Generated organ analyses using LangChain prompts",
            {"organs_count": len(analyses)},
        )
        return analyses

    @track_cost("Phase 3: Summary (LangChain)")
    def _summarize_procedure(
        self, medical_input: MedicalInput, organs: List[OrganAnalysis]
    ) -> _ProcedureSummary:
        system_prompt = (
            "You are a medical analyst summarizing findings. Return ONLY valid JSON."
        )
        user_prompt = """
Summarize the overall procedure analysis.

Procedure: {procedure}
Details: {details}
Organs analyzed: {organs}

Return JSON matching this schema:
{schema}
"""
        response = self._call_llm(
            system_prompt,
            user_prompt,
            procedure=medical_input.procedure,
            details=medical_input.details,
            organs=", ".join([o.organ_name for o in organs]),
            schema=_ProcedureSummary.model_json_schema(),
        )
        parsed = self._parse_json(response)

        if isinstance(parsed, dict):
            try:
                summary = _ProcedureSummary.model_validate(parsed)
            except Exception:
                summary = None
        else:
            summary = None

        if summary is None:
            summary = _ProcedureSummary(
                procedure_summary=f"{medical_input.procedure} - {medical_input.details}",
                confidence_score=0.7,
                general_recommendations=[],
                research_gaps=[],
            )

        self._log_step(
            ReasoningStage.CRITICAL_EVALUATION,
            {"procedure": medical_input.procedure},
            "Synthesized final summary using LangChain prompts",
            {"confidence_score": summary.confidence_score},
        )
        return summary

    def _fallback_organs(self, medical_input: MedicalInput) -> List[str]:
        details = medical_input.details.lower()
        if "contrast" in details:
            return ["kidneys", "liver", "brain"]
        return ["kidneys"]

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

    def _generate_practitioner_report(self, output: MedicalOutput) -> str:
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
- **At Risk:** {'Yes' if organ.at_risk else 'No'}
- **Evidence Quality:** {organ.evidence_quality.upper()}

**Biological Pathways Involved:**
"""
            for pathway in organ.pathways_involved:
                report += f"- {pathway}\n"

            if organ.known_recommendations:
                report += "\n**Evidence-Based Recommendations:**\n"
                for j, rec in enumerate(organ.known_recommendations, 1):
                    report += f"{j}. {rec}\n"

            if organ.potential_recommendations:
                report += "\n**Investigational Approaches:**\n"
                for j, rec in enumerate(organ.potential_recommendations, 1):
                    report += f"{j}. {rec}\n"

            if organ.debunked_claims:
                report += "\n**Debunked/Harmful Claims:**\n"
                for j, claim in enumerate(organ.debunked_claims, 1):
                    report += f"{j}. {claim}\n"

            report += "\n---\n\n"

        report += "## General Recommendations\n\n"
        for i, rec in enumerate(output.general_recommendations, 1):
            report += f"{i}. {rec}\n"

        report += "\n## Research Gaps & Future Directions\n\n"
        for i, gap in enumerate(output.research_gaps, 1):
            report += f"{i}. {gap}\n"

        report += f"""

---

**Report Generated:** {datetime.now().isoformat()}
**For Medical Professional Use Only**
"""
        return report
