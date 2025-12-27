"""
LangChain-based medical fact checker.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from cost_tracker import print_cost_summary, reset_tracking, track_cost
from medical_fact_checker.medical_fact_checker_agent import (
    AnalysisPhase,
    FactCheckSession,
    OutputType,
    PhaseResult,
)

from .base import LangChainAgentBase, LangChainAgentConfig


class _Phase1Model(BaseModel):
    official_narrative: str
    counter_narrative: str
    key_conflicts: str
    references: List[str] = Field(default_factory=list)


class _Phase2Model(BaseModel):
    industry_funded_studies: str
    independent_research: str
    methodology_quality: str
    anecdotal_signals: str
    time_weighted_evidence: str
    references: List[str] = Field(default_factory=list)


class _Phase3Model(BaseModel):
    biological_truth: str
    industry_bias: str
    grey_zone: str
    references: List[str] = Field(default_factory=list)


class LangChainMedicalFactChecker(LangChainAgentBase):
    """
    LangChain-based fact checker following the same phase protocol.
    """

    def __init__(
        self,
        primary_llm_provider: str = "claude-sonnet",
        fallback_providers: Optional[List[str]] = None,
        enable_logging: bool = True,
        interactive: bool = True,
        enable_reference_validation: bool = False,
    ):
        config = LangChainAgentConfig(
            primary_llm_provider=primary_llm_provider,
            fallback_providers=fallback_providers or ["openai", "ollama"],
            enable_logging=enable_logging,
            enable_reference_validation=enable_reference_validation,
        )
        super().__init__(config)
        self.interactive = interactive
        self.current_session: Optional[FactCheckSession] = None

    def start_analysis(self, subject: str, clarifying_info: str = "") -> FactCheckSession:
        reset_tracking()
        self.current_session = FactCheckSession(subject=subject)

        phase1 = self._phase1_conflict_scan(subject, clarifying_info)
        self.current_session.phase_results.append(phase1)

        if self.interactive:
            phase1.user_choice = self._prompt_user_phase1()
        else:
            phase1.user_choice = "Both"

        phase2 = self._phase2_evidence_stress_test(subject, phase1.content, phase1.user_choice)
        self.current_session.phase_results.append(phase2)

        if self.interactive:
            phase2.user_choice = self._prompt_user_phase2()
        else:
            phase2.user_choice = "Proceed"

        phase3 = self._phase3_synthesis_menu(subject, phase1.content, phase2.content)
        self.current_session.phase_results.append(phase3)

        if self.interactive:
            phase3.user_choice = self._prompt_user_phase3()
        else:
            phase3.user_choice = "P"

        output_type = OutputType(phase3.user_choice)
        phase4 = self._phase4_generate_output(subject, phase3.content, output_type)
        self.current_session.phase_results.append(phase4)
        self.current_session.practitioner_report = phase4.content.get("output", "")

        phase5 = self._phase5_simplify_output(phase4.content.get("output", ""))
        self.current_session.phase_results.append(phase5)
        self.current_session.final_output = phase5.content.get("simplified_output", "")

        if self.enable_reference_validation and self.reference_validator:
            self.current_session.validation_report = self.reference_validator.validate_analysis(
                self.current_session
            )

        print_cost_summary()
        return self.current_session

    @track_cost("Phase 1: Conflict Scan (LangChain)")
    def _phase1_conflict_scan(self, subject: str, context: str) -> PhaseResult:
        system_prompt = (
            "You are an independent medical researcher. Return ONLY valid JSON."
        )
        user_prompt = """
Analyze the health subject below.

Subject: {subject}
Context: {context}

Return JSON with:
- official_narrative
- counter_narrative
- key_conflicts
- references (list of 3-5 citations)

Schema:
{schema}
"""
        response = self._call_llm(
            system_prompt,
            user_prompt,
            audit_step="factcheck_phase_1",
            subject=subject,
            context=context or "General investigation",
            schema=_Phase1Model.model_json_schema(),
        )
        model = self._parse_phase_model(response, _Phase1Model, subject)
        return PhaseResult(
            phase=AnalysisPhase.CONFLICT_SCAN,
            timestamp=datetime.now(),
            content={
                "official_narrative": model.official_narrative,
                "counter_narrative": model.counter_narrative,
                "key_conflicts": model.key_conflicts,
            },
            references=self._normalize_references(model.references),
        )

    @track_cost("Phase 2: Evidence Stress Test (LangChain)")
    def _phase2_evidence_stress_test(
        self, subject: str, phase1_content: Dict[str, Any], angle: str
    ) -> PhaseResult:
        system_prompt = "You are a medical research auditor. Return ONLY valid JSON."
        user_prompt = """
Evaluate evidence for the subject.

Subject: {subject}
Priority angle: {angle}
Official narrative: {official}
Counter-narrative: {counter}

Return JSON with:
- industry_funded_studies
- independent_research
- methodology_quality
- anecdotal_signals
- time_weighted_evidence
- references

Schema:
{schema}
"""
        response = self._call_llm(
            system_prompt,
            user_prompt,
            audit_step="factcheck_phase_2",
            subject=subject,
            angle=angle,
            official=phase1_content.get("official_narrative", ""),
            counter=phase1_content.get("counter_narrative", ""),
            schema=_Phase2Model.model_json_schema(),
        )
        model = self._parse_phase_model(response, _Phase2Model, subject)
        return PhaseResult(
            phase=AnalysisPhase.EVIDENCE_STRESS_TEST,
            timestamp=datetime.now(),
            content={
                "industry_funded_studies": model.industry_funded_studies,
                "independent_research": model.independent_research,
                "methodology_quality": model.methodology_quality,
                "anecdotal_signals": model.anecdotal_signals,
                "time_weighted_evidence": model.time_weighted_evidence,
            },
            references=self._normalize_references(model.references),
        )

    @track_cost("Phase 3: Synthesis (LangChain)")
    def _phase3_synthesis_menu(
        self, subject: str, phase1_content: Dict[str, Any], phase2_content: Dict[str, Any]
    ) -> PhaseResult:
        system_prompt = "You are a medical synthesizer. Return ONLY valid JSON."
        user_prompt = """
Synthesize findings into a concise summary.

Subject: {subject}
Phase 1 summary: {phase1}
Phase 2 summary: {phase2}

Return JSON with:
- biological_truth
- industry_bias
- grey_zone
- references

Schema:
{schema}
"""
        response = self._call_llm(
            system_prompt,
            user_prompt,
            audit_step="factcheck_phase_3",
            subject=subject,
            phase1=phase1_content,
            phase2=phase2_content,
            schema=_Phase3Model.model_json_schema(),
        )
        model = self._parse_phase_model(response, _Phase3Model, subject)
        return PhaseResult(
            phase=AnalysisPhase.SYNTHESIS_MENU,
            timestamp=datetime.now(),
            content={
                "biological_truth": model.biological_truth,
                "industry_bias": model.industry_bias,
                "grey_zone": model.grey_zone,
            },
            references=self._normalize_references(model.references),
        )

    @track_cost("Phase 4: Complex Output (LangChain)")
    def _phase4_generate_output(
        self, subject: str, synthesis: Dict[str, Any], output_type: OutputType
    ) -> PhaseResult:
        system_prompt = "You are a clinical researcher. Write a detailed report."
        user_prompt = """
Write a detailed report for {subject}.

Output type: {output_type}
Synthesis data: {synthesis}

Use clear headings and include a short references section.
"""
        response = self._call_llm(
            system_prompt,
            user_prompt,
            audit_step="factcheck_phase_4",
            subject=subject,
            output_type=output_type.value,
            synthesis=synthesis,
        )
        references = self._extract_references_from_text(response)
        return PhaseResult(
            phase=AnalysisPhase.COMPLEX_OUTPUT,
            timestamp=datetime.now(),
            content={"output": response, "output_type": output_type.value},
            references=references,
        )

    @track_cost("Phase 5: Simplified Output (LangChain)")
    def _phase5_simplify_output(self, complex_output: str) -> PhaseResult:
        system_prompt = (
            "You are a medical writer simplifying content for a general audience."
        )
        user_prompt = """
Simplify the following content for a general audience.

Content:
{content}
"""
        response = self._call_llm(
            system_prompt,
            user_prompt,
            audit_step="factcheck_phase_5",
            content=complex_output,
        )
        return PhaseResult(
            phase=AnalysisPhase.SIMPLIFIED_OUTPUT,
            timestamp=datetime.now(),
            content={"simplified_output": response},
        )

    def _parse_phase_model(
        self, response: str, model_cls: type[BaseModel], subject: str
    ) -> BaseModel:
        parsed = self._parse_json(response)
        if isinstance(parsed, dict):
            try:
                return model_cls.model_validate(parsed)
            except Exception:
                pass
        if model_cls is _Phase1Model:
            return _Phase1Model(
                official_narrative=f"Analysis unavailable for {subject}.",
                counter_narrative="",
                key_conflicts="",
                references=[],
            )
        if model_cls is _Phase2Model:
            return _Phase2Model(
                industry_funded_studies="",
                independent_research="",
                methodology_quality="",
                anecdotal_signals="",
                time_weighted_evidence="",
                references=[],
            )
        return _Phase3Model(
            biological_truth="",
            industry_bias="",
            grey_zone="",
            references=[],
        )

    def _normalize_references(self, references: List[str]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for ref in references:
            if not ref:
                continue
            normalized.append({"raw_citation": ref})
        return normalized

    def _extract_references_from_text(self, text: str) -> List[Dict[str, Any]]:
        references: List[Dict[str, Any]] = []
        if "REFERENCES" not in text.upper():
            return references

        lines = text.splitlines()
        capture = False
        for line in lines:
            if line.strip().upper().startswith("REFERENCES"):
                capture = True
                continue
            if capture and line.strip():
                references.append({"raw_citation": line.strip()})
        return references

    def _prompt_user_phase1(self) -> str:
        print("\nPHASE 1 COMPLETE")
        print("Choose angle: Official / Independent / Both")
        while True:
            choice = input("Your choice: ").strip().lower()
            if choice in ["official", "independent", "both"]:
                return choice.capitalize()
            print("Invalid choice.")

    def _prompt_user_phase2(self) -> str:
        print("\nPHASE 2 COMPLETE")
        print("Proceed to synthesis? Dig / Proceed")
        while True:
            choice = input("Your choice: ").strip().lower()
            if choice in ["dig", "proceed"]:
                return choice.capitalize()
            print("Invalid choice.")

    def _prompt_user_phase3(self) -> str:
        print("\nPHASE 3 COMPLETE")
        print("Select output format: A / B / C / D / P")
        while True:
            choice = input("Your choice: ").strip().upper()
            if choice in ["A", "B", "C", "D", "P"]:
                return choice
            print("Invalid choice.")
