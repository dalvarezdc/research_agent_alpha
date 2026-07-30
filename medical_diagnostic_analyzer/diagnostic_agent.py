"""
LangChain-based medical diagnostic agent.

A 5-level pipeline that combines LLM NLP (symptom extraction, question phrasing,
report synthesis) with a deterministic Naive Bayes engine. Migrated onto
``LangChainAgentBase`` so it shares cost tracking, audit logging, robust JSON
parsing, web research, and the layered/lossless report helpers used by the other
agents.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from cost_tracker import CostTracker, reset_tracking, track_cost
from langchain_agents.base import LangChainAgentBase, LangChainAgentConfig

from .bayesian_engine import NaiveBayesDiagnosticEngine
from .dspy_schemas import DiagnosticReport, SymptomExtraction


class MedicalDiagnosticAgent(LangChainAgentBase):
    """
    A 5-level diagnostic agent that combines LLM NLP with Bayesian math.

    Public interface is preserved for the orchestrator:
        __init__(primary_llm_provider, fallback_providers, enable_logging, interactive)
        run_diagnostic_pipeline(user_query) -> {"extraction", "probabilities", "report"}
    """

    def __init__(
        self,
        primary_llm_provider: str = "claude-sonnet",
        fallback_providers: Optional[List[str]] = None,
        enable_logging: bool = True,
        interactive: bool = True,
        enable_web_research: bool = False,
    ):
        config = LangChainAgentConfig(
            primary_llm_provider=primary_llm_provider,
            fallback_providers=fallback_providers or ["openai", "ollama"],
            enable_logging=enable_logging,
            enable_web_research=enable_web_research,
        )
        super().__init__(config)
        self.interactive = interactive
        self.engine = NaiveBayesDiagnosticEngine()
        self.logger = logging.getLogger(__name__)
        self.cost_tracker = CostTracker()

    def run_diagnostic_pipeline(self, user_query: str) -> Dict[str, Any]:
        """Executes the 5-level diagnostic pipeline."""
        reset_tracking()
        self.cost_tracker.reset()
        self.logger.info(f"Starting diagnostic pipeline for: {user_query}")
        self.web_context = self._build_web_context(user_query)

        # --- Level 1: Symptom Extraction ---
        extraction = self._level1_extract_symptoms(user_query)

        if extraction.is_vague and self.interactive:
            print(f"\n[Diagnostic Agent] {extraction.clarification_question}")
            new_input = input("Your response: ")
            extraction = self._level1_extract_symptoms(
                f"{user_query}. Context: {new_input}"
            )

        # --- Level 2: Initial Bayesian Scoring (deterministic) ---
        results = self.engine.calculate_probabilities(
            extraction.symptoms, extraction.negative_symptoms
        )

        # --- Level 3: Differentiating Questions & Exams ---
        diff_symptoms = self.engine.get_differentiating_symptoms(
            results, extraction.symptoms
        )
        recommended_exams = self.engine.get_recommended_exams(results)

        exam_names = [e["name"] for e in recommended_exams]
        intervention_prompt = self._format_intervention_question(
            diff_symptoms, exam_names
        )

        # --- Level 4: Iterative Update (Interactive) ---
        if self.interactive and (diff_symptoms or recommended_exams):
            print(f"\n[Diagnostic Agent] {intervention_prompt}")
            print("Options:")
            print("  - Answer about symptoms (e.g., 'I have X but not Y')")
            print("  - Provide exam results (e.g., 'Positive Strep Test')")
            print("  - Press Enter to skip and generate report")

            user_response = input("Your response: ").strip()
            if user_response:
                new_extraction = self._level1_extract_symptoms(user_response)
                extraction.symptoms.extend(new_extraction.symptoms)
                extraction.negative_symptoms.extend(new_extraction.negative_symptoms)

                for exam in recommended_exams:
                    if (
                        exam["name"].lower() in user_response.lower()
                        or exam["id"].lower() in user_response.lower()
                    ):
                        is_pos = (
                            "positive" in user_response.lower()
                            or "yes" in user_response.lower()
                        )
                        results = self.engine.update_with_exam_result(
                            results, exam["id"], is_pos
                        )

                results = self.engine.calculate_probabilities(
                    extraction.symptoms, extraction.negative_symptoms
                )

        # --- Level 5: Final Report & Routing ---
        report = self._level5_generate_report(results, extraction)

        # Layered, lossless report (Conclusions → Reasoning → Statistical Appendix).
        patient_report, practitioner_report = self._build_diagnostic_layered_reports(
            report, results
        )

        # Sync cost tracker.
        from cost_tracker import get_cost_summary as _module_summary
        self.cost_tracker._phase_costs = _module_summary()["phases"][:]

        return {
            "extraction": extraction.model_dump(),
            "probabilities": results,
            "report": report.model_dump(),
            "references": [r.get("raw_citation", "") for r in self._references(report)],
            "patient_report": patient_report,
            "practitioner_report": practitioner_report,
        }

    # ── Level 1: symptom extraction ───────────────────────────────────────────
    @track_cost("Level 1: Symptom Extraction (Diagnostic)")
    def _level1_extract_symptoms(self, query: str) -> SymptomExtraction:
        system_prompt = (
            "You are a medical NLP specialist. Extract symptoms from the user query.\n"
            f"Available symptoms in database: {', '.join(self.engine.all_symptoms)}\n"
            "Return ONLY valid JSON matching the schema."
        )
        user_prompt = (
            "User Query: {query}\n"
            "Schema: {schema}\n"
            "Only use symptoms from the available list if they match. If a symptom is "
            "mentioned as absent, put it in negative_symptoms."
        )
        try:
            response = self._call_llm(
                system_prompt,
                user_prompt,
                audit_step="diagnostic_level1_extraction",
                query=query,
                schema=json.dumps(SymptomExtraction.model_json_schema()),
            )
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            return SymptomExtraction(
                symptoms=[],
                is_vague=True,
                clarification_question=(
                    "I encountered an error while processing your request. "
                    "Could you please describe your symptoms again?"
                ),
            )

        parsed = self._parse_json(response)
        if isinstance(parsed, dict):
            try:
                return SymptomExtraction.model_validate(parsed)
            except Exception as e:
                self.logger.error(f"Failed to validate Level 1 response: {e}")

        return SymptomExtraction(
            symptoms=[],
            is_vague=True,
            clarification_question="Could you describe your symptoms in more detail?",
        )

    # ── Level 3: intervention question phrasing ───────────────────────────────
    @track_cost("Level 3: Differentiation (Diagnostic)")
    def _format_intervention_question(
        self, diff_symptoms: List[str], exams: List[str]
    ) -> str:
        if not diff_symptoms and not exams:
            return "Is there anything else you would like to add?"
        system_prompt = "You are a helpful medical assistant."
        user_prompt = (
            "I have calculated that the following symptoms and exams would help "
            "differentiate the diagnosis:\n"
            "Symptoms: {symptoms}\n"
            "Exams: {exams}\n\n"
            "Create a single, polite, patient-friendly question asking if they have "
            "these symptoms or results."
        )
        try:
            response = self._call_llm(
                system_prompt,
                user_prompt,
                audit_step="diagnostic_level3_question",
                symptoms=", ".join(diff_symptoms),
                exams=", ".join(exams),
            )
            return response.strip()
        except Exception as e:
            self.logger.error(f"Intervention question generation failed: {e}")
            return "Do you have any of these additional symptoms or test results?"

    # ── Level 5: report synthesis ─────────────────────────────────────────────
    @track_cost("Level 5: Report Generation (Diagnostic)")
    def _level5_generate_report(
        self, results: List[Dict[str, Any]], extraction: SymptomExtraction
    ) -> DiagnosticReport:
        top_candidates = results[:5]
        most_probable = top_candidates[0]
        most_serious = max(top_candidates, key=lambda x: x["severity"])

        system_prompt = (
            "You are a senior diagnostic physician. Generate a structured report "
            "based on the provided mathematical data. Return ONLY valid JSON."
        )
        user_prompt = (
            "Diagnostic Data:\n"
            "Top 5 Candidates: {top_candidates}\n"
            "Most Probable: {most_probable} ({most_probable_pct})\n"
            "Most Serious: {most_serious} (Severity: {most_serious_sev}/5)\n\n"
            "Extracted Symptoms: {symptoms}\n"
            "Duration: {duration}\n\n"
            "Generate a report following the schema:\n{schema}\n\n"
            "For 'references': provide 3-6 APA 7 citations supporting the reasoning, "
            "each MUST include a DOI, PMID, or direct URL.\n"
            "Ensure 'suggested_agent' is 'medication_agent' if the solution is "
            "drug-based, or 'procedure_agent' if it requires interventional treatment."
        )
        try:
            response = self._call_llm(
                system_prompt,
                user_prompt,
                audit_step="diagnostic_level5_report",
                top_candidates=json.dumps(top_candidates, indent=2),
                most_probable=most_probable["name"],
                most_probable_pct=f"{most_probable['probability']:.2%}",
                most_serious=most_serious["name"],
                most_serious_sev=most_serious["severity"],
                symptoms=", ".join(extraction.symptoms),
                duration=extraction.duration or "not specified",
                schema=json.dumps(DiagnosticReport.model_json_schema()),
            )
            parsed = self._parse_json(response)
            if isinstance(parsed, dict):
                return DiagnosticReport.model_validate(parsed)
        except Exception as e:
            self.logger.error(f"Failed to parse Level 5 report: {e}")

        return DiagnosticReport(
            top_5_candidates=[c["name"] for c in top_candidates],
            most_probable=most_probable["name"],
            most_serious=most_serious["name"],
            reasoning_summary="Based on your symptoms and clinical probability models.",
            recommended_next_steps=["Consult with a healthcare professional."],
            suggested_agent="medication_agent",
            routing_rationale="General follow-up.",
        )

    # ── Layered report + references helpers ────────────────────────────────────
    def _references(self, report: DiagnosticReport) -> List[Dict[str, str]]:
        """Normalize DiagnosticReport.references into raw_citation dicts."""
        refs = getattr(report, "references", None) or []
        return [
            {"raw_citation": c.strip()}
            for c in refs
            if isinstance(c, str) and c.strip()
        ]

    def _build_diagnostic_layered_reports(
        self, report: DiagnosticReport, results: List[Dict[str, Any]]
    ) -> tuple[str, str]:
        """
        Build patient + practitioner layered documents. The precise Bayesian
        probabilities are placed in a deterministic Statistical Appendix so they
        are guaranteed present in the practitioner report, not just the console.
        """
        # Deterministic source for the plain-language layers.
        source = (
            f"Most probable condition: {report.most_probable}\n"
            f"Most serious condition to rule out: {report.most_serious}\n"
            f"Top candidates: {', '.join(report.top_5_candidates)}\n"
            f"Reasoning: {report.reasoning_summary}\n"
            f"Recommended next steps:\n"
            + "\n".join(f"- {s}" for s in report.recommended_next_steps)
            + f"\nSuggested follow-up: {report.suggested_agent} "
            f"({report.routing_rationale})\n"
        )
        framing = (
            "reassuring, clear tone. Explain what the most likely and most serious "
            "possibilities are and what to do next, in plain words."
        )
        try:
            plain_layers = self._layer_plain_language(
                source, framing=framing, audit_step="diagnostic_layering"
            )
        except Exception as e:
            self.logger.error(f"Diagnostic layering call failed: {e}")
            plain_layers = source

        # Statistical Appendix (deterministic): exact posterior probabilities.
        prob_lines = [
            f"{r['name']}: {r['probability']:.1%} (severity {r['severity']}/5)"
            for r in results[:10]
        ]
        appendix = self._build_statistical_appendix(
            {"Condition Probabilities (Bayesian posteriors)": prob_lines}
        )

        patient, practitioner = self._build_layered_report(
            conclusions_and_reasoning=plain_layers,
            appendix=appendix,
        )

        # Guard: the most probable/serious conditions must survive.
        self._verify_no_silent_loss(
            practitioner,
            [report.most_probable, report.most_serious],
            audit_step="diagnostic_layering_loss_check",
        )
        return patient, practitioner
