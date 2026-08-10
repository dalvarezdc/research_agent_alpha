"""
LangChain-based medical diagnostic agent (Diagnostic Specialist).

Router ID: ``diagnostic_agent``.

A 5-level pipeline that uses LLM clinical reasoning (free-form symptom
extraction + common-sense differential diagnosis) rather than a fixed
symptom/disease database. Not a multi-perspective fact-checker: no
Mainstream/Naturist/Biohacker assembly.

Shares cost tracking, audit logging, robust JSON parsing, web research, and
the layered/lossless report helpers used by the other agents.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from cost_tracker import CostTracker, reset_tracking, track_cost
from langchain_agents.base import LangChainAgentBase, LangChainAgentConfig

from .dspy_schemas import (
    ConditionCandidate,
    DiagnosticReport,
    DifferentialAssessment,
    SymptomExtraction,
)

# Shared system framing for clinical reasoning steps (Levels 2 and 5).
# Identity: Diagnostic Specialist — decision-support differential only.
_CLINICAL_COMMON_SENSE = (
    "You are the Diagnostic Specialist (router id: diagnostic_agent), an "
    "experienced clinician forming a differential for decision-support "
    "(NOT a final diagnosis, NOT a medication review, NOT a procedure plan, "
    "and NOT multi-perspective fact-checking).\n"
    "Rules:\n"
    "1. GROUNDING — Ground EVERY conclusion in the patient's ACTUAL stated "
    "positive findings, explicitly denied findings, and stated context. "
    "Do not invent symptoms, timeline, or exam findings they did not describe.\n"
    "2. FREE-FORM — Use free-form clinical language; there is NO fixed symptom "
    "or disease list. Record what the patient actually describes.\n"
    "3. PRESENTATION FIDELITY — Keep the differential anatomically and "
    "pathophysiologically relevant to THIS presentation "
    "(e.g. knee pain → MSK/rheum/vascular/joint-infectious causes; "
    "NOT primary respiratory infection or migraine).\n"
    "4. DUAL METRICS — Rank by estimated relative likelihood for THIS "
    "presentation; still include important cannot-miss diagnoses even when "
    "less likely. Severity (1=benign/self-limited … 5=life- or limb-threatening) "
    "is separate from probability.\n"
    "5. HONEST UNCERTAINTY — Never claim certainty or that this replaces a "
    "clinician evaluation. Prefer 'most likely among considered options', "
    "'cannot exclude', 'consider evaluation for…'.\n"
    "6. RELATIVE PROBABILITIES — Scores are rough relative estimates among "
    "the candidates you list (not calibrated population posteriors). They "
    "should sum to approximately 1.0; the host system may renormalize.\n"
)


class MedicalDiagnosticAgent(LangChainAgentBase):
    """
    A 5-level diagnostic agent driven by LLM clinical common sense.

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
        self.logger = logging.getLogger(__name__)
        self.cost_tracker = CostTracker()

    def run_diagnostic_pipeline(self, user_query: str) -> Dict[str, Any]:
        """Executes the 5-level diagnostic pipeline."""
        reset_tracking()
        self.cost_tracker.reset()
        self.logger.info(f"Starting diagnostic pipeline for: {user_query}")
        self.web_context = self._build_web_context(user_query)

        # --- Level 1: Free-form symptom extraction ---
        extraction = self._level1_extract_symptoms(user_query)

        if extraction.is_vague and self.interactive:
            print(f"\n[Diagnostic Agent] {extraction.clarification_question}")
            new_input = input("Your response: ")
            extraction = self._level1_extract_symptoms(
                f"{user_query}. Context: {new_input}"
            )
            user_query = f"{user_query}. Context: {new_input}"

        # --- Level 2: LLM differential (common-sense clinical reasoning) ---
        assessment = self._level2_differential(user_query, extraction)
        results = self._candidates_to_probabilities(assessment.candidates)

        # --- Levels 3–4: clarifying question + iterative update (interactive only) ---
        # API / batch runs set interactive=False and skip these LLM/user steps.
        if self.interactive and (
            assessment.differentiating_symptoms or assessment.recommended_exams
        ):
            intervention_prompt = self._format_intervention_question(
                assessment.differentiating_symptoms,
                assessment.recommended_exams,
            )
            print(f"\n[Diagnostic Agent] {intervention_prompt}")
            print("Options:")
            print("  - Answer about symptoms (e.g., 'I have X but not Y')")
            print("  - Provide exam results (e.g., 'X-ray normal', 'positive Lachman')")
            print("  - Press Enter to skip and generate report")

            user_response = input("Your response: ").strip()
            if user_response:
                new_extraction = self._level1_extract_symptoms(user_response)
                extraction.symptoms = list(
                    dict.fromkeys(extraction.symptoms + new_extraction.symptoms)
                )
                extraction.negative_symptoms = list(
                    dict.fromkeys(
                        extraction.negative_symptoms + new_extraction.negative_symptoms
                    )
                )
                if new_extraction.duration:
                    extraction.duration = new_extraction.duration
                if new_extraction.severity:
                    extraction.severity = new_extraction.severity
                if new_extraction.clinical_context:
                    ctx = extraction.clinical_context or ""
                    extraction.clinical_context = (
                        f"{ctx}; {new_extraction.clinical_context}".strip("; ")
                    )

                refined_query = (
                    f"{user_query}\n\nAdditional patient information: {user_response}"
                )
                assessment = self._level2_differential(refined_query, extraction)
                results = self._candidates_to_probabilities(assessment.candidates)

        # --- Level 5: Final report & routing ---
        report = self._level5_generate_report(
            user_query, results, extraction, assessment
        )

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
            "assessment": assessment.model_dump(),
        }

    # ── Level 1: free-form symptom extraction ─────────────────────────────────
    @track_cost("Level 1: Symptom Extraction (Diagnostic)")
    def _level1_extract_symptoms(self, query: str) -> SymptomExtraction:
        system_prompt = (
            "You are the Diagnostic Specialist's extraction step (Level 1). "
            "Extract symptoms and clinical context using free-form clinical "
            "language — there is NO fixed symptom vocabulary.\n"
            "Record what the patient actually describes "
            "(e.g. 'right knee pain', 'locking', 'giving way', 'fever', "
            "'unilateral throbbing headache').\n"
            "Do not add symptoms that were not stated or clearly implied.\n"
            "Return ONLY valid JSON matching the schema."
        )
        user_prompt = (
            "User Query: {query}\n"
            "Schema: {schema}\n"
            "If a symptom is mentioned as absent, put it in negative_symptoms.\n"
            "Capture duration, severity (patient's words), and clinical_context "
            "(age, sex, trauma, meds, comorbidities, occupation, etc.) only when stated.\n"
            "Set is_vague=true only when the query is too non-specific to form "
            "any differential (e.g. 'I feel bad'). A focused complaint like "
            "'knee pain' is NOT vague. If is_vague is true, provide one polite "
            "clarification_question."
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

    # ── Level 2: common-sense differential ────────────────────────────────────
    @track_cost("Level 2: Differential Assessment (Diagnostic)")
    def _level2_differential(
        self, user_query: str, extraction: SymptomExtraction
    ) -> DifferentialAssessment:
        system_prompt = (
            f"{_CLINICAL_COMMON_SENSE}\n"
            "This is Level 2: Differential Assessment. "
            "Return ONLY valid JSON matching the schema."
        )
        user_prompt = (
            "Original patient query:\n{query}\n\n"
            "Extracted structure:\n"
            "- Positive symptoms: {symptoms}\n"
            "- Denied findings: {negative_symptoms}\n"
            "- Duration: {duration}\n"
            "- Severity: {severity}\n"
            "- Other context: {clinical_context}\n\n"
            "Build a differential diagnosis for THIS presentation only.\n"
            "Include 3–8 plausible candidates. For each: name, relative "
            "probability (0–1 among the list, roughly sum to 1.0), severity "
            "1–5 (independent of probability), and a brief rationale.\n"
            "Always include important cannot-miss items even if less likely.\n"
            "Also provide differentiating_symptoms (short history items that "
            "would best narrow the list), recommended_exams (focused maneuvers/"
            "tests), and clinical_summary (ranking logic + key uncertainties).\n"
            "If you cannot produce a reliable differential, return an empty "
            "candidates list rather than inventing unrelated conditions.\n"
            "Schema:\n{schema}"
        )
        try:
            response = self._call_llm(
                system_prompt,
                user_prompt,
                audit_step="diagnostic_level2_differential",
                query=user_query,
                symptoms=", ".join(extraction.symptoms) or "(none extracted)",
                negative_symptoms=(
                    ", ".join(extraction.negative_symptoms) or "(none stated)"
                ),
                duration=extraction.duration or "not specified",
                severity=extraction.severity or "not specified",
                clinical_context=extraction.clinical_context or "not specified",
                schema=json.dumps(DifferentialAssessment.model_json_schema()),
            )
            parsed = self._parse_json(response)
            if isinstance(parsed, dict):
                assessment = DifferentialAssessment.model_validate(parsed)
                if assessment.candidates:
                    return assessment
        except Exception as e:
            self.logger.error(f"Level 2 differential failed: {e}")

        # Safe fallback: fail closed — do not invent unrelated conditions.
        return self._fallback_differential()

    @staticmethod
    def _fallback_differential() -> DifferentialAssessment:
        """Fail-closed Level 2 result when the model cannot build a differential."""
        return DifferentialAssessment(
            candidates=[
                ConditionCandidate(
                    name="Undifferentiated presentation — clinician evaluation needed",
                    probability=1.0,
                    severity=2,
                    rationale=(
                        "Automatic differential generation failed or returned no "
                        "candidates. The original presentation should be assessed "
                        "by a clinician rather than guessed."
                    ),
                )
            ],
            differentiating_symptoms=[],
            recommended_exams=["In-person clinical assessment"],
            clinical_summary=(
                "Could not build a reliable differential from the model output. "
                "Recommend direct clinical evaluation."
            ),
        )

    # ── Level 3: intervention question phrasing (interactive only in practice) ─
    @track_cost("Level 3: Differentiation (Diagnostic)")
    def _format_intervention_question(
        self, diff_symptoms: List[str], exams: List[str]
    ) -> str:
        if not diff_symptoms and not exams:
            return "Is there anything else you would like to add?"
        system_prompt = (
            "You are the Diagnostic Specialist writing one short clarifying "
            "question for a patient. Be polite, plain-language, and non-alarmist. "
            "Do not diagnose. Do not list more than one combined question."
        )
        user_prompt = (
            "The following history points and exams would help narrow the "
            "differential (decision-support only):\n"
            "History / symptoms to clarify: {symptoms}\n"
            "Exams / tests: {exams}\n\n"
            "Write a single, polite, patient-friendly question asking whether "
            "they have any of these symptoms or already know any of these results."
        )
        try:
            response = self._call_llm(
                system_prompt,
                user_prompt,
                audit_step="diagnostic_level3_question",
                symptoms=", ".join(diff_symptoms) or "none",
                exams=", ".join(exams) or "none",
            )
            return response.strip()
        except Exception as e:
            self.logger.error(f"Intervention question generation failed: {e}")
            return "Do you have any of these additional symptoms or test results?"

    # ── Level 5: report synthesis ─────────────────────────────────────────────
    @track_cost("Level 5: Report Generation (Diagnostic)")
    def _level5_generate_report(
        self,
        user_query: str,
        results: List[Dict[str, Any]],
        extraction: SymptomExtraction,
        assessment: DifferentialAssessment,
    ) -> DiagnosticReport:
        top_candidates = results[:5]
        if not top_candidates:
            return DiagnosticReport(
                top_5_candidates=[],
                most_probable="Unable to determine",
                most_serious="Unable to determine",
                reasoning_summary=(
                    "No candidate conditions were available to generate a report."
                ),
                recommended_next_steps=["Consult with a healthcare professional."],
                suggested_agent="medication_agent",
                routing_rationale="General follow-up.",
            )

        most_probable = top_candidates[0]
        most_serious = max(top_candidates, key=lambda x: x["severity"])

        system_prompt = (
            f"{_CLINICAL_COMMON_SENSE}\n"
            "This is Level 5: Structured Diagnostic Report. "
            "Generate JSON from the provided differential and presentation only. "
            "Do not invent new conditions or unstated findings. "
            "Return ONLY valid JSON matching the schema."
        )
        user_prompt = (
            "Original patient query (must stay faithful to this):\n{query}\n\n"
            "Diagnostic data (host-ranked; probabilities already relative/"
            "normalized among candidates):\n"
            "Top candidates: {top_candidates}\n"
            "Most Probable (highest relative likelihood): {most_probable} "
            "({most_probable_pct})\n"
            "Most Serious (highest severity among top candidates): {most_serious} "
            "(Severity: {most_serious_sev}/5)\n"
            "Clinical summary from differential: {clinical_summary}\n\n"
            "Extracted symptoms: {symptoms}\n"
            "Denied findings: {negative_symptoms}\n"
            "Duration: {duration}\n"
            "Context: {clinical_context}\n\n"
            "Fill the schema fields carefully:\n"
            "- top_5_candidates: names from the ranked list\n"
            "- most_probable / most_serious: use the host-provided labels above "
            "unless the list is empty\n"
            "- reasoning_summary: how THIS presentation drove the ranking\n"
            "- recommended_next_steps: concrete actions (urgency, who to see, tests)\n"
            "- suggested_agent: 'medication_agent' if follow-up is mainly "
            "pharmacologic, or 'procedure_agent' if interventional/procedural "
            "workup or treatment is central (routing hint only)\n"
            "- routing_rationale: one short justification for suggested_agent\n"
            "- references: 3–6 APA 7 citations for THIS presentation; each MUST "
            "include a DOI, PMID, or direct URL\n\n"
            "Schema:\n{schema}"
        )
        try:
            response = self._call_llm(
                system_prompt,
                user_prompt,
                audit_step="diagnostic_level5_report",
                query=user_query,
                top_candidates=json.dumps(top_candidates, indent=2),
                most_probable=most_probable["name"],
                most_probable_pct=f"{most_probable['probability']:.2%}",
                most_serious=most_serious["name"],
                most_serious_sev=most_serious["severity"],
                clinical_summary=assessment.clinical_summary,
                symptoms=", ".join(extraction.symptoms) or "(none extracted)",
                negative_symptoms=(
                    ", ".join(extraction.negative_symptoms) or "(none stated)"
                ),
                duration=extraction.duration or "not specified",
                clinical_context=extraction.clinical_context or "not specified",
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
            reasoning_summary=(
                assessment.clinical_summary
                or "Based on your symptoms and clinical probability estimates."
            ),
            recommended_next_steps=[
                "Consult with a healthcare professional for examination and next steps."
            ]
            + [f"Consider: {e}" for e in assessment.recommended_exams[:3]],
            suggested_agent="medication_agent",
            routing_rationale="General clinical follow-up.",
        )

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _candidates_to_probabilities(
        candidates: List[ConditionCandidate],
    ) -> List[Dict[str, Any]]:
        """Convert Pydantic candidates into normalized probability dicts."""
        if not candidates:
            return []

        raw = [max(0.0, float(c.probability)) for c in candidates]
        total = sum(raw)
        if total <= 0:
            raw = [1.0] * len(candidates)
            total = float(len(candidates))

        results: List[Dict[str, Any]] = []
        for cand, score in zip(candidates, raw):
            results.append(
                {
                    "id": cand.name.lower().replace(" ", "_")[:64],
                    "name": cand.name,
                    "severity": int(cand.severity),
                    "probability": score / total,
                    "rationale": cand.rationale,
                }
            )
        return sorted(results, key=lambda x: x["probability"], reverse=True)

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
        Build patient + practitioner layered documents. Estimated likelihoods
        are placed in a deterministic Statistical Appendix so they are
        guaranteed present in the practitioner report.
        """
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
            "reassuring, clear decision-support tone (not a final diagnosis). "
            "Conclusions first: most likely possibility, most serious cannot-miss "
            "to rule out, and what to do next. Then brief reasoning. Stay faithful "
            "to the conditions listed — do not introduce unrelated diagnoses. "
            "Do not invent multi-perspective (mainstream/naturist/biohacker) sections."
        )
        try:
            plain_layers = self._layer_plain_language(
                source, framing=framing, audit_step="diagnostic_layering"
            )
        except Exception as e:
            self.logger.error(f"Diagnostic layering call failed: {e}")
            plain_layers = source

        # Statistical Appendix (deterministic): estimated relative likelihoods.
        prob_lines = [
            f"{r['name']}: {r['probability']:.1%} (severity {r['severity']}/5)"
            for r in results[:10]
        ]
        appendix = self._build_statistical_appendix(
            {"Condition Likelihood Estimates (relative)": prob_lines}
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
