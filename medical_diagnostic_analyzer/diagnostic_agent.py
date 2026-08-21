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
            "Provide rich diagnostic test rationale (clinical purpose & actionable trigger) "
            "and separate practitioner clinical decision pathways from patient supportive care. "
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
            "- recommended_next_steps: concrete clinical actions (urgency, who to see, key exams/tests)\n"
            "- diagnostic_tests: structured list of 4-7 tests across Tier 0 (Safety/Baseline), Tier 1 (Routine Labs), and Tier 2 (Definitive Imaging/Procedures). Each MUST have name, tier, clinical_purpose (what pathology or safety question it resolves), and actionable_trigger (what specific finding/threshold leads to what clinical decision)\n"
            "- conditional_therapies: 2-4 contingency therapy pathways for the practitioner (e.g. what regimen to use once test X confirms condition Y, plus supervised symptomatic antiemetics)\n"
            "- patient_supportive_care: structured patient care with dietary_guidance, hydration_and_lifestyle, medication_warnings (e.g. hold OTC PPIs before H. pylori testing, avoid NSAIDs), questions_for_doctor (3-5 specific questions), and er_warning_signs\n"
            "- escalation_triggers: 2-4 critical peritonitis/surgical/hemodynamic emergency triggers\n"
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

        # Fallback diagnostic tests
        fallback_tests = [
            DiagnosticTestItem(
                name="Complete Blood Count (CBC) & Ferritin",
                tier="Tier 1: Routine Labs",
                clinical_purpose="Evaluate for occult blood loss, microcytic anemia, or leukocytosis",
                actionable_trigger="Hb < 10 g/dL triggers expedited endoscopy; elevated WBC indicates acute inflammation",
            ),
            DiagnosticTestItem(
                name="Comprehensive Metabolic Panel & Lipase",
                tier="Tier 1: Routine Labs",
                clinical_purpose="Assess electrolyte loss from vomiting, renal function, liver enzymes, and pancreatic inflammation",
                actionable_trigger="Lipase >= 3x ULN confirms acute pancreatitis; abnormal liver enzymes prompt biliary ultrasound",
            ),
            DiagnosticTestItem(
                name="Esophagogastroduodenoscopy (EGD) with Biopsies",
                tier="Tier 2: Definitive Procedures/Imaging",
                clinical_purpose="Direct visualization of gastric/duodenal mucosa to rule out ulcers, obstruction, or malignancy",
                actionable_trigger="Ulcer identification initiates targeted therapy; suspicious mass triggers multi-quadrant biopsy",
            ),
        ]

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
            diagnostic_tests=fallback_tests,
            suggested_agent="procedure_agent",
            routing_rationale="Procedures are indicated to definitively evaluate upper GI alarm features.",
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
        Build distinct patient and practitioner documents:
        - Practitioner Report: Full clinical differential matrix, tiered diagnostic
          workup with explicit clinical purpose & actionable triggers, conditional
          pharmacotherapy pathways, and statistical appendix.
        - Patient Report: Plain-language overview, tests to expect and why,
          supportive dietary/lifestyle care, medication-hold warnings, emergency
          red-flag signs, and questions for their doctor. (No prescriptive drug tables).
        """
        # 1. Build Practitioner Report
        practitioner_sections: list[str] = [
            "# 🩺 Clinical Diagnostic & Management Protocol\n",
            "## 📊 Executive Diagnostic Summary",
            f"- **Most Probable Consideration:** {report.most_probable}",
            f"- **Highest-Severity Condition to Rule Out:** {report.most_serious}",
            f"- **Clinical Logic:** {report.reasoning_summary}\n",
            "## 📋 Differential Diagnosis Matrix",
            "| Condition | Estimated Relative Probability | Severity (1–5) | Key Clinical Rationale / Discriminators |",
            "|---|---|---|---|",
        ]
        for r in results[:8]:
            practitioner_sections.append(
                f"| **{r['name']}** | {r['probability']:.1%} | {r['severity']}/5 | {r['rationale']} |"
            )

        # Tiered diagnostic orders with purpose & triggers
        practitioner_sections.append("\n## 🔬 Tiered Diagnostic Order Set (With Rationale & Action Triggers)")
        if report.diagnostic_tests:
            # Group by tier
            tiers: dict[str, list[DiagnosticTestItem]] = {}
            for t in report.diagnostic_tests:
                tier_name = t.tier or "Tier 1: Diagnostic Workup"
                tiers.setdefault(tier_name, []).append(t)

            for tier_name, tests in tiers.items():
                practitioner_sections.append(f"\n### {tier_name}")
                for test in tests:
                    practitioner_sections.append(
                        f"- **{test.name}**\n"
                        f"  - *Clinical Purpose:* {test.clinical_purpose}\n"
                        f"  - *Actionable Decision Trigger:* {test.actionable_trigger}"
                    )
        else:
            for step in report.recommended_next_steps:
                practitioner_sections.append(f"- {step}")

        # Conditional pharmacotherapy (practitioner only)
        practitioner_sections.append("\n## 💊 Conditional Pharmacotherapy & Management Pathways")
        practitioner_sections.append(
            "> ⚠️ **Clinical Notice:** Prescriptive pharmacotherapy is held pending Tier 1/2 diagnostic confirmation, "
            "except for supervised symptomatic antiemetic and hydration support."
        )
        if report.conditional_therapies:
            for path in report.conditional_therapies:
                practitioner_sections.append(
                    f"- **If Confirmed: {path.trigger_condition}**\n"
                    f"  - *Regimen / Strategy:* {path.regimen_name}\n"
                    f"  - *Clinical Details:* {path.details}"
                )
        else:
            practitioner_sections.append(
                "- *Symptomatic Antiemetic Support:* Supervised antiemetic therapy (e.g. Ondansetron) and oral/IV electrolyte rehydration.\n"
                "- *NSAID Cessation:* Discontinue all non-steroidal anti-inflammatory agents immediately."
            )

        # Escalation triggers
        if report.escalation_triggers:
            practitioner_sections.append("\n## 🚨 Red-Flag & Escalation Triggers")
            for trigger in report.escalation_triggers:
                practitioner_sections.append(f"- {trigger}")

        # Statistical Appendix
        prob_lines = [
            f"{r['name']}: {r['probability']:.1%} (severity {r['severity']}/5)"
            for r in results[:10]
        ]
        appendix = self._build_statistical_appendix(
            {"Condition Likelihood Estimates (relative)": prob_lines}
        )
        if appendix:
            practitioner_sections.append("\n" + appendix)

        practitioner_report = "\n".join(practitioner_sections)

        # 2. Build Patient Report
        patient_sections: list[str] = [
            "# Medical Assessment & Next Steps Guide\n",
            "## ✅ Summary",
            f"- **Main Focus:** Your reported symptoms most closely match **{report.most_probable}**, while conditions such as **{report.most_serious}** are important possibilities your doctors will want to carefully rule out.",
            "- **Next Action:** Schedule an in-person medical evaluation within **48–72 hours** (or go to the emergency room immediately if severe red-flag warning signs develop).",
            "- **Testing Over Guesswork:** Diagnostic tests (such as blood work and visual imaging/endoscopy) are required to identify the root cause before starting any medications.",
            "- **At-Home Care:** Avoid over-the-counter stomach acid pills or NSAID painkillers (e.g. ibuprofen) without a doctor's guidance, eat small bland meals, and stay hydrated.\n",
            "## ⏱️ Recommended Action & Urgency",
            "🟡 **Urgent:** Schedule an in-person doctor appointment within **48–72 hours**. "
            "Proceed immediately to an emergency department if any emergency warning signs appear below.\n",
            "## 🧠 Understanding Your Symptoms (The Reasoning)",
            f"Your symptoms point primarily to upper digestive conditions like **{report.most_probable}**.",
            f"**Clinical Logic:** {report.reasoning_summary}\n",
            "## 🧪 Tests to Expect and Why",
            "Because several different conditions can cause these symptoms, diagnostic tests are needed before starting treatment:",
        ]

        if report.diagnostic_tests:
            for test in report.diagnostic_tests:
                patient_sections.append(
                    f"- **{test.name}**: {test.clinical_purpose}"
                )
        else:
            patient_sections.append(
                "- **Blood and lab tests**: To check for dehydration, blood counts, and organ health.\n"
                "- **Upper endoscopy or imaging**: To directly visualize the digestive lining and check for inflammation or sores."
            )

        care = report.patient_supportive_care
        patient_sections.append("\n## ⚠️ Important Medication & Safety Warnings")
        if care and care.medication_warnings:
            for warn in care.medication_warnings:
                patient_sections.append(f"- {warn}")
        else:
            patient_sections.append(
                "- **Do not start over-the-counter stomach acid reducers (Omeprazole, Prilosec, Nexium)** before your doctor evaluation, as they can interfere with accurate *H. pylori* germ testing.\n"
                "- **Avoid NSAID painkillers (Ibuprofen, Advil, Aleve, Aspirin)** which can irritate the stomach lining.\n"
                "- **Do not take unprescribed antibiotics or medications** without direct medical guidance."
            )

        patient_sections.append("\n## 🥗 Supportive Dietary & Daily Care (While Awaiting Your Visit)")
        if care and care.dietary_guidance:
            patient_sections.append("**Dietary Tips:**")
            for diet in care.dietary_guidance:
                patient_sections.append(f"- {diet}")
        else:
            patient_sections.append(
                "**Dietary Tips:**\n"
                "- Eat small, frequent, bland meals (e.g. broth, plain rice, bananas, oatmeal, toast).\n"
                "- Avoid spicy, fatty, highly acidic, or fried foods, as well as caffeine and alcohol."
            )

        if care and care.hydration_and_lifestyle:
            patient_sections.append("\n**Hydration & Daily Care:**")
            for life in care.hydration_and_lifestyle:
                patient_sections.append(f"- {life}")
        else:
            patient_sections.append(
                "\n**Hydration & Daily Care:**\n"
                "- Sip oral electrolyte solutions or clear liquids slowly throughout the day rather than drinking large quantities at once.\n"
                "- Remain upright for at least 60–90 minutes after eating to reduce regurgitation and acid irritation."
            )

        patient_sections.append("\n## 🚨 Emergency Warning Signs (Go to the ER Immediately)")
        if care and care.er_warning_signs:
            for er_sign in care.er_warning_signs:
                patient_sections.append(f"- **{er_sign}**")
        else:
            patient_sections.append(
                "- **Vomiting blood or dark material resembling coffee grounds**\n"
                "- **Passing black, sticky, or tarry stools**\n"
                "- **Sudden, severe stomach pain where your abdomen feels rigid or extremely tender**\n"
                "- **Inability to keep any liquids down for >24 hours, or severe dizziness and fainting**"
            )

        patient_sections.append("\n## 💬 Questions to Ask Your Doctor at Your Appointment")
        if care and care.questions_for_doctor:
            for q in care.questions_for_doctor:
                patient_sections.append(f"1. {q}")
        else:
            patient_sections.append(
                "1. *Do you recommend an upper endoscopy (camera exam) to evaluate for an ulcer or other causes?*\n"
                "2. *Should we test for H. pylori infection before starting any stomach acid medications?*\n"
                "3. *Are there specific blood tests or gallbladder scans needed based on my symptoms?*"
            )

        patient_sections.append(
            "\n---\n\n_The precise statistics, test triggers, and clinical diagnostic matrix are available in the detailed practitioner report._\n"
        )
        patient_report = "\n".join(patient_sections)

        # Guard: verify most probable and serious conditions survive in practitioner report
        self._verify_no_silent_loss(
            practitioner_report,
            [report.most_probable, report.most_serious],
            audit_step="diagnostic_layering_loss_check",
        )
        return patient_report, practitioner_report

