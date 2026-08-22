"""
LangChain-based medical fact checker.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from cost_tracker import print_cost_summary, reset_tracking, track_cost, CostTracker
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


class PerspectiveLens(Enum):
    """User-chosen perspective lens for Phase 4 and Phase 5."""
    MAINSTREAM = "M"   # Evidence-based medicine and clinical guidelines
    NATURIST   = "N"   # Evolutionary biology and ancestral health
    BIOHACKER  = "B"   # Optimization, cutting-edge, n=1 protocols
    BALANCED   = "A"   # All perspectives weighted equally (default)


class _PerspectiveOutput(BaseModel):
    """Output from one perspective agent in Phase 4."""
    findings: str
    recommendations: List[str] = Field(default_factory=list)
    key_insight: str = ""
    citations: List[str] = Field(default_factory=list)
    # Precise quantitative evidence (effect sizes, CIs, p-values, sample sizes).
    # Kept separate so it can be surfaced verbatim in the practitioner Statistical
    # Appendix instead of being stripped when simplifying for a general audience.
    statistical_details: List[str] = Field(default_factory=list)


class _Phase4PerspectivesModel(BaseModel):
    """Assembled output from all three perspective agents."""
    mainstream: _PerspectiveOutput
    naturist: _PerspectiveOutput
    biohacker: _PerspectiveOutput


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
        self.interactive = interactive
        self.current_session: Optional[FactCheckSession] = None
        self.cost_tracker = CostTracker()

    def start_analysis(self, subject: str, clarifying_info: str = "") -> FactCheckSession:
        reset_tracking()
        self.cost_tracker.reset()
        self.current_session = FactCheckSession(subject=subject)
        self.web_context = self._build_web_context(subject)

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

        # Pick perspective lens (replaces old A/B/C/D/P output-type selection)
        if self.interactive:
            lens_str = self._prompt_user_lens()
        else:
            lens_str = PerspectiveLens.BALANCED.value  # "A"

        try:
            lens = PerspectiveLens(lens_str)
        except ValueError:
            lens = PerspectiveLens.BALANCED
        phase3.user_choice = lens.value

        # Phase 4: three parallel perspective agents → assembled report
        phase4 = self._phase4_generate_output(subject, phase3.content, lens)
        self.current_session.phase_results.append(phase4)

        # Split body from references before Phase 5 so citations survive simplification
        assembled = phase4.content.get("output", "")
        ref_separator = "\n## 📚 References\n"
        if ref_separator in assembled:
            body, refs_block = assembled.split(ref_separator, 1)
        else:
            body, refs_block = assembled, ""

        # Phase 5: build layered (Conclusions → Reasoning → Statistical Appendix)
        # documents. Pass structured upstream content so nothing is silently lost.
        phase5 = self._phase5_simplify_output(
            body,
            lens=lens,
            phase2_content=phase2.content,
            perspectives=phase4.content.get("perspectives"),
        )
        self.current_session.phase_results.append(phase5)

        # Practitioner report: layered doc (incl. Statistical Appendix) if produced,
        # else fall back to the full assembled Phase 4 report.
        practitioner_body = phase5.content.get("practitioner_layered") or body
        practitioner_report = practitioner_body
        if refs_block:
            practitioner_report = practitioner_report + ref_separator + refs_block
        self.current_session.practitioner_report = practitioner_report

        # Patient report: plain-language layers. Reattach references verbatim —
        # they are never touched by Phase 5.
        simplified = phase5.content.get("simplified_output", body)
        if refs_block:
            simplified = simplified + ref_separator + refs_block
        self.current_session.final_output = simplified

        if self.enable_reference_validation and self.reference_validator:
            self.current_session.validation_report = self.reference_validator.validate_analysis(
                self.current_session
            )

        from cost_tracker import get_cost_summary as _module_summary
        self.cost_tracker._phase_costs = _module_summary()["phases"][:]
        self.cost_tracker.print_summary()
        return self.current_session

    @track_cost("Phase 1: Conflict Scan (LangChain)")
    def _phase1_conflict_scan(self, subject: str, context: str) -> PhaseResult:
        system_prompt = (
            "You are an independent medical researcher. Return ONLY valid JSON."
        )
        _doc_ctx_block = (
            "Document context (from an attached file):\n{document_context}\n"
            if self.document_context
            else ""
        )
        user_prompt = """
Analyze the health subject below.

Subject: {subject}
Context: {context}
Web research context:
{web_context}
""" + _doc_ctx_block + """
Return JSON with:
- official_narrative
- counter_narrative
- key_conflicts
- references (list of 3-5 citations)

Schema:
{schema}
"""
        _call_kwargs: dict = dict(
            audit_step="factcheck_phase_1",
            subject=subject,
            context=context or "General investigation",
            web_context=self.web_context or "None",
            schema=_Phase1Model.model_json_schema(),
        )
        if self.document_context:
            _call_kwargs["document_context"] = self.document_context
        response = self._call_llm(
            system_prompt,
            user_prompt,
            **_call_kwargs,
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
        _doc_ctx_block = (
            "Document context (from an attached file):\n{document_context}\n"
            if self.document_context
            else ""
        )
        user_prompt = """
Evaluate evidence for the subject.

Subject: {subject}
Priority angle: {angle}
Official narrative: {official}
Counter-narrative: {counter}
Web research context:
{web_context}
""" + _doc_ctx_block + """
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
        _call_kwargs: dict = dict(
            audit_step="factcheck_phase_2",
            subject=subject,
            angle=angle,
            official=phase1_content.get("official_narrative", ""),
            counter=phase1_content.get("counter_narrative", ""),
            web_context=self.web_context or "None",
            schema=_Phase2Model.model_json_schema(),
        )
        if self.document_context:
            _call_kwargs["document_context"] = self.document_context
        response = self._call_llm(
            system_prompt,
            user_prompt,
            **_call_kwargs,
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
        _doc_ctx_block = (
            "Document context (from an attached file):\n{document_context}\n"
            if self.document_context
            else ""
        )
        user_prompt = """
Synthesize findings into a concise summary.

Subject: {subject}
Phase 1 summary: {phase1}
Phase 2 summary: {phase2}
Web research context:
{web_context}
""" + _doc_ctx_block + """
Return JSON with:
- biological_truth
- industry_bias
- grey_zone
- references

Schema:
{schema}
"""
        _call_kwargs: dict = dict(
            audit_step="factcheck_phase_3",
            subject=subject,
            phase1=phase1_content,
            phase2=phase2_content,
            web_context=self.web_context or "None",
            schema=_Phase3Model.model_json_schema(),
        )
        if self.document_context:
            _call_kwargs["document_context"] = self.document_context
        response = self._call_llm(
            system_prompt,
            user_prompt,
            **_call_kwargs,
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

    def _call_perspective(
        self,
        perspective: str,
        subject: str,
        synthesis: Dict[str, Any],
        lens: "PerspectiveLens",
    ) -> "_PerspectiveOutput":
        """
        Call the LLM for one perspective (mainstream / naturist / biohacker).
        Returns _PerspectiveOutput. Falls back to empty output on any failure.
        Thread-safe: _call_llm holds the GIL during network I/O.
        """
        _SYSTEM_PROMPTS = {
            "mainstream": (
                "You are a clinical researcher writing evidence-based medical analysis. "
                "Prioritize RCTs, Cochrane reviews, FDA/EMA guidance, and GRADE-A evidence. "
                "Third-person, objective tone. Cite DOI/PMID for every claim. "
                "Return ONLY valid JSON matching the schema provided."
            ),
            "naturist": (
                "You are an evolutionary medicine researcher. Prioritize ancestral biology, "
                "circadian alignment, whole-food interventions, and small independent studies. "
                "Use evolutionary logic as a tiebreaker when RCT evidence is mixed. "
                "Cite peer-reviewed support where available. "
                "Return ONLY valid JSON matching the schema provided."
            ),
            "biohacker": (
                "You are an optimization researcher. Prioritize recent cutting-edge findings, "
                "promising n=1 protocols, quantified self data, and emerging mechanisms even "
                "with limited RCT backing. Label evidence level explicitly (e.g. 'Limited RCT', "
                "'Anecdotal', 'Mechanistic'). "
                "Return ONLY valid JSON matching the schema provided."
            ),
        }

        system_prompt = _SYSTEM_PROMPTS.get(perspective, _SYSTEM_PROMPTS["mainstream"])
        _doc_ctx_block = (
            "Document context (from an attached file):\n{document_context}\n"
            if self.document_context
            else ""
        )
        user_prompt = (
            "Analyze {subject} from the {perspective} perspective.\n\n"
            "Synthesis context:\n{synthesis}\n\n"
            "Web research context:\n{web_context}\n\n"
            + _doc_ctx_block
            + "Return JSON with this exact schema:\n{schema}\n\n"
            "Requirements:\n"
            "- findings: 3-5 paragraphs (<= ~350 words total) covering evidence, "
            "mechanisms, and context\n"
            "- recommendations: 3-5 concrete, actionable items\n"
            "- key_insight: single sentence capturing the most important takeaway\n"
            "- citations: 5-10 APA 7 references, each MUST include a DOI, PMID, or direct URL\n"
            "- statistical_details: list the precise quantitative evidence behind the "
            "findings (effect sizes, relative/absolute risk, odds ratios, confidence "
            "intervals, p-values, sample sizes, NNT/NNH). One concise entry per item, "
            "each ending with its citation marker or DOI/PMID. Empty list only if truly "
            "no quantitative evidence exists.\n"
        )

        try:
            _call_kwargs: dict = dict(
                audit_step=f"factcheck_phase4_{perspective}",
                subject=subject,
                perspective=perspective,
                synthesis=synthesis,
                web_context=self.web_context or "None",
                schema=_PerspectiveOutput.model_json_schema(),
            )
            if self.document_context:
                _call_kwargs["document_context"] = self.document_context
            response = self._call_llm(
                system_prompt,
                user_prompt,
                **_call_kwargs,
            )
            parsed = self._parse_json(response)
            if isinstance(parsed, dict):
                try:
                    return _PerspectiveOutput.model_validate(parsed)
                except Exception:
                    pass
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                f"Perspective '{perspective}' LLM call failed: {exc}"
            )

        return _PerspectiveOutput(
            findings=f"Analysis unavailable for {perspective} perspective.",
            recommendations=[],
            key_insight="",
            citations=[],
        )

    @track_cost("Phase 4: Multi-Perspective Output (LangChain)")
    def _phase4_generate_output(
        self,
        subject: str,
        synthesis: Dict[str, Any],
        lens: "PerspectiveLens",
    ) -> PhaseResult:
        """
        Run three perspective agents in parallel, then assemble into a single report.
        Total LLM calls: 3 parallel + 1 assembler = 4.
        """
        # ── 1. Run three perspectives in parallel ────────────────────────────
        perspectives = ("mainstream", "naturist", "biohacker")
        results: Dict[str, _PerspectiveOutput] = {}

        with ThreadPoolExecutor(max_workers=3) as pool:
            future_map = {
                pool.submit(self._call_perspective, p, subject, synthesis, lens): p
                for p in perspectives
            }
            for future in as_completed(future_map):
                name = future_map[future]
                try:
                    results[name] = future.result()
                except Exception as exc:
                    import logging
                    logging.getLogger(__name__).warning(f"Perspective {name} failed: {exc}")
                    results[name] = _PerspectiveOutput(
                        findings="Analysis unavailable.",
                        recommendations=[],
                        key_insight="",
                        citations=[],
                    )

        mainstream = results["mainstream"]
        naturist   = results["naturist"]
        biohacker  = results["biohacker"]

        # ── 2. Collect and deduplicate all citations ─────────────────────────
        all_citations = mainstream.citations + naturist.citations + biohacker.citations
        seen_keys: set = set()
        unique_refs: List[Dict[str, Any]] = []
        for c in all_citations:
            key = c.strip().lower()[:100]
            if key and key not in seen_keys:
                seen_keys.add(key)
                unique_refs.append({"raw_citation": c.strip()})

        # ── 3. Assemble the final report via LLM ─────────────────────────────
        lens_label = {
            "M": "Mainstream", "N": "Naturist", "B": "Biohacker", "A": "All Perspectives",
        }.get(lens.value, "Balanced")

        assembler_system = (
            "You are a medical report editor. Combine the three perspective summaries "
            "into a clear, well-structured markdown report. Write concisely. "
            "Do NOT add information — only organize what is given."
        )
        assembler_user = (
            "Combine these three perspectives on {subject} into a markdown report.\n\n"
            "Chosen lens: {lens_label}\n\n"
            "MAINSTREAM PERSPECTIVE:\n"
            "Key insight: {mainstream_insight}\n"
            "Findings: {mainstream_findings}\n"
            "Recommendations:\n{mainstream_recs}\n\n"
            "NATURIST PERSPECTIVE:\n"
            "Key insight: {naturist_insight}\n"
            "Findings: {naturist_findings}\n"
            "Recommendations:\n{naturist_recs}\n\n"
            "BIOHACKER PERSPECTIVE:\n"
            "Key insight: {biohacker_insight}\n"
            "Findings: {biohacker_findings}\n"
            "Recommendations:\n{biohacker_recs}\n\n"
            "ALL CITATIONS (include all, deduplicated):\n{all_citations}\n\n"
            "Required output structure (use exactly these markdown headings):\n\n"
            "## 🎯 Your Focus: {lens_label} Perspective\n"
            "[2-3 sentences: the chosen perspective key_insight + top 3 recommendations]\n\n"
            "---\n\n"
            "## 🏥 Mainstream Medicine View\n"
            "[findings + recommendations for mainstream]\n\n"
            "## 🌿 Naturist / Evolutionary View\n"
            "[findings + recommendations for naturist]\n\n"
            "## 🚀 Biohacker / Optimization View\n"
            "[findings + recommendations for biohacker]\n\n"
            "\n## 📚 References\n"
            "[All citations, numbered, APA 7 format]\n"
        )

        assembled = self._call_llm(
            assembler_system,
            assembler_user,
            audit_step="factcheck_phase4_assembler",
            subject=subject,
            lens_label=lens_label,
            mainstream_insight=mainstream.key_insight,
            mainstream_findings=mainstream.findings,
            mainstream_recs="\n".join(f"- {r}" for r in mainstream.recommendations),
            naturist_insight=naturist.key_insight,
            naturist_findings=naturist.findings,
            naturist_recs="\n".join(f"- {r}" for r in naturist.recommendations),
            biohacker_insight=biohacker.key_insight,
            biohacker_findings=biohacker.findings,
            biohacker_recs="\n".join(f"- {r}" for r in biohacker.recommendations),
            all_citations="\n".join(c["raw_citation"] for c in unique_refs),
        )

        # Carry the structured perspective outputs forward so Phase 5 can build a
        # layered, lossless report (conclusions -> reasoning -> statistical appendix)
        # instead of re-summarizing an already-summarized markdown blob.
        perspectives_payload = {
            name: {
                "findings": p.findings,
                "recommendations": list(p.recommendations),
                "key_insight": p.key_insight,
                "statistical_details": list(p.statistical_details),
            }
            for name, p in (
                ("mainstream", mainstream),
                ("naturist", naturist),
                ("biohacker", biohacker),
            )
        }

        return PhaseResult(
            phase=AnalysisPhase.COMPLEX_OUTPUT,
            timestamp=datetime.now(),
            content={
                "output": assembled,
                "output_type": lens.value,
                "perspectives": perspectives_payload,
            },
            references=unique_refs,
        )

    # Lens-specific framing reused across the patient-facing layer.
    _LENS_FRAMING = {
        "M": (
            "clinical, evidence-graded tone. Use 'your doctor recommends' framing. "
            "Emphasize the strength of the evidence behind each recommendation."
        ),
        "N": (
            "warm, nature-first tone. Use 'your body evolved to...' framing. "
            "Emphasize ancestral wisdom and natural approaches."
        ),
        "B": (
            "optimization mindset tone. Use 'here is your protocol' framing. "
            "Emphasize measurable outcomes, n=1 experimentation, and cutting-edge insights."
        ),
        "A": (
            "balanced, neutral tone. Cover all perspectives equally. "
            "Let the reader decide which approach suits them."
        ),
    }

    @track_cost("Phase 5: Simplified Output (LangChain)")
    def _phase5_simplify_output(
        self,
        body: str,
        lens: "PerspectiveLens" = None,
        phase2_content: Optional[Dict[str, Any]] = None,
        perspectives: Optional[Dict[str, Any]] = None,
    ) -> PhaseResult:
        """
        Build a LAYERED, lossless report from the Phase 4 body plus structured
        upstream content, using progressive disclosure:

            Layer 1 — Conclusions   (plain language; what readers value; first)
            Layer 2 — The Reasoning (the logic behind the conclusions)
            Layer 3 — Statistical Appendix (precise numbers; optional; last)

        Two documents are produced and returned in ``content``:
          * ``simplified_output``     — patient-facing: Layers 1-2 in plain language.
          * ``practitioner_layered``  — Layers 1-2 + the full Statistical Appendix.

        The Statistical Appendix is assembled DETERMINISTICALLY in Python from
        Phase 2 evidence and each perspective's ``statistical_details`` so precise
        numbers can never be silently dropped or hallucinated by the LLM. The LLM
        is used only to phrase the plain-language Conclusions/Reasoning layers.

        References are NOT passed in — they are re-attached verbatim by
        ``start_analysis`` after this method returns.
        """
        if lens is None:
            lens = PerspectiveLens.BALANCED
        framing = self._LENS_FRAMING.get(lens.value, self._LENS_FRAMING["A"])

        # ── Layers 1-2: plain-language conclusions + reasoning (LLM) ──────────
        system_prompt = (
            f"You are a medical writer producing a layered report for a general "
            f"audience. Use a {framing} "
            f"Write at a 6th grade reading level with short sentences and common words. "
            f"State conclusions FIRST, then explain the reasoning behind them so the "
            f"reader can understand the logic. "
            f"When you mention a statistic, describe it in plain language (e.g. "
            f"'about a third lower risk') AND keep any inline citation markers like "
            f"[1], [2] exactly as they appear so claims stay traceable. "
            f"Do NOT invent facts and do NOT drop any recommendation or key point "
            f"present in the content. "
            f"Do NOT include a References section — that is added separately. "
            f"Do NOT write a Statistical Appendix — that is added separately."
        )

        _doc_ctx_block = (
            "Document context (from an attached file):\n{document_context}\n"
            if self.document_context
            else ""
        )
        user_prompt = (
            "Rewrite this medical content as a layered guide for a non-medical reader.\n\n"
            "Content:\n{body}\n\n"
            "Web research context:\n{web_context}\n\n"
            + _doc_ctx_block
            + "Structure the output with EXACTLY these two layers, in this order:\n\n"
            "# [topic from content]\n\n"
            "## ✅ Report Summary\n"
            "[The bottom line first: what the reader should take away, plus the top "
            "practical recommendations. Plain language.]\n\n"
            "## 🧠 The Reasoning\n"
            "[Explain the logic behind the conclusions: what each perspective found, "
            "where they agree or disagree, the grey zones and uncertainties, and what "
            "to watch out for. Keep inline citation markers.]\n\n"
            "Do NOT include a References section or a Statistical Appendix."
        )

        _call_kwargs: dict = dict(
            audit_step="factcheck_phase_5",
            body=body,
            web_context=self.web_context or "None",
        )
        if self.document_context:
            _call_kwargs["document_context"] = self.document_context
        plain_layers = self._call_llm(
            system_prompt,
            user_prompt,
            **_call_kwargs,
        )

        # ── Layer 3: statistical appendix (deterministic, lossless) ──────────
        appendix = self._build_statistical_appendix(phase2_content, perspectives)

        # Compose patient (Layers 1-2) and practitioner (Layers 1-3) variants via
        # the shared base helper.
        patient_output, practitioner_output = self._build_layered_report(
            conclusions_and_reasoning=plain_layers,
            appendix=appendix,
        )

        # ── Verification guard: warn on any silently dropped key content ─────
        must_survive = [
            (p or {}).get("key_insight", "")
            for p in (perspectives or {}).values()
        ]
        self._verify_no_silent_loss(
            practitioner_output,
            [m for m in must_survive if m],
            audit_step="factcheck_phase5_loss_check",
        )

        return PhaseResult(
            phase=AnalysisPhase.SIMPLIFIED_OUTPUT,
            timestamp=datetime.now(),
            content={
                "simplified_output": patient_output,
                "practitioner_layered": practitioner_output,
            },
        )

    def _build_statistical_appendix(
        self,
        phase2_content: Optional[Dict[str, Any]],
        perspectives: Optional[Dict[str, Any]],
    ) -> str:
        """
        Assemble the fact-checker Statistical Appendix (Layer 3) by grouping the
        structured upstream data into labelled sections and delegating to the
        shared deterministic builder in ``LangChainAgentBase``. Returns "" if
        there is nothing quantitative to show.
        """
        _labels = {
            "mainstream": "🏥 Mainstream Medicine",
            "naturist": "🌿 Naturist / Evolutionary",
            "biohacker": "🚀 Biohacker / Optimization",
        }
        sections: Dict[str, List[str]] = {}

        # Per-perspective quantitative evidence.
        for name in ("mainstream", "naturist", "biohacker"):
            details = ((perspectives or {}).get(name) or {}).get(
                "statistical_details"
            ) or []
            details = [str(d).strip() for d in details if str(d).strip()]
            if details:
                sections[_labels[name]] = details

        # Phase 2 evidence-quality fields (methodology, funding, recency).
        _p2_fields = [
            ("methodology_quality", "Methodology quality"),
            ("industry_funded_studies", "Industry-funded studies"),
            ("independent_research", "Independent research"),
            ("time_weighted_evidence", "Recency-weighted evidence"),
            ("anecdotal_signals", "Anecdotal signals"),
        ]
        p2_lines: List[str] = []
        for key, label in _p2_fields:
            value = str((phase2_content or {}).get(key, "") or "").strip()
            if value:
                p2_lines.append(f"**{label}:** {value}")
        if p2_lines:
            sections["Evidence Quality & Grading"] = p2_lines

        return super()._build_statistical_appendix(sections)

    def _parse_phase_model(
        self, response: str, model_cls: type[BaseModel], subject: str
    ) -> BaseModel:
        parsed = self._parse_json(response)

        if not isinstance(parsed, dict):
            raise RuntimeError(
                f"Fact-check phase failed for '{subject}': LLM response could not be parsed "
                f"as JSON (model: {model_cls.__name__}). "
                f"Response length: {len(response)} chars. "
                f"This usually means the response was truncated (max_tokens too low) "
                f"or the model returned plain text instead of JSON. "
                f"Response preview: {response[:200]!r}"
            )

        try:
            return model_cls.model_validate(parsed)
        except Exception as exc:
            raise RuntimeError(
                f"Fact-check phase failed for '{subject}': LLM returned valid JSON but it "
                f"did not match the expected schema ({model_cls.__name__}). "
                f"Validation error: {exc}. "
                f"JSON keys returned: {list(parsed.keys())}"
            ) from exc

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

    def _prompt_user_lens(self) -> str:
        print("\nPHASE 3 COMPLETE: Synthesis")
        print("Which perspective resonates most with you?\n")
        print("  [M] Mainstream   — Clinical guidelines and established evidence")
        print("  [N] Naturist     — Evolutionary biology and ancestral health")
        print("  [B] Biohacker    — Optimization, cutting-edge, n=1 protocols")
        print("  [A] All equal    — Balanced report, no preference\n")
        while True:
            choice = input("Your choice (M/N/B/A): ").strip().upper()
            if choice in ("M", "N", "B", "A"):
                return choice
            print("Invalid choice. Enter M, N, B, or A.")
