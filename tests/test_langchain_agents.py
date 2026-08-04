#!/usr/bin/env python3
"""
Unit tests for LangChain-based agents (mocked LLM responses).
"""

import json

import pytest

import langchain_agents.base as lc_base
from langchain_agents import (
    LangChainMedicalFactChecker,
    LangChainMedicalReasoningAgent,
    LangChainMedicationAnalyzer,
)
from medical_procedure_analyzer import MedicalInput
from medical_procedure_analyzer.medication_analyzer import MedicationInput


class _DummyProvider:
    def generate_response(self, prompt: str, system_prompt: str | None = None):
        from llm_integrations import TokenUsage

        return "{}", TokenUsage()


class _DummyManager:
    def get_available_provider(self):
        return _DummyProvider()


@pytest.fixture(autouse=True)
def _mock_llm_manager(monkeypatch):
    monkeypatch.setattr(lc_base, "create_llm_manager", lambda *args, **kwargs: _DummyManager())


def test_langchain_procedure_agent(monkeypatch):
    responses = iter(
        [
            json.dumps({"organs": ["kidneys", "brain"]}),
            json.dumps(
                [
                    {
                        "organ_name": "kidneys",
                        "affected_by_procedure": True,
                        "at_risk": True,
                        "risk_level": "moderate",
                        "pathways_involved": ["renal_excretion"],
                        "known_recommendations": ["Hydration"],
                        "potential_recommendations": [],
                        "debunked_claims": [],
                        "evidence_quality": "moderate",
                    },
                    {
                        "organ_name": "brain",
                        "affected_by_procedure": False,
                        "at_risk": False,
                        "risk_level": "low",
                        "pathways_involved": [],
                        "known_recommendations": [],
                        "potential_recommendations": [],
                        "debunked_claims": [],
                        "evidence_quality": "limited",
                    },
                ]
            ),
            json.dumps(
                {
                    "procedure_summary": "MRI Scanner - With contrast",
                    "confidence_score": 0.82,
                    "general_recommendations": ["Hydrate before and after procedure"],
                    "research_gaps": ["Long-term gadolinium retention studies"],
                }
            ),
        ]
    )

    agent = LangChainMedicalReasoningAgent(enable_logging=False)

    def _proc_call(*args, **kwargs):
        # The layered patient/practitioner build issues an extra plain-language
        # call; return markdown for it rather than consuming a JSON response.
        if kwargs.get("audit_step") == "procedure_layering":
            return "## ✅ Conclusions\nHydrate.\n\n## 🧠 The Reasoning\nContrast affects kidneys."
        return next(responses)

    monkeypatch.setattr(agent, "_call_llm", _proc_call)

    result = agent.analyze_medical_procedure(
        MedicalInput(
            procedure="MRI Scanner",
            details="With contrast",
            objectives=("risks", "post-procedure care"),
        )
    )

    assert result.procedure_summary == "MRI Scanner - With contrast"
    assert len(result.organs_analyzed) == 2
    assert result.reasoning_trace
    assert result.practitioner_report
    # Layered outputs produced.
    assert result.patient_report
    assert "Statistical Appendix" in result.practitioner_report
    assert "Statistical Appendix" not in result.patient_report


def test_langchain_medication_agent(monkeypatch):
    response = json.dumps(
        {
            "medication_name": "Metformin",
            "drug_class": "Biguanide",
            "mechanism_of_action": "Decreases hepatic glucose production.",
            "absorption": "Oral",
            "metabolism": "Minimal hepatic metabolism",
            "elimination": "Renal",
            "half_life": "6 hours",
            "approved_indications": ["Type 2 diabetes"],
            "off_label_uses": [],
            "standard_dosing": "500 mg twice daily",
            "dose_adjustments": {"renal": "Avoid if eGFR <30"},
            "common_adverse_effects": ["GI upset"],
            "serious_adverse_effects": ["Lactic acidosis"],
            "contraindications": [{"condition": "Severe renal impairment", "severity": "absolute"}],
            "black_box_warnings": ["Lactic acidosis"],
            "drug_interactions": [
                {
                    "interaction_type": "drug-drug",
                    "interacting_agent": "Cimetidine",
                    "severity": "moderate",
                    "mechanism": "Reduced clearance",
                    "clinical_effect": "Increased metformin levels",
                    "management": "Monitor levels",
                    "time_separation": None,
                    "evidence_level": "moderate",
                }
            ],
            "food_interactions": [],
            "environmental_considerations": [],
            "evidence_based_recommendations": [{"intervention": "Titrate slowly"}],
            "what_not_to_do": [{"action": "Do not stop abruptly"}],
            "debunked_claims": [],
            "monitoring_requirements": ["Check eGFR annually"],
            "warning_signs": [{"sign": "Fatigue"}],
            "evidence_quality": "moderate",
            "analysis_confidence": 0.8,
        }
    )

    agent = LangChainMedicationAnalyzer(enable_logging=False)

    def _med_call(*args, **kwargs):
        if kwargs.get("audit_step") == "medication_layering":
            return "## ✅ Conclusions\nTake with food.\n\n## 🧠 The Reasoning\nReduces GI upset."
        return response

    monkeypatch.setattr(agent, "_call_llm", _med_call)

    result = agent.analyze_medication(MedicationInput(medication_name="Metformin"))

    assert result.medication_name == "Metformin"
    assert result.drug_class == "Biguanide"
    assert result.drug_interactions
    assert result.analysis_confidence == 0.8
    # Layered outputs produced; black-box warning survives into practitioner layer.
    assert result.patient_report
    assert "Statistical Appendix" in result.practitioner_report
    assert "Statistical Appendix" not in result.patient_report
    assert "Lactic acidosis" in result.practitioner_report


def test_langchain_factcheck_agent(monkeypatch):
    """Smoke test: start_analysis completes and produces 5 phase results."""
    phase_json = {
        "official_narrative": "Official view",
        "counter_narrative": "Counter view",
        "key_conflicts": "Key conflicts",
        "industry_funded_studies": "Industry studies",
        "independent_research": "Independent studies",
        "methodology_quality": "Mixed",
        "anecdotal_signals": "Anecdotes",
        "time_weighted_evidence": "Recent data",
        "biological_truth": "Most likely truth",
        "industry_bias": "Biases",
        "grey_zone": "Open questions",
        "references": [],
    }

    def fake_call_llm(*args, **kwargs):
        audit_step = kwargs.get("audit_step", "")
        # Perspective calls return a _PerspectiveOutput-shaped JSON
        if "phase4_" in audit_step and "assembler" not in audit_step:
            return json.dumps({
                "findings": "findings",
                "recommendations": ["Rec"],
                "key_insight": "insight",
                "citations": [],
            })
        # Assembler returns markdown
        if "assembler" in audit_step:
            return "## 🎯 Your Focus\nKey insight.\n\n## 🏥 Mainstream View\nFindings."
        # Phase 5 simplification
        if audit_step == "factcheck_phase_5":
            return "Simplified output"
        # Phases 1-3 return JSON
        return json.dumps(phase_json)

    agent = LangChainMedicalFactChecker(enable_logging=False, interactive=False)
    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)

    session = agent.start_analysis("Vitamin D")

    assert session.practitioner_report
    assert len(session.phase_results) == 5


# ── Task 1: PerspectiveLens enum and Pydantic models ──────────────────────────

def test_perspective_lens_enum():
    from langchain_agents.factcheck_agent import PerspectiveLens
    assert PerspectiveLens("M") == PerspectiveLens.MAINSTREAM
    assert PerspectiveLens("N") == PerspectiveLens.NATURIST
    assert PerspectiveLens("B") == PerspectiveLens.BIOHACKER
    assert PerspectiveLens("A") == PerspectiveLens.BALANCED


def test_perspective_output_model_validates():
    from langchain_agents.factcheck_agent import _PerspectiveOutput
    out = _PerspectiveOutput(
        findings="Test findings",
        recommendations=["Rec 1", "Rec 2"],
        key_insight="Test insight",
        citations=["Author (2024). Title. Journal. https://doi.org/10.0/x"],
    )
    assert out.key_insight == "Test insight"
    assert len(out.citations) == 1


def test_perspective_output_model_empty_fallback():
    from langchain_agents.factcheck_agent import _PerspectiveOutput
    out = _PerspectiveOutput(
        findings="Analysis unavailable",
        recommendations=[],
        key_insight="",
        citations=[],
    )
    assert out.findings == "Analysis unavailable"


# ── Task 2: Lens picker in start_analysis ─────────────────────────────────────

def test_factchecker_noninteractive_uses_balanced_lens(monkeypatch):
    """Non-interactive mode must default to PerspectiveLens.BALANCED ('A')."""
    from langchain_agents import LangChainMedicalFactChecker
    from langchain_agents.factcheck_agent import PerspectiveLens
    from medical_fact_checker.medical_fact_checker_agent import (
        AnalysisPhase, PhaseResult,
    )
    from datetime import datetime

    captured = {}

    def fake_phase4(subject, synthesis, lens):
        captured["lens"] = lens
        return PhaseResult(
            phase=AnalysisPhase.COMPLEX_OUTPUT,
            timestamp=datetime.now(),
            content={"output": "report body"},
            references=[],
        )

    dummy_phase = lambda phase_enum, **kw: PhaseResult(
        phase=phase_enum, timestamp=datetime.now(),
        content={
            "official_narrative": "", "counter_narrative": "", "key_conflicts": "",
            "industry_funded_studies": "", "independent_research": "",
            "methodology_quality": "", "anecdotal_signals": "", "time_weighted_evidence": "",
            "biological_truth": "", "industry_bias": "", "grey_zone": "",
        },
        references=[],
    )

    agent = LangChainMedicalFactChecker(enable_logging=False, interactive=False)
    monkeypatch.setattr(agent, "_phase1_conflict_scan",
        lambda *a, **kw: dummy_phase(AnalysisPhase.CONFLICT_SCAN))
    monkeypatch.setattr(agent, "_phase2_evidence_stress_test",
        lambda *a, **kw: dummy_phase(AnalysisPhase.EVIDENCE_STRESS_TEST))
    monkeypatch.setattr(agent, "_phase3_synthesis_menu",
        lambda *a, **kw: dummy_phase(AnalysisPhase.SYNTHESIS_MENU))
    monkeypatch.setattr(agent, "_phase4_generate_output", fake_phase4)
    monkeypatch.setattr(agent, "_phase5_simplify_output",
        lambda *a, **kw: PhaseResult(
            phase=AnalysisPhase.SIMPLIFIED_OUTPUT, timestamp=datetime.now(),
            content={"simplified_output": "simple"}, references=[],
        ))

    agent.start_analysis("test subject")

    assert "lens" in captured, "Phase 4 was not called"
    assert captured["lens"] == PerspectiveLens.BALANCED


# ── Task 3: _call_perspective helper ──────────────────────────────────────────

def test_call_perspective_returns_perspective_output(monkeypatch):
    """_call_perspective must parse valid JSON and return a _PerspectiveOutput."""
    from langchain_agents import LangChainMedicalFactChecker
    from langchain_agents.factcheck_agent import _PerspectiveOutput, PerspectiveLens
    from llm_integrations import TokenUsage

    good_json = json.dumps({
        "findings": "Test findings for mainstream",
        "recommendations": ["Rec 1", "Rec 2"],
        "key_insight": "Take statins",
        "citations": ["Smith (2024). Title. NEJM. https://doi.org/10.1/x"],
    })

    agent = LangChainMedicalFactChecker(enable_logging=False, interactive=False)
    monkeypatch.setattr(agent, "_call_llm", lambda *a, **kw: good_json)

    synthesis = {"biological_truth": "Sugar causes inflammation", "industry_bias": "", "grey_zone": ""}
    result = agent._call_perspective("mainstream", "Sugar and cancer", synthesis, PerspectiveLens.BALANCED)

    assert isinstance(result, _PerspectiveOutput)
    assert result.findings == "Test findings for mainstream"
    assert "Rec 1" in result.recommendations
    assert result.key_insight == "Take statins"
    assert len(result.citations) == 1


def test_call_perspective_fallback_on_bad_json(monkeypatch):
    """_call_perspective must return a fallback _PerspectiveOutput when LLM returns non-JSON."""
    from langchain_agents import LangChainMedicalFactChecker
    from langchain_agents.factcheck_agent import _PerspectiveOutput, PerspectiveLens

    agent = LangChainMedicalFactChecker(enable_logging=False, interactive=False)
    monkeypatch.setattr(agent, "_call_llm", lambda *a, **kw: "This is not JSON at all.")

    result = agent._call_perspective("naturist", "Vitamin D", {}, PerspectiveLens.NATURIST)

    assert isinstance(result, _PerspectiveOutput)
    assert "unavailable" in result.findings.lower()
    assert result.citations == []


# ── Task 4: _phase4_generate_output with parallel perspectives ────────────────

def test_phase4_generates_three_perspective_report(monkeypatch):
    """Phase 4 must produce a report containing all three perspective sections."""
    from langchain_agents import LangChainMedicalFactChecker
    from langchain_agents.factcheck_agent import PerspectiveLens
    from medical_fact_checker.medical_fact_checker_agent import AnalysisPhase

    call_log = []

    def fake_call_llm(*args, **kwargs):
        audit_step = kwargs.get("audit_step", "")
        call_log.append(audit_step)
        if audit_step and "phase4_" in audit_step and "assembler" not in audit_step:
            return json.dumps({
                "findings": f"Findings for {audit_step}",
                "recommendations": ["Rec A"],
                "key_insight": f"Insight {audit_step}",
                "citations": [f"Author (2024). Title. J. https://doi.org/10.1/{audit_step}"],
            })
        # Assembler call — return markdown with all three sections
        return (
            "## 🎯 Your Focus: Balanced\nKey insight.\n\n"
            "## 🏥 Mainstream View\nMainstream findings.\n\n"
            "## 🌿 Naturist View\nNaturist findings.\n\n"
            "## 🚀 Biohacker View\nBiohacker findings.\n\n"
            "\n## 📚 References\n[1] Author (2024). https://doi.org/10.1/x"
        )

    agent = LangChainMedicalFactChecker(enable_logging=False, interactive=False)
    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)

    synthesis = {"biological_truth": "test", "industry_bias": "", "grey_zone": ""}
    result = agent._phase4_generate_output("Vitamin D", synthesis, PerspectiveLens.BALANCED)

    assert result.phase == AnalysisPhase.COMPLEX_OUTPUT
    report = result.content.get("output", "")
    assert "Mainstream" in report
    assert "Naturist" in report
    assert "Biohacker" in report
    # Assembler must have been called
    assert any("assembler" in s for s in call_log)
    # References must be stored in PhaseResult
    assert len(result.references) >= 1


def test_phase4_references_stored_in_phase_result(monkeypatch):
    """All citations from three perspectives must be in PhaseResult.references."""
    from langchain_agents import LangChainMedicalFactChecker
    from langchain_agents.factcheck_agent import PerspectiveLens

    CITATION_A = "Smith (2024). Title. NEJM. https://doi.org/10.1/mainstream"
    CITATION_B = "Jones (2023). Title. Nature. https://doi.org/10.1/nature"

    def fake_call_llm(*args, **kwargs):
        audit_step = kwargs.get("audit_step", "")
        if audit_step and "phase4_" in audit_step and "assembler" not in audit_step:
            return json.dumps({
                "findings": "findings",
                "recommendations": [],
                "key_insight": "insight",
                "citations": [CITATION_A, CITATION_B],
            })
        return f"## 📚 References\n[1] {CITATION_A}"

    agent = LangChainMedicalFactChecker(enable_logging=False, interactive=False)
    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)

    synthesis = {"biological_truth": "test", "industry_bias": "", "grey_zone": ""}
    result = agent._phase4_generate_output("Sugar", synthesis, PerspectiveLens.MAINSTREAM)

    raw_citations = [r["raw_citation"] for r in result.references]
    assert any("mainstream" in c for c in raw_citations)
    assert any("nature" in c for c in raw_citations)


# ── Task 5: Lens-aware Phase 5 ────────────────────────────────────────────────

def test_phase5_uses_lens_framing(monkeypatch):
    """_phase5_simplify_output must pass lens framing in the system prompt."""
    from langchain_agents import LangChainMedicalFactChecker
    from langchain_agents.factcheck_agent import PerspectiveLens

    captured = {}

    def fake_call_llm(system_prompt, user_prompt, **kwargs):
        captured["system"] = system_prompt
        return "Simplified content."

    agent = LangChainMedicalFactChecker(enable_logging=False, interactive=False)
    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)

    agent._phase5_simplify_output("Some complex body text.", lens=PerspectiveLens.BIOHACKER)
    assert "biohack" in captured["system"].lower() or "optim" in captured["system"].lower(), (
        f"Expected biohacker framing in system prompt, got: {captured['system'][:300]}"
    )


def test_phase5_references_not_in_body(monkeypatch):
    """Phase 5 body must not contain a references section (split happens upstream)."""
    from langchain_agents import LangChainMedicalFactChecker
    from langchain_agents.factcheck_agent import PerspectiveLens

    captured = {}

    def fake_call_llm(system_prompt, user_prompt, **kwargs):
        captured["user"] = user_prompt
        return "Simplified."

    agent = LangChainMedicalFactChecker(enable_logging=False, interactive=False)
    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)

    body_only = "## Key Findings\nSome findings here."
    agent._phase5_simplify_output(body_only, lens=PerspectiveLens.NATURIST)

    assert "📚 References" not in captured.get("user", ""), (
        "References section should not be in Phase 5 user prompt — split happens upstream"
    )


# ── Task 6: End-to-end reference preservation ─────────────────────────────────

def test_factchecker_end_to_end_references_preserved(monkeypatch):
    """Full start_analysis: sentinel reference must appear in final_output."""
    from langchain_agents import LangChainMedicalFactChecker

    SENTINEL_REF = "[1] Author (2024). Title. NEJM. https://doi.org/10.1/sentinel"

    phase_json = {
        "official_narrative": "Official view",
        "counter_narrative": "Counter view",
        "key_conflicts": "Conflicts",
        "industry_funded_studies": "Industry",
        "independent_research": "Independent",
        "methodology_quality": "Good",
        "anecdotal_signals": "Anecdotes",
        "time_weighted_evidence": "Recent",
        "biological_truth": "Truth",
        "industry_bias": "Bias",
        "grey_zone": "Grey",
        "references": [],
    }

    def fake_call_llm(*args, **kwargs):
        audit_step = kwargs.get("audit_step", "")
        if "phase4_" in audit_step and "assembler" not in audit_step:
            return json.dumps({
                "findings": "findings",
                "recommendations": ["Do this"],
                "key_insight": "Important insight",
                "citations": [SENTINEL_REF],
            })
        if "assembler" in audit_step:
            # Assembler includes the sentinel in the references block
            return (
                "## 🎯 Your Focus: Balanced\nKey insight.\n\n"
                "## 🏥 Mainstream View\nFindings.\n\n"
                "## 🌿 Naturist View\nFindings.\n\n"
                "## 🚀 Biohacker View\nFindings.\n\n"
                f"\n## 📚 References\n{SENTINEL_REF}"
            )
        if audit_step == "factcheck_phase_5":
            return "# Simplified Guide\n\n## Key Findings\nSimplified body — no refs here."
        return json.dumps(phase_json)

    agent = LangChainMedicalFactChecker(enable_logging=False, interactive=False)
    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)

    session = agent.start_analysis("Vitamin D and bone health")

    # The sentinel reference must survive into final_output verbatim
    assert SENTINEL_REF in session.final_output, (
        f"Sentinel reference not found in final_output.\n"
        f"final_output:\n{session.final_output}"
    )

    # Phase 4 PhaseResult must have the reference
    phase4 = next(
        (p for p in session.phase_results if p.phase.value == "complex_output"), None
    )
    assert phase4 is not None
    assert any(SENTINEL_REF in r.get("raw_citation", "") for r in phase4.references)


# ── Task 7: Layered, lossless Phase 5 ─────────────────────────────────────────

def test_phase4_no_500_char_truncation(monkeypatch):
    """Full perspective findings (>500 chars) must reach the assembler prompt."""
    from langchain_agents import LangChainMedicalFactChecker
    from langchain_agents.factcheck_agent import PerspectiveLens

    LONG_FINDING = "Finding sentence. " * 60  # ~1080 chars, well over old 500 cap
    captured = {}

    def fake_call_llm(system_prompt, user_prompt, **kwargs):
        audit_step = kwargs.get("audit_step", "")
        if "phase4_" in audit_step and "assembler" not in audit_step:
            return json.dumps({
                "findings": LONG_FINDING,
                "recommendations": ["Rec"],
                "key_insight": "Insight",
                "citations": ["Author (2024). https://doi.org/10.1/x"],
                "statistical_details": [],
            })
        # assembler — capture the rendered kwargs to confirm full findings passed
        captured["kwargs"] = kwargs
        return "## 📚 References\n[1] x"

    agent = LangChainMedicalFactChecker(enable_logging=False, interactive=False)
    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)

    synthesis = {"biological_truth": "t", "industry_bias": "", "grey_zone": ""}
    agent._phase4_generate_output("Vitamin D", synthesis, PerspectiveLens.BALANCED)

    # The full (untruncated) finding must be passed to the assembler.
    assert captured["kwargs"]["mainstream_findings"] == LONG_FINDING


def test_phase5_produces_layered_documents(monkeypatch):
    """Phase 5 must return both a patient and a practitioner layered document."""
    from langchain_agents import LangChainMedicalFactChecker
    from langchain_agents.factcheck_agent import PerspectiveLens

    agent = LangChainMedicalFactChecker(enable_logging=False, interactive=False)
    monkeypatch.setattr(agent, "_call_llm",
                        lambda *a, **k: "## ✅ Conclusions\nDo X.\n\n## 🧠 The Reasoning\nBecause Y.")

    perspectives = {
        "mainstream": {"findings": "f", "recommendations": ["r"],
                       "key_insight": "Insight", "statistical_details": ["RR 0.70 (95% CI 0.60-0.82), p<0.001 [1]"]},
        "naturist": {"findings": "f", "recommendations": [], "key_insight": "", "statistical_details": []},
        "biohacker": {"findings": "f", "recommendations": [], "key_insight": "", "statistical_details": []},
    }
    phase2 = {"methodology_quality": "Two large RCTs, low risk of bias."}

    result = agent._phase5_simplify_output(
        "body text", lens=PerspectiveLens.MAINSTREAM,
        phase2_content=phase2, perspectives=perspectives,
    )

    patient = result.content["simplified_output"]
    practitioner = result.content["practitioner_layered"]

    # Precise statistics appear ONLY in the practitioner document.
    assert "RR 0.70" in practitioner
    assert "Statistical Appendix" in practitioner
    assert "Two large RCTs" in practitioner
    assert "RR 0.70" not in patient
    assert "Statistical Appendix" not in patient
    # Both keep the plain-language conclusions layer.
    assert "Conclusions" in patient and "Conclusions" in practitioner


def test_phase5_layer_ordering(monkeypatch):
    """Conclusions precede Reasoning, which precedes the Statistical Appendix."""
    from langchain_agents import LangChainMedicalFactChecker
    from langchain_agents.factcheck_agent import PerspectiveLens

    agent = LangChainMedicalFactChecker(enable_logging=False, interactive=False)
    monkeypatch.setattr(agent, "_call_llm",
                        lambda *a, **k: "## ✅ Conclusions\nC.\n\n## 🧠 The Reasoning\nR.")

    perspectives = {
        "mainstream": {"findings": "f", "recommendations": [], "key_insight": "",
                       "statistical_details": ["OR 1.5 (1.1-2.0) [1]"]},
    }
    result = agent._phase5_simplify_output(
        "body", lens=PerspectiveLens.BALANCED,
        phase2_content={}, perspectives=perspectives,
    )
    doc = result.content["practitioner_layered"]
    assert doc.index("Conclusions") < doc.index("Reasoning") < doc.index("Statistical Appendix")


def test_phase5_verification_guard_warns_on_dropped_insight(monkeypatch, caplog):
    """If a perspective key_insight is dropped, a loss warning must be logged."""
    import logging
    from langchain_agents import LangChainMedicalFactChecker
    from langchain_agents.factcheck_agent import PerspectiveLens

    agent = LangChainMedicalFactChecker(enable_logging=False, interactive=False)
    # LLM output that omits the unique key_insight entirely.
    monkeypatch.setattr(agent, "_call_llm", lambda *a, **k: "## ✅ Conclusions\nUnrelated text.")

    perspectives = {
        "mainstream": {"findings": "f", "recommendations": [],
                       "key_insight": "Zebra quantum flux paradox marker", "statistical_details": []},
    }
    with caplog.at_level(logging.WARNING):
        agent._phase5_simplify_output(
            "body", lens=PerspectiveLens.BALANCED,
            phase2_content={}, perspectives=perspectives,
        )
    assert any("may have dropped" in r.message for r in caplog.records)


def test_build_statistical_appendix_empty_when_no_stats():
    """Appendix is empty string when there is no quantitative content."""
    from langchain_agents import LangChainMedicalFactChecker

    agent = LangChainMedicalFactChecker(enable_logging=False, interactive=False)
    assert agent._build_statistical_appendix({}, {"mainstream": {"statistical_details": []}}) == ""


# ── Stage A: shared base layering helpers ─────────────────────────────────────

def _make_base_agent():
    """A minimal LangChainAgentBase instance for helper unit tests."""
    from langchain_agents.base import LangChainAgentBase, LangChainAgentConfig
    return LangChainAgentBase(LangChainAgentConfig(enable_audit=True))


def test_base_build_statistical_appendix_sections_and_ordering():
    agent = _make_base_agent()
    appendix = agent._build_statistical_appendix({
        "Group One": ["RR 0.70 (0.60-0.82), p<0.001 [1]"],
        "Group Two": ["OR 1.5 (1.1-2.0) [2]"],
        "Empty Group": [],
    })
    assert "📊 Statistical Appendix" in appendix
    assert "### Group One" in appendix and "### Group Two" in appendix
    assert "Empty Group" not in appendix  # empty groups omitted
    assert appendix.index("Group One") < appendix.index("Group Two")


def test_base_build_statistical_appendix_empty():
    agent = _make_base_agent()
    assert agent._build_statistical_appendix({"X": [], "Y": ["  "]}) == ""


def test_base_build_layered_report_patient_omits_appendix():
    agent = _make_base_agent()
    body = "## ✅ Conclusions\nC.\n\n## 🧠 The Reasoning\nR."
    appendix = "## 📊 Statistical Appendix\nStats here."
    patient, practitioner = agent._build_layered_report(
        conclusions_and_reasoning=body, appendix=appendix,
    )
    assert "Statistical Appendix" in practitioner
    assert "Statistical Appendix" not in patient
    assert "practitioner report" in patient.lower()  # pointer present
    # Both keep the conclusions/reasoning layers.
    assert "Conclusions" in patient and "Conclusions" in practitioner


def test_base_build_layered_report_no_appendix_identical():
    agent = _make_base_agent()
    body = "## ✅ Conclusions\nC."
    patient, practitioner = agent._build_layered_report(
        conclusions_and_reasoning=body, appendix="",
    )
    assert patient == body == practitioner


def test_base_verify_no_silent_loss_flags_and_audits(caplog):
    import logging
    agent = _make_base_agent()
    with caplog.at_level(logging.WARNING):
        missing = agent._verify_no_silent_loss(
            "output that omits it entirely",
            ["Zebra quantum flux paradox marker"],
        )
    assert missing  # something reported missing
    assert any("may have dropped" in r.message for r in caplog.records)
    assert any(e.get("step") == "layering_loss_check" for e in agent.audit_events)


def test_base_verify_no_silent_loss_passes_when_present():
    agent = _make_base_agent()
    missing = agent._verify_no_silent_loss(
        "The conclusion is that vitamin D supports bone health markedly.",
        ["The conclusion is that vitamin D"],
    )
    assert missing == []


# ── Stage B: procedure & medication layering ──────────────────────────────────

def test_medication_statistical_appendix_contains_grading(monkeypatch):
    """Medication practitioner report must carry evidence grading + safety in the appendix."""
    from langchain_agents import LangChainMedicationAnalyzer

    med_json = json.dumps({
        "medication_name": "Metformin", "drug_class": "Biguanide",
        "mechanism_of_action": "m", "absorption": "a", "metabolism": "b",
        "elimination": "e", "half_life": "6h",
        "approved_indications": ["T2DM"], "off_label_uses": [],
        "standard_dosing": "500mg BID", "dose_adjustments": {"renal": "avoid <30"},
        "common_adverse_effects": ["GI upset"], "serious_adverse_effects": ["Lactic acidosis"],
        "contraindications": [], "black_box_warnings": ["Lactic acidosis risk"],
        "drug_interactions": [{
            "interaction_type": "drug-drug", "interacting_agent": "Cimetidine",
            "severity": "moderate", "mechanism": "x", "clinical_effect": "raised levels",
            "management": "monitor", "time_separation": None, "evidence_level": "moderate",
        }],
        "food_interactions": [], "environmental_considerations": [],
        "evidence_based_recommendations": [{"intervention": "Titrate slowly"}],
        "what_not_to_do": [], "debunked_claims": [], "monitoring_requirements": ["eGFR"],
        "warning_signs": [], "evidence_quality": "moderate", "analysis_confidence": 0.8,
    })

    agent = LangChainMedicationAnalyzer(enable_logging=False)

    def _call(*a, **k):
        if k.get("audit_step") == "medication_layering":
            return "## ✅ Conclusions\nc\n\n## 🧠 The Reasoning\nr"
        return med_json

    monkeypatch.setattr(agent, "_call_llm", _call)
    result = agent.analyze_medication(MedicationInput(medication_name="Metformin"))

    prac = result.practitioner_report
    assert "Analysis confidence: 0.80/1.00" in prac
    assert "Cimetidine" in prac  # interaction evidence in appendix
    assert "Black Box Warnings" in prac
    assert "Lactic acidosis risk" in prac


def test_procedure_patient_report_has_no_stats(monkeypatch):
    """Procedure patient report is plain; stats live only in practitioner report."""
    from langchain_agents import LangChainMedicalReasoningAgent

    responses = iter([
        json.dumps({"organs": ["kidneys"]}),
        json.dumps([{
            "organ_name": "kidneys", "affected_by_procedure": True, "at_risk": True,
            "risk_level": "high", "pathways_involved": ["renal"], "known_recommendations": ["Hydrate"],
            "potential_recommendations": [], "debunked_claims": [], "evidence_quality": "moderate",
        }]),
        json.dumps({
            "procedure_summary": "CT with contrast", "confidence_score": 0.9,
            "general_recommendations": ["Hydrate well"], "research_gaps": [],
        }),
    ])

    agent = LangChainMedicalReasoningAgent(enable_logging=False)

    def _call(*a, **k):
        if k.get("audit_step") == "procedure_layering":
            return "## ✅ Conclusions\nHydrate.\n\n## 🧠 The Reasoning\nKidneys at risk."
        return next(responses)

    monkeypatch.setattr(agent, "_call_llm", _call)
    result = agent.analyze_medical_procedure(
        MedicalInput(procedure="CT", details="With contrast", objectives=("risks",))
    )

    assert "Per-Organ Evidence & Risk Grading" in result.practitioner_report
    assert "Statistical Appendix" not in result.patient_report
    assert "practitioner report" in result.patient_report.lower()


# ── Stage C: references collected by procedure & medication ───────────────────

def test_procedure_collects_references(monkeypatch):
    from langchain_agents import LangChainMedicalReasoningAgent

    CITATION = "Smith (2024). Contrast nephropathy. NEJM. https://doi.org/10.1/x"
    responses = iter([
        json.dumps({"organs": ["kidneys"]}),
        json.dumps([{
            "organ_name": "kidneys", "affected_by_procedure": True, "at_risk": True,
            "risk_level": "high", "pathways_involved": [], "known_recommendations": [],
            "potential_recommendations": [], "debunked_claims": [], "evidence_quality": "moderate",
        }]),
        json.dumps({
            "procedure_summary": "CT", "confidence_score": 0.9,
            "general_recommendations": ["Hydrate"], "research_gaps": [],
            "references": [CITATION],
        }),
    ])
    agent = LangChainMedicalReasoningAgent(enable_logging=False)

    def _call(*a, **k):
        if k.get("audit_step") == "procedure_layering":
            return "## ✅ Conclusions\nc\n\n## 🧠 The Reasoning\nr"
        return next(responses)

    monkeypatch.setattr(agent, "_call_llm", _call)
    result = agent.analyze_medical_procedure(
        MedicalInput(procedure="CT", details="contrast", objectives=("risks",))
    )
    assert any(CITATION in r["raw_citation"] for r in result.references)


def test_medication_collects_references(monkeypatch):
    from langchain_agents import LangChainMedicationAnalyzer

    CITATION = "Jones (2023). Metformin safety. Lancet. https://doi.org/10.1/y"
    med_json = json.dumps({
        "medication_name": "Metformin", "drug_class": "Biguanide",
        "mechanism_of_action": "m", "absorption": "a", "metabolism": "b",
        "elimination": "e", "half_life": "6h", "approved_indications": [],
        "off_label_uses": [], "standard_dosing": "", "dose_adjustments": {},
        "common_adverse_effects": [], "serious_adverse_effects": [],
        "contraindications": [], "black_box_warnings": [], "drug_interactions": [],
        "food_interactions": [], "environmental_considerations": [],
        "evidence_based_recommendations": [], "what_not_to_do": [], "debunked_claims": [],
        "monitoring_requirements": [], "warning_signs": [], "evidence_quality": "moderate",
        "analysis_confidence": 0.8, "references": [CITATION],
    })
    agent = LangChainMedicationAnalyzer(enable_logging=False)

    def _call(*a, **k):
        if k.get("audit_step") == "medication_layering":
            return "## ✅ Conclusions\nc\n\n## 🧠 The Reasoning\nr"
        return med_json

    monkeypatch.setattr(agent, "_call_llm", _call)
    result = agent.analyze_medication(MedicationInput(medication_name="Metformin"))
    assert any(CITATION in r["raw_citation"] for r in result.references)


def test_collect_validated_references_reads_flat_references():
    """Orchestrator collector must handle objects exposing a flat .references list."""
    import sys
    from unittest.mock import MagicMock
    sys.modules.setdefault("pdf_generator", MagicMock())
    from run_analysis import AgentOrchestrator

    orch = AgentOrchestrator.__new__(AgentOrchestrator)
    orch._reference_validation_cache = {}
    orch._citation_url_validator = None

    class _Result:
        references = [{"raw_citation": "Smith (2024). Title. J. https://doi.org/10.1/z"}]

    # Stub the URL resolver so no network call happens; treat citation as valid.
    orch._resolve_reference_url = lambda citation, url, validator: (
        "https://doi.org/10.1/z", None, 1.0, None,
    )
    orch._get_citation_url_validator = lambda: MagicMock()

    kept, removed = orch._collect_validated_references(_Result())
    assert kept and "Smith (2024)" in kept[0]
    assert removed == []
