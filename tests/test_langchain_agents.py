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
    monkeypatch.setattr(agent, "_call_llm", lambda *args, **kwargs: next(responses))

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
    monkeypatch.setattr(agent, "_call_llm", lambda *args, **kwargs: response)

    result = agent.analyze_medication(MedicationInput(medication_name="Metformin"))

    assert result.medication_name == "Metformin"
    assert result.drug_class == "Biguanide"
    assert result.drug_interactions
    assert result.analysis_confidence == 0.8


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
