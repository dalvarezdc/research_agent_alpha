#!/usr/bin/env python3
"""Unit tests for the LangChain-migrated MedicalDiagnosticAgent (mocked LLM)."""

import json

import pytest

import langchain_agents.base as lc_base
from llm_integrations import TokenUsage


class _DummyProvider:
    def generate_response(self, prompt: str, system_prompt: str | None = None):
        return "{}", TokenUsage()


class _DummyManager:
    configs = []

    def get_available_provider(self):
        return _DummyProvider()


@pytest.fixture(autouse=True)
def _mock_llm_manager(monkeypatch):
    monkeypatch.setattr(
        lc_base, "create_llm_manager", lambda *a, **k: _DummyManager()
    )


def _make_agent():
    from medical_diagnostic_analyzer.diagnostic_agent import MedicalDiagnosticAgent

    return MedicalDiagnosticAgent(enable_logging=False, interactive=False)


def _fake_llm_router():
    """Return a fake _call_llm that answers each diagnostic step."""

    def fake_call(*args, **kwargs):
        step = kwargs.get("audit_step", "")
        if step == "diagnostic_level1_extraction":
            return json.dumps(
                {
                    "symptoms": ["right knee pain", "swelling after activity"],
                    "negative_symptoms": ["fever"],
                    "duration": "2 weeks",
                    "severity": "moderate",
                    "is_vague": False,
                    "clarification_question": None,
                    "clinical_context": "started after a hike",
                }
            )
        if step == "diagnostic_level2_differential":
            return json.dumps(
                {
                    "candidates": [
                        {
                            "name": "Patellofemoral pain syndrome",
                            "probability": 0.4,
                            "severity": 1,
                            "rationale": "Activity-related anterior knee pain",
                        },
                        {
                            "name": "Meniscal irritation",
                            "probability": 0.25,
                            "severity": 2,
                            "rationale": "Post-activity swelling",
                        },
                        {
                            "name": "Septic arthritis",
                            "probability": 0.05,
                            "severity": 5,
                            "rationale": "Cannot-miss infectious arthritis",
                        },
                        {
                            "name": "Osteoarthritis flare",
                            "probability": 0.2,
                            "severity": 2,
                            "rationale": "Possible degenerative contribution",
                        },
                        {
                            "name": "Medial collateral ligament strain",
                            "probability": 0.1,
                            "severity": 2,
                            "rationale": "Possible soft-tissue strain",
                        },
                    ],
                    "differentiating_symptoms": [
                        "locking",
                        "giving way",
                        "true night pain",
                    ],
                    "recommended_exams": [
                        "Focused knee exam",
                        "Ottawa knee rules X-ray if indicated",
                    ],
                    "clinical_summary": (
                        "Activity-related knee pain with swelling most consistent "
                        "with mechanical/overuse causes; rule out infection."
                    ),
                }
            )
        if step == "diagnostic_level5_report":
            return json.dumps(
                {
                    "top_5_candidates": [
                        "Patellofemoral pain syndrome",
                        "Meniscal irritation",
                        "Osteoarthritis flare",
                        "Medial collateral ligament strain",
                        "Septic arthritis",
                    ],
                    "most_probable": "Patellofemoral pain syndrome",
                    "most_serious": "Septic arthritis",
                    "reasoning_summary": "Activity-related knee pain without fever.",
                    "recommended_next_steps": [
                        "See a clinician for knee exam within a few days"
                    ],
                    "diagnostic_tests": [
                        {
                            "name": "Focused Knee Exam & McMurray Test",
                            "tier": "Tier 0: Safety & Baseline",
                            "clinical_purpose": "Assess for mechanical joint line tenderness or meniscal tear",
                            "actionable_trigger": "Positive McMurray or joint line pain prompts knee MRI; absent signs favor patellofemoral therapy",
                        },
                        {
                            "name": "Knee Plain Radiographs (Ottawa Knee Rules)",
                            "tier": "Tier 1: Routine Imaging",
                            "clinical_purpose": "Rule out acute fracture or severe osteoarthritic joint space narrowing",
                            "actionable_trigger": "Joint space narrowing guides degenerative management; normal X-ray supports soft tissue etiology",
                        },
                    ],
                    "conditional_therapies": [
                        {
                            "trigger_condition": "Confirmed Patellofemoral Syndrome",
                            "regimen_name": "Targeted Physical Therapy & VMO Strengthening",
                            "details": "Quadriceps strengthening, patellar taping, activity modification for 6-8 weeks",
                        }
                    ],
                    "patient_supportive_care": {
                        "dietary_guidance": ["Maintain balanced anti-inflammatory nutrition and adequate hydration"],
                        "hydration_and_lifestyle": ["RICE protocol: rest, ice for 15-20 min, elevation after activity"],
                        "medication_warnings": ["Avoid taking high-dose NSAIDs for prolonged periods without clinical supervision"],
                        "questions_for_doctor": [
                            "Do you recommend physical therapy for my knee?",
                            "Are X-rays or an MRI necessary to evaluate my joint?",
                        ],
                        "er_warning_signs": [
                            "Inability to bear any weight on the knee",
                            "Rapid severe swelling with high fever, redness, or heat",
                        ],
                    },
                    "escalation_triggers": [
                        "Acute inability to bear weight with joint effusion and fever > 38.5C (suspect septic joint)",
                    ],
                    "suggested_agent": "procedure_agent",
                    "routing_rationale": "May need imaging or procedures",
                    "references": [
                        "Smith (2024). Knee pain evaluation. J. https://doi.org/10.1/x"
                    ],
                }
            )
        if step == "diagnostic_layering":
            return (
                "## ✅ Report Summary\nPatellofemoral pain likely.\n\n"
                "## 🧠 The Reasoning\nBased on activity-related knee symptoms."
            )
        return "A friendly question?"

    return fake_call


def test_pipeline_produces_layered_reports_and_probabilities(monkeypatch):
    agent = _make_agent()
    monkeypatch.setattr(agent, "_call_llm", _fake_llm_router())

    result = agent.run_diagnostic_pipeline("I have right knee pain after hiking")

    # Likelihood estimates present and normalized.
    assert result["probabilities"]
    assert all("probability" in r for r in result["probabilities"])
    assert abs(sum(r["probability"] for r in result["probabilities"]) - 1.0) < 1e-6
    assert result["probabilities"][0]["name"] == "Patellofemoral pain syndrome"

    # Free-form extraction, not a fixed vocabulary.
    assert "knee" in " ".join(result["extraction"]["symptoms"]).lower()

    # Differentiated reports:
    patient = result["patient_report"]
    practitioner = result["practitioner_report"]

    # Practitioner report checks:
    assert "Statistical Appendix" in practitioner
    assert "Condition Likelihood Estimates" in practitioner
    assert "Differential Diagnosis Matrix" in practitioner
    assert "Clinical Purpose:" in practitioner
    assert "Actionable Decision Trigger:" in practitioner
    assert "Conditional Pharmacotherapy & Management Pathways" in practitioner

    # Patient report checks:
    assert "Statistical Appendix" not in patient
    assert "Summary" in patient
    assert "Questions to Ask Your Doctor" in patient
    assert "Supportive Dietary & Daily Care" in patient
    assert "Emergency Warning Signs" in patient
    assert "Important Medication & Safety Warnings" in patient

    # References collected.
    assert any("doi.org/10.1/x" in c for c in result["references"])

    # No dependency on a fixed symptom engine/database.
    assert not hasattr(agent, "engine")


def test_cost_tracker_synced(monkeypatch):
    agent = _make_agent()
    monkeypatch.setattr(agent, "_call_llm", _fake_llm_router())
    agent.run_diagnostic_pipeline("knee pain")
    assert hasattr(agent.cost_tracker, "_phase_costs")


def test_default_provider_is_claude_sonnet():
    agent = _make_agent()
    assert agent.provider_name == "claude-sonnet"


def test_knee_pain_not_mapped_to_respiratory_priors(monkeypatch):
    """Regression: empty DB priors used to rank cold/migraine for any query."""
    agent = _make_agent()
    monkeypatch.setattr(agent, "_call_llm", _fake_llm_router())

    result = agent.run_diagnostic_pipeline("knee pain")
    names = " ".join(r["name"].lower() for r in result["probabilities"])
    assert "cold" not in names
    assert "migraine" not in names
    assert "flu" not in names
    assert "covid" not in names
    assert any("knee" in r["name"].lower() or "menisc" in r["name"].lower()
               or "patello" in r["name"].lower() or "arthritis" in r["name"].lower()
               for r in result["probabilities"])


def test_clinical_framing_is_diagnostic_specialist_not_factchecker():
    """Prompt contract: differential specialist, not multi-perspective fact-check."""
    from medical_diagnostic_analyzer import diagnostic_agent as da

    framing = da._CLINICAL_COMMON_SENSE.lower()
    assert "diagnostic specialist" in framing
    assert "decision-support" in framing or "decision support" in framing
    assert "not" in framing and "fact-check" in framing
    assert "relative" in framing
    assert "cannot-miss" in framing or "cannot miss" in framing


def test_fallback_differential_is_fail_closed():
    agent = _make_agent()
    fb = agent._fallback_differential()
    assert len(fb.candidates) == 1
    c = fb.candidates[0]
    assert c.name.startswith("Undifferentiated presentation")
    assert c.probability == 1.0
    assert c.severity == 2
    assert fb.recommended_exams == ["In-person clinical assessment"]
    assert fb.differentiating_symptoms == []


def test_level2_empty_candidates_uses_fallback(monkeypatch):
    agent = _make_agent()

    def fake_call(*args, **kwargs):
        step = kwargs.get("audit_step", "")
        if step == "diagnostic_level2_differential":
            return json.dumps(
                {
                    "candidates": [],
                    "differentiating_symptoms": [],
                    "recommended_exams": [],
                    "clinical_summary": "empty",
                }
            )
        return _fake_llm_router()(*args, **kwargs)

    monkeypatch.setattr(agent, "_call_llm", fake_call)
    result = agent.run_diagnostic_pipeline("vague knee discomfort after a hike")
    assert result["probabilities"][0]["name"].startswith("Undifferentiated")
    assert result["probabilities"][0]["severity"] == 2


def test_noninteractive_skips_level3_question(monkeypatch):
    """API mode: interactive=False should not call Level 3 question phrasing."""
    agent = _make_agent()
    assert agent.interactive is False
    steps: list[str] = []

    base = _fake_llm_router()

    def tracking_call(*args, **kwargs):
        steps.append(kwargs.get("audit_step", ""))
        return base(*args, **kwargs)

    monkeypatch.setattr(agent, "_call_llm", tracking_call)
    agent.run_diagnostic_pipeline("right knee pain after hiking")
    assert "diagnostic_level3_question" not in steps
    assert "diagnostic_level1_extraction" in steps
    assert "diagnostic_level2_differential" in steps
    assert "diagnostic_level5_report" in steps
