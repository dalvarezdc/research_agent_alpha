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
                    "suggested_agent": "procedure_agent",
                    "routing_rationale": "May need imaging or procedures",
                    "references": [
                        "Smith (2024). Knee pain evaluation. J. https://doi.org/10.1/x"
                    ],
                }
            )
        if step == "diagnostic_layering":
            return (
                "## ✅ Conclusions\nPatellofemoral pain likely.\n\n"
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

    # Layered reports present; stats only in practitioner.
    assert result["patient_report"]
    assert "Statistical Appendix" in result["practitioner_report"]
    assert "Statistical Appendix" not in result["patient_report"]
    assert "Condition Likelihood Estimates" in result["practitioner_report"]

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
