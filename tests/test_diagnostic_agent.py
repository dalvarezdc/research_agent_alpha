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


def test_pipeline_produces_layered_reports_and_probabilities(monkeypatch):
    agent = _make_agent()

    # Use a real symptom from the database so scoring runs.
    a_symptom = next(iter(agent.engine.all_symptoms))

    def fake_call(*args, **kwargs):
        step = kwargs.get("audit_step", "")
        if step == "diagnostic_level1_extraction":
            return json.dumps({
                "symptoms": [a_symptom], "negative_symptoms": [],
                "duration": "2 days", "severity": "mild", "is_vague": False,
                "clarification_question": None,
            })
        if step == "diagnostic_level5_report":
            return json.dumps({
                "top_5_candidates": ["Cond A", "Cond B"],
                "most_probable": "Cond A",
                "most_serious": "Cond B",
                "reasoning_summary": "Reasoning here.",
                "recommended_next_steps": ["See a doctor"],
                "suggested_agent": "medication_agent",
                "routing_rationale": "Drug-based",
                "references": ["Smith (2024). Title. J. https://doi.org/10.1/x"],
            })
        if step == "diagnostic_layering":
            return "## ✅ Conclusions\nCond A likely.\n\n## 🧠 The Reasoning\nBased on symptoms."
        return "A friendly question?"

    monkeypatch.setattr(agent, "_call_llm", fake_call)

    result = agent.run_diagnostic_pipeline("I have symptoms")

    # Bayesian probabilities present.
    assert result["probabilities"]
    assert all("probability" in r for r in result["probabilities"])

    # Layered reports present; stats only in practitioner.
    assert result["patient_report"]
    assert "Statistical Appendix" in result["practitioner_report"]
    assert "Statistical Appendix" not in result["patient_report"]
    # Exact posteriors survive in practitioner appendix.
    assert "Condition Probabilities" in result["practitioner_report"]

    # References collected.
    assert any("doi.org/10.1/x" in c for c in result["references"])


def test_cost_tracker_synced(monkeypatch):
    agent = _make_agent()
    a_symptom = next(iter(agent.engine.all_symptoms))

    def fake_call(*args, **kwargs):
        step = kwargs.get("audit_step", "")
        if step == "diagnostic_level1_extraction":
            return json.dumps({
                "symptoms": [a_symptom], "negative_symptoms": [],
                "duration": None, "severity": None, "is_vague": False,
                "clarification_question": None,
            })
        if step == "diagnostic_level5_report":
            return json.dumps({
                "top_5_candidates": ["A"], "most_probable": "A", "most_serious": "A",
                "reasoning_summary": "r", "recommended_next_steps": ["x"],
                "suggested_agent": "procedure_agent", "routing_rationale": "y",
                "references": [],
            })
        if step == "diagnostic_layering":
            return "## ✅ Conclusions\nc\n\n## 🧠 The Reasoning\nr"
        return "q"

    monkeypatch.setattr(agent, "_call_llm", fake_call)
    agent.run_diagnostic_pipeline("symptoms")
    # Cost tracker was synced from the module-level tracker without error.
    assert hasattr(agent.cost_tracker, "_phase_costs")


def test_default_provider_is_claude_sonnet():
    agent = _make_agent()
    assert agent.provider_name == "claude-sonnet"
