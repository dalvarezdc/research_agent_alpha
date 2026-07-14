#!/usr/bin/env python3
"""
Integration tests for LangChain agent wiring through AgentOrchestrator.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import langchain_agents.base as lc_base
from medical_procedure_analyzer import MedicalOutput
from medical_procedure_analyzer.medication_analyzer import MedicationOutput
from run_analysis import AgentOrchestrator


class _DummyManager:
    configs: list = []

    def get_available_provider(self):
        return object()


def _fake_call_llm(self, system_prompt: str, user_prompt: str, **kwargs):
    schema = kwargs.get("schema", "")
    if isinstance(schema, dict):
        schema_text = json.dumps(schema)
    else:
        schema_text = str(schema)

    if "procedure_summary" in schema_text:
        return json.dumps(
            {
                "procedure_summary": "MRI Scanner - With contrast",
                "confidence_score": 0.82,
                "general_recommendations": ["Hydrate before and after procedure"],
                "research_gaps": ["Long-term gadolinium retention studies"],
            }
        )

    if "organ_name" in schema_text:
        return json.dumps(
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
                }
            ]
        )

    if "organs" in schema_text:
        return json.dumps({"organs": ["kidneys", "brain"]})

    if "medication_name" in schema_text:
        return json.dumps(
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
                "contraindications": [
                    {"condition": "Severe renal impairment", "severity": "absolute"}
                ],
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

    if "official_narrative" in schema_text:
        return json.dumps(
            {
                "official_narrative": "Official view",
                "counter_narrative": "Counter view",
                "key_conflicts": "Key conflicts",
                "references": ["Ref 1"],
            }
        )

    if "industry_funded_studies" in schema_text:
        return json.dumps(
            {
                "industry_funded_studies": "Industry studies",
                "independent_research": "Independent studies",
                "methodology_quality": "Mixed",
                "anecdotal_signals": "Anecdotes",
                "time_weighted_evidence": "Recent data",
                "references": ["Ref 2"],
            }
        )

    if "biological_truth" in schema_text:
        return json.dumps(
            {
                "biological_truth": "Most likely truth",
                "industry_bias": "Biases",
                "grey_zone": "Open questions",
                "references": ["Ref 3"],
            }
        )

    if "simplify" in user_prompt.lower():
        return "Simplified output"

    return "REPORT BODY\n\nREFERENCES\n[1] Ref 1"


@pytest.fixture(autouse=True)
def _patch_langchain(monkeypatch):
    monkeypatch.setattr(lc_base, "create_llm_manager", lambda *args, **kwargs: _DummyManager())
    monkeypatch.setattr(lc_base.LangChainAgentBase, "_call_llm", _fake_call_llm)


@pytest.fixture(autouse=True)
def _patch_pdf_generation(monkeypatch):
    import run_analysis as ra

    monkeypatch.setattr(ra, "convert_markdown_to_pdf_safe", lambda *args, **kwargs: None)


def test_orchestrator_langchain_procedure(tmp_path):
    orchestrator = AgentOrchestrator(output_dir=str(tmp_path))
    result, files = orchestrator.run_procedure_analyzer(
        procedure="MRI Scanner",
        details="With contrast",
        llm_provider="claude-sonnet",
        timeout=30,
        implementation="langchain",
    )

    assert isinstance(result, MedicalOutput)
    assert result.organs_analyzed
    assert "trace" in files
    assert "result" in files
    assert "summary" in files
    for key in ("trace", "result", "summary"):
        assert Path(files[key]).exists()


def test_orchestrator_langchain_medication(tmp_path):
    orchestrator = AgentOrchestrator(output_dir=str(tmp_path))
    result, files = orchestrator.run_medication_analyzer(
        medication="Metformin",
        indication="Type 2 diabetes",
        other_medications=["Cimetidine"],
        llm_provider="claude-sonnet",
        timeout=30,
        implementation="langchain",
    )

    assert isinstance(result, MedicationOutput)
    assert result.medication_name == "Metformin"
    assert "result" in files
    assert "summary" in files
    assert "detailed" in files
    for key in ("result", "summary", "detailed"):
        assert Path(files[key]).exists()


def test_orchestrator_langchain_factcheck(tmp_path):
    orchestrator = AgentOrchestrator(output_dir=str(tmp_path))
    session, files = orchestrator.run_fact_checker(
        subject="Vitamin D",
        context="",
        llm_provider="claude-sonnet",
        timeout=30,
        implementation="langchain",
    )

    assert session.final_output
    assert "session" in files
    assert "patient_report" in files
    assert "summary" in files
    for key in ("session", "patient_report", "summary"):
        assert Path(files[key]).exists()


# ===========================================================================
# Tests: document_context threading through run_* methods
# ===========================================================================

def test_document_context_field_declared_on_base():
    """document_context must be declared in LangChainAgentBase.__init__."""
    import inspect
    import langchain_agents.base as lc_base

    source = inspect.getsource(lc_base.LangChainAgentBase.__init__)
    assert "document_context" in source, (
        "LangChainAgentBase.__init__ must declare self.document_context"
    )


def test_procedure_document_context_set_on_agent(monkeypatch, tmp_path):
    """run_procedure_analyzer sets agent.document_context = stripped value."""
    captured = {}

    import langchain_agents as lc_pkg

    OrigClass = lc_pkg.LangChainMedicalReasoningAgent

    class _CapturingAgent(OrigClass):
        def analyze_medical_procedure(self, medical_input):
            captured["document_context"] = self.document_context
            return super().analyze_medical_procedure(medical_input)

    monkeypatch.setattr(
        lc_pkg, "LangChainMedicalReasoningAgent", _CapturingAgent
    )

    orchestrator = AgentOrchestrator(output_dir=str(tmp_path))
    orchestrator.run_procedure_analyzer(
        procedure="MRI Scanner",
        details="With contrast",
        llm_provider="claude-sonnet",
        timeout=30,
        implementation="langchain",
        document_context="Some document context",
    )

    assert captured.get("document_context") == "Some document context"


def test_procedure_document_context_whitespace_becomes_none(monkeypatch, tmp_path):
    """Whitespace-only document_context is coerced to None."""
    captured = {}

    import langchain_agents as lc_pkg

    OrigClass = lc_pkg.LangChainMedicalReasoningAgent

    class _CapturingAgent(OrigClass):
        def analyze_medical_procedure(self, medical_input):
            captured["document_context"] = self.document_context
            return super().analyze_medical_procedure(medical_input)

    monkeypatch.setattr(lc_pkg, "LangChainMedicalReasoningAgent", _CapturingAgent)

    orchestrator = AgentOrchestrator(output_dir=str(tmp_path))
    orchestrator.run_procedure_analyzer(
        procedure="MRI Scanner",
        details="With contrast",
        llm_provider="claude-sonnet",
        timeout=30,
        implementation="langchain",
        document_context="   ",
    )

    assert captured.get("document_context") is None


def test_procedure_document_context_empty_becomes_none(monkeypatch, tmp_path):
    """Empty string document_context is coerced to None."""
    captured = {}

    import langchain_agents as lc_pkg

    OrigClass = lc_pkg.LangChainMedicalReasoningAgent

    class _CapturingAgent(OrigClass):
        def analyze_medical_procedure(self, medical_input):
            captured["document_context"] = self.document_context
            return super().analyze_medical_procedure(medical_input)

    monkeypatch.setattr(lc_pkg, "LangChainMedicalReasoningAgent", _CapturingAgent)

    orchestrator = AgentOrchestrator(output_dir=str(tmp_path))
    orchestrator.run_procedure_analyzer(
        procedure="MRI Scanner",
        details="With contrast",
        llm_provider="claude-sonnet",
        timeout=30,
        implementation="langchain",
        document_context="",
    )

    assert captured.get("document_context") is None


def test_medication_document_context_whitespace_becomes_none(monkeypatch, tmp_path):
    """Whitespace-only document_context is coerced to None for medication agent."""
    captured = {}

    import langchain_agents as lc_pkg

    OrigClass = lc_pkg.LangChainMedicationAnalyzer

    class _CapturingAgent(OrigClass):
        def analyze_medication(self, medication_input):
            captured["document_context"] = self.document_context
            return super().analyze_medication(medication_input)

    monkeypatch.setattr(lc_pkg, "LangChainMedicationAnalyzer", _CapturingAgent)

    orchestrator = AgentOrchestrator(output_dir=str(tmp_path))
    orchestrator.run_medication_analyzer(
        medication="Metformin",
        indication="Type 2 diabetes",
        other_medications=["Cimetidine"],
        llm_provider="claude-sonnet",
        timeout=30,
        implementation="langchain",
        document_context="   ",
    )

    assert captured.get("document_context") is None


def test_medication_document_context_empty_becomes_none(monkeypatch, tmp_path):
    """Empty string document_context is coerced to None for medication agent."""
    captured = {}

    import langchain_agents as lc_pkg

    OrigClass = lc_pkg.LangChainMedicationAnalyzer

    class _CapturingAgent(OrigClass):
        def analyze_medication(self, medication_input):
            captured["document_context"] = self.document_context
            return super().analyze_medication(medication_input)

    monkeypatch.setattr(lc_pkg, "LangChainMedicationAnalyzer", _CapturingAgent)

    orchestrator = AgentOrchestrator(output_dir=str(tmp_path))
    orchestrator.run_medication_analyzer(
        medication="Metformin",
        indication="Type 2 diabetes",
        other_medications=["Cimetidine"],
        llm_provider="claude-sonnet",
        timeout=30,
        implementation="langchain",
        document_context="",
    )

    assert captured.get("document_context") is None


def test_factcheck_document_context_whitespace_becomes_none(monkeypatch, tmp_path):
    """Whitespace-only document_context is coerced to None for fact-checker agent."""
    captured = {}

    import langchain_agents as lc_pkg

    OrigClass = lc_pkg.LangChainMedicalFactChecker

    class _CapturingAgent(OrigClass):
        def start_analysis(self, subject, clarifying_info=""):
            captured["document_context"] = self.document_context
            return super().start_analysis(subject, clarifying_info)

    monkeypatch.setattr(lc_pkg, "LangChainMedicalFactChecker", _CapturingAgent)

    orchestrator = AgentOrchestrator(output_dir=str(tmp_path))
    orchestrator.run_fact_checker(
        subject="Vitamin D",
        context="",
        llm_provider="claude-sonnet",
        timeout=30,
        implementation="langchain",
        document_context="   ",
    )

    assert captured.get("document_context") is None


def test_factcheck_document_context_empty_becomes_none(monkeypatch, tmp_path):
    """Empty string document_context is coerced to None for fact-checker agent."""
    captured = {}

    import langchain_agents as lc_pkg

    OrigClass = lc_pkg.LangChainMedicalFactChecker

    class _CapturingAgent(OrigClass):
        def start_analysis(self, subject, clarifying_info=""):
            captured["document_context"] = self.document_context
            return super().start_analysis(subject, clarifying_info)

    monkeypatch.setattr(lc_pkg, "LangChainMedicalFactChecker", _CapturingAgent)

    orchestrator = AgentOrchestrator(output_dir=str(tmp_path))
    orchestrator.run_fact_checker(
        subject="Vitamin D",
        context="",
        llm_provider="claude-sonnet",
        timeout=30,
        implementation="langchain",
        document_context="",
    )

    assert captured.get("document_context") is None


def test_medication_document_context_set_on_agent(monkeypatch, tmp_path):
    """run_medication_analyzer threads document_context to agent."""
    captured = {}

    import langchain_agents as lc_pkg

    OrigClass = lc_pkg.LangChainMedicationAnalyzer

    class _CapturingAgent(OrigClass):
        def analyze_medication(self, medical_input):
            captured["document_context"] = self.document_context
            return super().analyze_medication(medical_input)

    monkeypatch.setattr(lc_pkg, "LangChainMedicationAnalyzer", _CapturingAgent)

    orchestrator = AgentOrchestrator(output_dir=str(tmp_path))
    orchestrator.run_medication_analyzer(
        medication="Metformin",
        indication="Type 2 diabetes",
        other_medications=["Cimetidine"],
        llm_provider="claude-sonnet",
        timeout=30,
        implementation="langchain",
        document_context="Medication context doc",
    )

    assert captured.get("document_context") == "Medication context doc"


def test_factcheck_document_context_set_on_agent(monkeypatch, tmp_path):
    """run_fact_checker threads document_context to LangChain agent."""
    captured = {}

    import langchain_agents as lc_pkg

    OrigClass = lc_pkg.LangChainMedicalFactChecker

    class _CapturingAgent(OrigClass):
        def start_analysis(self, subject, context=""):
            captured["document_context"] = self.document_context
            return super().start_analysis(subject, context)

    monkeypatch.setattr(lc_pkg, "LangChainMedicalFactChecker", _CapturingAgent)

    orchestrator = AgentOrchestrator(output_dir=str(tmp_path))
    orchestrator.run_fact_checker(
        subject="Vitamin D",
        context="",
        llm_provider="claude-sonnet",
        timeout=30,
        implementation="langchain",
        document_context="Fact check context",
    )

    assert captured.get("document_context") == "Fact check context"


def _make_stub_diagnostic_agent(captured: dict, return_query=True):
    """
    Build a MedicalDiagnosticAgent-compatible stub that:
    - Does NOT call any real LLM during __init__
    - Records the query passed to run_diagnostic_pipeline
    - Returns a minimal valid result dict
    """
    # We build a plain object that duck-types MedicalDiagnosticAgent.
    # Inheriting would trigger create_llm_manager in __init__, so we use
    # a completely independent class with the same public interface.

    class _StubDiagnosticAgent:
        def __init__(self, primary_llm_provider=None, fallback_providers=None,
                     enable_logging=True, interactive=True):
            self.interactive = interactive

        def run_diagnostic_pipeline(self, user_query: str) -> dict:
            if return_query:
                captured["query"] = user_query
            return {
                "probabilities": [
                    {"name": "Test condition", "probability": 0.75, "severity": 3}
                ],
                "report": {
                    "most_probable": "Test condition",
                    "most_serious": "Test serious",
                    "top_5_candidates": ["Test A"],
                    "reasoning_summary": "Test reasoning",
                    "recommended_next_steps": ["Step 1"],
                    "routing_rationale": "Test rationale",
                    "suggested_agent": "general_agent",
                },
            }

    return _StubDiagnosticAgent


def test_diagnostic_document_context_prepended_to_query(monkeypatch, tmp_path):
    """run_diagnostic_analyzer prepends document_context before the query."""
    captured = {}
    StubAgent = _make_stub_diagnostic_agent(captured)

    with patch("medical_diagnostic_analyzer.diagnostic_agent.MedicalDiagnosticAgent", StubAgent):
        orchestrator = AgentOrchestrator(output_dir=str(tmp_path))
        orchestrator.run_diagnostic_analyzer(
            query="fatigue and cold intolerance",
            llm_provider="claude-sonnet",
            timeout=30,
            interactive=False,
            document_context="Patient labs: TSH elevated.",
        )

    assert captured.get("query", "").startswith(
        "Document context:\nPatient labs: TSH elevated.\n\nQuestion:"
    ), f"Query was: {captured.get('query', '<not captured>')!r}"
    assert "fatigue and cold intolerance" in captured.get("query", "")


def test_diagnostic_empty_document_context_no_prepend(monkeypatch, tmp_path):
    """Empty document_context must NOT prepend anything to the diagnostic query."""
    captured = {}
    StubAgent = _make_stub_diagnostic_agent(captured)

    with patch("medical_diagnostic_analyzer.diagnostic_agent.MedicalDiagnosticAgent", StubAgent):
        orchestrator = AgentOrchestrator(output_dir=str(tmp_path))
        orchestrator.run_diagnostic_analyzer(
            query="headache",
            llm_provider="claude-sonnet",
            timeout=30,
            interactive=False,
            document_context="",
        )

    assert captured.get("query") == "headache", (
        f"Expected plain query 'headache', got {captured.get('query')!r}"
    )


def test_diagnostic_subject_text_uses_original_query(monkeypatch, tmp_path):
    """_persist_report_to_db must receive the original query, not the prepended string."""
    captured_subject = {}

    import run_analysis as ra

    def _capturing_persist(self, *, agent_type, subject_text, **kwargs):
        captured_subject["subject_text"] = subject_text

    monkeypatch.setattr(ra.AgentOrchestrator, "_persist_report_to_db", _capturing_persist)

    captured_query = {}
    StubAgent = _make_stub_diagnostic_agent(captured_query)

    with patch("medical_diagnostic_analyzer.diagnostic_agent.MedicalDiagnosticAgent", StubAgent):
        orchestrator = AgentOrchestrator(output_dir=str(tmp_path))
        orchestrator.run_diagnostic_analyzer(
            query="chest pain",
            llm_provider="claude-sonnet",
            timeout=30,
            interactive=False,
            document_context="Some patient file content here.",
        )

    assert captured_subject.get("subject_text") == "chest pain", (
        f"subject_text should be the original query, got {captured_subject.get('subject_text')!r}"
    )
