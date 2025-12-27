#!/usr/bin/env python3
"""
Integration tests for LangChain agent wiring through AgentOrchestrator.
"""

import json
from pathlib import Path

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
                "investigational_approaches": [],
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
