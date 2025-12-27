#!/usr/bin/env python3
"""
Router integration tests with LangChain implementation (mocked LLM routing + agent calls).
"""

import json
from pathlib import Path

import pytest

import langchain_agents.base as lc_base
import router
from router import AgentSpec
from run_analysis import AgentOrchestrator


class _DummyManager:
    configs: list = []

    def get_available_provider(self):
        return object()


def _fake_call_llm(self, system_prompt: str, user_prompt: str, **kwargs):
    schema = kwargs.get("schema", "")
    schema_text = json.dumps(schema) if isinstance(schema, dict) else str(schema)

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


def _sample_agents():
    return [
        AgentSpec(
            id="medication_agent",
            name="Medication Specialist",
            description="Handles medication queries",
        ),
        AgentSpec(
            id="procedure_agent",
            name="Medical Procedure Specialist",
            description="Handles procedure queries",
        ),
        AgentSpec(
            id="diagnostic_agent",
            name="Diagnostic Specialist",
            description="Handles diagnostic queries",
        ),
        AgentSpec(
            id="general_agent",
            name="General Medical Assistant",
            description="Handles general queries",
        ),
    ]


@pytest.mark.parametrize(
    ("agent_id", "expected_key"),
    [
        ("medication_agent", "result"),
        ("procedure_agent", "summary"),
        ("diagnostic_agent", "patient_report"),
    ],
)
def test_router_langchain_integration(monkeypatch, tmp_path, agent_id, expected_key):
    monkeypatch.setattr(router, "call_llm", lambda *args, **kwargs: agent_id)

    agents = _sample_agents()
    selected = router.route_agent("test query", agents, model=router.DEFAULT_ROUTING_MODEL)

    assert selected == agent_id

    orchestrator = AgentOrchestrator(output_dir=str(tmp_path))
    if selected == "medication_agent":
        _, files = orchestrator.run_medication_analyzer(
            medication="Metformin",
            indication="Type 2 diabetes",
            other_medications=["Cimetidine"],
            llm_provider="claude-sonnet",
            timeout=30,
            implementation="langchain",
        )
    elif selected == "procedure_agent":
        _, files = orchestrator.run_procedure_analyzer(
            procedure="MRI Scanner",
            details="With contrast",
            llm_provider="claude-sonnet",
            timeout=30,
            implementation="langchain",
        )
    else:
        _, files = orchestrator.run_fact_checker(
            subject="Vitamin D",
            context="",
            llm_provider="claude-sonnet",
            timeout=30,
            implementation="langchain",
        )

    assert expected_key in files
    assert Path(files[expected_key]).exists()
