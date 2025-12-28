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
    responses = iter(
        [
            json.dumps(
                {
                    "official_narrative": "Official view",
                    "counter_narrative": "Counter view",
                    "key_conflicts": "Key conflicts",
                    "references": ["Ref 1", "Ref 2"],
                }
            ),
            json.dumps(
                {
                    "industry_funded_studies": "Industry studies",
                    "independent_research": "Independent studies",
                    "methodology_quality": "Mixed",
                    "anecdotal_signals": "Anecdotes",
                    "time_weighted_evidence": "Recent data",
                    "references": ["Ref 3"],
                }
            ),
            json.dumps(
                {
                    "biological_truth": "Most likely truth",
                    "industry_bias": "Biases",
                    "grey_zone": "Open questions",
                    "references": ["Ref 4"],
                }
            ),
            "REPORT BODY\n\nREFERENCES\n[1] Ref 1",
            "Simplified output",
        ]
    )

    agent = LangChainMedicalFactChecker(enable_logging=False, interactive=False)
    monkeypatch.setattr(agent, "_call_llm", lambda *args, **kwargs: next(responses))

    session = agent.start_analysis("Vitamin D")

    assert session.final_output == "Simplified output"
    assert session.practitioner_report
    assert len(session.phase_results) == 5
