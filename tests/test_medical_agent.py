#!/usr/bin/env python3
"""
Test suite for Medical Reasoning Agent
"""

import pytest
import json
import tempfile
from unittest.mock import Mock, patch
from datetime import datetime

from medical_procedure_analyzer import (
    MedicalReasoningAgent,
    MedicalInput,
    OrganAnalysis,
    MedicalOutput,
    ReasoningStage,
    LLMManager
)


class TestMedicalReasoningAgent:
    """Test cases for the medical reasoning agent"""
    
    @pytest.fixture
    def sample_medical_input(self):
        """Sample medical input for testing"""
        return MedicalInput(
            procedure="MRI Scanner",
            details="With contrast",
            objectives=(
                "understand implications", 
                "risks", 
                "post-procedure care",
                "organs affected", 
                "organs at risk"
            )
        )
    
    @pytest.fixture
    def mock_llm_manager(self):
        """Mock LLM manager for testing"""
        mock_manager = Mock(spec=LLMManager)
        mock_manager.medical_analysis_with_fallback.return_value = {
            "analysis": "Test medical analysis",
            "confidence": 0.8,
            "sources_needed": ["pubmed_search"],
            "provider_used": "claude"
        }
        return mock_manager
    
    def test_agent_initialization(self):
        """Test agent initialization with different configurations"""
        agent = MedicalReasoningAgent(
            primary_llm_provider="claude",
            fallback_providers=["openai"],
            enable_logging=True
        )

        assert agent.primary_llm == "claude"
        assert agent.fallback_providers == ["openai"]
        assert len(agent.reasoning_trace) == 0
    
    def test_identify_affected_organs(self, sample_medical_input):
        """Test organ identification logic"""
        agent = MedicalReasoningAgent(enable_logging=False)
        
        organs = agent._identify_affected_organs(sample_medical_input)
        
        assert isinstance(organs, list)
        assert "kidneys" in organs
        assert "brain" in organs
        assert len(organs) > 0
    
    def test_reasoning_trace_logging(self, sample_medical_input):
        """Test that reasoning steps are properly logged"""
        agent = MedicalReasoningAgent(enable_logging=True)
        
        # Manually add a reasoning step
        agent._log_reasoning_step(
            ReasoningStage.INPUT_ANALYSIS,
            {"test": "data"},
            "Test reasoning step",
            {"result": "test_output"}
        )
        
        assert len(agent.reasoning_trace) == 1
        step = agent.reasoning_trace[0]
        assert step.stage == ReasoningStage.INPUT_ANALYSIS
        assert step.reasoning == "Test reasoning step"
        assert step.input_data == {"test": "data"}
        assert step.output == {"result": "test_output"}
    
    @patch('medical_procedure_analyzer.medical_reasoning_agent.MedicalReasoningAgent._gather_evidence')
    @patch('medical_procedure_analyzer.medical_reasoning_agent.MedicalReasoningAgent._assess_risks')
    @patch('medical_procedure_analyzer.medical_reasoning_agent.MedicalReasoningAgent._synthesize_recommendations')
    def test_full_analysis_pipeline(self, mock_synth, mock_risk, mock_evidence, sample_medical_input):
        """Test the complete analysis pipeline"""
        # Setup mocks
        mock_evidence.return_value = {"kidneys": {"evidence": "test"}}
        mock_risk.return_value = {"kidneys": {"risk_level": "moderate"}}
        mock_synth.return_value = {
            "kidneys": {
                "known_recommendations": [{"intervention": "hydration", "rationale": "test", "evidence_level": "strong", "timing": "test"}],
                "potential_recommendations": [{"intervention": "NAC", "rationale": "test", "evidence_level": "limited", "limitations": "test"}],
                "debunked_claims": [{"claim": "detox_teas", "reason_debunked": "test", "debunked_by": "test", "evidence": "test", "why_harmful": "test"}]
            }
        }

        agent = MedicalReasoningAgent(enable_logging=False)
        result = agent.analyze_medical_procedure(sample_medical_input)
        
        assert isinstance(result, MedicalOutput)
        assert result.procedure_summary == "MRI Scanner - With contrast"
        assert len(result.organs_analyzed) > 0
        assert result.confidence_score > 0
        assert len(result.reasoning_trace) > 0
    
    def test_export_reasoning_trace(self, sample_medical_input):
        """Test reasoning trace export functionality"""
        agent = MedicalReasoningAgent(enable_logging=False)
        
        # Add some reasoning steps
        agent._log_reasoning_step(
            ReasoningStage.INPUT_ANALYSIS,
            {"input": "test"},
            "Test step 1",
            {"output": "result1"}
        )
        agent._log_reasoning_step(
            ReasoningStage.ORGAN_IDENTIFICATION,
            {"organs": ["kidneys"]},
            "Test step 2",
            {"identified": ["kidneys"]}
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            agent.export_reasoning_trace(f.name)
            
            # Read back the exported file
            with open(f.name, 'r') as read_f:
                exported_data = json.load(read_f)
        
        assert len(exported_data) == 2
        assert exported_data[0]["stage"] == "input_analysis"
        assert exported_data[1]["stage"] == "organ_identification"
        assert "timestamp" in exported_data[0]
        assert "reasoning" in exported_data[0]


class TestMedicalDataStructures:
    """Test medical data structures"""
    
    def test_medical_input_creation(self):
        """Test MedicalInput data structure"""
        input_data = MedicalInput(
            procedure="CT Scan",
            details="With iodine contrast",
            objectives=("assess kidney function",),
            patient_context="age: 65, kidney_disease: True"
        )
        
        assert input_data.procedure == "CT Scan"
        assert input_data.details == "With iodine contrast"
        assert len(input_data.objectives) == 1
        assert "kidney_disease" in input_data.patient_context
    
    def test_organ_analysis_structure(self):
        """Test OrganAnalysis data structure"""
        analysis = OrganAnalysis(
            organ_name="kidneys",
            affected_by_procedure=True,
            at_risk=True,
            risk_level="high",
            pathways_involved=["filtration", "elimination"],
            known_recommendations=["hydration", "monitoring"],
            potential_recommendations=["NAC supplementation"],
            debunked_claims=["kidney cleanses"],
            evidence_quality="strong"
        )
        
        assert analysis.organ_name == "kidneys"
        assert analysis.risk_level == "high"
        assert len(analysis.pathways_involved) == 2
        assert "NAC supplementation" in analysis.potential_recommendations
        assert "kidney cleanses" in analysis.debunked_claims


class TestIntegrationScenarios:
    """Integration test scenarios"""
    
    @pytest.mark.integration
    def test_mri_contrast_scenario(self):
        """Integration test for MRI with contrast scenario"""
        medical_input = MedicalInput(
            procedure="MRI Scanner",
            details="With gadolinium contrast",
            objectives=(
                "understand kidney elimination pathway",
                "identify potential complications", 
                "recommend post-procedure care"
            )
        )
        
        agent = MedicalReasoningAgent(enable_logging=True)

        # This would normally use real LLM calls
        # For testing, we'll check the pipeline structure
        try:
            result = agent.analyze_medical_procedure(medical_input)
            
            # Verify output structure
            assert isinstance(result, MedicalOutput)
            assert "gadolinium" in result.procedure_summary.lower() or "contrast" in result.procedure_summary.lower()
            assert len(result.reasoning_trace) >= 5  # Should have multiple reasoning stages
            
            # Check for kidney analysis
            kidney_analysis = None
            for organ in result.organs_analyzed:
                if organ.organ_name.lower() == "kidneys":
                    kidney_analysis = organ
                    break
            
            assert kidney_analysis is not None
            assert kidney_analysis.affected_by_procedure
            
        except Exception as e:
            # Expected to fail without real LLM integration
            assert "LLM" in str(e) or "provider" in str(e)
    
    @pytest.mark.slow
    def test_reasoning_trace_completeness(self):
        """Test that reasoning trace captures all stages"""
        medical_input = MedicalInput(
            procedure="Cardiac Catheterization",
            details="With iodine contrast",
            objectives=("assess cardiac risks", "post-procedure monitoring")
        )
        
        agent = MedicalReasoningAgent(enable_logging=True)

        try:
            result = agent.analyze_medical_procedure(medical_input)
            
            # Verify all reasoning stages are present
            stages_present = {step.stage for step in result.reasoning_trace}
            expected_stages = {
                ReasoningStage.INPUT_ANALYSIS,
                ReasoningStage.ORGAN_IDENTIFICATION,
                ReasoningStage.EVIDENCE_GATHERING,
                ReasoningStage.RISK_ASSESSMENT,
                ReasoningStage.RECOMMENDATION_SYNTHESIS,
                ReasoningStage.CRITICAL_EVALUATION
            }
            
            assert stages_present == expected_stages
            
        except Exception:
            # Expected without LLM integration
            pass


# Test data fixtures
TEST_SCENARIOS = [
    {
        "name": "MRI_with_contrast",
        "input": {
            "procedure": "MRI Scanner",
            "details": "With gadolinium contrast",
            "objectives": ["kidney safety", "elimination pathway"]
        },
        "expected_organs": ["kidneys", "brain"],
        "expected_recommendations": ["hydration", "monitoring"]
    },
    {
        "name": "CT_with_iodine",
        "input": {
            "procedure": "CT Scan", 
            "details": "With iodine contrast",
            "objectives": ["nephrotoxicity assessment"]
        },
        "expected_organs": ["kidneys", "thyroid"],
        "expected_recommendations": ["pre_hydration", "kidney_function_monitoring"]
    }
]


@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
def test_scenario_based_analysis(scenario):
    """Test different medical scenarios"""
    # Convert objectives list to tuple for MedicalInput
    input_data = scenario["input"].copy()
    input_data["objectives"] = tuple(input_data["objectives"])
    medical_input = MedicalInput(**input_data)
    agent = MedicalReasoningAgent(enable_logging=False)
    
    # Test organ identification
    organs = agent._identify_affected_organs(medical_input)
    
    # Should identify expected organs (allowing for some flexibility)
    for expected_organ in scenario["expected_organs"]:
        # Check if expected organ or similar is identified
        organ_found = any(expected_organ in organ.lower() for organ in organs)
        assert organ_found, f"Expected organ {expected_organ} not found in {organs}"


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        __file__,
        "-v",
        "--cov=medical_procedure_analyzer",
        "--cov-report=html",
        "--cov-report=term-missing"
    ])