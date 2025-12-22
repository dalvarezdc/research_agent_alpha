#!/usr/bin/env python3
"""
Pytest tests for Medical Fact Checker Agent
"""

import pytest
import json
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medical_fact_checker.medical_fact_checker_agent import (
    MedicalFactChecker,
    AnalysisPhase,
    OutputType,
    PhaseResult,
    FactCheckSession
)
from llm_integrations import TokenUsage


# Fixtures
@pytest.fixture
def mock_llm_manager():
    """Mock LLM manager with fake responses"""
    manager = Mock()
    provider = Mock()

    # Mock generate_response to return (response, token_usage)
    def mock_generate_response(prompt, system_prompt=None):
        token_usage = TokenUsage(input_tokens=100, output_tokens=200, total_tokens=300)

        # Return different responses based on prompt content
        if "Official Narrative" in prompt or "Counter-Narrative" in prompt:
            response = """
            Official Narrative:
            Mainstream medicine recommends standard approach based on FDA guidelines.

            Counter-Narrative:
            Independent researchers suggest alternative mechanisms based on recent studies.

            Key Conflicts:
            - Official relies on older industry-funded studies
            - Independent findings show different results with better methodology
            """
        elif "Funding Filter" in prompt or "evidence analysis" in prompt.lower():
            response = """
            Industry-funded studies: Large trials funded by pharmaceutical companies.
            Independent research: Small lab studies with rigorous methodology.
            Methodology quality: Independent studies show better controls.
            Anecdotal signals: Patient reports suggest consistent patterns.
            Time-weighted evidence: 2023 studies contradict 1995 beliefs.
            """
        elif "Synthesize" in prompt or "Biological Truth" in prompt:
            response = """
            Biological Truth: Based on evolutionary evidence and recent independent research.
            Industry Bias: Profit motives may downplay natural alternatives.
            Grey Zone: Promising approaches lacking corporate funding for large trials.
            """
        elif "Evolutionary Protocol" in prompt:
            response = """
            # The Evolutionary Protocol: Test Subject

            ## The Ancestral Logic
            Humans evolved with this mechanism for millennia.

            ## The Toxic Load
            Modern synthetic compounds disrupt natural pathways.

            ## The Bio-Identical Swap
            Natural alternatives align with biological systems.

            ## The Protocol
            Daily routine following circadian rhythms.

            ## References
            Smith, J. (2023). Study title. Nature, 123(4), 567-890. https://pubmed.ncbi.nlm.nih.gov/12345678
            """
        elif "Bio-Hacker" in prompt:
            response = """
            # The Bio-Hacker's Guide

            ## The Optimization Target
            Specific cellular mechanism.

            ## The Underground Data
            Small lab findings showing promise.

            ## The Stack
            Compound A: 500mg daily
            Compound B: 1000mg twice daily

            ## Risk Management
            Monitor biomarkers monthly.

            ## References
            Jones, A. (2024). Research paper. Science, 456(7), 123-456. https://doi.org/10.1234/science.2024
            """
        else:
            response = "Generic response for testing"

        return response, token_usage

    provider.generate_response = Mock(side_effect=mock_generate_response)
    manager.get_available_provider = Mock(return_value=provider)

    return manager


@pytest.fixture
def agent_with_mock_llm(mock_llm_manager):
    """Create agent with mocked LLM"""
    with patch('medical_fact_checker.medical_fact_checker_agent.create_llm_manager', return_value=mock_llm_manager):
        agent = MedicalFactChecker(
            primary_llm_provider="claude",
            enable_logging=False,
            interactive=False
        )
        return agent


@pytest.fixture
def sample_subject():
    """Sample health subject for testing"""
    return "Vitamin D supplementation"


@pytest.fixture
def sample_context():
    """Sample context for testing"""
    return "optimal dosing for immune function"


# Test initialization
class TestInitialization:
    """Test agent initialization"""

    def test_agent_initialization_success(self, mock_llm_manager):
        """Test successful agent initialization"""
        with patch('medical_fact_checker.medical_fact_checker_agent.create_llm_manager', return_value=mock_llm_manager):
            agent = MedicalFactChecker(
                primary_llm_provider="claude",
                enable_logging=False,
                interactive=False
            )

            assert agent is not None
            assert agent.interactive is False
            assert agent.llm_manager is not None

    def test_agent_initialization_with_logging(self, mock_llm_manager):
        """Test agent initialization with logging enabled"""
        with patch('medical_fact_checker.medical_fact_checker_agent.create_llm_manager', return_value=mock_llm_manager):
            agent = MedicalFactChecker(
                primary_llm_provider="claude",
                enable_logging=True,
                interactive=True
            )

            assert agent.interactive is True
            assert agent.logger is not None

    def test_agent_initialization_failure(self):
        """Test agent initialization failure"""
        with patch('medical_fact_checker.medical_fact_checker_agent.create_llm_manager', side_effect=Exception("LLM init failed")):
            with pytest.raises(Exception) as exc_info:
                MedicalFactChecker(primary_llm_provider="invalid")

            assert "LLM init failed" in str(exc_info.value)


# Test Phase 1: Conflict Scan
class TestPhase1ConflictScan:
    """Test Phase 1: Conflict & Hypothesis Scan"""

    def test_phase1_execution(self, agent_with_mock_llm, sample_subject, sample_context):
        """Test Phase 1 executes successfully"""
        result = agent_with_mock_llm._phase1_conflict_scan(sample_subject, sample_context)

        assert isinstance(result, PhaseResult)
        assert result.phase == AnalysisPhase.CONFLICT_SCAN
        assert result.content is not None
        assert 'official_narrative' in result.content
        assert 'counter_narrative' in result.content
        assert 'key_conflicts' in result.content

    def test_phase1_content_parsing(self, agent_with_mock_llm):
        """Test Phase 1 content parsing"""
        response = """
        Official Narrative:
        The mainstream view is X.

        Counter-Narrative:
        Independent researchers suggest Y.

        Key Conflicts:
        Disagreement on methodology.
        """

        content = agent_with_mock_llm._parse_conflict_scan_response(response)

        assert 'official_narrative' in content
        assert 'counter_narrative' in content
        assert 'key_conflicts' in content
        assert len(content['official_narrative']) > 0

    def test_phase1_token_usage_tracking(self, agent_with_mock_llm, sample_subject):
        """Test Phase 1 tracks token usage"""
        result = agent_with_mock_llm._phase1_conflict_scan(sample_subject, "")

        assert result.token_usage is not None
        assert result.token_usage.total_tokens > 0


# Test Phase 2: Evidence Stress-Test
class TestPhase2EvidenceStressTest:
    """Test Phase 2: Evidence Stress-Test"""

    def test_phase2_execution(self, agent_with_mock_llm, sample_subject):
        """Test Phase 2 executes successfully"""
        phase1_content = {
            'official_narrative': 'Official view',
            'counter_narrative': 'Alternative view',
            'key_conflicts': 'Conflicts'
        }

        result = agent_with_mock_llm._phase2_evidence_stress_test(
            sample_subject, phase1_content, "Both"
        )

        assert isinstance(result, PhaseResult)
        assert result.phase == AnalysisPhase.EVIDENCE_STRESS_TEST
        assert result.content is not None

    def test_phase2_evidence_parsing(self, agent_with_mock_llm):
        """Test Phase 2 evidence parsing"""
        response = """
        Industry-funded studies: Big pharma trials.
        Independent research: Small lab findings.
        Methodology quality: Independent studies better.
        Anecdotal signals: Patient reports consistent.
        Recent evidence from 2023: New findings.
        """

        content = agent_with_mock_llm._parse_evidence_response(response)

        assert 'industry_funded_studies' in content
        assert 'independent_research' in content
        assert 'methodology_quality' in content
        assert 'anecdotal_signals' in content

    def test_phase2_with_different_angles(self, agent_with_mock_llm, sample_subject):
        """Test Phase 2 with different analysis angles"""
        phase1_content = {'official_narrative': 'Test', 'counter_narrative': 'Test'}

        for angle in ['Official', 'Independent', 'Both']:
            result = agent_with_mock_llm._phase2_evidence_stress_test(
                sample_subject, phase1_content, angle
            )
            assert result is not None


# Test Phase 3: Synthesis & Menu
class TestPhase3Synthesis:
    """Test Phase 3: Synthesis & Menu"""

    def test_phase3_execution(self, agent_with_mock_llm, sample_subject):
        """Test Phase 3 executes successfully"""
        phase1_content = {'official_narrative': 'Official', 'counter_narrative': 'Counter'}
        phase2_content = {'independent_research': 'Research', 'anecdotal_signals': 'Signals'}

        result = agent_with_mock_llm._phase3_synthesis_menu(
            sample_subject, phase1_content, phase2_content
        )

        assert isinstance(result, PhaseResult)
        assert result.phase == AnalysisPhase.SYNTHESIS_MENU
        assert result.content is not None

    def test_phase3_synthesis_parsing(self, agent_with_mock_llm):
        """Test Phase 3 synthesis parsing"""
        response = """
        Biological Truth: The most plausible reality.
        Industry Bias: Profit motives distort data.
        Grey Zone: Promising but unproven approaches.
        """

        content = agent_with_mock_llm._parse_synthesis_response(response)

        assert 'biological_truth' in content
        assert 'industry_bias' in content
        assert 'grey_zone' in content


# Test Phase 4: Output Generation
class TestPhase4OutputGeneration:
    """Test Phase 4: Complex Output Generation"""

    def test_phase4_evolutionary_output(self, agent_with_mock_llm, sample_subject):
        """Test Phase 4 with Evolutionary output type"""
        synthesis = {
            'biological_truth': 'Test truth',
            'industry_bias': 'Test bias',
            'grey_zone': 'Test grey zone'
        }

        output = agent_with_mock_llm._phase4_generate_output(
            sample_subject, synthesis, OutputType.EVOLUTIONARY
        )

        assert output is not None
        assert len(output) > 0
        assert "Evolutionary Protocol" in output or "Ancestral Logic" in output

    def test_phase4_biohacker_output(self, agent_with_mock_llm, sample_subject):
        """Test Phase 4 with Bio-Hacker output type"""
        synthesis = {'biological_truth': 'Test'}

        output = agent_with_mock_llm._phase4_generate_output(
            sample_subject, synthesis, OutputType.BIOHACKER
        )

        assert output is not None
        assert len(output) > 0

    def test_phase4_all_output_types(self, agent_with_mock_llm, sample_subject):
        """Test Phase 4 with all output types"""
        synthesis = {'biological_truth': 'Test', 'industry_bias': 'Test'}

        output_types = [
            OutputType.EVOLUTIONARY,
            OutputType.BIOHACKER,
            OutputType.PARADIGM_SHIFT,
            OutputType.VILLAGE_WISDOM,
            OutputType.PROCEED
        ]

        for output_type in output_types:
            output = agent_with_mock_llm._phase4_generate_output(
                sample_subject, synthesis, output_type
            )
            assert output is not None
            assert len(output) > 0


# Test Phase 5: Simplified Output
class TestPhase5SimplifiedOutput:
    """Test Phase 5: Simplified Output Generation"""

    def test_phase5_simplification(self, agent_with_mock_llm):
        """Test Phase 5 simplifies complex output"""
        complex_output = """
        # Complex Medical Guide

        This guide discusses pathophysiological mechanisms and pharmacokinetic principles.
        The hypothalamic-pituitary-adrenal axis modulation occurs through receptor binding.
        """

        simplified = agent_with_mock_llm._phase5_simplify_output(complex_output)

        assert simplified is not None
        assert len(simplified) > 0

    def test_phase5_preserves_content(self, agent_with_mock_llm):
        """Test Phase 5 preserves important content"""
        complex_output = "Important medical information with references"

        simplified = agent_with_mock_llm._phase5_simplify_output(complex_output)

        # Should not be empty
        assert len(simplified) > 0


# Test Response Parsing
class TestResponseParsing:
    """Test response parsing methods"""

    def test_parse_conflict_scan_empty_response(self, agent_with_mock_llm):
        """Test parsing empty conflict scan response"""
        content = agent_with_mock_llm._parse_conflict_scan_response("")

        assert 'official_narrative' in content
        assert 'counter_narrative' in content
        assert 'key_conflicts' in content

    def test_parse_conflict_scan_partial_response(self, agent_with_mock_llm):
        """Test parsing partial conflict scan response"""
        response = "Official Narrative: Only this section exists."

        content = agent_with_mock_llm._parse_conflict_scan_response(response)

        assert len(content['official_narrative']) > 0

    def test_parse_evidence_response_malformed(self, agent_with_mock_llm):
        """Test parsing malformed evidence response"""
        response = "Random text without proper sections"

        content = agent_with_mock_llm._parse_evidence_response(response)

        # Should still return structure
        assert 'independent_research' in content

    def test_parse_synthesis_response_variations(self, agent_with_mock_llm):
        """Test parsing synthesis response with variations"""
        responses = [
            "Biological truth: Finding 1\nIndustry bias: Finding 2",
            "Reality: Finding 1\nProfit motive: Finding 2",
            "Truth: Finding 1\nBias: Finding 2"
        ]

        for response in responses:
            content = agent_with_mock_llm._parse_synthesis_response(response)
            assert isinstance(content, dict)


# Test Full Workflow
class TestFullWorkflow:
    """Test complete analysis workflow"""

    def test_full_analysis_workflow(self, agent_with_mock_llm, sample_subject, sample_context):
        """Test complete analysis from start to finish"""
        session = agent_with_mock_llm.start_analysis(sample_subject, sample_context)

        assert isinstance(session, FactCheckSession)
        assert session.subject == sample_subject
        assert len(session.phase_results) >= 3  # At least 3 phases
        assert session.final_output is not None
        assert len(session.final_output) > 0

    def test_workflow_phase_progression(self, agent_with_mock_llm, sample_subject):
        """Test phases execute in correct order"""
        session = agent_with_mock_llm.start_analysis(sample_subject, "")

        # Check phases are in order
        phases_executed = [pr.phase for pr in session.phase_results]

        assert AnalysisPhase.CONFLICT_SCAN in phases_executed
        assert AnalysisPhase.EVIDENCE_STRESS_TEST in phases_executed
        assert AnalysisPhase.SYNTHESIS_MENU in phases_executed

    def test_workflow_with_empty_context(self, agent_with_mock_llm, sample_subject):
        """Test workflow with empty context"""
        session = agent_with_mock_llm.start_analysis(sample_subject, "")

        assert session is not None
        assert session.subject == sample_subject


# Test Session Management
class TestSessionManagement:
    """Test session tracking and export"""

    def test_session_creation(self, agent_with_mock_llm, sample_subject):
        """Test session is created properly"""
        session = agent_with_mock_llm.start_analysis(sample_subject, "")

        assert session.subject == sample_subject
        assert isinstance(session.started_at, datetime)
        assert isinstance(session.phase_results, list)

    def test_session_export(self, agent_with_mock_llm, sample_subject, tmp_path):
        """Test session export to JSON"""
        session = agent_with_mock_llm.start_analysis(sample_subject, "")

        # Export to temporary file
        export_path = tmp_path / "test_session.json"
        agent_with_mock_llm.export_session(str(export_path))

        # Verify file exists
        assert export_path.exists()

        # Verify content
        with open(export_path, 'r') as f:
            data = json.load(f)

        assert data['subject'] == sample_subject
        assert 'phases' in data
        assert 'final_output' in data
        assert len(data['phases']) > 0

    def test_session_export_no_active_session(self, agent_with_mock_llm, tmp_path):
        """Test export with no active session"""
        export_path = tmp_path / "no_session.json"

        # Should not raise error, just log warning
        agent_with_mock_llm.export_session(str(export_path))

    def test_phase_result_structure(self, agent_with_mock_llm, sample_subject):
        """Test PhaseResult structure is correct"""
        session = agent_with_mock_llm.start_analysis(sample_subject, "")

        for phase_result in session.phase_results:
            assert isinstance(phase_result, PhaseResult)
            assert isinstance(phase_result.phase, AnalysisPhase)
            assert isinstance(phase_result.timestamp, datetime)
            assert isinstance(phase_result.content, dict)


# Test Error Handling
class TestErrorHandling:
    """Test error handling scenarios"""

    def test_llm_failure_handling(self, agent_with_mock_llm, sample_subject):
        """Test handling of LLM failures"""
        # Mock LLM to raise exception
        agent_with_mock_llm.llm_manager.get_available_provider().generate_response = Mock(
            side_effect=Exception("LLM API error")
        )

        with pytest.raises(Exception) as exc_info:
            agent_with_mock_llm._phase1_conflict_scan(sample_subject, "")

        assert "LLM API error" in str(exc_info.value)

    def test_invalid_output_type(self, agent_with_mock_llm, sample_subject):
        """Test handling of invalid output type"""
        synthesis = {'biological_truth': 'Test'}

        # OutputType enum should prevent invalid values, but test anyway
        valid_types = [OutputType.EVOLUTIONARY, OutputType.BIOHACKER,
                      OutputType.PARADIGM_SHIFT, OutputType.VILLAGE_WISDOM,
                      OutputType.PROCEED]

        for output_type in valid_types:
            output = agent_with_mock_llm._phase4_generate_output(
                sample_subject, synthesis, output_type
            )
            assert output is not None


# Test Enums
class TestEnums:
    """Test enum definitions"""

    def test_analysis_phase_enum(self):
        """Test AnalysisPhase enum values"""
        assert AnalysisPhase.CONFLICT_SCAN.value == "conflict_scan"
        assert AnalysisPhase.EVIDENCE_STRESS_TEST.value == "evidence_stress_test"
        assert AnalysisPhase.SYNTHESIS_MENU.value == "synthesis_menu"

    def test_output_type_enum(self):
        """Test OutputType enum values"""
        assert OutputType.EVOLUTIONARY.value == "A"
        assert OutputType.BIOHACKER.value == "B"
        assert OutputType.PARADIGM_SHIFT.value == "C"
        assert OutputType.VILLAGE_WISDOM.value == "D"
        assert OutputType.PROCEED.value == "P"


# Test Interactive Mode (mocked)
class TestInteractiveMode:
    """Test interactive mode with mocked user input"""

    @patch('builtins.input', side_effect=['Both', 'Proceed', 'P'])
    def test_interactive_prompts(self, mock_input, mock_llm_manager, sample_subject):
        """Test interactive prompts work correctly"""
        with patch('medical_fact_checker.medical_fact_checker_agent.create_llm_manager', return_value=mock_llm_manager):
            agent = MedicalFactChecker(
                primary_llm_provider="claude",
                enable_logging=False,
                interactive=True
            )

            session = agent.start_analysis(sample_subject, "")

            # Verify session completed
            assert session is not None
            assert len(session.phase_results) >= 3

    def test_prompt_user_phase1(self, agent_with_mock_llm):
        """Test Phase 1 user prompt"""
        with patch('builtins.input', return_value='Both'):
            choice = agent_with_mock_llm._prompt_user_phase1()
            assert choice in ['Official', 'Independent', 'Both']

    def test_prompt_user_phase2(self, agent_with_mock_llm):
        """Test Phase 2 user prompt"""
        phase2_content = {
            'industry_funded_studies': 'Test studies',
            'independent_research': 'Test research',
            'anecdotal_signals': 'Test signals'
        }

        with patch('builtins.input', return_value='Proceed'):
            choice = agent_with_mock_llm._prompt_user_phase2(phase2_content)
            assert choice in ['Dig', 'Proceed']

    def test_prompt_user_phase3(self, agent_with_mock_llm):
        """Test Phase 3 user prompt"""
        with patch('builtins.input', return_value='A'):
            choice = agent_with_mock_llm._prompt_user_phase3()
            assert choice in ['A', 'B', 'C', 'D', 'P']


# Test Token Usage
class TestTokenUsage:
    """Test token usage tracking"""

    def test_token_usage_accumulation(self, agent_with_mock_llm, sample_subject):
        """Test token usage accumulates across phases"""
        session = agent_with_mock_llm.start_analysis(sample_subject, "")

        total_tokens = 0
        for phase_result in session.phase_results:
            if phase_result.token_usage:
                total_tokens += phase_result.token_usage.total_tokens

        assert total_tokens > 0

    def test_phase_token_usage(self, agent_with_mock_llm, sample_subject):
        """Test each phase tracks token usage"""
        result = agent_with_mock_llm._phase1_conflict_scan(sample_subject, "")

        assert result.token_usage is not None
        assert result.token_usage.input_tokens > 0
        assert result.token_usage.output_tokens > 0
        assert result.token_usage.total_tokens == (
            result.token_usage.input_tokens + result.token_usage.output_tokens
        )


# Integration Tests
class TestIntegration:
    """Integration tests for complete workflows"""

    def test_end_to_end_analysis(self, agent_with_mock_llm, sample_subject, tmp_path):
        """Test complete end-to-end analysis with export"""
        # Run analysis
        session = agent_with_mock_llm.start_analysis(sample_subject, "test context")

        # Verify session
        assert session.subject == sample_subject
        assert len(session.phase_results) >= 3
        assert session.final_output is not None

        # Export session
        export_path = tmp_path / "integration_test.json"
        agent_with_mock_llm.export_session(str(export_path))

        # Verify export
        assert export_path.exists()
        with open(export_path, 'r') as f:
            data = json.load(f)

        assert data['subject'] == sample_subject

    def test_multiple_analyses_same_agent(self, agent_with_mock_llm):
        """Test running multiple analyses with same agent instance"""
        subjects = ["Vitamin D", "Omega-3", "Zinc"]

        for subject in subjects:
            session = agent_with_mock_llm.start_analysis(subject, "")
            assert session.subject == subject
            assert session.final_output is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
