#!/usr/bin/env python3
"""
Agent Orchestrator
Executes agents based on router decisions and manages agent lifecycle.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from llm_integrations import call_model, get_available_models
from medical_procedure_analyzer.medical_reasoning_agent import (
    MedicalReasoningAgent,
    MedicalInput,
    MedicalOutput
)
from medical_fact_checker.medical_fact_checker_agent import (
    MedicalFactChecker,
    FactCheckSession
)


@dataclass
class AgentExecutionResult:
    """Result from agent execution"""
    agent_id: str
    agent_name: str
    query: str
    summary: str
    full_output: Any
    success: bool
    error_message: Optional[str] = None


class AgentOrchestrator:
    """
    Orchestrates agent execution based on routing decisions.

    Manages the lifecycle of agents:
    1. Maps agent_id to actual agent classes
    2. Adapts user queries to agent input formats
    3. Executes agents with appropriate configuration
    4. Formats and returns results
    """

    def __init__(self, llm_model: str = "grok-4-1-fast-non-reasoning-latest"):
        """
        Initialize orchestrator.

        Args:
            llm_model: Default LLM model to use for agent execution
        """
        self.llm_model = llm_model

        # Map provider name from model
        available_models = get_available_models()
        if llm_model in available_models:
            self.llm_provider = available_models[llm_model]
        else:
            # Fallback
            self.llm_provider = "grok-4-1-fast"

        # Agent metadata for display
        self.agent_metadata = {
            "medication_agent": {
                "name": "Medication Specialist",
                "agent_class": MedicalReasoningAgent,
                "description": "Analyzes medications, drugs, dosages, and pharmaceutical information"
            },
            "procedure_agent": {
                "name": "Medical Procedure Specialist",
                "agent_class": MedicalReasoningAgent,
                "description": "Analyzes medical procedures, surgeries, and treatments"
            },
            "diagnostic_agent": {
                "name": "Diagnostic Specialist",
                "agent_class": MedicalFactChecker,
                "description": "Investigates medical conditions, symptoms, and diagnoses"
            },
            "general_agent": {
                "name": "General Medical Assistant",
                "agent_class": MedicalReasoningAgent,
                "description": "Handles general medical and health queries"
            }
        }

    def execute_agent(self, agent_id: str, user_query: str) -> AgentExecutionResult:
        """
        Execute the specified agent with the user query.

        Args:
            agent_id: Agent identifier from router
            user_query: User's original query

        Returns:
            AgentExecutionResult with summary and full output
        """
        if agent_id not in self.agent_metadata:
            return AgentExecutionResult(
                agent_id=agent_id,
                agent_name="Unknown",
                query=user_query,
                summary="",
                full_output=None,
                success=False,
                error_message=f"Unknown agent ID: {agent_id}"
            )

        metadata = self.agent_metadata[agent_id]
        agent_name = metadata["name"]
        agent_class = metadata["agent_class"]

        try:
            # Route to appropriate executor
            if agent_class == MedicalReasoningAgent:
                result = self._execute_medical_reasoning_agent(
                    agent_id, agent_name, user_query
                )
            elif agent_class == MedicalFactChecker:
                result = self._execute_fact_checker_agent(
                    agent_id, agent_name, user_query
                )
            else:
                raise ValueError(f"Unsupported agent class: {agent_class}")

            return result

        except Exception as e:
            return AgentExecutionResult(
                agent_id=agent_id,
                agent_name=agent_name,
                query=user_query,
                summary="",
                full_output=None,
                success=False,
                error_message=f"Agent execution failed: {str(e)}"
            )

    def _execute_medical_reasoning_agent(
        self,
        agent_id: str,
        agent_name: str,
        user_query: str
    ) -> AgentExecutionResult:
        """
        Execute MedicalReasoningAgent for medication/procedure/general queries.
        """
        # Adapt query to MedicalInput format
        # Use LLM to extract structured information
        extraction_prompt = f"""Extract medical information from this query: "{user_query}"

Provide the following in a structured format:
1. Procedure/Topic: (main medical topic - medication name, procedure, or health topic)
2. Details: (additional context or specific questions)
3. Objectives: (what the user wants to know - list 2-3 objectives)

Format your response as:
PROCEDURE: <procedure name>
DETAILS: <details>
OBJECTIVES: <objective 1> | <objective 2> | <objective 3>"""

        messages = [
            {"role": "system", "content": "You are a medical query analyzer. Extract structured information concisely."},
            {"role": "user", "content": extraction_prompt}
        ]

        try:
            extracted = call_model(self.llm_model, messages)

            # Parse extracted information
            procedure = self._extract_field(extracted, "PROCEDURE", user_query)
            details = self._extract_field(extracted, "DETAILS", user_query)
            objectives_str = self._extract_field(extracted, "OBJECTIVES", "Provide comprehensive medical information")

            # Split objectives
            objectives = tuple([obj.strip() for obj in objectives_str.split("|")])

        except Exception as e:
            # Fallback if extraction fails
            procedure = user_query
            details = f"User query: {user_query}"
            objectives = ("Provide medical information", "Explain key aspects", "List important considerations")

        # Create MedicalInput
        medical_input = MedicalInput(
            procedure=procedure,
            details=details,
            objectives=objectives,
            patient_context=None
        )

        # Initialize and execute agent
        agent = MedicalReasoningAgent(
            primary_llm_provider=self.llm_provider,
            fallback_providers=[],
            enable_logging=False,
            enable_reference_validation=False
        )

        output: MedicalOutput = agent.analyze_medical_procedure(medical_input)

        # Generate summary
        summary = self._generate_summary_from_medical_output(output)

        return AgentExecutionResult(
            agent_id=agent_id,
            agent_name=agent_name,
            query=user_query,
            summary=summary,
            full_output=output,
            success=True
        )

    def _execute_fact_checker_agent(
        self,
        agent_id: str,
        agent_name: str,
        user_query: str
    ) -> AgentExecutionResult:
        """
        Execute MedicalFactChecker for diagnostic/investigation queries.
        """
        # Initialize fact checker (non-interactive for CLI)
        agent = MedicalFactChecker(
            primary_llm_provider=self.llm_provider,
            fallback_providers=[],
            interactive=False,
            enable_logging=False,
            enable_reference_validation=False
        )

        # Execute fact checking
        session: FactCheckSession = agent.start_analysis(
            subject=user_query,
            clarifying_info=""
        )

        # Generate summary from session
        summary = self._generate_summary_from_fact_check(session)

        return AgentExecutionResult(
            agent_id=agent_id,
            agent_name=agent_name,
            query=user_query,
            summary=summary,
            full_output=session,
            success=True
        )

    def _extract_field(self, text: str, field_name: str, default: str) -> str:
        """Extract a field from LLM-structured output."""
        for line in text.split("\n"):
            if line.startswith(f"{field_name}:"):
                return line.split(":", 1)[1].strip()
        return default

    def _generate_summary_from_medical_output(self, output: MedicalOutput) -> str:
        """Generate a concise summary from MedicalOutput."""
        lines = []

        lines.append("Medical Analysis Complete")
        lines.append(f"Confidence: {output.confidence_score:.1%}")
        lines.append("")

        # Procedure summary
        if output.procedure_summary:
            lines.append("Summary:")
            summary_preview = output.procedure_summary[:300]
            lines.append(summary_preview + "..." if len(output.procedure_summary) > 300 else summary_preview)
            lines.append("")

        # Organs analyzed
        if output.organs_analyzed:
            lines.append(f"Organs Analyzed: {len(output.organs_analyzed)}")
            high_risk_organs = [o for o in output.organs_analyzed if o.risk_level == "high"]
            if high_risk_organs:
                lines.append(f"  High-risk organs: {', '.join(o.organ_name for o in high_risk_organs[:3])}")
            lines.append("")

        # Recommendations
        if output.general_recommendations:
            lines.append("Key Recommendations:")
            for i, rec in enumerate(output.general_recommendations[:3], 1):
                lines.append(f"  {i}. {rec}")
            if len(output.general_recommendations) > 3:
                lines.append(f"  ... and {len(output.general_recommendations) - 3} more")
            lines.append("")

        lines.append("💡 Type 'full' to see complete analysis")

        return "\n".join(lines)

    def _generate_summary_from_fact_check(self, session: FactCheckSession) -> str:
        """Generate a concise summary from FactCheckSession."""
        lines = []

        lines.append(f"Fact Check Analysis: {session.subject}")
        lines.append("")

        # Phase summaries
        for phase_result in session.phase_results[:2]:  # Show first 2 phases
            lines.append(f"Phase: {phase_result.phase}")
            if phase_result.summary:
                summary_preview = phase_result.summary[:200]
                lines.append(summary_preview + "..." if len(phase_result.summary) > 200 else summary_preview)
            lines.append("")

        if len(session.phase_results) > 2:
            lines.append(f"... and {len(session.phase_results) - 2} more phases")
            lines.append("")

        lines.append("💡 Type 'full' to see complete analysis")

        return "\n".join(lines)
