#!/usr/bin/env python3
"""
Unified Analysis Runner for Medical AI Agents
Orchestrates multiple medical analysis agents with a common interface.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Any, Dict, Tuple

# Import medical procedure analyzer
from medical_procedure_analyzer import MedicalReasoningAgent, MedicalInput

# Import medical fact checker
from medical_fact_checker import MedicalFactChecker


class AgentOrchestrator:
    """Orchestrates multiple medical analysis agents"""

    AGENTS = {
        "procedure": {
            "name": "Medical Procedure Analyzer",
            "description": "Analyzes medical procedures with organ-focused reasoning",
            "class": MedicalReasoningAgent,
        },
        "factcheck": {
            "name": "Medical Fact Checker",
            "description": "Independent bio-investigator for health subjects",
            "class": MedicalFactChecker,
        },
    }

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run_procedure_analyzer(
        self, procedure: str, details: str, llm_provider: str = "claude", timeout: int = 300
    ) -> Tuple[Any, Dict[str, str]]:
        """Run the Medical Procedure Analyzer"""
        print("=" * 80)
        print("🔬 Medical Procedure Analyzer")
        print("=" * 80)
        print()

        # Initialize agent
        print(f"🤖 Initializing agent with {llm_provider} (timeout: {timeout}s)...")
        agent = MedicalReasoningAgent(
            primary_llm_provider=llm_provider,
            fallback_providers=["openai"],
            enable_logging=True,
        )
        # Update timeout if agent's LLM manager exists
        if hasattr(agent, 'llm_manager') and agent.llm_manager:
            for config in agent.llm_manager.configs:
                config.timeout = timeout

        # Create input
        medical_input = MedicalInput(
            procedure=procedure,
            details=details,
            objectives=(
                "understand implications",
                "identify risks",
                "post-procedure care",
                "affected organs",
                "organs at risk",
            ),
            patient_context="Standard adult patient",
        )

        print(f"📋 Analyzing: {medical_input.procedure}")
        print(f"   Details: {medical_input.details}")
        print()
        print("⏳ Running 6-stage analysis pipeline...")
        print()

        # Run analysis
        result = agent.analyze_medical_procedure(medical_input)

        print()
        print("=" * 80)
        print("✅ Analysis Complete!")
        print("=" * 80)
        print(f"📊 Procedure: {result.procedure_summary}")
        print(f"🫀 Organs Analyzed: {len(result.organs_analyzed)}")
        print(f"📈 Confidence Score: {result.confidence_score:.2f}")
        print(f"🧠 Reasoning Steps: {len(result.reasoning_trace)}")
        print()

        # Display brief results
        print("🔍 Organs Identified:")
        for organ in result.organs_analyzed:
            print(f"   - {organ.organ_name.upper()}: {organ.risk_level} risk")
            print(
                f"     ✓ {len(organ.known_recommendations)} evidence-based recommendations"
            )
            print(
                f"     ⚗️ {len(organ.potential_recommendations)} investigational approaches"
            )
            print(f"     ❌ {len(organ.debunked_claims)} debunked claims identified")
        print()

        # Save outputs
        print("💾 Saving outputs...")
        files = self._save_procedure_analysis(result, procedure)

        return result, files

    def run_fact_checker(
        self, subject: str, context: str = "", llm_provider: str = "claude", timeout: int = 300
    ) -> Tuple[Any, Dict[str, str]]:
        """Run the Medical Fact Checker"""
        print("=" * 80)
        print("🔎 Medical Fact Checker - Independent Bio-Investigator")
        print("=" * 80)
        print()

        # Initialize agent
        print(f"🤖 Initializing agent with {llm_provider} (timeout: {timeout}s)...")
        agent = MedicalFactChecker(
            primary_llm_provider=llm_provider,
            fallback_providers=["openai"],
            enable_logging=True,
            interactive=False,  # Non-interactive mode for automation
        )
        # Update timeout if agent's LLM manager exists
        if hasattr(agent, 'llm_manager') and agent.llm_manager:
            for config in agent.llm_manager.configs:
                config.timeout = timeout

        print(f"📋 Investigating: {subject}")
        if context:
            print(f"   Context: {context}")
        print()
        print("⏳ Running 5-phase fact-checking protocol...")
        print("   Phase 1: Conflict & Hypothesis Scan")
        print("   Phase 2: Evidence Stress-Test")
        print("   Phase 3: Synthesis & Menu")
        print("   Phase 4: Complex Output Generation")
        print("   Phase 5: Simplified Output")
        print()

        # Run analysis
        session = agent.start_analysis(subject, context)

        print()
        print("=" * 80)
        print("✅ Analysis Complete!")
        print("=" * 80)
        print(f"📊 Subject: {session.subject}")
        print(f"📄 Phases Completed: {len(session.phase_results)}")
        print(f"📝 Output Length: {len(session.final_output)} characters")
        print()

        # Display phase summary
        print("🔍 Phase Summary:")
        for phase_result in session.phase_results:
            print(f"   - {phase_result.phase.value.replace('_', ' ').title()}")
            if phase_result.user_choice:
                print(f"     Choice: {phase_result.user_choice}")
        print()

        # Save outputs
        print("💾 Saving outputs...")
        files = self._save_fact_check_analysis(session, subject)

        return session, files

    def _save_procedure_analysis(
        self, result: Any, procedure_name: str
    ) -> Dict[str, str]:
        """Save procedure analysis results"""
        base_name = procedure_name.replace(" ", "_").lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        files = {}

        # 1. Reasoning trace
        trace_file = f"{self.output_dir}/{base_name}_reasoning_trace_{timestamp}.json"
        trace_data = []
        for step in result.reasoning_trace:
            trace_data.append(
                {
                    "stage": step.stage.value,
                    "timestamp": step.timestamp.isoformat(),
                    "reasoning": step.reasoning,
                    "input": step.input_data,
                    "output": step.output,
                    "confidence": step.confidence,
                    "sources": step.sources,
                }
            )

        with open(trace_file, "w") as f:
            json.dump(trace_data, f, indent=2)
        print(f"✓ Reasoning trace: {os.path.basename(trace_file)}")
        files["trace"] = trace_file

        # 2. Analysis result
        result_file = f"{self.output_dir}/{base_name}_analysis_result_{timestamp}.json"
        analysis_data = {
            "timestamp": datetime.now().isoformat(),
            "agent_type": "procedure_analyzer",
            "procedure_summary": result.procedure_summary,
            "confidence_score": result.confidence_score,
            "organs_analyzed": [
                {
                    "organ_name": organ.organ_name,
                    "affected_by_procedure": organ.affected_by_procedure,
                    "at_risk": organ.at_risk,
                    "risk_level": organ.risk_level,
                    "pathways_involved": organ.pathways_involved,
                    "known_recommendations": organ.known_recommendations,
                    "potential_recommendations": organ.potential_recommendations,
                    "debunked_claims": organ.debunked_claims,
                    "evidence_quality": organ.evidence_quality,
                }
                for organ in result.organs_analyzed
            ],
            "general_recommendations": result.general_recommendations,
            "research_gaps": result.research_gaps,
            "reasoning_steps_count": len(result.reasoning_trace),
        }

        with open(result_file, "w") as f:
            json.dump(analysis_data, f, indent=2)
        print(f"✓ Analysis result: {os.path.basename(result_file)}")
        files["result"] = result_file

        # 3. Summary report
        summary_file = f"{self.output_dir}/{base_name}_summary_report_{timestamp}.md"
        summary = self._generate_procedure_summary(result)

        with open(summary_file, "w") as f:
            f.write(summary)
        print(f"✓ Summary report: {os.path.basename(summary_file)}")
        files["summary"] = summary_file

        return files

    def _save_fact_check_analysis(
        self, session: Any, subject: str
    ) -> Dict[str, str]:
        """Save fact check analysis results"""
        base_name = subject.replace(" ", "_").lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        files = {}

        # 1. Session data (phases and choices)
        session_file = f"{self.output_dir}/{base_name}_session_{timestamp}.json"
        session_data = {
            "timestamp": datetime.now().isoformat(),
            "agent_type": "fact_checker",
            "subject": session.subject,
            "started_at": session.started_at.isoformat(),
            "phases": [
                {
                    "phase": pr.phase.value,
                    "timestamp": pr.timestamp.isoformat(),
                    "content": pr.content,
                    "user_choice": pr.user_choice,
                    "token_usage": (
                        {
                            "input": pr.token_usage.input_tokens,
                            "output": pr.token_usage.output_tokens,
                            "total": pr.token_usage.total_tokens,
                        }
                        if pr.token_usage
                        else None
                    ),
                }
                for pr in session.phase_results
            ],
        }

        with open(session_file, "w") as f:
            json.dump(session_data, f, indent=2)
        print(f"✓ Session data: {os.path.basename(session_file)}")
        files["session"] = session_file

        # 2. Final output (markdown)
        output_file = f"{self.output_dir}/{base_name}_output_{timestamp}.md"
        with open(output_file, "w") as f:
            f.write(session.final_output)
        print(f"✓ Final output: {os.path.basename(output_file)}")
        files["output"] = output_file

        # 3. Summary report
        summary_file = f"{self.output_dir}/{base_name}_summary_{timestamp}.md"
        summary = self._generate_fact_check_summary(session)

        with open(summary_file, "w") as f:
            f.write(summary)
        print(f"✓ Summary report: {os.path.basename(summary_file)}")
        files["summary"] = summary_file

        return files

    def _generate_procedure_summary(self, result: Any) -> str:
        """Generate markdown summary for procedure analysis"""
        summary = f"""# Medical Procedure Analysis Report
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Analysis System:** MedicalReasoningAgent (6-Stage Pipeline)

---

## Procedure Overview
**Procedure:** {result.procedure_summary}
**Analysis Confidence:** {result.confidence_score:.2f}/1.00
**Total Organs Analyzed:** {len(result.organs_analyzed)}
**Reasoning Steps Completed:** {len(result.reasoning_trace)}

---

## Detailed Organ-Specific Analysis

"""

        for i, organ in enumerate(result.organs_analyzed, 1):
            summary += f"""### {i}. {organ.organ_name.upper()}

**Risk Assessment:**
- Risk Level: **{organ.risk_level.upper()}**
- Procedure Impact: {'YES - Directly affected' if organ.affected_by_procedure else 'NO - Minimal impact'}
- At Risk: {'YES - Requires monitoring' if organ.at_risk else 'NO - Low concern'}
- Evidence Quality: **{organ.evidence_quality.upper()}**

**Biological Pathways:**
"""
            for pathway in organ.pathways_involved:
                summary += f"- {pathway.replace('_', ' ').title()}\n"

            if organ.known_recommendations:
                summary += f"\n**✅ EVIDENCE-BASED RECOMMENDATIONS** ({len(organ.known_recommendations)} items):\n"
                for j, rec in enumerate(organ.known_recommendations, 1):
                    summary += f"{j}. {rec}\n"

            if organ.potential_recommendations:
                summary += f"\n**🔬 INVESTIGATIONAL/POTENTIAL** ({len(organ.potential_recommendations)} items):\n"
                for j, rec in enumerate(organ.potential_recommendations, 1):
                    summary += f"{j}. {rec}\n"

            if organ.debunked_claims:
                summary += f"\n**❌ DEBUNKED/HARMFUL** ({len(organ.debunked_claims)} items):\n"
                for j, claim in enumerate(organ.debunked_claims, 1):
                    summary += f"{j}. {claim}\n"

            summary += "\n---\n\n"

        summary += """## General Recommendations

"""
        for i, rec in enumerate(result.general_recommendations, 1):
            summary += f"{i}. {rec}\n"

        summary += """
## Research Gaps

"""
        for i, gap in enumerate(result.research_gaps, 1):
            summary += f"{i}. {gap}\n"

        summary += f"""
---

**Report Generated By:** MedicalReasoningAgent
**Timestamp:** {datetime.now().isoformat()}

⚠️ **DISCLAIMER:** This analysis is for educational and research purposes only. Always consult qualified healthcare providers for medical decisions.
"""

        return summary

    def _generate_fact_check_summary(self, session: Any) -> str:
        """Generate markdown summary for fact check analysis"""
        summary = f"""# Medical Fact Check Report
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Analysis System:** MedicalFactChecker (Independent Bio-Investigator)

---

## Subject
**Topic:** {session.subject}
**Analysis Started:** {session.started_at.strftime('%Y-%m-%d %H:%M:%S')}
**Phases Completed:** {len(session.phase_results)}

---

## Analysis Pipeline

"""

        for i, phase_result in enumerate(session.phase_results, 1):
            phase_name = phase_result.phase.value.replace("_", " ").title()
            summary += f"### Phase {i}: {phase_name}\n\n"
            summary += f"**Timestamp:** {phase_result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"

            if phase_result.user_choice:
                summary += f"**User Choice:** {phase_result.user_choice}\n"

            summary += "\n**Key Findings:**\n"
            for key, value in phase_result.content.items():
                if value and len(str(value)) > 0:
                    preview = (
                        str(value)[:200] + "..." if len(str(value)) > 200 else str(value)
                    )
                    summary += f"- {key.replace('_', ' ').title()}: {preview}\n"

            if phase_result.token_usage:
                summary += f"\n**Token Usage:** {phase_result.token_usage.total_tokens} tokens\n"

            summary += "\n---\n\n"

        summary += """## Final Output

See the detailed output file for the complete analysis.

---

**Report Generated By:** MedicalFactChecker
**Timestamp:** {datetime.now().isoformat()}

⚠️ **DISCLAIMER:** This analysis is for research and educational purposes. It provides critical analysis of medical literature but does not constitute medical advice. Always consult qualified healthcare professionals.
"""

        return summary

    @classmethod
    def list_agents(cls):
        """List all available agents"""
        print("\n" + "=" * 80)
        print("Available Medical Analysis Agents")
        print("=" * 80 + "\n")

        for agent_id, agent_info in cls.AGENTS.items():
            print(f"📌 {agent_id}")
            print(f"   Name: {agent_info['name']}")
            print(f"   Description: {agent_info['description']}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Unified Medical Analysis Agent Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run procedure analyzer
  python run_analysis.py procedure --subject "MRI Scanner" --details "With gadolinium contrast"

  # Run fact checker
  python run_analysis.py factcheck --subject "Vitamin D supplementation" --context "optimal dosing"

  # List all agents
  python run_analysis.py --list

  # Use different LLM
  python run_analysis.py factcheck --subject "Coffee" --llm openai

  # Increase timeout for complex queries (default is 300s = 5min)
  python run_analysis.py factcheck --subject "Complex topic" --timeout 600
        """,
    )

    parser.add_argument(
        "agent",
        nargs="?",
        choices=["procedure", "factcheck"],
        help="Which agent to run (procedure or factcheck)",
    )

    parser.add_argument("--list", action="store_true", help="List all available agents")

    parser.add_argument(
        "--subject", type=str, help="Subject to analyze (procedure name or health topic)"
    )

    parser.add_argument(
        "--details",
        type=str,
        default="",
        help="Additional details for procedure analyzer",
    )

    parser.add_argument(
        "--context",
        type=str,
        default="",
        help="Context or scope for fact checker",
    )

    parser.add_argument(
        "--llm",
        type=str,
        default="claude",
        choices=["claude", "openai", "ollama"],
        help="LLM provider to use (default: claude)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for results (default: outputs/)",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="API timeout in seconds (default: 300 = 5 minutes)",
    )

    args = parser.parse_args()

    # Handle list command
    if args.list:
        AgentOrchestrator.list_agents()
        return

    # Validate arguments
    if not args.agent:
        parser.print_help()
        print("\n❌ Error: Please specify an agent (procedure or factcheck)")
        print("   Use --list to see available agents")
        sys.exit(1)

    if not args.subject:
        print("❌ Error: --subject is required")
        sys.exit(1)

    # Initialize orchestrator
    orchestrator = AgentOrchestrator(output_dir=args.output_dir)

    try:
        # Run selected agent
        if args.agent == "procedure":
            details = args.details or "Standard procedure"
            result, files = orchestrator.run_procedure_analyzer(
                procedure=args.subject, details=details, llm_provider=args.llm, timeout=args.timeout
            )

        elif args.agent == "factcheck":
            result, files = orchestrator.run_fact_checker(
                subject=args.subject, context=args.context, llm_provider=args.llm, timeout=args.timeout
            )

        # Display file locations
        print()
        print("=" * 80)
        print("📁 Output Files")
        print("=" * 80)
        for file_type, file_path in files.items():
            print(f"   {file_type}: {file_path}")
        print("=" * 80)
        print()
        print("✅ Analysis complete!")
        print()

    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
