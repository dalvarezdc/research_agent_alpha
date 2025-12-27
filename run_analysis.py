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

# Optional diagnostics
from check_llms import print_llm_status

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, skip

# Import cost tracking
from cost_tracker import get_cost_summary

# Import PDF generator
from pdf_generator import convert_markdown_to_pdf_safe

# Import medical procedure analyzer
from medical_procedure_analyzer import MedicalReasoningAgent, MedicalInput as ProcedureInput

# Import medical fact checker
from medical_fact_checker import MedicalFactChecker

# Import medication analyzer
from medical_procedure_analyzer.medication_analyzer import MedicationAnalyzer, MedicationInput


class AgentOrchestrator:
    """Orchestrates multiple medical analysis agents"""

    AGENTS = {
        "procedure": {
            "name": "Medical Procedure Analyzer",
            "description": "Analyzes medical procedures with organ-focused reasoning",
            "class": MedicalReasoningAgent,
        },
        "medication": {
            "name": "Medication Analyzer",
            "description": "Comprehensive medication analysis with interactions and recommendations",
            "class": MedicationAnalyzer,
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
        medical_input = ProcedureInput(
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
        print("⏳ Running 5-phase analysis pipeline...")
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

    def run_medication_analyzer(
        self,
        medication: str,
        indication: str = None,
        other_medications: list = None,
        llm_provider: str = "claude",
        timeout: int = 300,
    ) -> Tuple[Any, Dict[str, str]]:
        """Run the Medication Analyzer"""
        print("=" * 80)
        print("💊 Medication Analyzer - Comprehensive Drug Analysis")
        print("=" * 80)
        print()

        # Initialize agent
        print(f"🤖 Initializing agent with {llm_provider} (timeout: {timeout}s)...")
        agent = MedicationAnalyzer(
            primary_llm_provider=llm_provider,
            fallback_providers=["openai"],
            enable_logging=True,
        )
        # Update timeout if agent's LLM manager exists
        if hasattr(agent, "llm_manager") and agent.llm_manager:
            for config in agent.llm_manager.configs:
                config.timeout = timeout

        # Create input
        med_input = MedicationInput(
            medication_name=medication,
            indication=indication,
            patient_medications=other_medications or [],
        )

        print(f"💊 Analyzing: {med_input.medication_name}")
        if med_input.indication:
            print(f"   Indication: {med_input.indication}")
        if med_input.patient_medications:
            print(f"   Other Medications: {', '.join(med_input.patient_medications)}")
        print()
        print("⏳ Running comprehensive medication analysis...")
        print("   Phase 1: Pharmacology Analysis")
        print("   Phase 2: Interaction Analysis (Drug-Drug, Drug-Food, Environmental)")
        print("   Phase 3: Safety Profile Assessment")
        print("   Phase 4: Clinical Recommendations")
        print("   Phase 5: Monitoring Requirements")
        print()

        # Run analysis
        result = agent.analyze_medication(med_input)

        print()
        print("=" * 80)
        print("✅ Analysis Complete!")
        print("=" * 80)
        print(f"💊 Medication: {result.medication_name}")
        print(f"🧬 Drug Class: {result.drug_class}")
        print(f"📊 Analysis Confidence: {result.analysis_confidence:.2f}")
        print()

        # Display brief results
        print("🔍 Analysis Summary:")
        print(f"   🔗 Drug-Drug Interactions: {len(result.drug_interactions)}")
        if result.drug_interactions:
            severe = [i for i in result.drug_interactions if i.severity.value == "severe"]
            if severe:
                print(f"      ⚠️  SEVERE: {len(severe)} interactions requiring immediate attention")

        print(f"   🍎 Food Interactions: {len(result.food_interactions)}")
        print(f"   ⚕️  Contraindications: {len(result.contraindications)}")
        print(f"   ✅ Evidence-Based Recommendations: {len(result.evidence_based_recommendations)}")
        print(f"   🔬 Investigational Approaches: {len(result.investigational_approaches)}")
        print(f"   ❌ Debunked Claims: {len(result.debunked_claims)}")

        if result.black_box_warnings:
            print(f"   ⚠️  BLACK BOX WARNINGS: {len(result.black_box_warnings)}")

        print()

        # Save outputs
        print("💾 Saving outputs...")
        files = self._save_medication_analysis(result, medication)

        return result, files

    def _save_procedure_analysis(
        self, result: Any, procedure_name: str
    ) -> Dict[str, str]:
        """Save procedure analysis results"""
        base_name = procedure_name.replace(" ", "_").lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        files = {}

        # Get cost summary
        cost_summary = get_cost_summary()

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
            "cost_analysis": cost_summary,
        }

        with open(result_file, "w") as f:
            json.dump(analysis_data, f, indent=2)
        print(f"✓ Analysis result: {os.path.basename(result_file)}")
        files["result"] = result_file

        # 3. Cost report (JSON)
        cost_file = f"{self.output_dir}/{base_name}_cost_report_{timestamp}.json"
        with open(cost_file, "w") as f:
            json.dump(cost_summary, f, indent=2)
        print(f"✓ Cost report: {os.path.basename(cost_file)}")
        files["cost"] = cost_file

        # 3.5. Practitioner report (markdown + PDF) - Detailed technical report for medical professionals
        if result.practitioner_report:
            practitioner_file = f"{self.output_dir}/{base_name}_practitioner_report_{timestamp}.md"
            practitioner_complete = self._append_hardcoded_disclaimer(result.practitioner_report)
            with open(practitioner_file, "w") as f:
                f.write(practitioner_complete)
            print(f"✓ Practitioner report: {os.path.basename(practitioner_file)}")
            files["practitioner_report"] = practitioner_file

            # Generate PDF version of practitioner report
            practitioner_pdf = convert_markdown_to_pdf_safe(practitioner_file)
            if practitioner_pdf:
                print(f"✓ Practitioner PDF: {os.path.basename(practitioner_pdf)}")
                files["practitioner_pdf"] = practitioner_pdf

        # 4. Summary report (with disclaimer)
        summary_file = f"{self.output_dir}/{base_name}_summary_report_{timestamp}.md"
        summary = self._generate_procedure_summary(result, cost_summary)
        summary_complete = self._append_hardcoded_disclaimer(summary)

        with open(summary_file, "w") as f:
            f.write(summary_complete)
        print(f"✓ Summary report: {os.path.basename(summary_file)}")
        files["summary"] = summary_file

        # 5. Generate PDF version of summary
        pdf_file = convert_markdown_to_pdf_safe(summary_file)
        if pdf_file:
            print(f"✓ Summary PDF: {os.path.basename(pdf_file)}")
            files["summary_pdf"] = pdf_file

        return files

    def _save_fact_check_analysis(
        self, session: Any, subject: str
    ) -> Dict[str, str]:
        """Save fact check analysis results"""
        base_name = subject.replace(" ", "_").lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        files = {}

        # Get cost summary
        cost_summary = get_cost_summary()

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
            "cost_analysis": cost_summary,
        }

        with open(session_file, "w") as f:
            json.dump(session_data, f, indent=2)
        print(f"✓ Session data: {os.path.basename(session_file)}")
        files["session"] = session_file

        # 2. Cost report (JSON)
        cost_file = f"{self.output_dir}/{base_name}_cost_report_{timestamp}.json"
        with open(cost_file, "w") as f:
            json.dump(cost_summary, f, indent=2)
        print(f"✓ Cost report: {os.path.basename(cost_file)}")
        files["cost"] = cost_file

        # 2.5. Practitioner report (markdown + PDF) - Complex output for medical professionals
        if session.practitioner_report:
            practitioner_file = f"{self.output_dir}/{base_name}_practitioner_report_{timestamp}.md"
            practitioner_with_refs = self._append_references_section(session.practitioner_report, session)
            practitioner_complete = self._append_hardcoded_disclaimer(practitioner_with_refs)
            with open(practitioner_file, "w") as f:
                f.write(practitioner_complete)
            print(f"✓ Practitioner report: {os.path.basename(practitioner_file)}")
            files["practitioner_report"] = practitioner_file

            # Generate PDF version of practitioner report
            practitioner_pdf = convert_markdown_to_pdf_safe(practitioner_file)
            if practitioner_pdf:
                print(f"✓ Practitioner PDF: {os.path.basename(practitioner_pdf)}")
                files["practitioner_pdf"] = practitioner_pdf

        # 3. Final output (markdown) with references and disclaimer appended
        output_file = f"{self.output_dir}/{base_name}_output_{timestamp}.md"
        output_with_refs = self._append_references_section(session.final_output, session)
        output_complete = self._append_hardcoded_disclaimer(output_with_refs)
        with open(output_file, "w") as f:
            f.write(output_complete)
        print(f"✓ Final output: {os.path.basename(output_file)}")
        files["output"] = output_file

        # 3.5. Generate PDF version of output
        output_pdf = convert_markdown_to_pdf_safe(output_file)
        if output_pdf:
            print(f"✓ Output PDF: {os.path.basename(output_pdf)}")
            files["output_pdf"] = output_pdf

        # 4. Summary report (with disclaimer)
        summary_file = f"{self.output_dir}/{base_name}_summary_{timestamp}.md"
        summary = self._generate_fact_check_summary(session, cost_summary)
        summary_complete = self._append_hardcoded_disclaimer(summary)

        with open(summary_file, "w") as f:
            f.write(summary_complete)
        print(f"✓ Summary report: {os.path.basename(summary_file)}")
        files["summary"] = summary_file

        # 4.5. Generate PDF version of summary
        summary_pdf = convert_markdown_to_pdf_safe(summary_file)
        if summary_pdf:
            print(f"✓ Summary PDF: {os.path.basename(summary_pdf)}")
            files["summary_pdf"] = summary_pdf

        return files

    def _save_medication_analysis(
        self, result: Any, medication_name: str
    ) -> Dict[str, str]:
        """Save medication analysis results"""
        base_name = medication_name.replace(" ", "_").lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        files = {}

        # 1. Analysis result (JSON)
        result_file = f"{self.output_dir}/{base_name}_medication_analysis_{timestamp}.json"
        analysis_data = {
            "timestamp": datetime.now().isoformat(),
            "agent_type": "medication_analyzer",
            "medication_name": result.medication_name,
            "drug_class": result.drug_class,
            "mechanism_of_action": result.mechanism_of_action,
            "pharmacokinetics": {
                "absorption": result.absorption,
                "metabolism": result.metabolism,
                "elimination": result.elimination,
                "half_life": result.half_life,
            },
            "clinical_use": {
                "approved_indications": result.approved_indications,
                "off_label_uses": result.off_label_uses,
                "standard_dosing": result.standard_dosing,
                "dose_adjustments": result.dose_adjustments,
            },
            "safety_profile": {
                "common_adverse_effects": result.common_adverse_effects,
                "serious_adverse_effects": result.serious_adverse_effects,
                "contraindications": result.contraindications,
                "black_box_warnings": result.black_box_warnings,
            },
            "interactions": {
                "drug_interactions": [
                    {
                        "type": i.interaction_type.value,
                        "agent": i.interacting_agent,
                        "severity": i.severity.value,
                        "mechanism": i.mechanism,
                        "clinical_effect": i.clinical_effect,
                        "management": i.management,
                        "time_separation": i.time_separation,
                        "evidence_level": i.evidence_level,
                    }
                    for i in result.drug_interactions
                ],
                "food_interactions": [
                    {
                        "type": i.interaction_type.value,
                        "agent": i.interacting_agent,
                        "severity": i.severity.value,
                        "mechanism": i.mechanism,
                        "clinical_effect": i.clinical_effect,
                        "management": i.management,
                    }
                    for i in result.food_interactions
                ],
                "environmental_considerations": result.environmental_considerations,
            },
            "recommendations": {
                "evidence_based": result.evidence_based_recommendations,
                "investigational": result.investigational_approaches,
                "debunked_claims": result.debunked_claims,
            },
            "monitoring": {
                "requirements": result.monitoring_requirements,
                "warning_signs": result.warning_signs,
            },
            "metadata": {
                "evidence_quality": result.evidence_quality,
                "analysis_confidence": result.analysis_confidence,
                "reasoning_steps_count": len(result.reasoning_trace),
            },
        }

        # Add cost information
        cost_summary = get_cost_summary()
        analysis_data["cost_analysis"] = cost_summary

        with open(result_file, "w") as f:
            json.dump(analysis_data, f, indent=2)
        print(f"✓ Analysis result: {os.path.basename(result_file)}")
        files["result"] = result_file

        # 2. Cost report (JSON)
        cost_file = f"{self.output_dir}/{base_name}_cost_report_{timestamp}.json"
        with open(cost_file, "w") as f:
            json.dump(cost_summary, f, indent=2)
        print(f"✓ Cost report: {os.path.basename(cost_file)}")
        files["cost"] = cost_file

        # 2.5. Practitioner report (markdown + PDF) - Comprehensive report for medical professionals
        if result.practitioner_report:
            practitioner_file = f"{self.output_dir}/{base_name}_practitioner_report_{timestamp}.md"
            practitioner_complete = self._append_hardcoded_disclaimer(result.practitioner_report)
            with open(practitioner_file, "w") as f:
                f.write(practitioner_complete)
            print(f"✓ Practitioner report: {os.path.basename(practitioner_file)}")
            files["practitioner_report"] = practitioner_file

            # Generate PDF version of practitioner report
            practitioner_pdf = convert_markdown_to_pdf_safe(practitioner_file)
            if practitioner_pdf:
                print(f"✓ Practitioner PDF: {os.path.basename(practitioner_pdf)}")
                files["practitioner_pdf"] = practitioner_pdf

        # 3. Summary report (Markdown with disclaimer)
        summary_file = f"{self.output_dir}/{base_name}_medication_summary_{timestamp}.md"
        summary = self._generate_medication_summary(result, cost_summary)
        summary_complete = self._append_hardcoded_disclaimer(summary)

        with open(summary_file, "w") as f:
            f.write(summary_complete)
        print(f"✓ Summary report: {os.path.basename(summary_file)}")
        files["summary"] = summary_file

        # 3.5. Generate PDF version of summary
        summary_pdf = convert_markdown_to_pdf_safe(summary_file)
        if summary_pdf:
            print(f"✓ Summary PDF: {os.path.basename(summary_pdf)}")
            files["summary_pdf"] = summary_pdf

        # 4. Comprehensive report (detailed Markdown with disclaimer)
        detailed_file = f"{self.output_dir}/{base_name}_medication_detailed_{timestamp}.md"
        detailed = self._generate_medication_detailed_report(result, cost_summary)
        detailed_complete = self._append_hardcoded_disclaimer(detailed)

        with open(detailed_file, "w") as f:
            f.write(detailed_complete)
        print(f"✓ Detailed report: {os.path.basename(detailed_file)}")
        files["detailed"] = detailed_file

        # 4.5. Generate PDF version of detailed report
        detailed_pdf = convert_markdown_to_pdf_safe(detailed_file)
        if detailed_pdf:
            print(f"✓ Detailed PDF: {os.path.basename(detailed_pdf)}")
            files["detailed_pdf"] = detailed_pdf

        return files

    def _generate_procedure_summary(self, result: Any, cost_summary: Dict = None) -> str:
        """Generate markdown summary for procedure analysis"""
        cost_info = ""
        if cost_summary and cost_summary.get('total_cost', 0) > 0:
            cost_info = f"""
**Analysis Cost:** ${cost_summary['total_cost']:.4f}
**Duration:** {cost_summary['total_duration']:.1f}s"""

        summary = f"""# 🔬 Medical Procedure Analysis Report
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Analysis System:** MedicalReasoningAgent (5-Phase Pipeline){cost_info}

---

## 📋 Procedure Overview
**Procedure:** {result.procedure_summary}
**Analysis Confidence:** {result.confidence_score:.2f}/1.00
**Total Organs Analyzed:** {len(result.organs_analyzed)}
**Reasoning Steps Completed:** {len(result.reasoning_trace)}

---

## 🫀 Detailed Organ-Specific Analysis

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

        summary += """## 💡 General Recommendations

"""
        for i, rec in enumerate(result.general_recommendations, 1):
            summary += f"{i}. {rec}\n"

        summary += """
## 🔬 Research Gaps

"""
        for i, gap in enumerate(result.research_gaps, 1):
            summary += f"{i}. {gap}\n"

        # Add references section
        summary += """

---

## 📚 References

_Note: This analysis synthesizes information from medical literature, clinical guidelines, and evidence-based medicine databases. Specific citations would be included for claims about individual studies and recommendations._

"""

        # Add cost breakdown if available
        if cost_summary and cost_summary.get('total_cost', 0) > 0:
            summary += """

---

## 💰 Cost Analysis

"""
            summary += f"**Total Cost:** ${cost_summary['total_cost']:.4f}\n"
            summary += f"**Total Duration:** {cost_summary['total_duration']:.1f}s\n\n"
            summary += "### Phase Breakdown:\n\n"
            for phase in cost_summary.get('phases', []):
                pct = (phase['cost'] / cost_summary['total_cost'] * 100) if cost_summary['total_cost'] > 0 else 0
                summary += f"- **{phase['phase']}**: ${phase['cost']:.4f} ({pct:.1f}%) - {phase['duration']:.1f}s\n"

        summary += f"""

---

**Report Generated By:** MedicalReasoningAgent
**Timestamp:** {datetime.now().isoformat()}

⚠️ **DISCLAIMER:** This analysis is for educational and research purposes only. Always consult qualified healthcare providers for medical decisions.
"""

        return summary

    def _append_hardcoded_disclaimer(self, output: str) -> str:
        """
        Append mandatory hardcoded disclaimer to ALL outputs.

        This saves tokens by not requiring the LLM to generate disclaimers,
        and ensures consistency and completeness across all reports.
        """
        # Check if disclaimer already exists (avoid duplication)
        if "⚠️ **DISCLAIMER:**" in output or "DISCLAIMER:" in output:
            return output

        disclaimer = """

---

⚠️ **DISCLAIMER:** This analysis is for research and educational purposes only. It provides critical analysis of medical literature and evidence-based information but does **not** constitute medical advice, diagnosis, or treatment recommendations.

**Always consult qualified healthcare professionals** for medical decisions, treatment plans, and health-related questions. The information presented here should not replace professional medical judgment or be used as the sole basis for healthcare choices.

**Key Limitations:**
- Medical knowledge evolves rapidly; information may become outdated
- Individual health situations vary significantly
- Not all studies are equal in quality or applicability
- Risk-benefit assessments must be personalized
- Drug interactions and contraindications require professional evaluation

This analysis aims to inform and educate, not to direct medical care. When in doubt, seek professional medical guidance.
"""

        return output + disclaimer

    def _append_references_section(self, output: str, session: Any) -> str:
        """Append aggregated references section from all phases"""
        # Check if references already embedded in output
        if "## 📚 References" in output or "## References" in output or "## REFERENCES" in output.upper():
            # References already present in the LLM-generated output
            # But still append phase-collected references if available
            pass

        # Aggregate references from all phases
        all_references = []
        seen_refs = set()  # Deduplicate by DOI, PMID, or title

        for phase_result in session.phase_results:
            if hasattr(phase_result, 'references') and phase_result.references:
                for ref in phase_result.references:
                    if isinstance(ref, dict):
                        # Create unique key for deduplication
                        unique_key = (
                            ref.get('doi') or
                            ref.get('pmid') or
                            ref.get('raw_citation', '')[:100].lower()
                        )

                        if unique_key and unique_key not in seen_refs:
                            seen_refs.add(unique_key)
                            all_references.append(ref)

        # If no phase references and no embedded references, add note
        if not all_references and ("## 📚 References" not in output and "## References" not in output):
            refs_section = "\n\n---\n\n## 📚 References\n\n"
            refs_section += "_Note: This analysis synthesizes information from medical literature, clinical guidelines, and evidence-based medicine databases. Specific citations are included for individual studies and recommendations throughout the analysis._\n"
            return output + refs_section

        # If we have collected references from phases, append them
        if all_references:
            refs_section = "\n\n---\n\n## 📚 Additional Phase References\n\n"
            refs_section += "_References collected during analysis phases:_\n\n"

            for i, ref in enumerate(all_references[:30], 1):  # Limit to 30 references
                citation = ref.get('raw_citation', '')

                # Enhance with extracted metadata
                if ref.get('doi'):
                    citation += f" https://doi.org/{ref['doi']}"
                if ref.get('pmid'):
                    citation += f" PMID: {ref['pmid']}"
                if ref.get('url') and 'doi.org' not in citation:
                    citation += f" {ref['url']}"

                refs_section += f"[{i}] {citation}\n\n"

            return output + refs_section

        return output

    def _append_cost_section(self, output: str, cost_summary: Dict) -> str:
        """Append cost analysis section to output"""
        if cost_summary.get('total_cost', 0) == 0:
            return output

        cost_section = f"""

---

## 💰 Analysis Cost Summary

**Total Cost:** ${cost_summary['total_cost']:.4f}
**Total Duration:** {cost_summary['total_duration']:.1f}s

### Phase Breakdown:
"""
        for phase in cost_summary.get('phases', []):
            pct = (phase['cost'] / cost_summary['total_cost'] * 100) if cost_summary['total_cost'] > 0 else 0
            cost_section += f"- **{phase['phase']}**: ${phase['cost']:.4f} ({pct:.1f}%) - {phase['duration']:.1f}s\n"

        return output + cost_section

    def _generate_fact_check_summary(self, session: Any, cost_summary: Dict = None) -> str:
        """Generate markdown summary for fact check analysis"""
        cost_info = ""
        if cost_summary and cost_summary.get('total_cost', 0) > 0:
            cost_info = f"""
**Analysis Cost:** ${cost_summary['total_cost']:.4f}
**Duration:** {cost_summary['total_duration']:.1f}s"""

        summary = f"""# 🔎 Medical Fact Check Report
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Analysis System:** MedicalFactChecker (Independent Bio-Investigator){cost_info}

---

## 📋 Subject
**Topic:** {session.subject}
**Analysis Started:** {session.started_at.strftime('%Y-%m-%d %H:%M:%S')}
**Phases Completed:** {len(session.phase_results)}

---

## 🔬 Analysis Pipeline

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

        summary += """## 📄 Final Output

See the detailed output file for the complete analysis.

---

**Report Generated By:** MedicalFactChecker
**Timestamp:** {datetime.now().isoformat()}

⚠️ **DISCLAIMER:** This analysis is for research and educational purposes. It provides critical analysis of medical literature but does not constitute medical advice. Always consult qualified healthcare professionals.
"""

        return summary

    def _generate_medication_summary(self, result: Any, cost_summary: Dict = None) -> str:
        """Generate markdown summary for medication analysis"""
        cost_info = ""
        if cost_summary and cost_summary.get('total_cost', 0) > 0:
            cost_info = f"""
**Analysis Cost:** ${cost_summary['total_cost']:.4f}
**Duration:** {cost_summary['total_duration']:.1f}s
"""

        summary = f"""# 💊 Medication Analysis Report
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Analysis System:** MedicationAnalyzer (Comprehensive Drug Analysis){cost_info}

---

## 📋 Medication Overview
**Name:** {result.medication_name}
**Drug Class:** {result.drug_class}
**Analysis Confidence:** {result.analysis_confidence:.2f}/1.00

---

## 🧬 Mechanism of Action
{result.mechanism_of_action[:300] + '...' if len(result.mechanism_of_action) > 300 else result.mechanism_of_action}

---

## ⚗️ Pharmacokinetics Summary
- **Absorption:** {result.absorption[:100] + '...' if len(result.absorption) > 100 else result.absorption}
- **Metabolism:** {result.metabolism[:100] + '...' if len(result.metabolism) > 100 else result.metabolism}
- **Elimination:** {result.elimination[:100] + '...' if len(result.elimination) > 100 else result.elimination}
- **Half-Life:** {result.half_life}

---

## ⚠️ Key Safety Information

"""

        if result.black_box_warnings:
            summary += "### ⚠️ BLACK BOX WARNINGS\n\n"
            for i, warning in enumerate(result.black_box_warnings, 1):
                summary += f"{i}. {warning}\n"
            summary += "\n"

        if result.contraindications:
            summary += f"### Contraindications ({len(result.contraindications)} identified)\n\n"
            for contra in result.contraindications[:3]:  # Show top 3
                if isinstance(contra, dict):
                    summary += f"- **{contra.get('condition', 'N/A')}** ({contra.get('severity', 'N/A')})\n"
            if len(result.contraindications) > 3:
                summary += f"- _(and {len(result.contraindications) - 3} more - see detailed report)_\n"
            summary += "\n"

        summary += f"""---

## 🔗 Interactions Summary

"""

        if result.drug_interactions:
            severe = [i for i in result.drug_interactions if i.severity.value == "severe"]
            moderate = [i for i in result.drug_interactions if i.severity.value == "moderate"]

            if severe:
                summary += f"### 🔴 SEVERE Drug Interactions ({len(severe)})\n\n"
                for interaction in severe[:3]:
                    summary += f"- **{interaction.interacting_agent}**\n"
                    summary += f"  - Effect: {interaction.clinical_effect[:100]}...\n"
                    summary += f"  - Management: {interaction.management[:100]}...\n\n"

            if moderate:
                summary += f"### 🟡 Moderate Drug Interactions ({len(moderate)})\n\n"
                for interaction in moderate[:3]:
                    summary += f"- **{interaction.interacting_agent}**: {interaction.clinical_effect[:80]}...\n"

        if result.food_interactions:
            summary += f"\n### Food Interactions ({len(result.food_interactions)})\n\n"
            for interaction in result.food_interactions[:3]:
                summary += f"- **{interaction.interacting_agent}**: {interaction.management[:100]}...\n"

        summary += """
---

## ✅ Evidence-Based Recommendations

"""

        if result.evidence_based_recommendations:
            for i, rec in enumerate(result.evidence_based_recommendations[:5], 1):
                if isinstance(rec, dict):
                    summary += f"{i}. **{rec.get('intervention', 'N/A')}**\n"
                    summary += f"   - {rec.get('rationale', 'N/A')[:150]}...\n\n"

        summary += """
---

**For complete details including all interactions, dosing adjustments, and comprehensive recommendations, see the detailed report.**

⚠️ **DISCLAIMER:** This analysis is for educational and research purposes only. Always consult qualified healthcare providers for medication decisions.
"""

        return summary

    def _generate_medication_detailed_report(self, result: Any, cost_summary: Dict = None) -> str:
        """Generate comprehensive detailed report for medication"""
        cost_header = ""
        if cost_summary and cost_summary.get('total_cost', 0) > 0:
            cost_header = f"\n**Analysis Cost:** ${cost_summary['total_cost']:.4f}\n**Duration:** {cost_summary['total_duration']:.1f}s"

        report = f"""# 💊 Comprehensive Medication Analysis: {result.medication_name}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Analysis Confidence:** {result.analysis_confidence:.2f}
**Evidence Quality:** {result.evidence_quality}{cost_header}

---

## 📑 Table of Contents
1. [Overview](#overview)
2. [Pharmacology](#pharmacology)
3. [Clinical Use](#clinical-use)
4. [Interactions](#interactions)
5. [Safety Profile](#safety-profile)
6. [Recommendations](#recommendations)
7. [Monitoring](#monitoring)

---

## 📋 Overview

### 🧬 Drug Classification
**Drug Class:** {result.drug_class}

### 🔬 Mechanism of Action
{result.mechanism_of_action}

---

## ⚗️ Pharmacology

### Absorption
{result.absorption}

### Distribution & Metabolism
{result.metabolism}

### Elimination
{result.elimination}

**Half-Life:** {result.half_life}

---

## 💉 Clinical Use

### Approved Indications
"""

        for i, indication in enumerate(result.approved_indications, 1):
            report += f"{i}. {indication}\n"

        if result.off_label_uses:
            report += "\n### Off-Label Uses\n"
            for i, use in enumerate(result.off_label_uses, 1):
                report += f"{i}. {use}\n"

        report += f"""
### Standard Dosing
{result.standard_dosing}

"""

        if result.dose_adjustments:
            report += "### Dose Adjustments\n"
            for adjustment_type, adjustment_info in result.dose_adjustments.items():
                report += f"**{adjustment_type.replace('_', ' ').title()}:**\n{adjustment_info}\n\n"

        report += """
---

## 🔗 Interactions

### 💊 Drug-Drug Interactions
"""

        if result.drug_interactions:
            for interaction in result.drug_interactions:
                severity_emoji = "🔴" if interaction.severity.value == "severe" else "🟡" if interaction.severity.value == "moderate" else "🟢"
                report += f"\n#### {severity_emoji} {interaction.interacting_agent} ({interaction.severity.value.upper()})\n\n"
                report += f"**Mechanism:** {interaction.mechanism}\n\n"
                report += f"**Clinical Effect:** {interaction.clinical_effect}\n\n"
                report += f"**Management:** {interaction.management}\n\n"
                if interaction.time_separation:
                    report += f"**Time Separation:** {interaction.time_separation}\n\n"
                report += f"**Evidence Level:** {interaction.evidence_level}\n\n"
        else:
            report += "No significant drug-drug interactions identified.\n\n"

        report += "### 🍎 Food & Lifestyle Interactions\n\n"

        if result.food_interactions:
            for interaction in result.food_interactions:
                report += f"#### {interaction.interacting_agent}\n\n"
                report += f"**Mechanism:** {interaction.mechanism}\n\n"
                report += f"**Clinical Effect:** {interaction.clinical_effect}\n\n"
                report += f"**Management:** {interaction.management}\n\n"
        else:
            report += "No significant food interactions identified.\n\n"

        if result.environmental_considerations:
            report += "### Environmental Considerations\n\n"
            for consideration in result.environmental_considerations:
                if isinstance(consideration, dict):
                    report += f"- **{consideration.get('type', 'N/A')}:** {consideration.get('description', 'N/A')}\n"
                else:
                    report += f"- {consideration}\n"

        report += """
---

## ⚠️ Safety Profile

"""

        if result.black_box_warnings:
            report += "### 🚨 BLACK BOX WARNINGS\n\n"
            for i, warning in enumerate(result.black_box_warnings, 1):
                report += f"{i}. {warning}\n\n"

        report += "### Adverse Effects\n\n"

        if result.common_adverse_effects:
            report += "**Common (>10%):**\n"
            for effect in result.common_adverse_effects:
                report += f"- {effect}\n"
            report += "\n"

        if result.serious_adverse_effects:
            report += "**Serious (Any Frequency):**\n"
            for effect in result.serious_adverse_effects:
                report += f"- {effect}\n"
            report += "\n"

        if result.contraindications:
            report += "### Contraindications\n\n"
            for contra in result.contraindications:
                if isinstance(contra, dict):
                    report += f"**{contra.get('condition', 'N/A')}** ({contra.get('severity', 'N/A')})\n"
                    report += f"- Reason: {contra.get('reason', 'N/A')}\n"
                    if contra.get('alternative'):
                        report += f"- Alternative: {contra.get('alternative')}\n"
                    report += "\n"

        if result.warning_signs:
            report += "### Warning Signs\n\n"
            for sign in result.warning_signs:
                if isinstance(sign, dict):
                    report += f"**{sign.get('sign', 'N/A')}** ({sign.get('severity', 'N/A')})\n"
                    report += f"- Action: {sign.get('action', 'N/A')}\n\n"

        report += """
---

## 💡 Recommendations

### ✅ What TO DO: Evidence-Based Recommendations

"""

        if result.evidence_based_recommendations:
            for i, rec in enumerate(result.evidence_based_recommendations, 1):
                if isinstance(rec, dict):
                    report += f"#### {i}. {rec.get('intervention', 'N/A')}\n\n"
                    report += f"**Rationale:** {rec.get('rationale', 'N/A')}\n\n"
                    report += f"**Evidence Level:** {rec.get('evidence_level', 'N/A')}\n\n"
                    report += f"**Implementation:** {rec.get('implementation', 'N/A')}\n\n"
                    if rec.get('expected_outcome'):
                        report += f"**Expected Outcome:** {rec.get('expected_outcome')}\n\n"

        if result.investigational_approaches:
            report += "### 🔬 Investigational Approaches (Limited Evidence)\n\n"
            for i, rec in enumerate(result.investigational_approaches, 1):
                if isinstance(rec, dict):
                    report += f"#### {i}. {rec.get('intervention', 'N/A')}\n\n"
                    report += f"**Rationale:** {rec.get('rationale', 'N/A')}\n\n"
                    report += f"**Limitations:** {rec.get('limitations', 'N/A')}\n\n"

        if result.debunked_claims:
            report += "### ❌ What NOT TO DO: Debunked Claims\n\n"
            for i, claim in enumerate(result.debunked_claims, 1):
                if isinstance(claim, dict):
                    report += f"#### ❌ {i}. {claim.get('claim', 'N/A')}\n\n"
                    report += f"**Why Debunked:** {claim.get('reason_debunked', 'N/A')}\n\n"
                    report += f"**Evidence Against:** {claim.get('evidence', 'N/A')}\n\n"
                    report += f"**Why Harmful:** {claim.get('why_harmful', 'N/A')}\n\n"

        report += """
---

## 📊 Monitoring Requirements

"""

        if result.monitoring_requirements:
            for i, req in enumerate(result.monitoring_requirements, 1):
                if isinstance(req, dict):
                    report += f"{i}. **{req.get('parameter', 'N/A')}**\n"
                    report += f"   - Frequency: {req.get('frequency', 'N/A')}\n"
                    report += f"   - Rationale: {req.get('rationale', 'N/A')}\n\n"
                else:
                    report += f"{i}. {req}\n"

        report += f"""
---

**Analysis Completed:** {datetime.now().isoformat()}
**Reasoning Steps:** {len(result.reasoning_trace)}
"""

        # Add cost breakdown if available
        if cost_summary and cost_summary.get('total_cost', 0) > 0:
            report += f"""
---

## 💰 Cost Analysis

**Total Cost:** ${cost_summary['total_cost']:.4f}
**Total Duration:** {cost_summary['total_duration']:.1f}s

### Phase Breakdown

"""
            for phase in cost_summary.get('phases', []):
                pct = (phase['cost'] / cost_summary['total_cost'] * 100) if cost_summary['total_cost'] > 0 else 0
                report += f"- **{phase['phase']}**: ${phase['cost']:.4f} ({pct:.1f}%) - {phase['duration']:.1f}s\n"

        report += """
---

⚠️ **IMPORTANT DISCLAIMER:** This analysis is for educational and research purposes only.
It does not constitute medical advice. Always consult qualified healthcare professionals for
medication decisions, dosing, and management of health conditions.

---

*Generated by Medical Analysis Agent*
"""

        return report

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

  # Run medication analyzer
  python run_analysis.py medication --subject "Metformin" --indication "Type 2 Diabetes" --other-meds "Lisinopril" "Atorvastatin"

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
        choices=["procedure", "medication", "factcheck"],
        help="Which agent to run (procedure, medication, or factcheck)",
    )

    parser.add_argument("--list", action="store_true", help="List all available agents")
    parser.add_argument(
        "--check-llms",
        action="store_true",
        help="Print which LLM providers are configured and exit",
    )
    parser.add_argument(
        "--models",
        action="store_true",
        help="List supported model identifiers and exit",
    )

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
        "--indication",
        type=str,
        help="Primary indication for medication analyzer",
    )

    parser.add_argument(
        "--other-meds",
        nargs="*",
        help="Other medications patient is taking (for medication analyzer)",
    )

    parser.add_argument(
        "--llm",
        type=str,
        default="claude-sonnet",
        choices=["claude-sonnet", "claude-opus", "openai", "ollama", "grok-4-1-fast", "grok-4-1-code", "grok-4-1-reasoning"],
        help="LLM provider to use (default: claude-sonnet). Options: claude-sonnet (Sonnet 4.5), claude-opus (Opus 4.5), openai (GPT-4), ollama (local), grok-4-1-fast (fast non-reasoning), grok-4-1-code (code optimized), grok-4-1-reasoning (reasoning optimized)",
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

    if args.check_llms:
        print_llm_status(load_env=True)
        return

    if args.models:
        from llm_integrations import get_available_models

        models = get_available_models()
        print("\nAvailable model identifiers:")
        for model_name, provider in sorted(models.items()):
            print(f"  - {model_name} ({provider})")
        print()
        return

    # Handle list command
    if args.list:
        AgentOrchestrator.list_agents()
        return

    # Validate arguments
    if not args.agent:
        parser.print_help()
        print("\n❌ Error: Please specify an agent (procedure, medication, or factcheck)")
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

        elif args.agent == "medication":
            result, files = orchestrator.run_medication_analyzer(
                medication=args.subject,
                indication=args.indication,
                other_medications=args.other_meds,
                llm_provider=args.llm,
                timeout=args.timeout,
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
