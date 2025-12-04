#!/usr/bin/env python3
"""
Test Runner for MedicalReasoningAgent
Runs analysis and saves comprehensive outputs to the outputs/ directory.
"""

import os
import json
from datetime import datetime
from medical_procedure_analyzer import MedicalReasoningAgent, MedicalInput

def save_analysis_results(result, procedure_name: str, output_dir: str = "outputs"):
    """Save comprehensive analysis results"""

    # Create outputs directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate base filename
    base_name = procedure_name.replace(" ", "_").lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Save reasoning trace (detailed step-by-step reasoning)
    trace_file = f"{output_dir}/{base_name}_reasoning_trace_{timestamp}.json"
    trace_data = []
    for step in result.reasoning_trace:
        trace_data.append({
            "stage": step.stage.value,
            "timestamp": step.timestamp.isoformat(),
            "reasoning": step.reasoning,
            "input": step.input_data,
            "output": step.output,
            "confidence": step.confidence,
            "sources": step.sources
        })

    with open(trace_file, 'w') as f:
        json.dump(trace_data, f, indent=2)
    print(f"✓ Reasoning trace saved: {trace_file}")

    # 2. Save analysis result (structured output)
    result_file = f"{output_dir}/{base_name}_analysis_result_{timestamp}.json"
    analysis_data = {
        "timestamp": datetime.now().isoformat(),
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
                "evidence_quality": organ.evidence_quality
            }
            for organ in result.organs_analyzed
        ],
        "general_recommendations": result.general_recommendations,
        "research_gaps": result.research_gaps,
        "reasoning_steps_count": len(result.reasoning_trace)
    }

    with open(result_file, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    print(f"✓ Analysis result saved: {result_file}")

    # 3. Save human-readable summary report
    summary_file = f"{output_dir}/{base_name}_summary_report_{timestamp}.md"
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

    summary += f"""## General Recommendations

"""
    for i, rec in enumerate(result.general_recommendations, 1):
        summary += f"{i}. {rec}\n"

    summary += f"""
## Research Gaps

"""
    for i, gap in enumerate(result.research_gaps, 1):
        summary += f"{i}. {gap}\n"

    summary += f"""
## Reasoning Pipeline Summary

"""

    reasoning_stages = {}
    for step in result.reasoning_trace:
        stage_name = step.stage.value.replace('_', ' ').title()
        if stage_name not in reasoning_stages:
            reasoning_stages[stage_name] = step.reasoning

    for stage, reasoning in reasoning_stages.items():
        summary += f"**{stage}:** {reasoning}\n\n"

    summary += f"""
---

**Report Generated By:** MedicalReasoningAgent v2.0
**Analysis Methodology:** 6-Stage Reasoning Pipeline
**Timestamp:** {datetime.now().isoformat()}

⚠️ **DISCLAIMER:** This analysis is for educational and research purposes only. Always consult qualified healthcare providers for medical decisions.
"""

    with open(summary_file, 'w') as f:
        f.write(summary)
    print(f"✓ Summary report saved: {summary_file}")

    return trace_file, result_file, summary_file


def main():
    print("=" * 60)
    print("Medical Reasoning Agent - Test Runner")
    print("=" * 60)
    print()

    # Create agent with full 6-stage reasoning pipeline
    print("🤖 Initializing MedicalReasoningAgent with Claude...")
    agent = MedicalReasoningAgent(
        primary_llm_provider="claude",
        fallback_providers=["openai"],
        enable_logging=True
    )
    print()

    # Define test input
    medical_input = MedicalInput(
        procedure="MRI Scanner",
        details="With gadolinium contrast",
        objectives=("understand implications", "identify risks", "post-procedure care",
                   "affected organs", "organs at risk"),
        patient_context="Standard adult patient"
    )

    print(f"📋 Analyzing: {medical_input.procedure}")
    print(f"   Details: {medical_input.details}")
    print(f"   Objectives: {', '.join(medical_input.objectives)}")
    print()
    print("⏳ Running 6-stage analysis pipeline...")
    print("   Stage 1: Input Analysis")
    print("   Stage 2: Organ Identification (LLM)")
    print("   Stage 3: Evidence Gathering (LLM per organ)")
    print("   Stage 4: Risk Assessment")
    print("   Stage 5: Recommendation Synthesis (LLM per organ)")
    print("   Stage 6: Critical Evaluation")
    print()

    # Run analysis
    result = agent.analyze_medical_procedure(medical_input)

    print()
    print("=" * 60)
    print("✅ Analysis Complete!")
    print("=" * 60)
    print(f"📊 Procedure: {result.procedure_summary}")
    print(f"🫀 Organs Analyzed: {len(result.organs_analyzed)}")
    print(f"📈 Confidence Score: {result.confidence_score:.2f}")
    print(f"🧠 Reasoning Steps: {len(result.reasoning_trace)}")
    print()

    # Display brief results
    print("🔍 Organs Identified:")
    for organ in result.organs_analyzed:
        print(f"   - {organ.organ_name.upper()}: {organ.risk_level} risk")
        print(f"     ✓ {len(organ.known_recommendations)} evidence-based recommendations")
        print(f"     ⚗️ {len(organ.potential_recommendations)} investigational approaches")
        print(f"     ❌ {len(organ.debunked_claims)} debunked claims identified")
    print()

    # Save outputs
    print("💾 Saving outputs...")
    trace_file, result_file, summary_file = save_analysis_results(
        result,
        medical_input.procedure
    )

    print()
    print("=" * 60)
    print("📁 All outputs saved to outputs/ directory:")
    print(f"   1. Reasoning Trace (JSON): {os.path.basename(trace_file)}")
    print(f"   2. Analysis Result (JSON): {os.path.basename(result_file)}")
    print(f"   3. Summary Report (MD): {os.path.basename(summary_file)}")
    print("=" * 60)
    print()
    print(f"💡 View the summary report for human-readable results:")
    print(f"   cat {summary_file}")
    print()
    print(f"💡 View the reasoning trace to see all 6 stages:")
    print(f"   cat {trace_file} | jq '.[] | {{stage: .stage, reasoning: .reasoning}}'")
    print()


if __name__ == "__main__":
    main()
