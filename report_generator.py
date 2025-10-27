#!/usr/bin/env python3
"""
Medical Analysis Report Generator
Generates comprehensive reports in JSON and Markdown formats from analysis results.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

from medical_reasoning_agent import MedicalOutput, OrganAnalysis, ReasoningStep


@dataclass
class ReportConfig:
    """Configuration for report generation"""
    include_reasoning_trace: bool = True
    include_validation_scores: bool = True
    include_timestamps: bool = True
    include_confidence_analysis: bool = True
    include_recommendations_summary: bool = True
    markdown_style: str = "detailed"  # detailed, summary, clinical


class MedicalReportGenerator:
    """Generates medical analysis reports in multiple formats"""
    
    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or ReportConfig()
    
    def generate_json_report(self, analysis_result: MedicalOutput, 
                           validation_report: Optional[Dict] = None,
                           output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate machine-readable JSON report.
        
        Args:
            analysis_result: Medical analysis output
            validation_report: Optional validation report
            output_path: Optional file path to save report
            
        Returns:
            Dict containing structured report data
        """
        report = {
            "metadata": {
                "report_generated_at": datetime.now().isoformat(),
                "report_version": "1.0",
                "analysis_type": "medical_procedure_reasoning"
            },
            "procedure": {
                "summary": analysis_result.procedure_summary,
                "confidence_score": analysis_result.confidence_score,
                "organs_analyzed_count": len(analysis_result.organs_analyzed),
                "reasoning_steps_count": len(analysis_result.reasoning_trace)
            },
            "organs_analysis": self._format_organs_for_json(analysis_result.organs_analyzed),
            "general_recommendations": analysis_result.general_recommendations,
            "research_gaps": analysis_result.research_gaps
        }
        
        # Add reasoning trace if enabled
        if self.config.include_reasoning_trace:
            report["reasoning_trace"] = self._format_reasoning_trace_for_json(
                analysis_result.reasoning_trace
            )
        
        # Add validation scores if provided
        if validation_report and self.config.include_validation_scores:
            report["validation"] = {
                "overall_score": validation_report.get("overall_score", 0),
                "safety_score": validation_report.get("safety_score", 0),
                "accuracy_score": validation_report.get("accuracy_score", 0),
                "completeness_score": validation_report.get("completeness_score", 0),
                "issues_count": len(validation_report.get("issues", [])),
                "critical_issues": len([
                    i for i in validation_report.get("issues", []) 
                    if i.get("severity") == "critical"
                ])
            }
        
        # Add confidence analysis
        if self.config.include_confidence_analysis:
            report["confidence_analysis"] = self._analyze_confidence(analysis_result)
        
        # Add recommendations summary
        if self.config.include_recommendations_summary:
            report["recommendations_summary"] = self._summarize_recommendations(
                analysis_result.organs_analyzed
            )
        
        # Save to file if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report
    
    def generate_markdown_report(self, analysis_result: MedicalOutput,
                               validation_report: Optional[Dict] = None,
                               output_path: Optional[str] = None) -> str:
        """
        Generate human-readable Markdown report.
        
        Args:
            analysis_result: Medical analysis output
            validation_report: Optional validation report
            output_path: Optional file path to save report
            
        Returns:
            String containing Markdown report
        """
        if self.config.markdown_style == "clinical":
            report = self._generate_clinical_markdown(analysis_result, validation_report)
        elif self.config.markdown_style == "summary":
            report = self._generate_summary_markdown(analysis_result, validation_report)
        else:
            report = self._generate_detailed_markdown(analysis_result, validation_report)
        
        # Save to file if path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report
    
    def _format_organs_for_json(self, organs: List[OrganAnalysis]) -> List[Dict[str, Any]]:
        """Format organ analyses for JSON output"""
        return [
            {
                "organ_name": organ.organ_name,
                "affected_by_procedure": organ.affected_by_procedure,
                "at_risk": organ.at_risk,
                "risk_level": organ.risk_level,
                "pathways_involved": organ.pathways_involved,
                "evidence_quality": organ.evidence_quality,
                "recommendations": {
                    "known": organ.known_recommendations,
                    "potential": organ.potential_recommendations,
                    "debunked_claims": organ.debunked_claims
                },
                "recommendation_counts": {
                    "known_count": len(organ.known_recommendations),
                    "potential_count": len(organ.potential_recommendations),
                    "debunked_count": len(organ.debunked_claims)
                }
            }
            for organ in organs
        ]
    
    def _format_reasoning_trace_for_json(self, trace: List[ReasoningStep]) -> List[Dict[str, Any]]:
        """Format reasoning trace for JSON output"""
        return [
            {
                "stage": step.stage.value,
                "timestamp": step.timestamp.isoformat() if self.config.include_timestamps else None,
                "reasoning": step.reasoning,
                "confidence": step.confidence,
                "input_summary": str(step.input_data)[:200] + "..." if len(str(step.input_data)) > 200 else str(step.input_data),
                "output_summary": str(step.output)[:200] + "..." if len(str(step.output)) > 200 else str(step.output)
            }
            for step in trace
        ]
    
    def _analyze_confidence(self, analysis_result: MedicalOutput) -> Dict[str, Any]:
        """Analyze confidence patterns in the analysis"""
        trace_confidences = [step.confidence for step in analysis_result.reasoning_trace]
        
        return {
            "overall_confidence": analysis_result.confidence_score,
            "reasoning_confidence": {
                "average": sum(trace_confidences) / len(trace_confidences) if trace_confidences else 0,
                "minimum": min(trace_confidences) if trace_confidences else 0,
                "maximum": max(trace_confidences) if trace_confidences else 0,
                "consistency": max(trace_confidences) - min(trace_confidences) if trace_confidences else 0
            },
            "confidence_interpretation": self._interpret_confidence(analysis_result.confidence_score)
        }
    
    def _interpret_confidence(self, score: float) -> str:
        """Interpret confidence score"""
        if score >= 0.9:
            return "Very High - Strong evidence-based analysis"
        elif score >= 0.75:
            return "High - Good evidence with some limitations"
        elif score >= 0.6:
            return "Moderate - Mixed evidence quality"
        elif score >= 0.4:
            return "Low - Limited evidence available"
        else:
            return "Very Low - Insufficient evidence"
    
    def _summarize_recommendations(self, organs: List[OrganAnalysis]) -> Dict[str, Any]:
        """Summarize recommendations across all organs"""
        total_known = sum(len(organ.known_recommendations) for organ in organs)
        total_potential = sum(len(organ.potential_recommendations) for organ in organs)
        total_debunked = sum(len(organ.debunked_claims) for organ in organs)
        
        high_risk_organs = [organ for organ in organs if organ.risk_level == "high"]
        moderate_risk_organs = [organ for organ in organs if organ.risk_level == "moderate"]
        
        return {
            "totals": {
                "known_recommendations": total_known,
                "potential_recommendations": total_potential,
                "debunked_claims": total_debunked
            },
            "risk_distribution": {
                "high_risk_organs": len(high_risk_organs),
                "moderate_risk_organs": len(moderate_risk_organs),
                "low_risk_organs": len(organs) - len(high_risk_organs) - len(moderate_risk_organs)
            },
            "organs_with_most_recommendations": sorted(
                [(organ.organ_name, len(organ.known_recommendations) + len(organ.potential_recommendations)) 
                 for organ in organs],
                key=lambda x: x[1],
                reverse=True
            )[:3]
        }
    
    def _generate_detailed_markdown(self, analysis_result: MedicalOutput,
                                  validation_report: Optional[Dict] = None) -> str:
        """Generate detailed Markdown report"""
        report_lines = []
        
        # Header
        report_lines.extend([
            "# Medical Procedure Analysis Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Procedure:** {analysis_result.procedure_summary}",
            f"**Overall Confidence:** {analysis_result.confidence_score:.2f} ({self._interpret_confidence(analysis_result.confidence_score)})",
            ""
        ])
        
        # Executive Summary
        report_lines.extend([
            "## 📋 Executive Summary",
            "",
            f"This analysis examined **{len(analysis_result.organs_analyzed)} organ systems** affected by the specified medical procedure.",
            f"The systematic reasoning process completed **{len(analysis_result.reasoning_trace)} analytical stages** following evidence-based medical principles.",
            ""
        ])
        
        # Validation Scores (if available)
        if validation_report and self.config.include_validation_scores:
            report_lines.extend([
                "## 🎯 Quality Assessment",
                "",
                f"- **Overall Score:** {validation_report.get('overall_score', 0):.2f}/1.00",
                f"- **Safety Score:** {validation_report.get('safety_score', 0):.2f}/1.00",
                f"- **Medical Accuracy:** {validation_report.get('accuracy_score', 0):.2f}/1.00",
                f"- **Completeness:** {validation_report.get('completeness_score', 0):.2f}/1.00",
                "",
                f"**Issues Identified:** {len(validation_report.get('issues', []))} total",
                ""
            ])
        
        # Organ-by-Organ Analysis
        report_lines.extend([
            "## 🔍 Organ System Analysis",
            ""
        ])
        
        for organ in analysis_result.organs_analyzed:
            risk_emoji = "🔴" if organ.risk_level == "high" else "🟡" if organ.risk_level == "moderate" else "🟢"
            
            report_lines.extend([
                f"### {risk_emoji} {organ.organ_name.title()}",
                "",
                f"**Risk Level:** {organ.risk_level.title()}  ",
                f"**Evidence Quality:** {organ.evidence_quality.title()}  ",
                f"**Pathways Involved:** {', '.join(organ.pathways_involved) if organ.pathways_involved else 'Not specified'}",
                ""
            ])
            
            # Known Recommendations
            if organ.known_recommendations:
                report_lines.extend([
                    "#### ✅ Evidence-Based Recommendations",
                    ""
                ])
                for rec in organ.known_recommendations:
                    report_lines.append(f"- {rec}")
                report_lines.append("")
            
            # Potential Recommendations
            if organ.potential_recommendations:
                report_lines.extend([
                    "#### 🔬 Potential Interventions (Limited Evidence)",
                    ""
                ])
                for rec in organ.potential_recommendations:
                    report_lines.append(f"- {rec}")
                report_lines.append("")
            
            # Debunked Claims
            if organ.debunked_claims:
                report_lines.extend([
                    "#### ❌ Debunked/Ineffective Approaches",
                    ""
                ])
                for claim in organ.debunked_claims:
                    report_lines.append(f"- {claim}")
                report_lines.append("")
        
        # General Recommendations
        if analysis_result.general_recommendations:
            report_lines.extend([
                "## 🎯 General Recommendations",
                ""
            ])
            for rec in analysis_result.general_recommendations:
                report_lines.append(f"- {rec}")
            report_lines.append("")
        
        # Research Gaps
        if analysis_result.research_gaps:
            report_lines.extend([
                "## 🔬 Research Gaps Identified",
                ""
            ])
            for gap in analysis_result.research_gaps:
                report_lines.append(f"- {gap}")
            report_lines.append("")
        
        # Reasoning Trace (if enabled)
        if self.config.include_reasoning_trace and analysis_result.reasoning_trace:
            report_lines.extend([
                "## 🧠 Reasoning Process",
                "",
                "The analysis followed this systematic approach:",
                ""
            ])
            
            for i, step in enumerate(analysis_result.reasoning_trace, 1):
                confidence_bar = "█" * int(step.confidence * 10) + "░" * (10 - int(step.confidence * 10))
                report_lines.extend([
                    f"### {i}. {step.stage.value.replace('_', ' ').title()}",
                    "",
                    f"**Confidence:** `{confidence_bar}` {step.confidence:.2f}",
                    "",
                    f"{step.reasoning}",
                    ""
                ])
        
        # Footer
        report_lines.extend([
            "---",
            "",
            "**⚠️ Important Notice:** This analysis is for educational and research purposes only.",
            "Always consult qualified healthcare professionals for medical decisions.",
            "",
            f"*Report generated by Medical Reasoning Agent v1.0 on {datetime.now().strftime('%Y-%m-%d')}*"
        ])
        
        return "\n".join(report_lines)
    
    def _generate_summary_markdown(self, analysis_result: MedicalOutput,
                                 validation_report: Optional[Dict] = None) -> str:
        """Generate summary Markdown report"""
        report_lines = []
        
        # Header
        report_lines.extend([
            f"# {analysis_result.procedure_summary} - Analysis Summary",
            "",
            f"**Confidence:** {analysis_result.confidence_score:.2f} | **Organs:** {len(analysis_result.organs_analyzed)} | **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            ""
        ])
        
        # Quick Overview
        high_risk = [o for o in analysis_result.organs_analyzed if o.risk_level == "high"]
        moderate_risk = [o for o in analysis_result.organs_analyzed if o.risk_level == "moderate"]
        
        if high_risk:
            report_lines.extend([
                "## 🔴 High Risk Organs",
                ""
            ])
            for organ in high_risk:
                report_lines.append(f"- **{organ.organ_name.title()}**: {len(organ.known_recommendations)} evidence-based recommendations")
            report_lines.append("")
        
        if moderate_risk:
            report_lines.extend([
                "## 🟡 Moderate Risk Organs", 
                ""
            ])
            for organ in moderate_risk:
                report_lines.append(f"- **{organ.organ_name.title()}**: {len(organ.known_recommendations)} known + {len(organ.potential_recommendations)} potential interventions")
            report_lines.append("")
        
        # Key Recommendations
        all_known = []
        for organ in analysis_result.organs_analyzed:
            all_known.extend(organ.known_recommendations)
        
        if all_known:
            report_lines.extend([
                "## ✅ Key Evidence-Based Actions",
                ""
            ])
            for rec in list(set(all_known))[:5]:  # Top 5 unique recommendations
                report_lines.append(f"- {rec}")
            report_lines.append("")
        
        # Common Debunked Claims
        all_debunked = []
        for organ in analysis_result.organs_analyzed:
            all_debunked.extend(organ.debunked_claims)
        
        if all_debunked:
            report_lines.extend([
                "## ❌ Avoid These Debunked Approaches",
                ""
            ])
            for claim in list(set(all_debunked))[:3]:  # Top 3 unique debunked claims
                report_lines.append(f"- {claim}")
            report_lines.append("")
        
        report_lines.extend([
            "---",
            "*For detailed analysis, generate full report. Educational use only.*"
        ])
        
        return "\n".join(report_lines)
    
    def _generate_clinical_markdown(self, analysis_result: MedicalOutput,
                                  validation_report: Optional[Dict] = None) -> str:
        """Generate clinical-style Markdown report"""
        report_lines = []
        
        # Clinical Header
        report_lines.extend([
            "# MEDICAL PROCEDURE ANALYSIS",
            "",
            f"**PROCEDURE:** {analysis_result.procedure_summary}",
            f"**ANALYSIS DATE:** {datetime.now().strftime('%d %b %Y %H:%M')}",
            f"**CONFIDENCE LEVEL:** {analysis_result.confidence_score:.2f}",
            ""
        ])
        
        # Clinical Assessment
        report_lines.extend([
            "## CLINICAL ASSESSMENT",
            ""
        ])
        
        for organ in analysis_result.organs_analyzed:
            risk_level = organ.risk_level.upper()
            report_lines.extend([
                f"**{organ.organ_name.upper()}** - {risk_level} RISK",
                ""
            ])
            
            if organ.known_recommendations:
                report_lines.append("*Recommended interventions:*")
                for rec in organ.known_recommendations:
                    report_lines.append(f"  • {rec}")
                report_lines.append("")
            
            if organ.debunked_claims:
                report_lines.append("*Contraindicated/ineffective:*")
                for claim in organ.debunked_claims:
                    report_lines.append(f"  • {claim}")
                report_lines.append("")
        
        # Clinical Notes
        if analysis_result.general_recommendations:
            report_lines.extend([
                "## CLINICAL NOTES",
                ""
            ])
            for rec in analysis_result.general_recommendations:
                report_lines.append(f"- {rec}")
            report_lines.append("")
        
        report_lines.extend([
            "---",
            "**DISCLAIMER:** This analysis is generated for educational purposes.",
            "Clinical decisions should always involve qualified healthcare professionals."
        ])
        
        return "\n".join(report_lines)


def generate_reports(analysis_result: MedicalOutput,
                    validation_report: Optional[Dict] = None,
                    output_dir: str = "reports",
                    base_filename: str = None) -> Dict[str, str]:
    """
    Convenience function to generate both JSON and Markdown reports.
    
    Args:
        analysis_result: Medical analysis output
        validation_report: Optional validation report
        output_dir: Directory to save reports
        base_filename: Base filename (auto-generated if None)
        
    Returns:
        Dict with paths to generated files
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    if not base_filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        procedure_clean = "".join(c for c in analysis_result.procedure_summary if c.isalnum() or c.isspace()).strip()
        procedure_clean = "_".join(procedure_clean.split())[:30]
        base_filename = f"{procedure_clean}_{timestamp}"
    
    generator = MedicalReportGenerator()
    
    # Generate reports
    json_path = output_path / f"{base_filename}_report.json"
    markdown_path = output_path / f"{base_filename}_report.md"
    summary_path = output_path / f"{base_filename}_summary.md"
    
    # JSON report
    generator.generate_json_report(analysis_result, validation_report, str(json_path))
    
    # Detailed Markdown report
    generator.generate_markdown_report(analysis_result, validation_report, str(markdown_path))
    
    # Summary Markdown report
    generator.config.markdown_style = "summary"
    generator.generate_markdown_report(analysis_result, validation_report, str(summary_path))
    
    return {
        "json_report": str(json_path),
        "detailed_markdown": str(markdown_path),
        "summary_markdown": str(summary_path)
    }


if __name__ == "__main__":
    # Example usage
    from medical_reasoning_agent import MedicalReasoningAgent, MedicalInput
    from validation_scoring import validate_medical_output
    
    # Generate sample analysis
    agent = MedicalReasoningAgent(enable_logging=False)
    input_data = MedicalInput("MRI Scanner", "With contrast", ("test", "report", "generation"))
    result = agent.analyze_medical_procedure(input_data)
    
    # Validate
    validation = validate_medical_output(result)
    
    # Convert validation report to dict format
    validation_dict = {
        "overall_score": validation.overall_score,
        "safety_score": validation.safety_score,
        "accuracy_score": validation.accuracy_score,
        "completeness_score": validation.completeness_score,
        "issues": [
            {
                "severity": issue.severity.value,
                "category": issue.category.value,
                "message": issue.message
            }
            for issue in validation.issues
        ]
    }
    
    # Generate reports
    report_paths = generate_reports(result, validation_dict, "sample_reports")
    
    print("Sample reports generated:")
    for report_type, path in report_paths.items():
        print(f"  {report_type}: {path}")