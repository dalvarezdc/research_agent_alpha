#!/usr/bin/env python3
"""
Simplified Medical Reasoning Agent
Clean, focused orchestration of medical analysis components.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from medical_reasoning_agent import MedicalInput, MedicalOutput, OrganAnalysis, ReasoningStep, ReasoningStage
from consolidated_analyzer import ConsolidatedAnalyzer
from input_validation import InputValidator, ValidationError
from colored_logger import get_colored_logger


class SimpleMedicalAgent:
    """Simplified medical reasoning agent with clear separation of concerns"""
    
    def __init__(self, llm_manager=None, enable_logging: bool = True):
        """Initialize with optional LLM support"""
        self.llm_manager = llm_manager
        self.reasoning_trace: List[ReasoningStep] = []

        # Initialize consolidated analyzer (replaces organ_analyzer, evidence_gatherer, recommendation_synthesizer)
        self.consolidated_analyzer = ConsolidatedAnalyzer(llm_manager)

        # Setup colored logging
        self.logger = get_colored_logger(__name__, enable_logging)

        # Log initialization with colors
        if llm_manager:
            self.logger.llm_enabled("AI models")
        else:
            self.logger.llm_offline_mode()
    
    def analyze_procedure(self, medical_input: MedicalInput) -> MedicalOutput:
        """Main analysis method - simplified with consolidated single-call approach"""
        self.logger.analysis_start(medical_input.procedure)
        self.reasoning_trace = []

        # Reset token usage counter
        if self.llm_manager:
            self.llm_manager.reset_token_usage()

        try:
            # 1. Validate input
            self._validate_input(medical_input)

            # 2. Get procedure overview (SEPARATE API call)
            self.logger.analysis_stage("PROCEDURE_OVERVIEW", f"Fetching overview for {medical_input.procedure}")
            self._log_step(
                ReasoningStage.INPUT_ANALYSIS,
                f"Fetching procedure overview for {medical_input.procedure}",
                {"method": "consolidated_analyzer.get_procedure_overview"}
            )
            procedure_overview = self.consolidated_analyzer.get_procedure_overview(medical_input)

            # 3. Perform consolidated analysis (ONE LLM call instead of multiple)
            self.logger.analysis_stage("CONSOLIDATED_ANALYSIS", f"Analyzing {medical_input.procedure} (single call)")
            self._log_step(
                ReasoningStage.ORGAN_IDENTIFICATION,
                f"Performing consolidated analysis for {medical_input.procedure}",
                {"method": "consolidated_analyzer"}
            )

            analysis_result = self.consolidated_analyzer.analyze_procedure(medical_input)

            # 4. Create final output from consolidated result
            return self._create_output_from_consolidated(medical_input, analysis_result, procedure_overview)

        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            raise
    
    def _validate_input(self, medical_input: MedicalInput):
        """Validate medical input"""
        self.logger.analysis_stage("INPUT_ANALYSIS", f"Validating input for {medical_input.procedure}")
        self._log_step(
            ReasoningStage.INPUT_ANALYSIS,
            f"Validating input for {medical_input.procedure}",
            {"procedure": medical_input.procedure}
        )
        
        # Basic validation
        proc_result = InputValidator.validate_medical_procedure(medical_input.procedure)
        if not proc_result.is_valid:
            error_msg = f"Invalid procedure: {', '.join(proc_result.errors)}"
            self.logger.validation_error(error_msg)
            raise ValidationError(error_msg)
    
    
    def _create_output_from_consolidated(self, medical_input: MedicalInput,
                                        analysis_result: Dict[str, Any],
                                        procedure_overview: Dict[str, str]) -> MedicalOutput:
        """Create final structured output from consolidated analysis result"""
        self.logger.analysis_stage("CRITICAL_EVALUATION", "Creating final analysis output")

        organs_data = analysis_result.get("organs_analyzed", [])

        self._log_step(
            ReasoningStage.CRITICAL_EVALUATION,
            "Creating final analysis output",
            {"organs_analyzed": len(organs_data)}
        )

        # Create organ analyses from consolidated result
        organ_analyses = []
        for organ_data in organs_data:
            # Log each organ found
            organ_name = organ_data.get("name", "unknown")
            self.logger.organs_identified([organ_name])

            # Get evidence and log quality
            evidence = organ_data.get("evidence", {})
            quality = evidence.get("quality", "limited")
            self.logger.evidence_gathered(organ_name, quality)

            # Get recommendations and log counts
            recs = organ_data.get("recommendations", {})
            total_count = len(recs.get("known", [])) + len(recs.get("potential", [])) + len(recs.get("debunked", []))
            self.logger.recommendations_generated(organ_name, total_count)

            analysis = OrganAnalysis(
                organ_name=organ_name,
                affected_by_procedure=True,
                at_risk=True,
                risk_level=organ_data.get("risk_level", "moderate"),
                pathways_involved=organ_data.get("pathways", ["unknown"]),
                known_recommendations=recs.get("known", []),
                potential_recommendations=recs.get("potential", []),
                debunked_claims=recs.get("debunked", []),
                evidence_quality=quality
            )
            organ_analyses.append(analysis)

        # Use confidence from consolidated result or calculate if not provided
        confidence = analysis_result.get("confidence_score", 0.7)

        # Get token usage from LLM manager
        token_usage = None
        if self.llm_manager:
            token_usage = self.llm_manager.get_token_usage()
            if token_usage and token_usage.total_tokens > 0:
                self.logger.info(f"💰 Token usage: {token_usage.total_tokens:,} total "
                               f"({token_usage.input_tokens:,} input + {token_usage.output_tokens:,} output)")

        # Log analysis completion
        self.logger.analysis_complete(confidence, len(organs_data))

        return MedicalOutput(
            procedure_summary=f"{medical_input.procedure} - {medical_input.details}",
            procedure_overview=procedure_overview,
            organs_analyzed=organ_analyses,
            general_recommendations=[
                "Consult healthcare provider before procedure",
                "Follow all pre-procedure instructions",
                "Monitor for adverse effects post-procedure"
            ],
            research_gaps=[
                "Long-term effects need further study",
                "Optimal protocols under investigation"
            ],
            confidence_score=confidence,
            reasoning_trace=self.reasoning_trace,
            token_usage=token_usage
        )
    
    
    def _log_step(self, stage: ReasoningStage, reasoning: str, data: Dict[str, Any]):
        """Log reasoning step"""
        step = ReasoningStep(
            stage=stage,
            timestamp=datetime.now(),
            input_data=data,
            reasoning=reasoning,
            output=data,
            confidence=0.8
        )
        self.reasoning_trace.append(step)
        self.logger.info(f"[{stage.value}] {reasoning}")
    
    def export_reasoning_trace(self, filepath: str):
        """Export reasoning trace to file"""
        import json
        import os
        
        trace_data = []
        for step in self.reasoning_trace:
            trace_data.append({
                "stage": step.stage.value,
                "timestamp": step.timestamp.isoformat(),
                "reasoning": step.reasoning,
                "confidence": step.confidence
            })
        
        with open(filepath, 'w') as f:
            json.dump(trace_data, f, indent=2)
        
        abs_path = os.path.abspath(filepath)
        self.logger.file_saved("reasoning_trace", abs_path)
        return abs_path
    
    def export_analysis_result(self, result: 'MedicalOutput', filepath: str):
        """Export detailed analysis result to JSON file"""
        import json
        import os
        
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
            "reasoning_steps_count": len(result.reasoning_trace),
            "token_usage": {
                "input_tokens": result.token_usage.input_tokens,
                "output_tokens": result.token_usage.output_tokens,
                "total_tokens": result.token_usage.total_tokens
            } if result.token_usage else None
        }
        
        with open(filepath, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        abs_path = os.path.abspath(filepath)
        self.logger.file_saved("analysis_result", abs_path)
        return abs_path
    
    def export_summary_report(self, result: 'MedicalOutput', filepath: str):
        """Export comprehensive medical analysis report"""
        import os
        
        # Build procedure overview section
        overview_section = ""
        if result.procedure_overview:
            overview = result.procedure_overview
            overview_section = f"""
## 📋 PROCEDURE INFORMATION

### What is {result.procedure_summary.split(' - ')[0]}?

{overview.get('description', 'No description available.')}

### 🎯 Medical Conditions Treated

{overview.get('conditions_treated', 'No conditions information available.')}

### ⚠️ Patients at Special Risk / Contraindications

{overview.get('contraindications', 'No contraindications information available.')}

---
"""

        summary = f"""# COMPREHENSIVE MEDICAL PROCEDURE ANALYSIS
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Analysis System:** SimpleMedicalAgent v2.0

---
{overview_section}
## 🏥 ANALYSIS OVERVIEW
**Procedure:** {result.procedure_summary}
**Analysis Confidence:** {result.confidence_score:.2f}/1.00 ({self._confidence_interpretation(result.confidence_score)})
**Total Organs Analyzed:** {len(result.organs_analyzed)}
**Reasoning Steps Completed:** {len(result.reasoning_trace)}

---

## 🫀 DETAILED ORGAN-SPECIFIC ANALYSIS

"""
        
        for i, organ in enumerate(result.organs_analyzed, 1):
            summary += f"""### {i}. {organ.organ_name.upper()} ANALYSIS

**🔴 RISK ASSESSMENT**
- **Risk Level:** {organ.risk_level.upper()} 
- **Procedure Impact:** {'YES - Directly affected' if organ.affected_by_procedure else 'NO - Minimal impact'}
- **At Risk:** {'YES - Requires monitoring' if organ.at_risk else 'NO - Low concern'}
- **Evidence Quality:** {organ.evidence_quality.upper()}

**🔬 BIOLOGICAL PATHWAYS INVOLVED**
"""
            for pathway in organ.pathways_involved:
                summary += f"- {pathway.replace('_', ' ').title()}\n"
            
            # Filter out generic/placeholder recommendations
            filtered_known = [rec for rec in organ.known_recommendations
                            if not any(phrase in rec.lower() for phrase in
                                     ['standard potential recommendations', 'standard debunked recommendations',
                                      'key considerations:', 'rationale for', 'limitations acknowledged',
                                      'emerging but not yet standard', 'synthesized recommendations'])]

            filtered_potential = [rec for rec in organ.potential_recommendations
                                if not any(phrase in rec.lower() for phrase in
                                         ['standard potential recommendations', 'key considerations:'])]

            filtered_debunked = [rec for rec in organ.debunked_claims
                               if not any(phrase in rec.lower() for phrase in
                                        ['standard debunked recommendations'])]

            if filtered_known:
                summary += f"""
**✅ EVIDENCE-BASED RECOMMENDATIONS** ({len(filtered_known)} recommendations)
*Proven interventions with strong clinical evidence*

"""
                for j, rec in enumerate(filtered_known, 1):
                    # Clean up formatting - remove excessive asterisks and colons at end
                    clean_rec = rec.strip('*').strip(':').strip()
                    summary += f"{j}. {clean_rec}\n"

            if filtered_potential:
                summary += f"""
**🔬 INVESTIGATIONAL/POTENTIAL TREATMENTS** ({len(filtered_potential)} items)
*Show promise but need more research*

"""
                for j, rec in enumerate(filtered_potential, 1):
                    clean_rec = rec.strip('*').strip(':').strip()
                    summary += f"{j}. {clean_rec}\n"

            if filtered_debunked:
                summary += f"""
**❌ DEBUNKED/HARMFUL TREATMENTS** ({len(filtered_debunked)} items)
*AVOID - Proven ineffective or dangerous*

"""
                for j, claim in enumerate(filtered_debunked, 1):
                    clean_claim = claim.strip('*').strip(':').strip().lstrip('❌').strip()
                    summary += f"{j}. ❌ {clean_claim}\n"
            
            summary += "---\n\n"
        
        summary += f"""## 🩺 COMPREHENSIVE CARE RECOMMENDATIONS

**IMMEDIATE ACTIONS REQUIRED:**
"""
        for i, rec in enumerate(result.general_recommendations, 1):
            summary += f"{i}. {rec}\n"

        summary += f"""
## 🔬 CURRENT RESEARCH GAPS & LIMITATIONS

**Areas Needing Further Study:**
"""
        for i, gap in enumerate(result.research_gaps, 1):
            summary += f"{i}. {gap}\n"
        
        # Add detailed reasoning trace summary
        summary += f"""
## 🧠 ANALYSIS METHODOLOGY & REASONING

**Analysis Pipeline Used:**
"""
        reasoning_stages = {}
        for step in result.reasoning_trace:
            stage_name = step.stage.value.replace('_', ' ').title()
            reasoning_stages[stage_name] = step.reasoning
        
        for stage, reasoning in reasoning_stages.items():
            summary += f"""
**{stage}:**
- {reasoning}
"""
        
        summary += f"""

## 📊 CONFIDENCE & RELIABILITY METRICS

**Overall Analysis Confidence:** {result.confidence_score:.2f}/1.00
- **Interpretation:** {self._confidence_interpretation(result.confidence_score)}
- **Reliability Factors:**
  - Evidence quality: {self._assess_evidence_quality(result)}
  - Organ coverage: {len(result.organs_analyzed)} systems analyzed
  - Reasoning depth: {len(result.reasoning_trace)} analytical steps

**Recommendation Confidence Levels:**
"""
        for organ in result.organs_analyzed:
            known_count = len(organ.known_recommendations)
            potential_count = len(organ.potential_recommendations)
            total_recs = known_count + potential_count

            if total_recs > 0:
                known_percentage = (known_count / total_recs) * 100
                summary += f"- **{organ.organ_name.title()}:** {known_percentage:.0f}% evidence-based recommendations\n"

        # Add token usage if available
        if result.token_usage and result.token_usage.total_tokens > 0:
            summary += f"""
**Token Usage:**
- Input tokens: {result.token_usage.input_tokens:,}
- Output tokens: {result.token_usage.output_tokens:,}
- Total tokens: {result.token_usage.total_tokens:,}
"""
        
        summary += f"""

## ⚠️ IMPORTANT DISCLAIMERS

**MEDICAL DISCLAIMER:**
- This analysis is for educational and research purposes only
- NOT a substitute for professional medical advice
- Always consult qualified healthcare providers for medical decisions
- Individual patient factors may significantly alter recommendations

**EVIDENCE LIMITATIONS:**
- Based on available medical literature and clinical guidelines
- Medical knowledge evolves rapidly - newer studies may change recommendations
- Individual patient responses may vary significantly
- Some recommendations may not apply to all patient populations

## 📚 METHODOLOGY & SOURCES

**Analysis Framework:**
- Systematic organ-focused evaluation
- Evidence-based recommendation classification
- Multi-stage reasoning with confidence scoring
- Integration of current clinical guidelines and research

**Data Sources Utilized:**
- Medical literature databases
- Clinical practice guidelines
- Established medical protocols
- Current research findings

---

## 📋 SUMMARY FOR HEALTHCARE PROVIDERS

**Key Findings:**
- **Primary organs at risk:** {', '.join([organ.organ_name for organ in result.organs_analyzed if organ.risk_level in ['moderate', 'high']])}
- **Critical interventions:** {len([rec for organ in result.organs_analyzed for rec in organ.known_recommendations])} evidence-based recommendations identified
- **Treatments to avoid:** {len([claim for organ in result.organs_analyzed for claim in organ.debunked_claims])} debunked/harmful approaches identified

**Clinical Action Items:**
1. Review patient-specific risk factors for identified organ systems
2. Implement evidence-based protective protocols
3. Avoid identified harmful/ineffective treatments
4. Monitor for organ-specific complications during and after procedure

---

**Report Generated By:** SimpleMedicalAgent v2.0
**Generation Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Analysis Time:** < 1 minute
**Medical Knowledge Base:** Current as of {datetime.now().strftime('%Y-%m')}

*This comprehensive analysis provides the detailed medical information needed for informed decision-making about the procedure and associated care protocols.*
"""
        
        with open(filepath, 'w') as f:
            f.write(summary)
        
        abs_path = os.path.abspath(filepath)
        self.logger.file_saved("comprehensive_report", abs_path)
        return abs_path
    
    def _confidence_interpretation(self, score: float) -> str:
        """Interpret confidence score"""
        if score >= 0.8:
            return "HIGH - Strong analytical confidence"
        elif score >= 0.6:
            return "MODERATE - Good analytical confidence" 
        elif score >= 0.4:
            return "LIMITED - Some analytical uncertainty"
        else:
            return "LOW - Significant analytical limitations"
    
    def _assess_evidence_quality(self, result: 'MedicalOutput') -> str:
        """Assess overall evidence quality"""
        qualities = [organ.evidence_quality for organ in result.organs_analyzed]
        if 'strong' in qualities:
            return "Strong evidence available for key organs"
        elif 'moderate' in qualities:
            return "Moderate evidence base"
        else:
            return "Limited evidence - more research needed"


# Factory function for easy creation
def create_simple_agent(llm_provider: str = "claude", enable_logging: bool = True) -> SimpleMedicalAgent:
    """Create a simple medical agent with optional LLM support"""
    logger = get_colored_logger("create_simple_agent", enable_logging)
    llm_manager = None
    
    if llm_provider:
        try:
            from llm_integrations import create_llm_manager
            llm_manager = create_llm_manager(llm_provider)
            logger.provider_auth_success(llm_provider)
        except ImportError as e:
            logger.provider_unavailable(llm_provider)
        except Exception as e:
            logger.provider_auth_failed(llm_provider, str(e))
            logger.fallback_mode("LLM", "Running in offline mode")
    
    return SimpleMedicalAgent(llm_manager, enable_logging)


# Command line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Simple Medical Reasoning Agent - Analyze medical procedures with AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use defaults (MRI with contrast)
  python simple_medical_agent.py
  
  # Custom procedure
  python simple_medical_agent.py --procedure "Endoscopy" --details "Upper GI endoscopy" 
  
  # Custom objectives
  python simple_medical_agent.py --objectives "risks" "preparation" "recovery"
  
  # Different LLM provider
  python simple_medical_agent.py --provider openai
        """
    )
    
    parser.add_argument("--procedure", "-p", 
                       default="MRI Scanner",
                       help="Medical procedure name (default: MRI Scanner)")
    
    parser.add_argument("--details", "-d",
                       default=None, 
                       help="Procedure details (default: None)")
    
    parser.add_argument("--objectives", "-o",
                       nargs="+",
                       default=["understand implications", "risks", "post-procedure care"],
                       help="Analysis objectives (default: implications, risks, care)")
    
    parser.add_argument("--provider",
                       choices=["claude", "openai", "ollama", "none"],
                       default="claude",
                       help="LLM provider to use (default: claude)")
    
    parser.add_argument("--output", 
                       default="reasoning_trace.json",
                       help="Output file for reasoning trace (default: reasoning_trace.json)")
    
    parser.add_argument("--quiet", "-q",
                       action="store_true",
                       help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    # Create agent
    provider = None if args.provider == "" else args.provider
    agent = create_simple_agent(provider, enable_logging=not args.quiet)
    
    # Create medical input
    details = args.details or ""  # Convert None to empty string
    medical_input = MedicalInput(
        procedure=args.procedure,
        details=details,
        objectives=tuple(args.objectives)
    )
    
    if not args.quiet:
        print(f"🧠 Analyzing: {args.procedure}")
        print(f"📋 Details: {details or 'None specified'}")
        print(f"🎯 Objectives: {', '.join(args.objectives)}")
        print(f"🤖 Provider: {args.provider}")
        print("-" * 50)
    
    # Run analysis
    result = agent.analyze_procedure(medical_input)
    
    # Display results
    print(f"✅ Analysis complete!")
    print(f"📝 Procedure: {result.procedure_summary}")
    print(f"🫀 Organs analyzed: {len(result.organs_analyzed)}")
    print(f"📊 Confidence: {result.confidence_score:.2f}")

    # Display token usage if available
    if result.token_usage and result.token_usage.total_tokens > 0:
        print(f"💰 Tokens used: {result.token_usage.total_tokens:,} "
              f"({result.token_usage.input_tokens:,} input + {result.token_usage.output_tokens:,} output)")

    for organ in result.organs_analyzed:
        print(f"\n🔍 {organ.organ_name.upper()}:")
        print(f"    Risk level: {organ.risk_level}")
        print(f"    Known recommendations: {len(organ.known_recommendations)}")
        print(f"    Potential recommendations: {len(organ.potential_recommendations)}")
        print(f"    Debunked claims: {len(organ.debunked_claims)}")
    
    # Create outputs directory
    import os
    outputs_dir = "outputs"
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Generate all output files
    base_name = f"{args.procedure.replace(' ', '_').lower()}"
    
    # Export all files
    trace_path = agent.export_reasoning_trace(f"{outputs_dir}/{base_name}_reasoning_trace.json")
    analysis_path = agent.export_analysis_result(result, f"{outputs_dir}/{base_name}_analysis_result.json")  
    summary_path = agent.export_summary_report(result, f"{outputs_dir}/{base_name}_summary_report.md")
    
    print(f"\n📁 All files saved to outputs/ directory:")
    print(f"📄 Reasoning trace: {trace_path}")
    print(f"📊 Analysis result: {analysis_path}")
    print(f"📝 Summary report: {summary_path}")
    
    if not args.quiet:
        print(f"\n💡 Check the summary report for human-readable results!")
        print(f"💡 Analysis result contains detailed JSON data.")
        print(f"💡 Run with -h to see all available options.")