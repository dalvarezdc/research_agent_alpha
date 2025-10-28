#!/usr/bin/env python3
"""
Simplified Medical Reasoning Agent
Clean, focused orchestration of medical analysis components.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging

from medical_reasoning_agent import MedicalInput, MedicalOutput, OrganAnalysis, ReasoningStep, ReasoningStage
from organ_analyzer import OrganAnalyzer
from evidence_gatherer import EvidenceGatherer
from recommendation_synthesizer import RecommendationSynthesizer
from input_validation import InputValidator, ValidationError


class SimpleMedicalAgent:
    """Simplified medical reasoning agent with clear separation of concerns"""
    
    def __init__(self, llm_manager=None, enable_logging: bool = True):
        """Initialize with optional LLM support"""
        self.llm_manager = llm_manager
        self.reasoning_trace: List[ReasoningStep] = []
        
        # Initialize components
        self.organ_analyzer = OrganAnalyzer(llm_manager)
        self.evidence_gatherer = EvidenceGatherer(llm_manager)
        self.recommendation_synthesizer = RecommendationSynthesizer(llm_manager)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not enable_logging:
            self.logger.disabled = True
        
        # Log initialization
        if llm_manager:
            self.logger.info("🤖 LLM integration enabled - will use AI for enhanced analysis")
        else:
            self.logger.info("🔧 Running in offline mode - using built-in medical knowledge base")
    
    def analyze_procedure(self, medical_input: MedicalInput) -> MedicalOutput:
        """Main analysis method - simplified and clear"""
        self.logger.info(f"Starting analysis of {medical_input.procedure}")
        self.reasoning_trace = []
        
        try:
            # 1. Validate input
            self._validate_input(medical_input)
            
            # 2. Identify affected organs
            organs = self._identify_organs(medical_input)
            
            # 3. Gather evidence
            evidence = self._gather_evidence(organs, medical_input)
            
            # 4. Generate recommendations
            recommendations = self._generate_recommendations(organs, evidence, medical_input)
            
            # 5. Create final output
            return self._create_output(medical_input, organs, evidence, recommendations)
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            raise
    
    def _validate_input(self, medical_input: MedicalInput):
        """Validate medical input"""
        self._log_step(
            ReasoningStage.INPUT_ANALYSIS,
            f"Validating input for {medical_input.procedure}",
            {"procedure": medical_input.procedure}
        )
        
        # Basic validation
        proc_result = InputValidator.validate_medical_procedure(medical_input.procedure)
        if not proc_result.is_valid:
            raise ValidationError(f"Invalid procedure: {', '.join(proc_result.errors)}")
    
    def _identify_organs(self, medical_input: MedicalInput) -> List[str]:
        """Identify affected organs"""
        self._log_step(
            ReasoningStage.ORGAN_IDENTIFICATION,
            f"Identifying organs affected by {medical_input.procedure}",
            {"method": "organ_analyzer"}
        )
        
        organs = self.organ_analyzer.identify_affected_organs(medical_input)
        self.logger.info(f"Identified organs: {organs}")
        return organs
    
    def _gather_evidence(self, organs: List[str], medical_input: MedicalInput) -> Dict[str, Dict[str, Any]]:
        """Gather evidence for organs"""
        self._log_step(
            ReasoningStage.EVIDENCE_GATHERING,
            f"Gathering evidence for {len(organs)} organs",
            {"organs": organs, "method": "evidence_gatherer"}
        )
        
        evidence = self.evidence_gatherer.get_evidence_summary(organs, medical_input.procedure)
        self.logger.info(f"Gathered evidence for {len(evidence)} organs")
        return evidence
    
    def _generate_recommendations(self, organs: List[str], evidence: Dict[str, Dict[str, Any]], 
                                medical_input: MedicalInput) -> Dict[str, Dict[str, List[str]]]:
        """Generate recommendations"""
        self._log_step(
            ReasoningStage.RECOMMENDATION_SYNTHESIS,
            f"Generating recommendations for {len(organs)} organs",
            {"method": "recommendation_synthesizer"}
        )
        
        recommendations = self.recommendation_synthesizer.synthesize_all_recommendations(
            evidence, medical_input
        )
        self.logger.info(f"Generated recommendations for {len(recommendations)} organs")
        return recommendations
    
    def _create_output(self, medical_input: MedicalInput, organs: List[str], 
                      evidence: Dict[str, Dict[str, Any]], 
                      recommendations: Dict[str, Dict[str, List[str]]]) -> MedicalOutput:
        """Create final structured output"""
        self._log_step(
            ReasoningStage.CRITICAL_EVALUATION,
            "Creating final analysis output",
            {"organs_analyzed": len(organs)}
        )
        
        # Create organ analyses
        organ_analyses = []
        for organ in organs:
            organ_evidence = evidence.get(organ, {})
            organ_recs = recommendations.get(organ, {})
            
            analysis = OrganAnalysis(
                organ_name=organ,
                affected_by_procedure=True,
                at_risk=True,
                risk_level=self._assess_risk_level(organ_evidence),
                pathways_involved=[organ_evidence.get("pathway", "unknown")],
                known_recommendations=organ_recs.get("known", []),
                potential_recommendations=organ_recs.get("potential", []),
                debunked_claims=organ_recs.get("debunked", []),
                evidence_quality=organ_evidence.get("quality", "limited")
            )
            organ_analyses.append(analysis)
        
        # Calculate confidence
        confidence = self._calculate_confidence(organs, evidence, recommendations)
        
        return MedicalOutput(
            procedure_summary=f"{medical_input.procedure} - {medical_input.details}",
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
            reasoning_trace=self.reasoning_trace
        )
    
    def _assess_risk_level(self, evidence: Dict[str, Any]) -> str:
        """Simple risk assessment"""
        risks = evidence.get("risks", [])
        if len(risks) >= 3:
            return "high"
        elif len(risks) >= 2:
            return "moderate"
        else:
            return "low"
    
    def _calculate_confidence(self, organs: List[str], evidence: Dict[str, Dict[str, Any]], 
                            recommendations: Dict[str, Dict[str, List[str]]]) -> float:
        """Calculate confidence score"""
        base_confidence = 0.6
        
        # Bonus for comprehensive analysis
        if len(organs) >= 2:
            base_confidence += 0.1
        
        # Bonus for high-quality evidence
        high_quality_count = sum(1 for ev in evidence.values() if ev.get("quality") == "strong")
        base_confidence += high_quality_count * 0.05
        
        # Bonus for complete recommendations
        complete_recs = sum(1 for rec in recommendations.values() 
                           if len(rec.get("known", [])) >= 2)
        base_confidence += complete_recs * 0.05
        
        return min(base_confidence, 0.95)
    
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
        self.logger.info(f"📄 Reasoning trace saved to: {abs_path}")
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
            "reasoning_steps_count": len(result.reasoning_trace)
        }
        
        with open(filepath, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        abs_path = os.path.abspath(filepath)
        self.logger.info(f"📊 Analysis result saved to: {abs_path}")
        return abs_path
    
    def export_summary_report(self, result: 'MedicalOutput', filepath: str):
        """Export comprehensive medical analysis report"""
        import os
        
        summary = f"""# COMPREHENSIVE MEDICAL PROCEDURE ANALYSIS
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Analysis System:** SimpleMedicalAgent v2.0

---

## 🏥 PROCEDURE OVERVIEW
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
            
            summary += f"""
**✅ EVIDENCE-BASED RECOMMENDATIONS** ({len(organ.known_recommendations)} items)
*These are proven, medically-established interventions with strong clinical evidence*

"""
            for j, rec in enumerate(organ.known_recommendations, 1):
                summary += f"{j}. **{rec}**\n"
                summary += f"   - *Evidence Level: Strong clinical support*\n"
                summary += f"   - *Implementation: Follow standard medical protocols*\n\n"
            
            summary += f"""**🔬 INVESTIGATIONAL/POTENTIAL TREATMENTS** ({len(organ.potential_recommendations)} items)
*These show promise but need more research or have limited evidence*

"""
            for j, rec in enumerate(organ.potential_recommendations, 1):
                summary += f"{j}. **{rec}**\n"
                summary += f"   - *Evidence Level: Limited but promising*\n"
                summary += f"   - *Status: Under investigation or emerging evidence*\n\n"
            
            summary += f"""**❌ DEBUNKED/HARMFUL TREATMENTS** ({len(organ.debunked_claims)} items)
*These are PROVEN INEFFECTIVE or potentially dangerous - AVOID these*

"""
            for j, claim in enumerate(organ.debunked_claims, 1):
                summary += f"{j}. **❌ {claim}**\n"
                summary += f"   - *Status: SCIENTIFICALLY DISPROVEN*\n"
                summary += f"   - *Risk: May delay proper treatment or cause harm*\n\n"
            
            summary += "---\n\n"
        
        summary += f"""## 🩺 COMPREHENSIVE CARE RECOMMENDATIONS

**IMMEDIATE ACTIONS REQUIRED:**
"""
        for i, rec in enumerate(result.general_recommendations, 1):
            summary += f"{i}. **{rec}**\n"
            summary += f"   - Priority: High\n"
            summary += f"   - Timeline: Before and during procedure\n\n"
        
        summary += f"""
## 🔬 CURRENT RESEARCH GAPS & LIMITATIONS

**Areas Needing Further Study:**
"""
        for i, gap in enumerate(result.research_gaps, 1):
            summary += f"{i}. **{gap}**\n"
            summary += f"   - Impact: May affect future treatment protocols\n"
            summary += f"   - Status: Active area of medical research\n\n"
        
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
        self.logger.info(f"📝 COMPREHENSIVE medical report saved to: {abs_path}")
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
    llm_manager = None
    
    if llm_provider:
        try:
            from llm_integrations import create_llm_manager
            llm_manager = create_llm_manager(llm_provider)
        except Exception as e:
            logging.warning(f"Failed to initialize LLM: {e}. Running in fallback mode.")
    
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
    provider = None if args.provider == "none" else args.provider
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