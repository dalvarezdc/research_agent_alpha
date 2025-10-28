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
        
        self.logger.info(f"Reasoning trace exported to {filepath}")


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
                       default="With contrast", 
                       help="Procedure details (default: With contrast)")
    
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
    medical_input = MedicalInput(
        procedure=args.procedure,
        details=args.details,
        objectives=tuple(args.objectives)
    )
    
    if not args.quiet:
        print(f"🧠 Analyzing: {args.procedure}")
        print(f"📋 Details: {args.details}")
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
    
    # Export reasoning trace
    agent.export_reasoning_trace(args.output)
    print(f"\n📄 Reasoning trace exported to: {args.output}")
    
    if not args.quiet:
        print(f"\n💡 To see detailed recommendations, check the trace file!")
        print(f"💡 Run with -h to see all available options.")