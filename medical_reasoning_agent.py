#!/usr/bin/env python3
"""
Medical Reasoning Agent
Follows systematic analysis pattern for medical procedures with organ-focused reasoning.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from datetime import datetime

# Add caching for efficiency
from functools import lru_cache


class ReasoningStage(Enum):
    """Stages of medical reasoning pipeline"""
    INPUT_ANALYSIS = "input_analysis"
    ORGAN_IDENTIFICATION = "organ_identification" 
    EVIDENCE_GATHERING = "evidence_gathering"
    RISK_ASSESSMENT = "risk_assessment"
    RECOMMENDATION_SYNTHESIS = "recommendation_synthesis"
    CRITICAL_EVALUATION = "critical_evaluation"


@dataclass
class ReasoningStep:
    """Individual step in reasoning process"""
    stage: ReasoningStage
    timestamp: datetime
    input_data: Dict[str, Any]
    reasoning: str
    output: Dict[str, Any]
    confidence: float
    sources: List[str] = field(default_factory=list)


@dataclass(frozen=True)  # Make hashable for caching
class MedicalInput:
    """Structured medical procedure input"""
    procedure: str
    details: str
    objectives: tuple  # Use tuple instead of list for immutability
    patient_context: Optional[str] = None  # Simplified for hashing


@dataclass
class OrganAnalysis:
    """Analysis for a specific organ system"""
    organ_name: str
    affected_by_procedure: bool
    at_risk: bool
    risk_level: str  # low, moderate, high
    pathways_involved: List[str]
    known_recommendations: List[str]
    potential_recommendations: List[str]
    debunked_claims: List[str]
    evidence_quality: str  # strong, moderate, limited, poor


@dataclass
class MedicalOutput:
    """Final structured output"""
    procedure_summary: str
    organs_analyzed: List[OrganAnalysis]
    general_recommendations: List[str]
    research_gaps: List[str]
    confidence_score: float
    reasoning_trace: List[ReasoningStep]


class MedicalReasoningAgent:
    """
    DEPRECATED: Use SimpleMedicalAgent instead.
    This class is kept for backwards compatibility only.
    """
    
    def __init__(self, 
                 primary_llm_provider: str = "claude",
                 fallback_providers: List[str] = None,
                 enable_logging: bool = True):
        """DEPRECATED: Use create_simple_agent() from simple_medical_agent instead."""
        import warnings
        warnings.warn(
            "MedicalReasoningAgent is deprecated. Use SimpleMedicalAgent from simple_medical_agent.py instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Minimal initialization for backwards compatibility
        from simple_medical_agent import create_simple_agent
        self._simple_agent = create_simple_agent(primary_llm_provider, enable_logging)
        self.reasoning_trace = []
            
    def analyze_medical_procedure(self, medical_input: MedicalInput) -> MedicalOutput:
        """DEPRECATED: Use analyze_procedure() from SimpleMedicalAgent instead."""
        result = self._simple_agent.analyze_procedure(medical_input)
        self.reasoning_trace = self._simple_agent.reasoning_trace
        return result
    
    def export_reasoning_trace(self, filepath: str):
        """Export reasoning trace to JSON file for analysis."""
        return self._simple_agent.export_reasoning_trace(filepath)


if __name__ == "__main__":
    # Example usage - DEPRECATED, use simple_medical_agent.py instead
    print("⚠️  This file is deprecated. Use simple_medical_agent.py instead.")
    print("Example: python simple_medical_agent.py")
    
    # Still works for backwards compatibility
    agent = MedicalReasoningAgent(enable_logging=True)
    
    medical_input = MedicalInput(
        procedure="MRI Scanner",
        details="With contrast",
        objectives=("understand implications", "risks", "post-procedure care")
    )
    
    result = agent.analyze_medical_procedure(medical_input)
    
    print("Medical Analysis Complete:")
    print(f"Procedure: {result.procedure_summary}")
    print(f"Organs Analyzed: {len(result.organs_analyzed)}")
    print(f"Confidence Score: {result.confidence_score}")
    
    # Export reasoning trace
    agent.export_reasoning_trace("reasoning_trace.json")
    print(f"Reasoning trace exported to: reasoning_trace.json")