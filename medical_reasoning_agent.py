#!/usr/bin/env python3
"""
Medical Reasoning Data Classes
Contains the core data structures used by the simplified medical reasoning system.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


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


@dataclass(frozen=True)
class MedicalInput:
    """Structured medical procedure input"""
    procedure: str
    details: str
    objectives: tuple  # Use tuple for immutability
    patient_context: Optional[str] = None


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


if __name__ == "__main__":
    print("This module contains data classes only.")
    print("Use simple_medical_agent.py for medical analysis.")
    print("Run: python simple_medical_agent.py -h")