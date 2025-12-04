"""
Medical Fact Checker - Independent Bio-Investigator

An AI agent for critical analysis of health subjects, skeptical of
consensus driven by inertia or corporate interest.
"""

from .medical_fact_checker_agent import (
    MedicalFactChecker,
    AnalysisPhase,
    OutputType,
    PhaseResult,
    FactCheckSession
)

__version__ = "0.1.0"
__all__ = [
    "MedicalFactChecker",
    "AnalysisPhase",
    "OutputType",
    "PhaseResult",
    "FactCheckSession"
]
