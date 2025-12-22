"""
Medical Procedure Analyzer Package

A comprehensive toolkit for analyzing medical procedures with AI-powered reasoning.
"""

from .medical_reasoning_agent import (
    MedicalInput,
    MedicalOutput,
    OrganAnalysis,
    ReasoningStep,
    ReasoningStage,
    MedicalReasoningAgent,
)
from .input_validation import InputValidator, ValidationError
from .validation_scoring import validate_medical_output
from .colored_logger import get_colored_logger

# Import LLMManager from root-level llm_integrations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from llm_integrations import LLMManager

__all__ = [
    "MedicalInput",
    "MedicalOutput",
    "OrganAnalysis",
    "ReasoningStep",
    "ReasoningStage",
    "MedicalReasoningAgent",
    "InputValidator",
    "ValidationError",
    "validate_medical_output",
    "LLMManager",
    "get_colored_logger",
]

__version__ = "2.0.0"
