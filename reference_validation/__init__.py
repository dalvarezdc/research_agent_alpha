"""
Enhanced Reference Validation System
Validates citations, sources, and URLs across all medical AI agents.
"""

from .models import (
    ValidationResult,
    ValidationReport,
    ValidationLevel,
    ValidationConfig,
    ValidationIssue,
    SourceType,
)
from .orchestrator import ReferenceValidator

__all__ = [
    "ReferenceValidator",
    "ValidationResult",
    "ValidationReport",
    "ValidationLevel",
    "ValidationConfig",
    "ValidationIssue",
    "SourceType",
]

__version__ = "1.0.0"
