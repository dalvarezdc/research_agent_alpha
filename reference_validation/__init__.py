"""
Reference Validation System — Path B (agent-level, currently dormant).

NOTE: This package is the agent-level validation pipeline (Path B). It is only
activated when agents are initialised with enable_reference_validation=True,
which is never done in production code. The active validation path is Path A:
  reference_validation.core.citation_url_correspondence_validator
  (imported directly by run_analysis.AgentOrchestrator._collect_validated_references)

See pending.md item 7 for the planned reconciliation of these two paths.
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
