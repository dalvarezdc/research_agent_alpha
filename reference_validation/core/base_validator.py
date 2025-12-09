"""
Base validator class that all specific validators inherit from.
Follows the repository's pattern of clear abstractions and type safety.
"""

import hashlib
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from datetime import datetime

from ..models import ValidationResult, SourceType


class BaseValidator(ABC):
    """
    Abstract base class for all validators.
    Provides common functionality and interface.
    """

    name: str = "base_validator"

    def __init__(self, timeout: int = 10, logger: Optional[logging.Logger] = None):
        """
        Initialize base validator.

        Args:
            timeout: Timeout in seconds for external API calls
            logger: Optional logger instance
        """
        self.timeout = timeout
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def validate(self, reference: str, **kwargs) -> ValidationResult:
        """
        Validate a reference.

        Args:
            reference: Reference text to validate
            **kwargs: Additional validator-specific parameters

        Returns:
            ValidationResult with validation details
        """
        pass

    @abstractmethod
    def can_validate(self, reference: str) -> bool:
        """
        Check if this validator can handle the given reference.

        Args:
            reference: Reference text to check

        Returns:
            True if validator can handle this reference
        """
        pass

    def generate_id(self, reference: str) -> str:
        """
        Generate a unique ID for a reference.

        Args:
            reference: Reference text

        Returns:
            MD5 hash of reference (for caching)
        """
        return hashlib.md5(reference.encode()).hexdigest()

    def _create_base_result(self, reference: str) -> ValidationResult:
        """
        Create a base ValidationResult with common fields populated.

        Args:
            reference: Reference text

        Returns:
            ValidationResult with defaults
        """
        return ValidationResult(
            citation=reference,
            reference_id=self.generate_id(reference),
            is_valid=False,  # Default to invalid
            credibility_score=0.0,
            confidence=0.0,
            validated_at=datetime.now(),
            validators_used=[self.name],
        )

    def _log_validation_start(self, reference: str) -> None:
        """Log start of validation"""
        self.logger.debug(f"[{self.name}] Starting validation: {reference[:100]}")

    def _log_validation_end(
        self, reference: str, result: ValidationResult, duration_ms: float
    ) -> None:
        """Log end of validation"""
        self.logger.debug(
            f"[{self.name}] Completed validation: "
            f"valid={result.is_valid}, "
            f"score={result.credibility_score:.1f}, "
            f"time={duration_ms:.1f}ms"
        )

    def _handle_error(
        self, reference: str, error: Exception
    ) -> ValidationResult:
        """
        Handle validation errors gracefully.

        Args:
            reference: Reference being validated
            error: Exception that occurred

        Returns:
            ValidationResult marked as invalid with error details
        """
        self.logger.error(f"[{self.name}] Error validating reference: {error}")

        result = self._create_base_result(reference)
        result.issues.append({
            "severity": "high",
            "message": f"Validation failed: {str(error)}",
            "field": "validation_error"
        })
        result.warnings.append(f"Could not complete validation: {str(error)}")

        return result
