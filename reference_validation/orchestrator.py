"""
Reference Validation Orchestrator - main interface for validation.
Coordinates all validators and provides simple API for agents.
"""

import logging
from typing import List, Optional, Any
from datetime import datetime

from .models import (
    ValidationResult,
    ValidationReport,
    ValidationLevel,
    ValidationConfig,
    ExtractedReference,
)
from .core import (
    CitationValidator,
    URLChecker,
    ReferenceExtractor,
    ScoringEngine,
)
from .validators import UnifiedReferenceValidator
from .cache import CacheManager


class ReferenceValidator:
    """
    Main interface for reference validation.
    Use this class in your agents to validate references.
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize reference validator.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or ValidationConfig()

        # Setup logging
        self.logger = self._setup_logging()

        # Initialize components
        self.citation_validator = CitationValidator(
            timeout=self.config.timeout_seconds,
            logger=self.logger
        )

        self.url_checker = URLChecker(
            timeout=self.config.timeout_seconds,
            logger=self.logger
        )

        self.unified_validator = UnifiedReferenceValidator(
            api_key=self.config.pubmed_api_key,
            email=self.config.pubmed_email,
            timeout=self.config.timeout_seconds,
            logger=self.logger
        )

        self.reference_extractor = ReferenceExtractor()
        self.scoring_engine = ScoringEngine()

        # Initialize cache
        self.cache = CacheManager(
            backend=self.config.cache_backend,
            cache_path=self.config.cache_path,
            ttl_days=self.config.cache_ttl_days
        )

        self.logger.info(f"ReferenceValidator initialized with {self.config.cache_backend} cache")

    def validate_reference(
        self,
        citation: str,
        expected_claim: Optional[str] = None,
        validation_level: Optional[ValidationLevel] = None
    ) -> ValidationResult:
        """
        Validate a single reference.

        Args:
            citation: Citation text to validate
            expected_claim: Optional claim that reference should support
            validation_level: Level of validation (QUICK/STANDARD/THOROUGH)

        Returns:
            ValidationResult with validation details

        Example:
            >>> validator = ReferenceValidator()
            >>> result = validator.validate_reference(
            ...     "Smith et al. (2020). Nature. DOI: 10.1234/example"
            ... )
            >>> print(f"Valid: {result.is_valid}, Score: {result.credibility_score}")
        """
        level = validation_level or self.config.validation_level

        # Check cache
        cache_key = self._make_cache_key(citation)
        cached_result = self.cache.get(cache_key)

        if cached_result:
            cached_result.cache_hit = True
            self.logger.debug(f"Cache hit for: {citation[:50]}")
            return cached_result

        # Perform validation based on level
        if level == ValidationLevel.QUICK:
            result = self._validate_quick(citation)
        elif level == ValidationLevel.STANDARD:
            result = self._validate_standard(citation)
        else:  # THOROUGH
            result = self._validate_thorough(citation)

        # Cache result
        self.cache.set(cache_key, result)

        return result

    def validate_batch(
        self,
        citations: List[str],
        level: Optional[ValidationLevel] = None,
        parallel: bool = True
    ) -> ValidationReport:
        """
        Validate multiple references at once.

        Args:
            citations: List of citations to validate
            level: Validation level
            parallel: Whether to validate in parallel (not implemented yet)

        Returns:
            ValidationReport with aggregated results

        Example:
            >>> validator = ReferenceValidator()
            >>> citations = [
            ...     "Paper 1. PMID: 12345678",
            ...     "Paper 2. DOI: 10.1234/example"
            ... ]
            >>> report = validator.validate_batch(citations)
            >>> print(f"Overall score: {report.overall_score}")
        """
        validation_level = level or self.config.validation_level

        self.logger.info(f"Validating {len(citations)} references at {validation_level.value} level")

        results = []
        for citation in citations:
            result = self.validate_reference(
                citation,
                validation_level=validation_level
            )
            results.append(result)

        # Generate report
        report = self.scoring_engine.generate_report(results, validation_level)

        self.logger.info(
            f"Validation complete: {report.valid_references}/{report.total_references} valid, "
            f"score: {report.overall_score:.1f}"
        )

        return report

    def validate_analysis(
        self,
        analysis_result: Any,
        level: Optional[ValidationLevel] = None
    ) -> ValidationReport:
        """
        Validate all references in an analysis result.
        Automatically extracts references from the result.

        Args:
            analysis_result: Analysis result object (e.g., MedicalOutput)
            level: Validation level

        Returns:
            ValidationReport

        Example:
            >>> agent = MedicalReasoningAgent()
            >>> result = agent.analyze_medical_procedure(input_data)
            >>> validator = ReferenceValidator()
            >>> validation_report = validator.validate_analysis(result)
        """
        # Extract references from result
        # This is a generic method - specific implementations may override

        if hasattr(analysis_result, 'reasoning_trace'):
            text = str(analysis_result.reasoning_trace)
        elif hasattr(analysis_result, 'text'):
            text = analysis_result.text
        else:
            text = str(analysis_result)

        extracted_refs = self.reference_extractor.extract_from_text(text)
        citations = [ref.raw_text for ref in extracted_refs]

        return self.validate_batch(citations, level=level)

    def extract_references(self, text: str) -> List[ExtractedReference]:
        """
        Extract references from text without validating.

        Args:
            text: Text containing references

        Returns:
            List of ExtractedReference objects
        """
        return self.reference_extractor.extract_from_text(text)

    def _validate_quick(self, citation: str) -> ValidationResult:
        """Quick validation - format check only"""
        start_time = datetime.now()

        # Just check citation format
        result = self.citation_validator.validate(citation)

        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        result.validation_time_ms = duration_ms

        return result

    def _validate_standard(self, citation: str) -> ValidationResult:
        """Standard validation - format + identifier verification"""
        start_time = datetime.now()

        # Start with format check
        result = self.citation_validator.validate(citation)

        # If we have identifiers, verify them
        if result.doi or result.pmid or result.arxiv_id or result.url:
            # Use unified validator to verify existence
            unified_result = self.unified_validator.validate(citation)

            # Merge results
            result = self._merge_results(result, unified_result)

        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        result.validation_time_ms = duration_ms

        return result

    def _validate_thorough(self, citation: str) -> ValidationResult:
        """Thorough validation - all checks including URL accessibility"""
        start_time = datetime.now()

        # Format check
        result = self.citation_validator.validate(citation)

        # Identifier verification
        if result.doi or result.pmid or result.arxiv_id or result.url:
            unified_result = self.unified_validator.validate(citation)
            result = self._merge_results(result, unified_result)

        # URL accessibility check
        if result.url:
            url_result = self.url_checker.validate(citation)

            # Merge URL results
            if url_result.url_accessible is not None:
                result.url_accessible = url_result.url_accessible

                # Adjust score based on URL accessibility
                if url_result.url_accessible:
                    result.credibility_score += 5
                else:
                    result.credibility_score -= 10
                    result.warnings.append("URL is not accessible")

        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        result.validation_time_ms = duration_ms

        return result

    def _merge_results(
        self,
        base_result: ValidationResult,
        additional_result: ValidationResult
    ) -> ValidationResult:
        """Merge two validation results, taking the best information from each"""

        # Take higher credibility score
        if additional_result.credibility_score > base_result.credibility_score:
            base_result.credibility_score = additional_result.credibility_score

        # Merge validation flags
        if additional_result.pubmed_verified:
            base_result.pubmed_verified = True

        if additional_result.doi_valid:
            base_result.doi_valid = True

        if additional_result.url_accessible is not None:
            base_result.url_accessible = additional_result.url_accessible

        # Merge metadata
        if additional_result.publication_year:
            base_result.publication_year = additional_result.publication_year

        if additional_result.journal_name:
            base_result.journal_name = additional_result.journal_name

        if additional_result.authors:
            base_result.authors = additional_result.authors

        if additional_result.peer_reviewed:
            base_result.peer_reviewed = True

        # Update validity
        base_result.is_valid = (
            base_result.is_valid or
            additional_result.pubmed_verified or
            additional_result.doi_valid or
            (additional_result.url_accessible is True)
        )

        # Merge validators used
        base_result.validators_used.extend(additional_result.validators_used)
        base_result.validators_used = list(set(base_result.validators_used))

        # Merge issues and warnings
        base_result.issues.extend(additional_result.issues)
        base_result.warnings.extend(additional_result.warnings)

        # Take higher confidence
        if additional_result.confidence > base_result.confidence:
            base_result.confidence = additional_result.confidence

        return base_result

    def _make_cache_key(self, citation: str) -> str:
        """Generate cache key from citation"""
        import hashlib
        return hashlib.md5(citation.encode()).hexdigest()

    def _setup_logging(self) -> logging.Logger:
        """Setup logger"""
        logger = logging.getLogger("ReferenceValidator")

        if self.config.enable_logging:
            logger.setLevel(getattr(logging, self.config.log_level))

            if not logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                handler.setFormatter(formatter)
                logger.addHandler(handler)
        else:
            logger.addHandler(logging.NullHandler())

        return logger

    def clear_cache(self) -> None:
        """Clear validation cache"""
        self.cache.clear()
        self.logger.info("Cache cleared")

    def get_stats(self) -> dict:
        """Get validation statistics"""
        return {
            'cache_backend': self.config.cache_backend,
            'cache_size': self.cache.size(),
            'validation_level': self.config.validation_level.value,
        }
