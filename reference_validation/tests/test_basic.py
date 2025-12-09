"""
Basic tests for reference validation system.
Run with: python -m pytest reference_validation/tests/test_basic.py -v
"""

import pytest
from reference_validation import (
    ReferenceValidator,
    ValidationConfig,
    ValidationLevel,
)


class TestBasicValidation:
    """Basic validation tests"""

    def test_validator_initialization(self):
        """Test validator can be initialized"""
        validator = ReferenceValidator()
        assert validator is not None
        assert validator.config is not None

    def test_validator_with_custom_config(self):
        """Test validator with custom configuration"""
        config = ValidationConfig(
            cache_backend="memory",
            validation_level=ValidationLevel.QUICK,
            min_credibility_score=70
        )

        validator = ReferenceValidator(config)
        assert validator.config.cache_backend == "memory"
        assert validator.config.validation_level == ValidationLevel.QUICK

    def test_validate_reference_with_doi(self):
        """Test validation of reference with DOI"""
        validator = ReferenceValidator(ValidationConfig(cache_backend="memory"))

        citation = "Smith J et al. (2020). Important Research. Nature. DOI: 10.1038/s41586-020-0001-1"

        result = validator.validate_reference(citation, validation_level=ValidationLevel.QUICK)

        assert result is not None
        assert result.citation == citation
        assert result.doi is not None
        assert result.credibility_score > 0

    def test_validate_reference_with_pmid(self):
        """Test validation of reference with PMID"""
        validator = ReferenceValidator(ValidationConfig(cache_backend="memory"))

        citation = "Important medical research. PMID: 12345678"

        result = validator.validate_reference(citation, validation_level=ValidationLevel.QUICK)

        assert result is not None
        assert result.pmid == "12345678"

    def test_validate_reference_with_url(self):
        """Test validation of reference with URL"""
        validator = ReferenceValidator(ValidationConfig(cache_backend="memory"))

        citation = "Web resource. https://www.example.com/research"

        result = validator.validate_reference(citation, validation_level=ValidationLevel.QUICK)

        assert result is not None
        assert result.url is not None

    def test_validate_batch(self):
        """Test batch validation"""
        validator = ReferenceValidator(ValidationConfig(cache_backend="memory"))

        citations = [
            "Paper 1. DOI: 10.1234/example1",
            "Paper 2. PMID: 12345678",
            "Paper 3. https://example.com/paper3"
        ]

        report = validator.validate_batch(citations, level=ValidationLevel.QUICK)

        assert report is not None
        assert report.total_references == 3
        assert len(report.results) == 3
        assert report.overall_score >= 0
        assert report.overall_score <= 100

    def test_extract_references(self):
        """Test reference extraction"""
        validator = ReferenceValidator()

        text = """
        This study found important results [1].
        Another finding was reported [2].

        ## References
        [1] Smith J. (2020). Important paper. Nature. DOI: 10.1234/example
        [2] Jones A. (2021). Another paper. Science. PMID: 12345678
        """

        refs = validator.extract_references(text)

        assert len(refs) > 0

    def test_cache_functionality(self):
        """Test that caching works"""
        validator = ReferenceValidator(ValidationConfig(cache_backend="memory"))

        citation = "Test paper. DOI: 10.1234/cache-test"

        # First validation (not cached)
        result1 = validator.validate_reference(citation, validation_level=ValidationLevel.QUICK)
        assert not result1.cache_hit

        # Second validation (should be cached)
        result2 = validator.validate_reference(citation, validation_level=ValidationLevel.QUICK)
        assert result2.cache_hit

    def test_validation_levels(self):
        """Test different validation levels"""
        validator = ReferenceValidator(ValidationConfig(cache_backend="memory"))

        citation = "Test paper. DOI: 10.1234/test"

        # Quick validation
        result_quick = validator.validate_reference(citation, validation_level=ValidationLevel.QUICK)
        time_quick = result_quick.validation_time_ms

        # Standard validation
        validator.clear_cache()  # Clear cache to avoid cache hit
        result_standard = validator.validate_reference(citation, validation_level=ValidationLevel.STANDARD)

        assert result_quick is not None
        assert result_standard is not None

    def test_validation_report_statistics(self):
        """Test validation report statistics"""
        validator = ReferenceValidator(ValidationConfig(cache_backend="memory"))

        citations = [
            "Good paper with DOI. DOI: 10.1234/good",
            "Paper with PMID. PMID: 12345678",
            "Poor citation without identifiers",
        ]

        report = validator.validate_batch(citations, level=ValidationLevel.QUICK)

        assert report.total_references == 3
        assert report.average_credibility >= 0
        assert report.pass_rate >= 0
        assert report.pass_rate <= 100

    def test_clear_cache(self):
        """Test cache clearing"""
        validator = ReferenceValidator(ValidationConfig(cache_backend="memory"))

        citation = "Test. DOI: 10.1234/clear"

        # Validate and cache
        validator.validate_reference(citation)

        # Clear cache
        validator.clear_cache()

        # Should not be cached now
        result = validator.validate_reference(citation)
        assert not result.cache_hit


class TestCitationExtraction:
    """Test citation extraction and parsing"""

    def test_extract_doi(self):
        """Test DOI extraction"""
        validator = ReferenceValidator()

        citations = [
            "Paper. DOI: 10.1234/example",
            "Paper. doi:10.5678/test",
            "Paper. https://doi.org/10.9012/something"
        ]

        for citation in citations:
            result = validator.validate_reference(citation, validation_level=ValidationLevel.QUICK)
            assert result.doi is not None

    def test_extract_pmid(self):
        """Test PMID extraction"""
        validator = ReferenceValidator()

        citations = [
            "Paper. PMID: 12345678",
            "Paper. pmid:87654321",
        ]

        for citation in citations:
            result = validator.validate_reference(citation, validation_level=ValidationLevel.QUICK)
            assert result.pmid is not None

    def test_extract_year(self):
        """Test year extraction"""
        from reference_validation.core import CitationValidator

        cv = CitationValidator()

        citations = [
            "Smith (2020). Paper title.",
            "Jones et al. (2019). Another paper.",
            "Brown (2021). Yet another.",
        ]

        for citation in citations:
            year = cv.extract_year(citation)
            assert year is not None
            assert 2019 <= year <= 2021


if __name__ == "__main__":
    # Run a quick test
    print("Running basic validation test...")

    validator = ReferenceValidator(ValidationConfig(cache_backend="memory"))

    test_citations = [
        "Smith J, Doe A. (2020). COVID-19 treatment. NEJM. DOI: 10.1056/NEJMoa2001282",
        "Jones B. (2021). Vaccine efficacy. Lancet. PMID: 33378609",
        "Research paper. https://www.cdc.gov/coronavirus/2019-ncov/index.html"
    ]

    print("\nValidating test citations...")
    report = validator.validate_batch(test_citations, level=ValidationLevel.QUICK)

    print(f"\nResults:")
    print(f"  Total references: {report.total_references}")
    print(f"  Valid references: {report.valid_references}")
    print(f"  Overall score: {report.overall_score:.1f}/100")
    print(f"  Pass rate: {report.pass_rate:.1f}%")

    print("\nIndividual results:")
    for i, result in enumerate(report.results, 1):
        print(f"  {i}. Score: {result.credibility_score:.1f}, Valid: {result.is_valid}")
        print(f"     DOI: {result.doi}, PMID: {result.pmid}")

    print("\n✅ Basic test completed!")
