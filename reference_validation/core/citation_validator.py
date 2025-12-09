"""
Citation format validator - parses and validates citation formats.
Priority: Verify references exist (DOI/PMID/URL verification over authority)
"""

import re
from typing import Optional, List, Tuple
from datetime import datetime

from .base_validator import BaseValidator
from ..models import ValidationResult, ValidationIssue, SourceType


class CitationValidator(BaseValidator):
    """Validates citation format and extracts metadata"""

    name = "citation_validator"

    # Regex patterns for common citation elements
    DOI_PATTERN = r'(?:doi:|DOI:|https?://doi\.org/)?(10\.\d{4,}/[^\s]+)'
    PMID_PATTERN = r'(?:PMID:|pmid:)\s*(\d{7,8})'
    ARXIV_PATTERN = r'(?:arXiv:|arxiv:)\s*(\d{4}\.\d{4,5})'
    URL_PATTERN = r'https?://[^\s<>"{}|\\^`\[\]]+'
    YEAR_PATTERN = r'\((\d{4})\)'

    # Journal patterns
    JOURNAL_INDICATORS = [
        'journal', 'jama', 'nejm', 'lancet', 'bmj', 'nature',
        'science', 'cell', 'plos', 'pubmed'
    ]

    def can_validate(self, reference: str) -> bool:
        """All text can be validated for citation format"""
        return bool(reference and len(reference.strip()) > 10)

    def validate(self, reference: str, **kwargs) -> ValidationResult:
        """
        Validate citation format and extract metadata.
        Priority: Verify references exist (DOI/PMID/URL) over authority.

        Args:
            reference: Citation text to validate
            **kwargs: Optional parameters (expected_claim, etc.)

        Returns:
            ValidationResult with citation analysis
        """
        start_time = datetime.now()
        self._log_validation_start(reference)

        result = self._create_base_result(reference)

        try:
            # Extract identifiers
            doi = self.extract_doi(reference)
            pmid = self.extract_pmid(reference)
            arxiv_id = self.extract_arxiv_id(reference)
            urls = self.extract_urls(reference)
            year = self.extract_year(reference)
            authors = self.extract_authors(reference)

            # Populate result
            result.doi = doi
            result.pmid = pmid
            result.arxiv_id = arxiv_id
            result.url = urls[0] if urls else None
            result.publication_year = year
            result.authors = authors

            # Determine source type
            result.source_type = self._determine_source_type(reference, doi, pmid, arxiv_id)

            # Validate format
            format_valid, format_issues = self._validate_format(reference)
            result.citation_format_valid = format_valid

            # Add issues
            for issue in format_issues:
                result.issues.append(issue)

            # Calculate credibility based on verifiability (existence) not authority
            score = self._calculate_format_score(
                has_doi=bool(doi),
                has_pmid=bool(pmid),
                has_url=bool(urls),
                has_year=bool(year),
                has_authors=bool(authors),
                source_type=result.source_type
            )

            result.credibility_score = score
            result.confidence = 0.7  # Medium confidence from format alone
            result.is_valid = format_valid and score >= 40

            # Add recommendations
            if not doi and not pmid and not urls:
                result.recommendations.append(
                    "Add DOI, PMID, or URL for verifiability - cannot confirm reference exists"
                )
            if not year:
                result.warnings.append("Publication year not found")
            if not authors:
                result.warnings.append("No authors identified")

        except Exception as e:
            return self._handle_error(reference, e)

        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        result.validation_time_ms = duration_ms
        self._log_validation_end(reference, result, duration_ms)

        return result

    def extract_doi(self, text: str) -> Optional[str]:
        """Extract DOI from text"""
        match = re.search(self.DOI_PATTERN, text, re.IGNORECASE)
        if match:
            doi = match.group(1) if match.lastindex else match.group(0)
            # Clean up DOI
            doi = doi.rstrip('.,;)')
            return doi
        return None

    def extract_pmid(self, text: str) -> Optional[str]:
        """Extract PubMed ID from text"""
        match = re.search(self.PMID_PATTERN, text, re.IGNORECASE)
        return match.group(1) if match else None

    def extract_arxiv_id(self, text: str) -> Optional[str]:
        """Extract arXiv ID from text"""
        match = re.search(self.ARXIV_PATTERN, text, re.IGNORECASE)
        return match.group(1) if match else None

    def extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text"""
        return re.findall(self.URL_PATTERN, text)

    def extract_year(self, text: str) -> Optional[int]:
        """Extract publication year from text"""
        match = re.search(self.YEAR_PATTERN, text)
        if match:
            year = int(match.group(1))
            # Sanity check
            if 1900 <= year <= datetime.now().year + 1:
                return year
        return None

    def extract_authors(self, text: str) -> List[str]:
        """
        Extract author names from citation.
        This is a simple heuristic - more sophisticated parsing may be needed.
        """
        authors = []

        # Look for pattern like "Smith, J."  or "Smith J"
        author_pattern = r'\b([A-Z][a-z]+(?:-[A-Z][a-z]+)?),?\s+([A-Z]\.?)'
        matches = re.findall(author_pattern, text)

        for last, first_initial in matches:
            authors.append(f"{last}, {first_initial}")

        # Look for "et al."
        if 'et al' in text.lower():
            authors.append("et al.")

        return authors[:10]  # Limit to first 10 authors

    def _determine_source_type(
        self,
        text: str,
        doi: Optional[str],
        pmid: Optional[str],
        arxiv_id: Optional[str]
    ) -> SourceType:
        """Determine the type of source from available information"""

        text_lower = text.lower()

        # Regulatory sources
        if any(org in text_lower for org in ['fda.gov', 'ema.europa', 'who.int']):
            return SourceType.REGULATORY

        # Clinical guidelines
        if 'guideline' in text_lower or 'recommendation' in text_lower:
            return SourceType.CLINICAL_GUIDELINE

        # Preprints
        if arxiv_id or 'preprint' in text_lower or 'biorxiv' in text_lower:
            return SourceType.PREPRINT

        # Journal articles
        if doi or pmid or any(journal in text_lower for journal in self.JOURNAL_INDICATORS):
            return SourceType.JOURNAL_ARTICLE

        # Books
        if 'isbn' in text_lower or 'book' in text_lower or 'publisher' in text_lower:
            return SourceType.BOOK

        # Websites
        if 'http' in text_lower and not doi:
            return SourceType.WEBSITE

        return SourceType.UNKNOWN

    def _validate_format(self, citation: str) -> Tuple[bool, List[ValidationIssue]]:
        """
        Validate citation format.

        Returns:
            (is_valid, list_of_issues)
        """
        issues = []

        # Check minimum length
        if len(citation.strip()) < 20:
            issues.append(ValidationIssue(
                severity="high",
                message="Citation appears too short to be complete",
                field="length"
            ))

        # Check for some basic elements
        has_year = bool(self.extract_year(citation))
        has_identifiers = bool(
            self.extract_doi(citation) or
            self.extract_pmid(citation) or
            self.extract_urls(citation)
        )

        if not has_year:
            issues.append(ValidationIssue(
                severity="medium",
                message="No publication year found",
                field="year",
                recommendation="Add publication year in format (YYYY)"
            ))

        if not has_identifiers:
            issues.append(ValidationIssue(
                severity="high",
                message="No DOI, PMID, or URL found - cannot verify reference exists",
                field="identifiers",
                recommendation="Add DOI, PMID, or URL for verifiability"
            ))

        # Citation is valid if it has no high-severity issues
        high_severity_issues = [i for i in issues if i.severity == "high"]
        is_valid = len(high_severity_issues) == 0

        return is_valid, issues

    def _calculate_format_score(
        self,
        has_doi: bool,
        has_pmid: bool,
        has_url: bool,
        has_year: bool,
        has_authors: bool,
        source_type: SourceType
    ) -> float:
        """
        Calculate credibility score based on format elements.
        PRIORITY: Verifiable identifiers (DOI/PMID/URL) > metadata > authority
        """

        score = 20.0  # Base score for having text

        # HIGHEST PRIORITY: Verifiable identifiers (can we check if it exists?)
        if has_doi:
            score += 25  # DOI is gold standard for verification
        if has_pmid:
            score += 25  # PubMed ID is also excellent
        if has_url and not (has_doi or has_pmid):
            score += 15  # URL is good but less reliable

        # HIGH PRIORITY: Essential metadata
        if has_year:
            score += 15  # Year helps verify accuracy
        if has_authors:
            score += 15  # Authors help verify accuracy

        # LOWER PRIORITY: Source type (nice to have, but existence matters more)
        if source_type == SourceType.REGULATORY:
            score += 10  # Regulatory sources are reliable
        elif source_type == SourceType.JOURNAL_ARTICLE:
            score += 5  # Journal articles are common
        elif source_type == SourceType.CLINICAL_GUIDELINE:
            score += 5  # Guidelines are useful

        return min(score, 100.0)
