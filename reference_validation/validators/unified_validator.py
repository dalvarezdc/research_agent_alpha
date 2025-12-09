"""
Unified Reference Validator - handles all external validation in one class.
Validates DOI, PMID, arXiv, and URLs through their respective APIs.
Priority: Verify the reference exists.
"""

import requests
import xml.etree.ElementTree as ET
import time
import re
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
from urllib.parse import urlparse

from ..core.base_validator import BaseValidator
from ..models import ValidationResult, ValidationIssue, SourceType


class UnifiedReferenceValidator(BaseValidator):
    """
    Single validator that handles all external validation.
    Detects identifier type and uses appropriate API.
    """

    name = "unified_validator"

    # API endpoints
    PUBMED_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    DOI_RESOLVER = "https://doi.org"
    CROSSREF_API = "https://api.crossref.org/works"
    ARXIV_API = "http://export.arxiv.org/api/query"

    # Regex patterns
    DOI_PATTERN = r'(?:doi:|DOI:|https?://doi\.org/)?(10\.\d{4,}/[^\s]+)'
    PMID_PATTERN = r'(?:PMID:|pmid:)\s*(\d{7,8})'
    ARXIV_PATTERN = r'(?:arXiv:|arxiv:)\s*(\d{4}\.\d{4,5})'

    # Reliable domains
    RELIABLE_DOMAINS = {
        'pubmed.ncbi.nlm.nih.gov', 'doi.org', 'nature.com', 'sciencedirect.com',
        'springer.com', 'wiley.com', 'nih.gov', 'cdc.gov', 'fda.gov', 'who.int',
        'ema.europa.eu', 'bmj.com', 'thelancet.com', 'jamanetwork.com', 'nejm.org',
        'arxiv.org', 'biorxiv.org', 'medrxiv.org',
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        email: Optional[str] = None,
        timeout: int = 10,
        **kwargs
    ):
        super().__init__(timeout=timeout, **kwargs)
        self.api_key = api_key
        self.email = email or "research_agent@example.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Research Agent Alpha - Reference Validator)'
        })
        self.last_request_time = {}  # Track per-API rate limiting

    def can_validate(self, reference: str) -> bool:
        """Can validate any reference"""
        return bool(reference and len(reference.strip()) > 10)

    def validate(self, reference: str, **kwargs) -> ValidationResult:
        """
        Validate reference using appropriate method based on identifiers found.

        Args:
            reference: Citation to validate
            **kwargs: Optional parameters

        Returns:
            ValidationResult with comprehensive validation
        """
        start_time = datetime.now()
        self._log_validation_start(reference)

        result = self._create_base_result(reference)

        try:
            # Extract all identifiers
            doi = self._extract_doi(reference)
            pmid = self._extract_pmid(reference)
            arxiv_id = self._extract_arxiv_id(reference)
            url = self._extract_url(reference)

            result.doi = doi
            result.pmid = pmid
            result.arxiv_id = arxiv_id
            result.url = url

            # Validate in order of reliability: PMID > DOI > arXiv > URL
            validated = False

            # Try PMID first (most reliable for medical research)
            if pmid:
                validated = self._validate_pmid(pmid, result)
                if validated:
                    result.credibility_score += 20  # Bonus for PubMed verification

            # Try DOI if not validated yet
            if not validated and doi:
                validated = self._validate_doi(doi, result)
                if validated:
                    result.credibility_score += 15  # Bonus for DOI verification

            # Try arXiv
            if not validated and arxiv_id:
                validated = self._validate_arxiv(arxiv_id, result)
                if validated:
                    result.credibility_score += 10  # Preprints less reliable

            # Try URL as last resort
            if not validated and url:
                validated = self._validate_url(url, result)
                if validated:
                    result.credibility_score += 5  # URLs least reliable

            # Set final validity
            result.is_valid = validated

            if not validated:
                result.issues.append(ValidationIssue(
                    severity="high",
                    message="Could not verify reference exists in any database",
                    field="verification",
                    recommendation="Add DOI, PMID, or accessible URL"
                ))
                result.credibility_score = 20.0  # Base score for formatted text
                result.confidence = 0.8

        except Exception as e:
            return self._handle_error(reference, e)

        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        result.validation_time_ms = duration_ms
        self._log_validation_end(reference, result, duration_ms)

        return result

    def _validate_pmid(self, pmid: str, result: ValidationResult) -> bool:
        """
        Validate via PubMed.

        Returns:
            True if validated successfully
        """
        try:
            self._rate_limit('pubmed')

            params = {
                'db': 'pubmed',
                'id': pmid,
                'retmode': 'xml',
                'email': self.email
            }

            if self.api_key:
                params['api_key'] = self.api_key

            response = self.session.get(
                self.PUBMED_EFETCH,
                params=params,
                timeout=self.timeout
            )

            if response.status_code == 200 and len(response.text) > 100:
                # Parse metadata
                metadata = self._parse_pubmed_xml(response.text)

                if metadata:
                    result.pubmed_verified = True
                    result.source_type = SourceType.JOURNAL_ARTICLE
                    result.peer_reviewed = True
                    result.publication_year = metadata.get('year')
                    result.journal_name = metadata.get('journal')
                    result.authors = metadata.get('authors', [])
                    result.metadata['pubmed'] = metadata
                    result.credibility_score = 70.0  # Base for PubMed
                    result.confidence = 0.95
                    result.validators_used.append('pubmed')

                    self.logger.info(f"PubMed validated: PMID {pmid}")
                    return True

            result.pubmed_verified = False
            return False

        except Exception as e:
            self.logger.warning(f"PubMed validation error for PMID {pmid}: {e}")
            result.pubmed_verified = False
            return False

    def _validate_doi(self, doi: str, result: ValidationResult) -> bool:
        """
        Validate via DOI resolver and CrossRef.

        Returns:
            True if validated successfully
        """
        try:
            self._rate_limit('doi')

            # Try DOI resolver first (fastest)
            doi_url = f"{self.DOI_RESOLVER}/{doi}"
            response = self.session.head(
                doi_url,
                timeout=self.timeout,
                allow_redirects=True
            )

            if 200 <= response.status_code < 400:
                result.doi_valid = True
                result.url = response.url
                result.credibility_score = 65.0  # Base for DOI
                result.confidence = 0.90
                result.validators_used.append('doi_resolver')

                # Try to get metadata from CrossRef
                metadata = self._fetch_crossref_metadata(doi)
                if metadata:
                    result.publication_year = metadata.get('year')
                    result.journal_name = metadata.get('journal')
                    result.authors = metadata.get('authors', [])
                    result.source_type = SourceType.JOURNAL_ARTICLE
                    result.metadata['crossref'] = metadata

                self.logger.info(f"DOI validated: {doi}")
                return True

            result.doi_valid = False
            return False

        except Exception as e:
            self.logger.warning(f"DOI validation error for {doi}: {e}")
            result.doi_valid = False
            return False

    def _validate_arxiv(self, arxiv_id: str, result: ValidationResult) -> bool:
        """
        Validate via arXiv API.

        Returns:
            True if validated successfully
        """
        try:
            self._rate_limit('arxiv')

            response = self.session.get(
                f"{self.ARXIV_API}?id_list={arxiv_id}",
                timeout=self.timeout
            )

            if response.status_code == 200 and '<entry>' in response.text:
                result.source_type = SourceType.PREPRINT
                result.peer_reviewed = False
                result.credibility_score = 60.0  # Preprints less reliable
                result.confidence = 0.85
                result.validators_used.append('arxiv')

                # Parse arXiv response for metadata
                metadata = self._parse_arxiv_xml(response.text)
                if metadata:
                    result.publication_year = metadata.get('year')
                    result.authors = metadata.get('authors', [])
                    result.metadata['arxiv'] = metadata

                self.logger.info(f"arXiv validated: {arxiv_id}")
                return True

            return False

        except Exception as e:
            self.logger.warning(f"arXiv validation error for {arxiv_id}: {e}")
            return False

    def _validate_url(self, url: str, result: ValidationResult) -> bool:
        """
        Validate via URL accessibility check.

        Returns:
            True if URL is accessible
        """
        try:
            self._rate_limit('url')

            response = self.session.head(
                url,
                timeout=self.timeout,
                allow_redirects=True
            )

            if 200 <= response.status_code < 400:
                result.url_accessible = True
                result.credibility_score = 50.0  # Base for accessible URL
                result.confidence = 0.75
                result.validators_used.append('url_check')

                # Bonus for reliable domains
                if self._is_reliable_domain(url):
                    result.credibility_score += 15
                    result.metadata['reliable_domain'] = True

                self.logger.info(f"URL validated: {url}")
                return True

            result.url_accessible = False
            return False

        except Exception as e:
            self.logger.warning(f"URL validation error for {url}: {e}")
            result.url_accessible = False
            return False

    def _fetch_crossref_metadata(self, doi: str) -> Optional[Dict[str, Any]]:
        """Fetch metadata from CrossRef API"""
        try:
            response = self.session.get(
                f"{self.CROSSREF_API}/{doi}",
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                message = data.get('message', {})

                metadata = {}

                # Extract year
                if 'published-print' in message:
                    year_parts = message['published-print'].get('date-parts', [[]])[0]
                    if year_parts:
                        metadata['year'] = year_parts[0]

                # Extract journal
                if 'container-title' in message:
                    titles = message['container-title']
                    if titles:
                        metadata['journal'] = titles[0]

                # Extract authors
                if 'author' in message:
                    authors = []
                    for author in message['author'][:10]:
                        if 'family' in author:
                            name = author['family']
                            if 'given' in author:
                                name += f", {author['given'][0]}"
                            authors.append(name)
                    metadata['authors'] = authors

                return metadata

        except Exception as e:
            self.logger.warning(f"CrossRef API error: {e}")

        return None

    def _parse_pubmed_xml(self, xml_text: str) -> Dict[str, Any]:
        """Parse PubMed XML response"""
        try:
            root = ET.fromstring(xml_text)
            metadata = {}

            # Title
            title_elem = root.find(".//ArticleTitle")
            if title_elem is not None:
                metadata['title'] = title_elem.text

            # Journal
            journal_elem = root.find(".//Journal/Title")
            if journal_elem is not None:
                metadata['journal'] = journal_elem.text

            # Year
            year_elem = root.find(".//PubDate/Year")
            if year_elem is not None:
                metadata['year'] = int(year_elem.text)

            # Authors
            authors = []
            for author in root.findall(".//Author"):
                last_name = author.find("LastName")
                initials = author.find("Initials")
                if last_name is not None:
                    name = last_name.text
                    if initials is not None:
                        name += f", {initials.text}"
                    authors.append(name)
            metadata['authors'] = authors[:10]

            return metadata

        except ET.ParseError as e:
            self.logger.error(f"XML parse error: {e}")
            return {}

    def _parse_arxiv_xml(self, xml_text: str) -> Dict[str, Any]:
        """Parse arXiv XML response"""
        try:
            # arXiv uses Atom format
            metadata = {}

            # Extract year from published date
            year_match = re.search(r'<published>(\d{4})', xml_text)
            if year_match:
                metadata['year'] = int(year_match.group(1))

            # Extract authors
            author_pattern = r'<name>([^<]+)</name>'
            authors = re.findall(author_pattern, xml_text)
            metadata['authors'] = authors[:10]

            return metadata

        except Exception as e:
            self.logger.error(f"arXiv parse error: {e}")
            return {}

    def _extract_doi(self, text: str) -> Optional[str]:
        """Extract DOI"""
        match = re.search(self.DOI_PATTERN, text, re.IGNORECASE)
        if match:
            doi = match.group(1) if match.lastindex else match.group(0)
            return doi.rstrip('.,;)')
        return None

    def _extract_pmid(self, text: str) -> Optional[str]:
        """Extract PMID"""
        match = re.search(self.PMID_PATTERN, text, re.IGNORECASE)
        return match.group(1) if match else None

    def _extract_arxiv_id(self, text: str) -> Optional[str]:
        """Extract arXiv ID"""
        match = re.search(self.ARXIV_PATTERN, text, re.IGNORECASE)
        return match.group(1) if match else None

    def _extract_url(self, text: str) -> Optional[str]:
        """Extract URL"""
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text)
        return urls[0] if urls else None

    def _is_reliable_domain(self, url: str) -> bool:
        """Check if URL is from reliable domain"""
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        return any(reliable in domain for reliable in self.RELIABLE_DOMAINS)

    def _rate_limit(self, api: str) -> None:
        """Simple rate limiting per API"""
        min_interval = 0.34  # ~3 requests/sec default

        if api == 'pubmed' and self.api_key:
            min_interval = 0.1  # 10 req/sec with API key

        last_time = self.last_request_time.get(api, 0)
        current_time = time.time()
        time_since_last = current_time - last_time

        if time_since_last < min_interval:
            time.sleep(min_interval - time_since_last)

        self.last_request_time[api] = time.time()
