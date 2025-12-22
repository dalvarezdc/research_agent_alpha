"""
Citation-URL Correspondence Validator

PRIORITY: The APA citation text is the source of truth.
This validator ensures URLs actually correspond to the cited work.

Workflow:
1. Parse APA citation → extract title, authors, year
2. Check if provided URL matches the citation metadata
3. If URL is wrong/broken, find the correct URL
4. Log all mismatches
"""

import re
import logging
import requests
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime
from urllib.parse import urlparse, quote
from dataclasses import dataclass
import json

from .base_validator import BaseValidator
from ..models import ValidationResult, ValidationIssue, SourceType


@dataclass
class CitationMetadata:
    """Structured citation metadata extracted from APA text"""
    raw_text: str
    title: Optional[str] = None
    authors: List[str] = None
    year: Optional[int] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    url: Optional[str] = None

    def __post_init__(self):
        if self.authors is None:
            self.authors = []


@dataclass
class URLCorrespondence:
    """Result of URL correspondence check"""
    matches: bool
    confidence: float  # 0.0-1.0
    found_title: Optional[str] = None
    found_authors: List[str] = None
    found_year: Optional[int] = None
    correct_url: Optional[str] = None
    mismatch_reasons: List[str] = None

    def __post_init__(self):
        if self.mismatch_reasons is None:
            self.mismatch_reasons = []


class CitationURLCorrespondenceValidator(BaseValidator):
    """
    Validates that URLs actually correspond to the cited work.

    The APA citation is the source of truth. If URL doesn't match,
    we search for the correct URL using citation metadata.
    """

    name = "citation_url_correspondence"

    # Enhanced regex patterns
    DOI_PATTERN = r'(?:doi:|DOI:|https?://doi\.org/)?(10\.\d{4,}/[^\s,\)\]]+)'
    PMID_PATTERN = r'(?:PMID:|pmid:)\s*(\d{7,8})'
    YEAR_PATTERN = r'\((\d{4})\)'

    # APA title pattern - text between period after authors and period before journal
    # Example: "Smith, J. (2020). This is the title. Journal Name, 10(2), 123-145."
    TITLE_PATTERN = r'\.([^.]+)\.\s*(?:[A-Z][^.,]+(?:,|\.))'

    # API endpoints
    CROSSREF_SEARCH = "https://api.crossref.org/works"
    SEMANTIC_SCHOLAR_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"
    OPENALEX_SEARCH = "https://api.openalex.org/works"

    def __init__(self, timeout: int = 15, **kwargs):
        super().__init__(timeout=timeout, **kwargs)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Research Agent Alpha - Citation Validator)',
            'Accept': 'application/json'
        })

        # Setup mismatch logger
        self.mismatch_logger = logging.getLogger(f"{__name__}.mismatches")
        self.mismatch_logger.setLevel(logging.WARNING)

        # Create file handler for mismatches if not exists
        if not self.mismatch_logger.handlers:
            handler = logging.FileHandler('reference_validation_mismatches.log')
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - MISMATCH - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            self.mismatch_logger.addHandler(handler)

    def can_validate(self, reference: str) -> bool:
        """Can validate if reference has both citation text and URL"""
        return bool(
            reference and
            len(reference.strip()) > 20 and
            'http' in reference.lower()
        )

    def validate(self, reference: str, **kwargs) -> ValidationResult:
        """
        Validate that URL corresponds to the cited work.

        Priority:
        1. Parse APA citation (source of truth)
        2. Check if URL matches citation
        3. Find correct URL if mismatch
        4. Log discrepancies

        Args:
            reference: Full APA citation with URL
            **kwargs: Optional parameters

        Returns:
            ValidationResult with correspondence check and corrected URL
        """
        start_time = datetime.now()
        self._log_validation_start(reference)

        result = self._create_base_result(reference)
        result.validators_used.append('citation_url_correspondence')

        try:
            # Step 1: Parse citation metadata (source of truth)
            citation_meta = self.parse_apa_citation(reference)

            if not citation_meta.title:
                result.warnings.append("Could not extract title from citation")
                result.credibility_score = 30.0
                result.confidence = 0.5
                return result

            # Populate result with citation metadata
            result.metadata['citation_title'] = citation_meta.title
            result.metadata['citation_authors'] = citation_meta.authors
            result.metadata['citation_year'] = citation_meta.year
            result.doi = citation_meta.doi
            result.pmid = citation_meta.pmid
            result.url = citation_meta.url

            # Step 2: Check URL correspondence
            if citation_meta.url:
                correspondence = self.check_url_correspondence(
                    citation_meta.url,
                    citation_meta
                )

                result.metadata['url_matches_citation'] = correspondence.matches
                result.metadata['match_confidence'] = correspondence.confidence

                if correspondence.matches:
                    # URL is correct!
                    result.is_valid = True
                    result.credibility_score = 85.0 + (correspondence.confidence * 15)
                    result.confidence = correspondence.confidence
                    self.logger.info(f"✓ URL matches citation: {citation_meta.title[:50]}")

                else:
                    # URL doesn't match - log mismatch
                    self._log_mismatch(citation_meta, correspondence)

                    result.issues.append(ValidationIssue(
                        severity="high",
                        message="URL does not correspond to cited work",
                        field="url",
                        recommendation="Verify URL or use suggested correct URL"
                    ))

                    # Step 3: Find correct URL
                    correct_url = self.find_correct_url(citation_meta)

                    if correct_url:
                        result.metadata['corrected_url'] = correct_url
                        result.recommendations.append(
                            f"Correct URL found: {correct_url}"
                        )
                        result.credibility_score = 70.0
                        result.confidence = 0.8
                        self.logger.info(f"✓ Found correct URL for: {citation_meta.title[:50]}")
                    else:
                        result.credibility_score = 40.0
                        result.confidence = 0.6
                        result.warnings.append(
                            "Could not find correct URL for this citation"
                        )

            else:
                # No URL provided - try to find one
                result.warnings.append("No URL provided in citation")

                correct_url = self.find_correct_url(citation_meta)

                if correct_url:
                    result.metadata['suggested_url'] = correct_url
                    result.recommendations.append(
                        f"Add URL: {correct_url}"
                    )
                    result.credibility_score = 65.0
                    result.confidence = 0.75
                else:
                    result.credibility_score = 50.0
                    result.confidence = 0.5

        except Exception as e:
            return self._handle_error(reference, e)

        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        result.validation_time_ms = duration_ms
        self._log_validation_end(reference, result, duration_ms)

        return result

    def parse_apa_citation(self, citation: str) -> CitationMetadata:
        """
        Parse APA citation to extract metadata.

        Priority: Extract title, authors, year (these are source of truth)

        Args:
            citation: Full APA citation text

        Returns:
            CitationMetadata with extracted fields
        """
        meta = CitationMetadata(raw_text=citation)

        # Extract DOI
        doi_match = re.search(self.DOI_PATTERN, citation, re.IGNORECASE)
        if doi_match:
            meta.doi = doi_match.group(1).rstrip('.,;)')

        # Extract PMID
        pmid_match = re.search(self.PMID_PATTERN, citation, re.IGNORECASE)
        if pmid_match:
            meta.pmid = pmid_match.group(1)

        # Extract year
        year_match = re.search(self.YEAR_PATTERN, citation)
        if year_match:
            meta.year = int(year_match.group(1))

        # Extract URL
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, citation)
        if urls:
            # Filter out DOI URLs if we have a separate DOI
            meta.url = urls[0]
            for url in urls:
                if 'doi.org' not in url:
                    meta.url = url
                    break

        # Extract authors (more comprehensive)
        meta.authors = self._extract_authors_comprehensive(citation)

        # Extract title (this is critical!)
        meta.title = self._extract_title(citation)

        # Extract journal name
        meta.journal = self._extract_journal(citation)

        return meta

    def _extract_title(self, citation: str) -> Optional[str]:
        """
        Extract title from APA citation.

        APA format: Author, A. (Year). Title of work. Journal Name, volume(issue), pages.
        Title is between first period after year and next period.
        """
        # Remove URLs first to avoid confusion
        cleaned = re.sub(r'https?://[^\s]+', '', citation)

        # Try pattern: .(Year). TITLE. Journal
        pattern1 = r'\(\d{4}\)\.\s*([^.]+)\.'
        match = re.search(pattern1, cleaned)
        if match:
            title = match.group(1).strip()
            # Clean up title
            title = re.sub(r'\s+', ' ', title)  # Normalize whitespace
            if len(title) > 10:  # Sanity check
                return title

        # Try pattern: . TITLE. (without year check)
        # Find text between two periods after author names
        pattern2 = r'(?:[A-Z][a-z]+,?\s+[A-Z]\..*?)\.\s*([^.]+)\.'
        match = re.search(pattern2, cleaned)
        if match:
            title = match.group(1).strip()
            title = re.sub(r'\s+', ' ', title)
            if len(title) > 10:
                return title

        # Fallback: try to find longest sentence-like structure
        sentences = re.split(r'\.\s+', cleaned)
        for sent in sentences:
            # Skip if it looks like author names or journal info
            if re.search(r'^[A-Z][a-z]+,\s+[A-Z]\.', sent):
                continue
            if re.search(r'\d+\(\d+\)', sent):  # Skip volume(issue)
                continue
            if len(sent) > 15 and len(sent) < 300:
                return sent.strip()

        return None

    def _extract_authors_comprehensive(self, citation: str) -> List[str]:
        """Extract authors from APA citation - more comprehensive than base"""
        authors = []

        # APA format: "LastName, F. I., LastName2, F. I., & LastName3, F. I."
        # Pattern for single author: LastName, Initials
        author_pattern = r'\b([A-Z][a-z]+(?:-[A-Z][a-z]+)?),\s+([A-Z]\.(?:\s*[A-Z]\.)?)'
        matches = re.findall(author_pattern, citation)

        for last, initials in matches:
            authors.append(f"{last}, {initials}")

        # Check for "et al."
        if 'et al' in citation.lower():
            if authors:  # Only add if we have at least one author
                authors.append("et al.")

        return authors[:10]  # Limit to first 10

    def _extract_journal(self, citation: str) -> Optional[str]:
        """Extract journal name from APA citation"""
        # After title (after 2nd period), before volume number
        # Example: "...Title. Journal Name, 10(2), 123."
        pattern = r'\.\s*([A-Z][^.,]+),\s*\d+'
        match = re.search(pattern, citation)
        if match:
            journal = match.group(1).strip()
            # Clean up
            journal = re.sub(r'\s+', ' ', journal)
            return journal
        return None

    def check_url_correspondence(
        self,
        url: str,
        citation_meta: CitationMetadata
    ) -> URLCorrespondence:
        """
        Check if URL actually corresponds to the cited work.

        Method:
        1. Fetch URL content
        2. Extract title, authors, DOI from page
        3. Compare with citation metadata
        4. Return match confidence

        Args:
            url: URL to check
            citation_meta: Metadata from APA citation (source of truth)

        Returns:
            URLCorrespondence with match result
        """
        correspondence = URLCorrespondence(matches=False, confidence=0.0)

        try:
            # Fetch URL content
            response = self.session.get(
                url,
                timeout=self.timeout,
                allow_redirects=True
            )

            if response.status_code != 200:
                correspondence.mismatch_reasons.append(
                    f"URL not accessible (status: {response.status_code})"
                )
                return correspondence

            html_content = response.text.lower()

            # Extract metadata from URL content
            correspondence.found_title = self._extract_title_from_html(response.text)
            correspondence.found_authors = self._extract_authors_from_html(response.text)
            correspondence.found_year = self._extract_year_from_html(response.text)

            # Compare title (most important!)
            title_match_score = 0.0
            if citation_meta.title and correspondence.found_title:
                title_match_score = self._calculate_title_similarity(
                    citation_meta.title,
                    correspondence.found_title
                )

            # Compare authors
            author_match_score = 0.0
            if citation_meta.authors and correspondence.found_authors:
                author_match_score = self._calculate_author_similarity(
                    citation_meta.authors,
                    correspondence.found_authors
                )

            # Compare year
            year_match = False
            if citation_meta.year and correspondence.found_year:
                year_match = abs(citation_meta.year - correspondence.found_year) <= 1

            # Calculate overall confidence
            # Title is weighted most heavily (60%), authors (30%), year (10%)
            confidence = (
                title_match_score * 0.6 +
                author_match_score * 0.3 +
                (1.0 if year_match else 0.0) * 0.1
            )

            correspondence.confidence = confidence
            correspondence.matches = confidence >= 0.7  # 70% threshold

            # Add mismatch reasons
            if title_match_score < 0.7:
                correspondence.mismatch_reasons.append(
                    f"Title mismatch (confidence: {title_match_score:.2f})"
                )
            if author_match_score < 0.5:
                correspondence.mismatch_reasons.append(
                    f"Author mismatch (confidence: {author_match_score:.2f})"
                )
            if not year_match and citation_meta.year:
                correspondence.mismatch_reasons.append(
                    f"Year mismatch (cited: {citation_meta.year}, found: {correspondence.found_year})"
                )

        except requests.RequestException as e:
            correspondence.mismatch_reasons.append(f"Error fetching URL: {str(e)}")
            self.logger.warning(f"URL fetch error: {e}")

        return correspondence

    def find_correct_url(self, citation_meta: CitationMetadata) -> Optional[str]:
        """
        Find the correct URL for a citation using various search APIs.

        Priority order:
        1. DOI (if available) → doi.org resolver
        2. PMID (if available) → PubMed link
        3. Search by title + authors → CrossRef
        4. Search by title + authors → Semantic Scholar
        5. Search by title → OpenAlex

        Args:
            citation_meta: Citation metadata (source of truth)

        Returns:
            Correct URL if found, None otherwise
        """
        # 1. Try DOI first (most reliable)
        if citation_meta.doi:
            doi_url = f"https://doi.org/{citation_meta.doi}"
            if self._verify_url_works(doi_url):
                return doi_url

        # 2. Try PMID
        if citation_meta.pmid:
            pmid_url = f"https://pubmed.ncbi.nlm.nih.gov/{citation_meta.pmid}/"
            if self._verify_url_works(pmid_url):
                return pmid_url

        # 3. Search CrossRef by title + authors
        if citation_meta.title:
            crossref_url = self._search_crossref(citation_meta)
            if crossref_url:
                return crossref_url

        # 4. Search Semantic Scholar
        if citation_meta.title:
            s2_url = self._search_semantic_scholar(citation_meta)
            if s2_url:
                return s2_url

        # 5. Search OpenAlex
        if citation_meta.title:
            openalex_url = self._search_openalex(citation_meta)
            if openalex_url:
                return openalex_url

        return None

    def _search_crossref(self, citation_meta: CitationMetadata) -> Optional[str]:
        """Search CrossRef for work by title and authors"""
        try:
            query = citation_meta.title
            if citation_meta.authors and len(citation_meta.authors) > 0:
                # Add first author to query
                first_author = citation_meta.authors[0].split(',')[0]  # Get last name
                query = f"{first_author} {citation_meta.title}"

            params = {
                'query': query,
                'rows': 1
            }

            response = self.session.get(
                self.CROSSREF_SEARCH,
                params=params,
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                items = data.get('message', {}).get('items', [])

                if items:
                    item = items[0]
                    # Check if this is a good match
                    found_title = ' '.join(item.get('title', []))
                    similarity = self._calculate_title_similarity(
                        citation_meta.title,
                        found_title
                    )

                    if similarity >= 0.8:
                        # Get DOI and construct URL
                        doi = item.get('DOI')
                        if doi:
                            return f"https://doi.org/{doi}"

        except Exception as e:
            self.logger.warning(f"CrossRef search error: {e}")

        return None

    def _search_semantic_scholar(self, citation_meta: CitationMetadata) -> Optional[str]:
        """Search Semantic Scholar for work by title"""
        try:
            params = {
                'query': citation_meta.title,
                'limit': 1,
                'fields': 'title,authors,year,url,externalIds'
            }

            response = self.session.get(
                self.SEMANTIC_SCHOLAR_SEARCH,
                params=params,
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                papers = data.get('data', [])

                if papers:
                    paper = papers[0]
                    found_title = paper.get('title', '')
                    similarity = self._calculate_title_similarity(
                        citation_meta.title,
                        found_title
                    )

                    if similarity >= 0.8:
                        # Try to get DOI first, fallback to S2 URL
                        external_ids = paper.get('externalIds', {})
                        if 'DOI' in external_ids:
                            return f"https://doi.org/{external_ids['DOI']}"
                        elif paper.get('url'):
                            return paper['url']

        except Exception as e:
            self.logger.warning(f"Semantic Scholar search error: {e}")

        return None

    def _search_openalex(self, citation_meta: CitationMetadata) -> Optional[str]:
        """Search OpenAlex for work by title"""
        try:
            params = {
                'filter': f'display_name.search:{citation_meta.title}',
                'per-page': 1
            }

            response = self.session.get(
                self.OPENALEX_SEARCH,
                params=params,
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])

                if results:
                    work = results[0]
                    found_title = work.get('display_name', '')
                    similarity = self._calculate_title_similarity(
                        citation_meta.title,
                        found_title
                    )

                    if similarity >= 0.8:
                        # Get DOI or landing page URL
                        doi = work.get('doi')
                        if doi:
                            return doi  # OpenAlex returns full URL
                        elif work.get('primary_location', {}).get('landing_page_url'):
                            return work['primary_location']['landing_page_url']

        except Exception as e:
            self.logger.warning(f"OpenAlex search error: {e}")

        return None

    def _verify_url_works(self, url: str) -> bool:
        """Quick check if URL is accessible"""
        try:
            response = self.session.head(url, timeout=5, allow_redirects=True)
            return 200 <= response.status_code < 400
        except:
            return False

    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """
        Calculate similarity between two titles.

        Uses simple token-based Jaccard similarity.
        More sophisticated options: difflib, fuzzywuzzy, etc.
        """
        # Normalize
        t1 = set(re.findall(r'\w+', title1.lower()))
        t2 = set(re.findall(r'\w+', title2.lower()))

        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
        t1 = t1 - stop_words
        t2 = t2 - stop_words

        if not t1 or not t2:
            return 0.0

        # Jaccard similarity
        intersection = len(t1 & t2)
        union = len(t1 | t2)

        return intersection / union if union > 0 else 0.0

    def _calculate_author_similarity(self, authors1: List[str], authors2: List[str]) -> float:
        """Calculate similarity between author lists"""
        if not authors1 or not authors2:
            return 0.0

        # Extract last names only
        def get_last_names(authors):
            last_names = set()
            for author in authors:
                # Extract last name (before comma or first word)
                if ',' in author:
                    last_name = author.split(',')[0].strip().lower()
                else:
                    last_name = author.split()[0].strip().lower()
                last_names.add(last_name)
            return last_names

        names1 = get_last_names(authors1)
        names2 = get_last_names(authors2)

        # Jaccard similarity
        intersection = len(names1 & names2)
        union = len(names1 | names2)

        return intersection / union if union > 0 else 0.0

    def _extract_title_from_html(self, html: str) -> Optional[str]:
        """Extract title from HTML content"""
        # Try meta tags first
        meta_patterns = [
            r'<meta[^>]+name=["\']citation_title["\'][^>]+content=["\']([^"\']+)',
            r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)',
            r'<meta[^>]+name=["\']DC.title["\'][^>]+content=["\']([^"\']+)',
        ]

        for pattern in meta_patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Try <title> tag
        title_match = re.search(r'<title>([^<]+)</title>', html, re.IGNORECASE)
        if title_match:
            title = title_match.group(1).strip()
            # Clean up common suffixes
            title = re.sub(r'\s*[\|\-]\s*(PubMed|NCBI|Journal|PMC).*$', '', title, flags=re.IGNORECASE)
            return title

        return None

    def _extract_authors_from_html(self, html: str) -> List[str]:
        """Extract authors from HTML content"""
        authors = []

        # Try meta tags
        author_patterns = [
            r'<meta[^>]+name=["\']citation_author["\'][^>]+content=["\']([^"\']+)',
            r'<meta[^>]+name=["\']DC.creator["\'][^>]+content=["\']([^"\']+)',
            r'<meta[^>]+property=["\']article:author["\'][^>]+content=["\']([^"\']+)',
        ]

        for pattern in author_patterns:
            matches = re.findall(pattern, html, re.IGNORECASE)
            authors.extend([m.strip() for m in matches])

        return authors[:10]  # Limit to first 10

    def _extract_year_from_html(self, html: str) -> Optional[int]:
        """Extract publication year from HTML content"""
        # Try meta tags
        year_patterns = [
            r'<meta[^>]+name=["\']citation_publication_date["\'][^>]+content=["\'](\d{4})',
            r'<meta[^>]+property=["\']article:published_time["\'][^>]+content=["\'](\d{4})',
            r'<meta[^>]+name=["\']DC.date["\'][^>]+content=["\'](\d{4})',
        ]

        for pattern in year_patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                year = int(match.group(1))
                if 1900 <= year <= datetime.now().year + 1:
                    return year

        return None

    def _log_mismatch(
        self,
        citation_meta: CitationMetadata,
        correspondence: URLCorrespondence
    ) -> None:
        """
        Log mismatch between citation and URL to dedicated log file.

        Format:
        TIMESTAMP - MISMATCH
        Citation: [title]
        Provided URL: [url]
        Reasons: [reasons]
        ---
        """
        log_entry = f"""
Citation Title: {citation_meta.title}
Citation Authors: {', '.join(citation_meta.authors) if citation_meta.authors else 'Unknown'}
Citation Year: {citation_meta.year or 'Unknown'}
Provided URL: {citation_meta.url}
Match Confidence: {correspondence.confidence:.2f}
Found Title: {correspondence.found_title or 'Not found'}
Found Authors: {', '.join(correspondence.found_authors) if correspondence.found_authors else 'Not found'}
Found Year: {correspondence.found_year or 'Not found'}
Mismatch Reasons: {'; '.join(correspondence.mismatch_reasons)}
---
"""

        self.mismatch_logger.warning(log_entry)
        self.logger.warning(f"URL mismatch logged for: {citation_meta.title[:60]}...")
