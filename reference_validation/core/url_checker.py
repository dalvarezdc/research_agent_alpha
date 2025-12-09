"""
URL accessibility checker - verifies URLs are accessible and valid.
Priority: Verify the reference exists and is accessible.
"""

import requests
from typing import Optional, Dict
from datetime import datetime
from urllib.parse import urlparse
import time

from .base_validator import BaseValidator
from ..models import ValidationResult, ValidationIssue


class URLChecker(BaseValidator):
    """Checks if URLs are accessible and valid"""

    name = "url_checker"

    def __init__(self, timeout: int = 10, **kwargs):
        super().__init__(timeout=timeout, **kwargs)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Research Agent Alpha - Reference Validator)'
        })

    def can_validate(self, reference: str) -> bool:
        """Can validate if reference contains a URL"""
        return 'http://' in reference or 'https://' in reference

    def validate(self, reference: str, **kwargs) -> ValidationResult:
        """
        Check if URL in reference is accessible.

        Args:
            reference: Citation text containing URL
            **kwargs: Optional parameters

        Returns:
            ValidationResult with URL accessibility status
        """
        start_time = datetime.now()
        self._log_validation_start(reference)

        result = self._create_base_result(reference)

        try:
            # Extract URL from reference
            url = self._extract_primary_url(reference)

            if not url:
                result.warnings.append("No URL found in reference")
                result.confidence = 0.5
                return result

            result.url = url

            # Check URL accessibility
            accessible, status_code, redirect_url = self._check_url_accessible(url)

            result.url_accessible = accessible
            result.metadata['status_code'] = status_code
            result.metadata['redirect_url'] = redirect_url

            if accessible:
                result.credibility_score = 70.0  # Base score for accessible URL
                result.is_valid = True
                result.confidence = 0.8

                if status_code == 200:
                    result.credibility_score = 80.0
                elif status_code in [301, 302, 303, 307, 308]:
                    result.warnings.append(f"URL redirects to: {redirect_url}")
                    result.credibility_score = 75.0

                # Check if it's a reliable domain
                if self._is_reliable_domain(url):
                    result.credibility_score += 15
                    result.metadata['reliable_domain'] = True

            else:
                result.is_valid = False
                result.credibility_score = 20.0
                result.confidence = 0.9  # High confidence that it's not accessible

                result.issues.append(ValidationIssue(
                    severity="high",
                    message=f"URL not accessible (status: {status_code})",
                    field="url",
                    recommendation="Verify URL or find alternative source"
                ))

        except Exception as e:
            return self._handle_error(reference, e)

        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        result.validation_time_ms = duration_ms
        self._log_validation_end(reference, result, duration_ms)

        return result

    def _extract_primary_url(self, text: str) -> Optional[str]:
        """Extract the primary URL from text"""
        import re
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text)
        return urls[0] if urls else None

    def _check_url_accessible(self, url: str) -> tuple[bool, Optional[int], Optional[str]]:
        """
        Check if URL is accessible.

        Returns:
            (is_accessible, status_code, redirect_url)
        """
        try:
            response = self.session.head(
                url,
                timeout=self.timeout,
                allow_redirects=True
            )

            status_code = response.status_code
            redirect_url = response.url if response.url != url else None

            # Consider 2xx and 3xx as accessible
            is_accessible = 200 <= status_code < 400

            return is_accessible, status_code, redirect_url

        except requests.exceptions.Timeout:
            self.logger.warning(f"Timeout checking URL: {url}")
            return False, None, None

        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Error checking URL {url}: {e}")
            return False, None, None

    def _is_reliable_domain(self, url: str) -> bool:
        """Check if URL is from a known reliable domain"""
        reliable_domains = [
            'pubmed.ncbi.nlm.nih.gov',
            'doi.org',
            'nature.com',
            'sciencedirect.com',
            'springer.com',
            'wiley.com',
            'nih.gov',
            'cdc.gov',
            'fda.gov',
            'who.int',
            'ema.europa.eu',
            'bmj.com',
            'thelancet.com',
            'jamanetwork.com',
            'nejm.org',
            'arxiv.org',
            'biorxiv.org',
            'medrxiv.org',
        ]

        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        return any(reliable in domain for reliable in reliable_domains)

    def check_multiple_urls(self, urls: list[str]) -> Dict[str, bool]:
        """
        Check multiple URLs at once.

        Args:
            urls: List of URLs to check

        Returns:
            Dict mapping URL to accessibility status
        """
        results = {}
        for url in urls:
            accessible, _, _ = self._check_url_accessible(url)
            results[url] = accessible
            time.sleep(0.1)  # Rate limiting

        return results
