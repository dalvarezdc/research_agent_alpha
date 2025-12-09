"""
Data models for reference validation system.
Uses Pydantic for validation and type safety.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class ValidationLevel(str, Enum):
    """Validation thoroughness levels"""
    QUICK = "quick"  # Format check + cache lookup (50ms)
    STANDARD = "standard"  # Format + DOI + URL (500ms)
    THOROUGH = "thorough"  # All validators + credibility (5s)


class SourceType(str, Enum):
    """Types of sources"""
    JOURNAL_ARTICLE = "journal_article"
    PREPRINT = "preprint"
    BOOK = "book"
    WEBSITE = "website"
    CLINICAL_GUIDELINE = "clinical_guideline"
    REGULATORY = "regulatory"  # FDA, EMA, etc.
    UNKNOWN = "unknown"


class ValidationIssue(BaseModel):
    """Individual validation issue"""
    severity: str = Field(..., description="low, medium, high, critical")
    message: str = Field(..., description="Human-readable issue description")
    field: Optional[str] = Field(None, description="Which field has the issue")
    recommendation: Optional[str] = Field(None, description="How to fix it")


class ValidationResult(BaseModel):
    """Result of validating a single reference"""

    # Identity
    citation: str = Field(..., description="Original citation text")
    reference_id: str = Field(..., description="Unique identifier")

    # Overall validation status
    is_valid: bool = Field(..., description="Whether reference passed validation")
    credibility_score: float = Field(..., ge=0, le=100, description="Score 0-100")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in validation")

    # Detailed validation results
    citation_format_valid: bool = Field(default=False)
    url_accessible: Optional[bool] = Field(default=None)
    doi_valid: Optional[bool] = Field(default=None)
    pubmed_verified: Optional[bool] = Field(default=None)
    content_matches_claim: Optional[bool] = Field(default=None)

    # Metadata
    source_type: SourceType = Field(default=SourceType.UNKNOWN)
    peer_reviewed: bool = Field(default=False)
    publication_year: Optional[int] = Field(default=None)
    journal_name: Optional[str] = Field(default=None)
    journal_impact_factor: Optional[float] = Field(default=None)
    authors: List[str] = Field(default_factory=list)

    # URLs and identifiers
    url: Optional[str] = Field(default=None)
    doi: Optional[str] = Field(default=None)
    pmid: Optional[str] = Field(default=None)
    arxiv_id: Optional[str] = Field(default=None)

    # Issues and recommendations
    issues: List[ValidationIssue] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    alternative_sources: List[str] = Field(default_factory=list)

    # Provenance
    validated_at: datetime = Field(default_factory=datetime.now)
    validators_used: List[str] = Field(default_factory=list)
    cache_hit: bool = Field(default=False)
    validation_time_ms: float = Field(default=0.0)

    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('credibility_score')
    @classmethod
    def validate_credibility_score(cls, v: float) -> float:
        """Ensure credibility score is in valid range"""
        return max(0.0, min(100.0, v))

    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is in valid range"""
        return max(0.0, min(1.0, v))


class ValidationReport(BaseModel):
    """Aggregated report for multiple reference validations"""

    # Summary statistics
    total_references: int = Field(..., description="Total references validated")
    valid_references: int = Field(..., description="Number of valid references")
    invalid_references: int = Field(..., description="Number of invalid references")
    overall_score: float = Field(..., ge=0, le=100, description="Aggregate score")

    # Detailed results
    results: List[ValidationResult] = Field(..., description="Individual results")

    # Issues by severity
    critical_issues: List[str] = Field(default_factory=list)
    high_priority_issues: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    # Recommendations
    recommendations: List[str] = Field(default_factory=list)

    # Statistics
    average_credibility: float = Field(..., ge=0, le=100)
    peer_reviewed_count: int = Field(default=0)
    recent_sources_count: int = Field(default=0, description="<5 years old")

    # Source type breakdown
    source_type_counts: Dict[str, int] = Field(default_factory=dict)

    # Performance
    total_validation_time_ms: float = Field(default=0.0)
    cache_hit_rate: float = Field(default=0.0, ge=0, le=1)

    # Metadata
    generated_at: datetime = Field(default_factory=datetime.now)
    validation_level: ValidationLevel = Field(default=ValidationLevel.STANDARD)

    def add_critical_issue(self, issue: str) -> None:
        """Add a critical issue to the report"""
        self.critical_issues.append(issue)

    def add_warning(self, warning: str) -> None:
        """Add a warning to the report"""
        self.warnings.append(warning)

    def add_recommendation(self, recommendation: str) -> None:
        """Add a recommendation to the report"""
        self.recommendations.append(recommendation)

    @property
    def pass_rate(self) -> float:
        """Calculate percentage of valid references"""
        if self.total_references == 0:
            return 0.0
        return (self.valid_references / self.total_references) * 100


class ValidationConfig(BaseModel):
    """Configuration for reference validation"""

    # Cache settings
    cache_backend: str = Field(default="sqlite", description="sqlite, redis, json, memory, none")
    cache_ttl_days: int = Field(default=30, ge=1, le=365)
    cache_path: str = Field(default="./cache/reference_validation.db")

    # Validation behavior
    validation_level: ValidationLevel = Field(default=ValidationLevel.STANDARD)
    timeout_seconds: int = Field(default=10, ge=1, le=60)
    retry_attempts: int = Field(default=3, ge=1, le=10)
    parallel_workers: int = Field(default=4, ge=1, le=20)

    # External APIs
    enable_pubmed: bool = Field(default=True)
    enable_crossref: bool = Field(default=True)
    enable_web_scraping: bool = Field(default=True)
    enable_doi_resolver: bool = Field(default=True)
    pubmed_api_key: Optional[str] = Field(default=None)
    pubmed_email: Optional[str] = Field(default=None)

    # Credibility thresholds
    min_credibility_score: float = Field(default=60, ge=0, le=100)
    require_peer_review: bool = Field(default=False)
    max_source_age_years: Optional[int] = Field(default=None, ge=1)

    # Rate limiting
    requests_per_minute: int = Field(default=30, ge=1, le=300)
    respect_robots_txt: bool = Field(default=True)

    # Logging
    enable_logging: bool = Field(default=True)
    log_level: str = Field(default="INFO")

    @field_validator('cache_backend')
    @classmethod
    def validate_cache_backend(cls, v: str) -> str:
        """Validate cache backend choice"""
        valid_backends = ["sqlite", "redis", "json", "memory", "none"]
        if v not in valid_backends:
            raise ValueError(f"cache_backend must be one of {valid_backends}")
        return v


class ExtractedReference(BaseModel):
    """A reference extracted from text"""
    raw_text: str = Field(..., description="Original reference text")
    citation_style: Optional[str] = Field(None, description="APA, MLA, Vancouver, etc.")
    context: Optional[str] = Field(None, description="Surrounding text")
    claim: Optional[str] = Field(None, description="Claim being supported")
    position: Optional[int] = Field(None, description="Position in document")
