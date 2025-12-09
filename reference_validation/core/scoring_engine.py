"""
Scoring engine - aggregates validation results and computes overall scores.
"""

from typing import List, Dict
from datetime import datetime
from statistics import mean, median

from ..models import ValidationResult, ValidationReport, ValidationLevel


class ScoringEngine:
    """Aggregates validation results and computes scores"""

    def generate_report(
        self,
        results: List[ValidationResult],
        validation_level: ValidationLevel = ValidationLevel.STANDARD
    ) -> ValidationReport:
        """
        Generate aggregated validation report.

        Args:
            results: List of individual validation results
            validation_level: Level of validation performed

        Returns:
            ValidationReport with aggregated statistics
        """
        if not results:
            return self._create_empty_report(validation_level)

        # Calculate summary statistics
        total_refs = len(results)
        valid_refs = sum(1 for r in results if r.is_valid)
        invalid_refs = total_refs - valid_refs

        # Calculate scores
        credibility_scores = [r.credibility_score for r in results]
        overall_score = mean(credibility_scores) if credibility_scores else 0.0
        average_credibility = overall_score

        # Count by source characteristics
        peer_reviewed_count = sum(1 for r in results if r.peer_reviewed)
        recent_sources = sum(
            1 for r in results
            if r.publication_year and r.publication_year >= (datetime.now().year - 5)
        )

        # Source type breakdown
        source_type_counts: Dict[str, int] = {}
        for result in results:
            source_type = result.source_type.value
            source_type_counts[source_type] = source_type_counts.get(source_type, 0) + 1

        # Collect issues by severity
        critical_issues = []
        high_priority_issues = []
        warnings = []

        for result in results:
            for issue in result.issues:
                if issue.severity == "critical":
                    critical_issues.append(f"{result.citation[:50]}: {issue.message}")
                elif issue.severity == "high":
                    high_priority_issues.append(f"{result.citation[:50]}: {issue.message}")

            warnings.extend(result.warnings)

        # Generate recommendations
        recommendations = self._generate_recommendations(results)

        # Calculate performance metrics
        total_time_ms = sum(r.validation_time_ms for r in results)
        cache_hits = sum(1 for r in results if r.cache_hit)
        cache_hit_rate = cache_hits / total_refs if total_refs > 0 else 0.0

        # Create report
        report = ValidationReport(
            total_references=total_refs,
            valid_references=valid_refs,
            invalid_references=invalid_refs,
            overall_score=overall_score,
            results=results,
            critical_issues=critical_issues,
            high_priority_issues=high_priority_issues,
            warnings=list(set(warnings)),  # Remove duplicates
            recommendations=recommendations,
            average_credibility=average_credibility,
            peer_reviewed_count=peer_reviewed_count,
            recent_sources_count=recent_sources,
            source_type_counts=source_type_counts,
            total_validation_time_ms=total_time_ms,
            cache_hit_rate=cache_hit_rate,
            validation_level=validation_level
        )

        return report

    def _create_empty_report(self, validation_level: ValidationLevel) -> ValidationReport:
        """Create an empty report when no results provided"""
        return ValidationReport(
            total_references=0,
            valid_references=0,
            invalid_references=0,
            overall_score=0.0,
            results=[],
            average_credibility=0.0,
            validation_level=validation_level
        )

    def _generate_recommendations(self, results: List[ValidationResult]) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []

        if not results:
            return recommendations

        # Check overall validity
        valid_count = sum(1 for r in results if r.is_valid)
        total_count = len(results)
        validity_rate = valid_count / total_count if total_count > 0 else 0

        if validity_rate < 0.7:
            recommendations.append(
                f"⚠️ Only {validity_rate*100:.0f}% of references could be validated. "
                "Consider adding DOI/PMID identifiers for better verification."
            )

        # Check credibility scores
        low_credibility = [r for r in results if r.credibility_score < 60]
        if low_credibility:
            recommendations.append(
                f"📊 {len(low_credibility)} reference(s) have low credibility scores (<60). "
                "Review these references carefully."
            )

        # Check for missing identifiers
        missing_identifiers = [
            r for r in results
            if not r.doi and not r.pmid and not r.url
        ]
        if missing_identifiers:
            recommendations.append(
                f"🔗 {len(missing_identifiers)} reference(s) lack DOI, PMID, or URL. "
                "Cannot verify these references exist."
            )

        # Check for inaccessible URLs
        inaccessible_urls = [
            r for r in results
            if r.url_accessible is False
        ]
        if inaccessible_urls:
            recommendations.append(
                f"⚠️ {len(inaccessible_urls)} URL(s) are not accessible. "
                "These references may be broken or behind paywalls."
            )

        # Check peer review status
        peer_reviewed_count = sum(1 for r in results if r.peer_reviewed)
        if peer_reviewed_count == 0:
            recommendations.append(
                "⚠️ No peer-reviewed sources detected. "
                "Consider including peer-reviewed research for higher credibility."
            )

        # Check publication recency
        old_sources = [
            r for r in results
            if r.publication_year and r.publication_year < (datetime.now().year - 10)
        ]
        if len(old_sources) > len(results) * 0.5:
            recommendations.append(
                f"📅 {len(old_sources)} source(s) are >10 years old. "
                "Consider supplementing with recent research."
            )

        return recommendations

    def calculate_aggregate_score(
        self,
        results: List[ValidationResult],
        weights: Dict[str, float] = None
    ) -> float:
        """
        Calculate weighted aggregate score.

        Args:
            results: List of validation results
            weights: Custom weights for different factors

        Returns:
            Weighted aggregate score (0-100)
        """
        if not results:
            return 0.0

        # Default weights
        if weights is None:
            weights = {
                'credibility': 0.4,
                'validity': 0.3,
                'verifiability': 0.2,
                'recency': 0.1
            }

        # Calculate component scores
        credibility = mean([r.credibility_score for r in results])

        validity = (sum(1 for r in results if r.is_valid) / len(results)) * 100

        verifiable = sum(
            1 for r in results
            if r.doi or r.pmid or (r.url_accessible is True)
        )
        verifiability = (verifiable / len(results)) * 100

        recent = sum(
            1 for r in results
            if r.publication_year and r.publication_year >= (datetime.now().year - 5)
        )
        recency = (recent / len(results)) * 100

        # Weighted score
        score = (
            weights['credibility'] * credibility +
            weights['validity'] * validity +
            weights['verifiability'] * verifiability +
            weights['recency'] * recency
        )

        return min(score, 100.0)
