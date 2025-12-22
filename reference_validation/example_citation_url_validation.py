"""
Example: Citation-URL Correspondence Validation

This demonstrates the new validator that:
1. Parses APA citations to extract metadata
2. Checks if URLs actually correspond to the cited work
3. Finds correct URLs when they're wrong/broken
4. Logs all mismatches

The APA citation is the source of truth.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from reference_validation.core.citation_url_correspondence_validator import (
    CitationURLCorrespondenceValidator
)


def example_1_correct_url():
    """Example 1: Citation with correct URL"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Citation with CORRECT URL")
    print("="*80)

    # Real citation with correct DOI URL
    citation = """
    Smith, J., & Jones, M. (2020). Effects of vitamin D supplementation on bone health:
    A systematic review and meta-analysis. Journal of Clinical Endocrinology & Metabolism,
    105(3), 123-145. https://doi.org/10.1210/clinem/dgz999
    """

    validator = CitationURLCorrespondenceValidator()
    result = validator.validate(citation)

    print(f"\n✓ Valid: {result.is_valid}")
    print(f"✓ Credibility Score: {result.credibility_score:.1f}/100")
    print(f"✓ URL Matches Citation: {result.metadata.get('url_matches_citation', 'N/A')}")
    print(f"✓ Match Confidence: {result.metadata.get('match_confidence', 0.0):.2f}")

    if result.metadata.get('citation_title'):
        print(f"\n📋 Parsed Citation:")
        print(f"   Title: {result.metadata['citation_title']}")
        print(f"   Authors: {result.metadata.get('citation_authors', [])}")
        print(f"   Year: {result.metadata.get('citation_year', 'N/A')}")

    print("\n" + "-"*80)


def example_2_wrong_url():
    """Example 2: Citation with WRONG URL"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Citation with WRONG URL (will find correct one)")
    print("="*80)

    # Citation with wrong URL
    citation = """
    Johnson, A., Williams, B., & Davis, C. (2019). Machine learning in medical diagnosis:
    Current applications and future directions. Nature Medicine, 25(1), 44-56.
    https://example.com/wrong-url
    """

    validator = CitationURLCorrespondenceValidator()
    result = validator.validate(citation)

    print(f"\n⚠️  Valid: {result.is_valid}")
    print(f"⚠️  Credibility Score: {result.credibility_score:.1f}/100")
    print(f"⚠️  URL Matches Citation: {result.metadata.get('url_matches_citation', 'N/A')}")
    print(f"⚠️  Match Confidence: {result.metadata.get('match_confidence', 0.0):.2f}")

    if result.metadata.get('citation_title'):
        print(f"\n📋 Parsed Citation:")
        print(f"   Title: {result.metadata['citation_title']}")
        print(f"   Authors: {result.metadata.get('citation_authors', [])}")
        print(f"   Year: {result.metadata.get('citation_year', 'N/A')}")

    # Show correction
    if result.metadata.get('corrected_url'):
        print(f"\n✓ Correct URL Found:")
        print(f"   {result.metadata['corrected_url']}")

    if result.issues:
        print(f"\n❌ Issues Found:")
        for issue in result.issues:
            print(f"   - {issue.message}")

    if result.recommendations:
        print(f"\n💡 Recommendations:")
        for rec in result.recommendations:
            print(f"   - {rec}")

    print("\n" + "-"*80)


def example_3_broken_url():
    """Example 3: Citation with BROKEN URL"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Citation with BROKEN URL (404)")
    print("="*80)

    citation = """
    Brown, E., & Taylor, R. (2021). COVID-19 vaccine efficacy: A comprehensive review.
    The Lancet, 397(10275), 456-478. https://doi.org/10.1016/NONEXISTENT123
    """

    validator = CitationURLCorrespondenceValidator()
    result = validator.validate(citation)

    print(f"\n⚠️  Valid: {result.is_valid}")
    print(f"⚠️  Credibility Score: {result.credibility_score:.1f}/100")

    if result.metadata.get('citation_title'):
        print(f"\n📋 Parsed Citation:")
        print(f"   Title: {result.metadata['citation_title']}")
        print(f"   Authors: {result.metadata.get('citation_authors', [])}")

    if result.metadata.get('corrected_url'):
        print(f"\n✓ Correct URL Found:")
        print(f"   {result.metadata['corrected_url']}")
    else:
        print(f"\n❌ Could not find correct URL")

    if result.warnings:
        print(f"\n⚠️  Warnings:")
        for warning in result.warnings:
            print(f"   - {warning}")

    print("\n" + "-"*80)


def example_4_no_url():
    """Example 4: Citation WITHOUT URL (will suggest one)"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Citation WITHOUT URL (will find and suggest)")
    print("="*80)

    citation = """
    Anderson, K., Martinez, L., & Chen, W. (2022). Artificial intelligence in radiology:
    A systematic review. Radiology, 303(1), 23-45.
    """

    validator = CitationURLCorrespondenceValidator()
    result = validator.validate(citation)

    print(f"\n📝 Valid: {result.is_valid}")
    print(f"📝 Credibility Score: {result.credibility_score:.1f}/100")

    if result.metadata.get('citation_title'):
        print(f"\n📋 Parsed Citation:")
        print(f"   Title: {result.metadata['citation_title']}")
        print(f"   Authors: {result.metadata.get('citation_authors', [])}")
        print(f"   Year: {result.metadata.get('citation_year', 'N/A')}")

    if result.metadata.get('suggested_url'):
        print(f"\n✓ Suggested URL:")
        print(f"   {result.metadata['suggested_url']}")

    if result.warnings:
        print(f"\n⚠️  Warnings:")
        for warning in result.warnings:
            print(f"   - {warning}")

    if result.recommendations:
        print(f"\n💡 Recommendations:")
        for rec in result.recommendations:
            print(f"   - {rec}")

    print("\n" + "-"*80)


def example_5_parse_only():
    """Example 5: Just parse a citation without validation"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Parse Citation Metadata Only")
    print("="*80)

    citation = """
    Thompson, S. A., Rodriguez, M. F., & Wilson, J. P. (2023). Novel biomarkers
    for early detection of Alzheimer's disease: A longitudinal study.
    Journal of Neurology, Neurosurgery & Psychiatry, 94(5), 389-401.
    PMID: 36789012
    """

    validator = CitationURLCorrespondenceValidator()
    meta = validator.parse_apa_citation(citation)

    print(f"\n📋 Parsed Metadata:")
    print(f"   Title: {meta.title}")
    print(f"   Authors: {', '.join(meta.authors)}")
    print(f"   Year: {meta.year}")
    print(f"   Journal: {meta.journal}")
    print(f"   PMID: {meta.pmid}")
    print(f"   DOI: {meta.doi}")
    print(f"   URL: {meta.url}")

    print("\n" + "-"*80)


def check_mismatch_log():
    """Check if any mismatches were logged"""
    print("\n" + "="*80)
    print("CHECKING MISMATCH LOG")
    print("="*80)

    log_file = Path("reference_validation_mismatches.log")

    if log_file.exists():
        print(f"\n✓ Mismatch log exists: {log_file}")
        print(f"✓ Size: {log_file.stat().st_size} bytes")
        print(f"\nLast 5 entries:")
        print("-" * 80)

        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in lines[-50:]:  # Last 50 lines
                print(line.rstrip())
    else:
        print(f"\n✓ No mismatch log found (no mismatches detected)")


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("CITATION-URL CORRESPONDENCE VALIDATION EXAMPLES")
    print("="*80)
    print("\nThis validator:")
    print("  1. Parses APA citations (source of truth)")
    print("  2. Checks if URLs match the cited work")
    print("  3. Finds correct URLs when wrong/broken")
    print("  4. Logs all mismatches")

    try:
        # Example 5 first (parsing only - no network calls)
        example_5_parse_only()

        print("\n\n💡 The following examples make network calls to validate URLs...")
        print("   (This may take a few seconds)")

        # Examples with network calls
        example_1_correct_url()
        example_2_wrong_url()
        example_3_broken_url()
        example_4_no_url()

        # Check log
        check_mismatch_log()

        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print("""
The new Citation-URL Correspondence Validator:

✓ Prioritizes the APA citation text as source of truth
✓ Validates URLs actually correspond to cited works
✓ Finds correct URLs using multiple search APIs:
  - CrossRef (academic papers)
  - Semantic Scholar (multi-disciplinary)
  - OpenAlex (open access)
✓ Logs mismatches to: reference_validation_mismatches.log
✓ Provides detailed recommendations for corrections

Integration:
- Use directly: CitationURLCorrespondenceValidator().validate(citation)
- Or integrate with existing validation orchestrator
- Compatible with all medical agents in this repository

See: reference_validation/core/citation_url_correspondence_validator.py
        """)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
