"""
Example: Integrating Reference Validation with Medical Agents
Shows how to add validation to existing agents.
"""

from reference_validation import (
    ReferenceValidator,
    ValidationConfig,
    ValidationLevel,
)


# ============================================================================
# Example 1: Basic Usage
# ============================================================================

def example_basic_usage():
    """Basic reference validation"""
    print("=" * 80)
    print("Example 1: Basic Usage")
    print("=" * 80)

    # Initialize validator
    validator = ReferenceValidator()

    # Test references
    references = [
        "Smith J, Doe A. (2020). COVID-19 treatment efficacy. N Engl J Med. DOI: 10.1056/NEJMoa2001282",
        "Jones B et al. (2021). Vaccine safety profile. Lancet. PMID: 33378609",
        "CDC Guidelines. https://www.cdc.gov/coronavirus/2019-ncov/index.html",
        "Unverifiable claim without any identifiers or URLs",
    ]

    print("\nValidating references...")
    report = validator.validate_batch(references, level=ValidationLevel.STANDARD)

    print(f"\n📊 Results:")
    print(f"   Total references: {report.total_references}")
    print(f"   Valid references: {report.valid_references}")
    print(f"   Overall score: {report.overall_score:.1f}/100")
    print(f"   Pass rate: {report.pass_rate:.1f}%")

    print(f"\n📝 Individual Results:")
    for i, result in enumerate(report.results, 1):
        print(f"\n   {i}. {result.citation[:60]}...")
        print(f"      Valid: {result.is_valid}")
        print(f"      Score: {result.credibility_score:.1f}")
        print(f"      DOI: {result.doi or 'None'}")
        print(f"      PMID: {result.pmid or 'None'}")
        if result.warnings:
            print(f"      ⚠️  Warnings: {', '.join(result.warnings[:2])}")

    if report.recommendations:
        print(f"\n💡 Recommendations:")
        for rec in report.recommendations:
            print(f"   - {rec}")


# ============================================================================
# Example 2: Integration with Existing Agent
# ============================================================================

def example_agent_integration():
    """Show how to integrate validation into an agent"""
    print("\n" + "=" * 80)
    print("Example 2: Agent Integration")
    print("=" * 80)

    # Configure validator with high standards for medical data
    config = ValidationConfig(
        cache_backend="memory",
        validation_level=ValidationLevel.STANDARD,
        min_credibility_score=75,  # High threshold
        require_peer_review=False,  # Would be True in production
        enable_logging=True
    )

    validator = ReferenceValidator(config)

    # Simulate agent output with references
    agent_output = """
    ## Analysis Results

    Doxycycline shows efficacy against various bacterial infections [1].
    The drug has a well-established safety profile [2].
    Recent studies suggest potential anti-inflammatory effects [3].

    ## References
    [1] Nelson ML, Levy SB. (2011). The history of tetracyclines. Ann NY Acad Sci. DOI: 10.1111/j.1749-6632.2011.06354.x
    [2] Agwuh KN, MacGowan A. (2006). Pharmacokinetics. Infection. PMID: 16896850
    [3] Website reference without proper citation
    """

    print("\n📄 Agent Output (with references):")
    print(agent_output[:200] + "...")

    # Extract and validate references
    print("\n🔍 Extracting references...")
    extracted_refs = validator.extract_references(agent_output)
    print(f"   Found {len(extracted_refs)} references")

    # Validate all extracted references
    citations = [ref.raw_text for ref in extracted_refs]
    if citations:
        print("\n✅ Validating extracted references...")
        report = validator.validate_batch(citations, level=ValidationLevel.STANDARD)

        print(f"\n   Overall credibility: {report.overall_score:.1f}/100")

        # Decision logic based on validation
        if report.overall_score >= 75:
            print("   ✅ All references meet quality standards")
        elif report.overall_score >= 60:
            print("   ⚠️  Some references need verification")
        else:
            print("   ❌ References do not meet quality standards")

        # Show which references passed/failed
        print(f"\n   Reference Quality:")
        for i, result in enumerate(report.results, 1):
            status = "✅" if result.is_valid and result.credibility_score >= 75 else "⚠️"
            print(f"   {status} [{i}] Score: {result.credibility_score:.0f}/100")


# ============================================================================
# Example 3: Real-Time Validation with Feedback
# ============================================================================

def example_realtime_validation():
    """Show real-time validation during LLM generation"""
    print("\n" + "=" * 80)
    print("Example 3: Real-Time Validation with Feedback")
    print("=" * 80)

    validator = ReferenceValidator(ValidationConfig(cache_backend="memory"))

    # Simulate LLM-generated response (first attempt)
    llm_response_v1 = """
    Treatment X is effective [unverified blog post].
    Study Y showed benefits [paper without identifiers].
    """

    print("\n📝 LLM Response (v1):")
    print(llm_response_v1)

    # Validate in real-time
    refs = validator.extract_references(llm_response_v1)
    citations = [ref.raw_text for ref in refs]

    if citations:
        validation = validator.validate_batch(citations, level=ValidationLevel.QUICK)

        print(f"\n🔍 Quick Validation:")
        print(f"   Score: {validation.overall_score:.1f}/100")

        # Provide feedback if score is low
        if validation.overall_score < 60:
            print(f"\n💬 Feedback to LLM:")
            print(f"   'References have low credibility. Please provide:")
            print(f"    - DOI or PMID for research papers")
            print(f"    - URLs from reliable sources")
            print(f"    - Peer-reviewed sources preferred'")

            # Simulate improved LLM response
            llm_response_v2 = """
            Treatment X shows efficacy in clinical trials. Smith et al. (2020). JAMA. DOI: 10.1001/jama.2020.1234
            Study Y demonstrated benefits. PMID: 12345678
            """

            print(f"\n📝 LLM Response (v2 - after feedback):")
            print(llm_response_v2)

            # Validate again
            refs_v2 = validator.extract_references(llm_response_v2)
            citations_v2 = [ref.raw_text for ref in refs_v2]

            if citations_v2:
                validation_v2 = validator.validate_batch(citations_v2, level=ValidationLevel.QUICK)
                print(f"\n🔍 Validation (v2):")
                print(f"   Score: {validation_v2.overall_score:.1f}/100 (improved!)")


# ============================================================================
# Example 4: Custom Validation Thresholds per Agent
# ============================================================================

def example_custom_thresholds():
    """Show different validation standards for different agents"""
    print("\n" + "=" * 80)
    print("Example 4: Custom Thresholds per Agent")
    print("=" * 80)

    test_citation = "Research paper. Jones et al. (2020). DOI: 10.1234/test"

    # Medication Analyzer - HIGH standards
    print("\n💊 Medication Analyzer (HIGH standards):")
    med_config = ValidationConfig(
        cache_backend="memory",
        min_credibility_score=85,  # Very high
        require_peer_review=True,
        validation_level=ValidationLevel.THOROUGH
    )
    med_validator = ReferenceValidator(med_config)
    result = med_validator.validate_reference(test_citation, validation_level=ValidationLevel.QUICK)
    print(f"   Score: {result.credibility_score:.1f}")
    print(f"   Required: ≥{med_config.min_credibility_score}")
    print(f"   Status: {'✅ Pass' if result.credibility_score >= med_config.min_credibility_score else '❌ Fail'}")

    # Procedure Analyzer - MEDIUM standards
    print("\n🏥 Procedure Analyzer (MEDIUM standards):")
    proc_config = ValidationConfig(
        cache_backend="memory",
        min_credibility_score=70,
        require_peer_review=False,
        validation_level=ValidationLevel.STANDARD
    )
    proc_validator = ReferenceValidator(proc_config)
    result = proc_validator.validate_reference(test_citation, validation_level=ValidationLevel.QUICK)
    print(f"   Score: {result.credibility_score:.1f}")
    print(f"   Required: ≥{proc_config.min_credibility_score}")
    print(f"   Status: {'✅ Pass' if result.credibility_score >= proc_config.min_credibility_score else '❌ Fail'}")

    # General Info - LOWER standards
    print("\n📚 General Information (LOWER standards):")
    gen_config = ValidationConfig(
        cache_backend="memory",
        min_credibility_score=50,
        require_peer_review=False,
        validation_level=ValidationLevel.QUICK
    )
    gen_validator = ReferenceValidator(gen_config)
    result = gen_validator.validate_reference(test_citation, validation_level=ValidationLevel.QUICK)
    print(f"   Score: {result.credibility_score:.1f}")
    print(f"   Required: ≥{gen_config.min_credibility_score}")
    print(f"   Status: {'✅ Pass' if result.credibility_score >= gen_config.min_credibility_score else '❌ Fail'}")


# ============================================================================
# Example 5: Cache Performance
# ============================================================================

def example_cache_performance():
    """Show cache performance benefits"""
    print("\n" + "=" * 80)
    print("Example 5: Cache Performance")
    print("=" * 80)

    import time

    validator = ReferenceValidator(ValidationConfig(cache_backend="memory"))

    citation = "Test paper. DOI: 10.1234/cache-test"

    # First validation (not cached)
    print("\n⏱️  First validation (uncached):")
    start = time.time()
    result1 = validator.validate_reference(citation, validation_level=ValidationLevel.QUICK)
    time1 = (time.time() - start) * 1000
    print(f"   Time: {time1:.1f}ms")
    print(f"   Cache hit: {result1.cache_hit}")

    # Second validation (cached)
    print("\n⏱️  Second validation (cached):")
    start = time.time()
    result2 = validator.validate_reference(citation, validation_level=ValidationLevel.QUICK)
    time2 = (time.time() - start) * 1000
    print(f"   Time: {time2:.1f}ms")
    print(f"   Cache hit: {result2.cache_hit}")

    speedup = time1 / time2 if time2 > 0 else float('inf')
    print(f"\n🚀 Speedup: {speedup:.1f}x faster with cache")

    # Show cache stats
    stats = validator.get_stats()
    print(f"\n📊 Cache Stats:")
    print(f"   Backend: {stats['cache_backend']}")
    print(f"   Size: {stats['cache_size']} entries")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "Reference Validation Integration Examples" + " " * 21 + "║")
    print("╚" + "=" * 78 + "╝")

    example_basic_usage()
    example_agent_integration()
    example_realtime_validation()
    example_custom_thresholds()
    example_cache_performance()

    print("\n" + "=" * 80)
    print("✅ All examples completed successfully!")
    print("=" * 80 + "\n")
