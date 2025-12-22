# Reference Validation Integration

✅ **Status:** UPDATED (2025-12-13)

## NEW: Citation-URL Correspondence Validation

**Priority:** The APA citation text is the source of truth.

### Problem Statement

The old approach validated that:
- DOIs/PMIDs exist in databases ✓
- URLs are accessible (return 200 status) ✓

But it **did not** validate that:
- ❌ The URL actually corresponds to the cited work
- ❌ The URL content matches the citation metadata
- ❌ Wrong URLs are corrected automatically

### The New Approach

The new `CitationURLCorrespondenceValidator` fixes this by:

1. **Parsing APA Citations** - Extract title, authors, year (source of truth)
2. **Validating URL Correspondence** - Check if URL content matches citation
3. **Finding Correct URLs** - Search for correct URL if wrong/broken
4. **Logging Mismatches** - Log all discrepancies to `reference_validation_mismatches.log`

---

## Quick Start

### Option 1: Use the New Validator Directly

```python
from reference_validation.core.citation_url_correspondence_validator import (
    CitationURLCorrespondenceValidator
)

# Create validator
validator = CitationURLCorrespondenceValidator()

# Validate a citation
citation = """
Smith, J., & Jones, M. (2020). Effects of vitamin D supplementation.
Journal of Medicine, 105(3), 123-145. https://doi.org/10.1210/example
"""

result = validator.validate(citation)

# Check results
print(f"URL Matches Citation: {result.metadata.get('url_matches_citation')}")
print(f"Match Confidence: {result.metadata.get('match_confidence'):.2f}")

if 'corrected_url' in result.metadata:
    print(f"Correct URL: {result.metadata['corrected_url']}")
```

### Option 2: Use Through Orchestrator (THOROUGH level)

```python
from reference_validation import ReferenceValidator
from reference_validation.models import ValidationLevel

# Enable THOROUGH validation (includes correspondence check)
validator = ReferenceValidator()

result = validator.validate_reference(
    citation,
    validation_level=ValidationLevel.THOROUGH
)

# Check correspondence
if 'url_matches_citation' in result.metadata:
    matches = result.metadata['url_matches_citation']
    confidence = result.metadata.get('match_confidence', 0.0)

    if not matches:
        print(f"⚠️ URL mismatch (confidence: {confidence:.2f})")
        if 'corrected_url' in result.metadata:
            print(f"Correct URL: {result.metadata['corrected_url']}")
```

### Option 3: Integration with Medical Agents

```python
from medical_procedure_analyzer import MedicalReasoningAgent

# Enable validation with THOROUGH level
agent = MedicalReasoningAgent(
    enable_reference_validation=True,
    # Validation level defaults to THOROUGH
)

# Run analysis
result = agent.analyze_medical_procedure(medical_input)

# Check validation report
if result.validation_report:
    for val_result in result.validation_report.results:
        if 'url_matches_citation' in val_result.metadata:
            if not val_result.metadata['url_matches_citation']:
                print(f"⚠️ Citation with wrong URL:")
                print(f"   Title: {val_result.metadata.get('citation_title')}")
                print(f"   Wrong URL: {val_result.url}")
                if 'corrected_url' in val_result.metadata:
                    print(f"   Correct URL: {val_result.metadata['corrected_url']}")
```

---

## How It Works

### Step 1: Parse APA Citation (Source of Truth)

```python
# Extract metadata from APA citation text
citation_meta = validator.parse_apa_citation(citation)

# Returns:
{
    'title': 'Effects of vitamin D supplementation on bone health',
    'authors': ['Smith, J.', 'Jones, M.'],
    'year': 2020,
    'journal': 'Journal of Clinical Endocrinology & Metabolism',
    'doi': '10.1210/clinem/dgz999',
    'url': 'https://doi.org/10.1210/clinem/dgz999'
}
```

### Step 2: Check URL Correspondence

```python
# Fetch URL content
# Extract title, authors, year from HTML
# Compare with citation metadata

correspondence = validator.check_url_correspondence(url, citation_meta)

# Returns:
{
    'matches': True/False,
    'confidence': 0.0-1.0,  # 0.7+ = match
    'found_title': 'Title extracted from URL',
    'found_authors': ['Authors from URL'],
    'mismatch_reasons': ['Title mismatch', ...]
}
```

### Step 3: Find Correct URL (if mismatch)

```python
# Searches multiple APIs in priority order:
# 1. DOI resolver (if DOI available)
# 2. PubMed (if PMID available)
# 3. CrossRef API (by title + authors)
# 4. Semantic Scholar API (by title)
# 5. OpenAlex API (by title)

correct_url = validator.find_correct_url(citation_meta)
# Returns: 'https://correct-url.com' or None
```

### Step 4: Log Mismatch

When URL doesn't match citation, details are logged to:
`reference_validation_mismatches.log`

```
2025-12-13 10:30:45 - MISMATCH
Citation Title: Effects of vitamin D supplementation on bone health
Citation Authors: Smith, J., Jones, M.
Citation Year: 2020
Provided URL: https://wrong-url.com
Match Confidence: 0.35
Found Title: Completely different paper
Mismatch Reasons: Title mismatch (confidence: 0.35); Author mismatch
---
```

---

## Validation Levels

### QUICK (Format Only)
- ✓ Parses citation format
- ✓ Extracts DOI/PMID/URL
- ✗ No correspondence check
- **Time:** ~50ms
- **Use case:** Quick format check

### STANDARD (Format + Verification)
- ✓ Parses citation format
- ✓ Verifies DOI/PMID exist
- ✓ Checks URL accessibility
- ✗ No correspondence check
- **Time:** ~500ms
- **Use case:** Default validation

### THOROUGH (Complete Validation) ⭐ **RECOMMENDED**
- ✓ Parses citation format
- ✓ Verifies DOI/PMID exist
- ✓ Checks URL accessibility
- ✓ **NEW:** Validates URL-citation correspondence
- ✓ **NEW:** Finds correct URL if mismatch
- ✓ **NEW:** Logs mismatches
- **Time:** ~2-5s (network calls)
- **Use case:** High-quality validation

---

## API Reference

### CitationURLCorrespondenceValidator

#### `parse_apa_citation(citation: str) -> CitationMetadata`

Parses APA citation to extract structured metadata.

```python
meta = validator.parse_apa_citation(citation)

# Returns CitationMetadata:
print(meta.title)      # Extracted title
print(meta.authors)    # List of authors
print(meta.year)       # Publication year
print(meta.doi)        # DOI if present
print(meta.pmid)       # PMID if present
print(meta.url)        # URL if present
```

#### `check_url_correspondence(url: str, citation_meta: CitationMetadata) -> URLCorrespondence`

Checks if URL actually corresponds to the cited work.

```python
correspondence = validator.check_url_correspondence(url, citation_meta)

print(correspondence.matches)           # True if URL matches
print(correspondence.confidence)        # 0.0-1.0 match confidence
print(correspondence.found_title)       # Title from URL
print(correspondence.mismatch_reasons)  # List of issues
```

#### `find_correct_url(citation_meta: CitationMetadata) -> Optional[str]`

Finds correct URL using multiple search APIs.

```python
correct_url = validator.find_correct_url(citation_meta)

if correct_url:
    print(f"Found correct URL: {correct_url}")
else:
    print("Could not find correct URL")
```

#### `validate(reference: str) -> ValidationResult`

Complete validation workflow.

```python
result = validator.validate(citation)

# Check metadata
print(result.metadata['citation_title'])
print(result.metadata['url_matches_citation'])
print(result.metadata.get('corrected_url'))

# Check issues
for issue in result.issues:
    print(f"{issue.severity}: {issue.message}")

# Check recommendations
for rec in result.recommendations:
    print(rec)
```

---

## Configuration

### Enable Correspondence Validation

Correspondence validation runs automatically in **THOROUGH** mode:

```python
from reference_validation import ReferenceValidator, ValidationConfig
from reference_validation.models import ValidationLevel

# Method 1: Set validation level
config = ValidationConfig(
    validation_level=ValidationLevel.THOROUGH
)
validator = ReferenceValidator(config)

# Method 2: Specify per validation
result = validator.validate_reference(
    citation,
    validation_level=ValidationLevel.THOROUGH
)
```

### Timeout Configuration

```python
config = ValidationConfig(
    timeout_seconds=15  # Increase for slow networks
)
```

### Logging Mismatch Details

Mismatches are automatically logged to:
- File: `reference_validation_mismatches.log`
- Level: WARNING
- Format: Structured with citation details, URL info, mismatch reasons

To view mismatches:
```bash
cat reference_validation_mismatches.log
```

---

## Integration with Medical Agents

### Medical Procedure Analyzer

```python
from medical_procedure_analyzer import MedicalReasoningAgent

agent = MedicalReasoningAgent(
    enable_reference_validation=True,
    # Uses THOROUGH validation by default
)

result = agent.analyze_medical_procedure(input_data)

# Check for URL mismatches
if result.validation_report:
    mismatches = [
        r for r in result.validation_report.results
        if not r.metadata.get('url_matches_citation', True)
    ]

    if mismatches:
        print(f"⚠️ Found {len(mismatches)} citations with wrong URLs")
```

### Medication Analyzer

```python
from medical_procedure_analyzer.medication_analyzer import MedicationAnalyzer

analyzer = MedicationAnalyzer(
    enable_reference_validation=True
)

result = analyzer.analyze_medication(med_input)

# Validation report includes correspondence checks
print(f"Overall credibility: {result.validation_report.overall_score}/100")
```

### Medical Fact Checker

```python
from medical_fact_checker import MedicalFactChecker

checker = MedicalFactChecker(
    enable_reference_validation=True,
    interactive=False
)

session = checker.start_analysis("topic")

# Validation runs on all extracted references
if session.validation_report:
    for result in session.validation_report.results:
        if 'corrected_url' in result.metadata:
            print(f"Corrected URL found for: {result.metadata['citation_title']}")
```

---

## Examples

### Example 1: Citation with Correct URL

```python
citation = """
Smith, J. (2020). Machine learning in medicine.
Nature, 580(1), 312-317. https://doi.org/10.1038/s41586-020-2649-2
"""

result = validator.validate(citation)
# result.metadata['url_matches_citation'] = True
# result.metadata['match_confidence'] = 0.95
```

### Example 2: Citation with Wrong URL

```python
citation = """
Jones, M. (2021). COVID-19 vaccine efficacy.
The Lancet, 397(1), 99-111. https://example.com/wrong-paper
"""

result = validator.validate(citation)
# result.metadata['url_matches_citation'] = False
# result.metadata['match_confidence'] = 0.20
# result.metadata['corrected_url'] = 'https://doi.org/10.1016/...'
# result.recommendations = ['Correct URL found: https://...']
```

### Example 3: Citation with Broken URL

```python
citation = """
Brown, E. (2019). Treatment guidelines.
BMJ, 365(1), 234-240. https://broken-link.com/404
"""

result = validator.validate(citation)
# result.metadata['url_matches_citation'] = False
# result.issues = [ValidationIssue('URL not accessible')]
# result.metadata['corrected_url'] = 'https://correct-url...'
```

### Example 4: Citation Without URL

```python
citation = """
Davis, C. (2022). Systematic review of treatments.
JAMA, 328(5), 450-465.
"""

result = validator.validate(citation)
# result.warnings = ['No URL provided in citation']
# result.metadata['suggested_url'] = 'https://doi.org/...'
# result.recommendations = ['Add URL: https://...']
```

---

## Performance

### Impact on Analysis Time

| Validation Level | Time Added | Network Calls |
|-----------------|------------|---------------|
| QUICK | +0ms | 0 |
| STANDARD | +500ms | 1-2 (DOI/PMID check) |
| THOROUGH | +2-5s | 3-5 (correspondence + search) |

### Caching

- ✓ Validation results cached for 30 days
- ✓ First validation: ~3-5s
- ✓ Subsequent validations: ~10ms
- ✓ Cache location: `./cache/reference_validation.db`

### Rate Limiting

- ✓ Automatic rate limiting for all APIs
- ✓ ~3 requests/second (PubMed without key)
- ✓ ~10 requests/second (PubMed with API key)
- ✓ No rate limits for CrossRef, Semantic Scholar, OpenAlex

---

## Testing

### Run Examples

```bash
# Run all correspondence validation examples
python reference_validation/example_citation_url_validation.py
```

### Check Mismatch Log

```bash
# View logged mismatches
cat reference_validation_mismatches.log

# Count mismatches
grep "MISMATCH" reference_validation_mismatches.log | wc -l
```

### Integration Test

```python
from reference_validation import ReferenceValidator
from reference_validation.models import ValidationLevel

validator = ReferenceValidator()

# Test with real citation
citation = """Your real citation here..."""

result = validator.validate_reference(
    citation,
    validation_level=ValidationLevel.THOROUGH
)

print(f"Valid: {result.is_valid}")
print(f"Credibility: {result.credibility_score:.1f}/100")
print(f"URL Matches: {result.metadata.get('url_matches_citation')}")
```

---

## Troubleshooting

### Issue: "Could not extract title from citation"

**Cause:** Citation format not recognized as APA

**Solution:** Ensure citation follows APA 7 format:
```
Author, A. (Year). Title of work. Journal Name, volume(issue), pages. URL
```

### Issue: "URL not accessible"

**Cause:** URL returns 404 or timeout

**Solution:** The validator will automatically search for correct URL. Check `result.metadata['corrected_url']` for suggestion.

### Issue: "Low match confidence"

**Cause:** URL content doesn't match citation

**Solution:**
1. Check mismatch log for details
2. Use suggested corrected URL
3. Verify citation metadata is correct

### Issue: "Validation too slow"

**Solutions:**
1. Use STANDARD instead of THOROUGH
2. Increase timeout: `ValidationConfig(timeout_seconds=20)`
3. Check network connectivity
4. Results are cached - subsequent runs faster

---

## Backward Compatibility

✅ **Fully backward compatible**

- Default: STANDARD validation (no correspondence check)
- Enable correspondence: Use THOROUGH level
- Existing code works without changes
- New features are opt-in

---

## Future Enhancements

- [ ] Support for citation styles beyond APA 7
- [ ] Machine learning for better title matching
- [ ] Bulk URL correction tool
- [ ] Integration with citation management tools
- [ ] Real-time validation during LLM generation
- [ ] Automated reference correction in reports

---

## Support

### Documentation

- **This file:** Integration guide
- **Code:** `reference_validation/core/citation_url_correspondence_validator.py`
- **Examples:** `reference_validation/example_citation_url_validation.py`
- **Dev Guide:** `README_FOR_LLM_DEVELOPMENT.md`

### Getting Help

```bash
# View detailed code documentation
python -c "
from reference_validation.core.citation_url_correspondence_validator import CitationURLCorrespondenceValidator
help(CitationURLCorrespondenceValidator)
"
```

### Reporting Issues

Check mismatch log for details:
```bash
cat reference_validation_mismatches.log
```

---

**Last Updated:** 2025-12-13
**Version:** 2.0.0 (NEW Citation-URL Correspondence Validation)

## Summary of Changes (v2.0.0)

**NEW FEATURES:**
- ✨ Citation-URL correspondence validation
- ✨ APA citation parsing (title, authors, year extraction)
- ✨ Automatic URL correction using multiple search APIs
- ✨ Mismatch logging to dedicated log file
- ✨ Confidence scoring for URL matches

**IMPROVEMENTS:**
- 🔧 THOROUGH validation now checks URL correspondence
- 🔧 Better error messages and recommendations
- 🔧 Support for CrossRef, Semantic Scholar, OpenAlex APIs

**PRIORITY CHANGE:**
- 📋 **APA citation text is now source of truth** (not URL)
- 📋 URLs are validated against citation metadata
- 📋 Wrong URLs are automatically corrected
