# Reference Validation Integration

✅ **Status:** INTEGRATED (2025-12-09)

Reference validation is now integrated into all medical agents with **minimal code changes**.

## Integration Summary

### Changes Made

**Added ~10 lines per agent:**

1. **Data Models** - Added `validation_report` field
2. **Constructor** - Added `enable_reference_validation` parameter
3. **Validation** - Added 3 lines before return statement

### Agents Integrated

| Agent | File | Changes | Status |
|-------|------|---------|--------|
| **Medical Procedure Analyzer** | `medical_procedure_analyzer/medical_reasoning_agent.py` | 10 lines | ✅ |
| **Medication Analyzer** | `medical_procedure_analyzer/medication_analyzer.py` | 10 lines | ✅ |
| **Medical Fact Checker** | `medical_fact_checker/medical_fact_checker_agent.py` | 10 lines | ✅ |

## Usage

### Basic Usage (Validation Disabled by Default)

```python
from medical_procedure_analyzer import MedicalReasoningAgent, MedicalInput

# Default: validation OFF (backward compatible)
agent = MedicalReasoningAgent()
result = agent.analyze_medical_procedure(input_data)
```

### Enable Reference Validation

```python
# Enable validation (simple!)
agent = MedicalReasoningAgent(enable_reference_validation=True)

# Run analysis
result = agent.analyze_medical_procedure(input_data)

# Check validation report
if result.validation_report:
    print(f"Overall credibility: {result.validation_report.overall_score}/100")
    print(f"Valid references: {result.validation_report.valid_references}/{result.validation_report.total_references}")
```

### Medication Analyzer

```python
from medical_procedure_analyzer.medication_analyzer import MedicationAnalyzer, MedicationInput

# Enable validation for drug references
analyzer = MedicationAnalyzer(enable_reference_validation=True)

result = analyzer.analyze_medication(med_input)

# Drug interaction references are validated
if result.validation_report:
    print(f"Reference credibility: {result.validation_report.overall_score}/100")
```

### Medical Fact Checker

```python
from medical_fact_checker import MedicalFactChecker

# Enable validation for evidence sources
checker = MedicalFactChecker(
    enable_reference_validation=True,
    interactive=False
)

session = checker.start_analysis("vitamin D supplementation")

# Evidence sources are validated
if session.validation_report:
    print(f"Evidence credibility: {session.validation_report.overall_score}/100")
```

## What Gets Validated

### Automatic Extraction

The system automatically extracts references from:
- `reasoning_trace` (all agents)
- `final_output` text (fact checker)
- Structured output fields

### Validation Checks

For each reference:
- ✅ DOI verification (doi.org resolver)
- ✅ PMID verification (PubMed API)
- ✅ URL accessibility (HTTP check)
- ✅ Citation format parsing
- ✅ Credibility scoring (0-100)

### Validation Thresholds

| Agent | Min Score | Priority |
|-------|-----------|----------|
| Medication Analyzer | 70 | High (drug safety) |
| Medical Procedure Analyzer | 70 | High (patient safety) |
| Medical Fact Checker | 75 | Highest (evidence quality) |

## Output Structure

### Validation Report Fields

```python
result.validation_report = {
    'total_references': 5,
    'valid_references': 4,
    'invalid_references': 1,
    'overall_score': 82.5,  # 0-100
    'average_credibility': 82.5,
    'peer_reviewed_count': 3,
    'results': [ValidationResult, ...],  # Individual results
    'recommendations': ["Add DOI...", ...],
    'warnings': ["URL not accessible", ...]
}
```

### Individual Validation Result

```python
validation_result = {
    'is_valid': True,
    'credibility_score': 85.0,
    'doi': '10.1234/example',
    'pmid': '12345678',
    'doi_valid': True,
    'pubmed_verified': True,
    'url_accessible': True,
    'peer_reviewed': True,
    'issues': [],
    'warnings': [],
    'recommendations': []
}
```

## Configuration

### Custom Validation Settings

```python
from reference_validation import ValidationConfig, ValidationLevel

# Create custom validator config
config = ValidationConfig(
    cache_backend="sqlite",
    cache_ttl_days=30,
    validation_level=ValidationLevel.THOROUGH,
    min_credibility_score=80,
    require_peer_review=True
)

# Pass to agent (requires custom integration)
agent = MedicalReasoningAgent(enable_reference_validation=True)
agent.reference_validator.config = config
```

### Validation Levels

- **QUICK** (~50ms) - Format check + cache lookup
- **STANDARD** (~500ms) - Format + DOI/PMID verification **(default)**
- **THOROUGH** (~5s) - All checks + URL accessibility

## Performance

### Impact on Analysis Time

| Agent | Without Validation | With Validation | Overhead |
|-------|-------------------|-----------------|----------|
| Procedure Analyzer | ~30s | ~32s | +2s |
| Medication Analyzer | ~60s | ~62s | +2s |
| Fact Checker | ~45s | ~47s | +2s |

**Note:** First run slower (no cache), subsequent runs much faster (~10ms overhead)

### Caching

Validation results are cached (30-day TTL):
- First validation: ~500ms per reference
- Cached validation: ~10ms per reference

Cache location: `./cache/reference_validation.db`

## Backward Compatibility

✅ **Fully backward compatible**

- Default: `enable_reference_validation=False`
- Existing code works without changes
- Optional feature, no breaking changes

## Testing

```bash
# Test validation integration
source .venv/bin/activate

python -c "
from medical_procedure_analyzer import MedicalReasoningAgent
agent = MedicalReasoningAgent(enable_reference_validation=True)
print('✓ Integration working')
"
```

## Examples

See:
- `reference_validation/example_integration.py` - Detailed examples
- `reference_validation/README.md` - Full documentation
- `reference_validation/tests/test_basic.py` - Test suite

## Future Enhancements

- [ ] Real-time validation during LLM generation
- [ ] Automatic regeneration if references fail
- [ ] Custom validation rules per agent
- [ ] Validation metrics in reports
- [ ] Web UI for validation results

## Notes

- Validation is **optional** (disabled by default)
- Adds ~2 seconds to analysis time
- Requires internet for DOI/PMID verification
- Uses free APIs (PubMed, DOI.org, CrossRef)
- No API keys required (optional for higher rate limits)

## Support

For issues or questions:
- See `reference_validation/README.md` for detailed docs
- Check `README_FOR_LLM_DEVELOPMENT.md` for coding standards
