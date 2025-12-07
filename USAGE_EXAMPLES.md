# Usage Examples for Enhanced Medical Analysis System

This guide provides practical examples for using the enhanced medical analysis system, including the new medication analyzer with comprehensive interaction analysis.

## Table of Contents
1. [Medical Procedure Analyzer](#medical-procedure-analyzer)
2. [Medication Analyzer](#medication-analyzer-new)
3. [Medical Fact Checker](#medical-fact-checker)
4. [Advanced Usage](#advanced-usage)

---

## Medical Procedure Analyzer

### Basic Procedure Analysis

```bash
# Analyze MRI with contrast
python run_analysis.py procedure --subject "MRI Scanner" --details "With gadolinium contrast"

# Analyze CT scan
python run_analysis.py procedure --subject "CT Scan" --details "Abdominal scan with iodinated contrast"

# Analyze cardiac procedure
python run_analysis.py procedure --subject "Cardiac Catheterization" --details "Right heart catheterization"
```

### Output Files Generated
- `{procedure}_reasoning_trace_{timestamp}.json` - Complete reasoning steps
- `{procedure}_analysis_result_{timestamp}.json` - Structured analysis data
- `{procedure}_summary_report_{timestamp}.md` - Human-readable summary with **detailed recommendations including rationales, evidence levels, timing, and implementation guidance**

### What's New in Procedure Analysis
✅ **Comprehensive recommendations** with detailed rationales (2-3 sentences explaining mechanism)
✅ **Evidence levels** with specific sources (e.g., "Strong - Multiple RCTs including Smith et al. 2023")
✅ **Precise timing** and duration (e.g., "Begin 24h before procedure, continue for 48h after")
✅ **Implementation guidance** with step-by-step instructions
✅ **Expected outcomes** and monitoring requirements
✅ **Cost considerations** when relevant

---

## Medication Analyzer (NEW!)

### Basic Medication Analysis

```bash
# Analyze a single medication
python run_analysis.py medication --subject "Metformin"

# Analyze with indication
python run_analysis.py medication --subject "Lisinopril" --indication "Hypertension"

# Analyze with other medications (interaction check)
python run_analysis.py medication --subject "Warfarin" \
    --indication "Atrial Fibrillation" \
    --other-meds "Aspirin" "Amoxicillin" "Simvastatin"
```

### Advanced Medication Analysis

```bash
# Comprehensive analysis with multiple medications
python run_analysis.py medication \
    --subject "Metformin" \
    --indication "Type 2 Diabetes" \
    --other-meds "Lisinopril" "Atorvastatin" "Levothyroxine" \
    --llm claude \
    --timeout 600

# Quick analysis with OpenAI
python run_analysis.py medication \
    --subject "Sertraline" \
    --indication "Major Depressive Disorder" \
    --llm openai
```

### Medication Output Files

The medication analyzer generates three comprehensive reports:

1. **`{medication}_medication_analysis_{timestamp}.json`** - Complete structured data including:
   - Pharmacokinetics (ADME)
   - All interactions with details
   - Safety profile
   - Dosing recommendations
   - Monitoring requirements

2. **`{medication}_medication_summary_{timestamp}.md`** - Executive summary including:
   - Mechanism of action
   - Key safety information
   - Severe interactions (highlighted)
   - Top recommendations

3. **`{medication}_medication_detailed_{timestamp}.md`** - Comprehensive report including:
   - Full pharmacology
   - Complete interaction analysis
   - Detailed recommendations with rationales
   - Monitoring schedule
   - Warning signs

### What the Medication Analyzer Provides

#### 1. **Comprehensive Interaction Analysis**
- **Drug-Drug Interactions**: Severity levels (severe/moderate/minor), mechanisms, clinical effects, management
- **Drug-Food Interactions**: What to eat/avoid, timing with meals, absorption effects
- **Drug-Supplement Interactions**: Herbs, vitamins, minerals that interact
- **Environmental Factors**: Light sensitivity, temperature, activity restrictions

#### 2. **Detailed Safety Profile**
- Common adverse effects (with frequencies)
- Serious adverse effects requiring attention
- Black box warnings (if any)
- Contraindications with severity levels and alternatives
- Warning signs with action required

#### 3. **Evidence-Based Recommendations**
- **What TO DO**: Interventions with strong evidence
  - Detailed rationale (mechanism)
  - Evidence level with sources
  - Implementation steps
  - Expected outcomes
  - Monitoring requirements

- **Investigational Approaches**: Promising but limited evidence
  - Theoretical basis
  - Current evidence level
  - Limitations
  - Safety profile

- **What NOT TO DO**: Debunked claims
  - Why debunked
  - Evidence against
  - Why harmful
  - Common misconceptions

#### 4. **Complete Pharmacology**
- Drug class and mechanism of action
- ADME (Absorption, Distribution, Metabolism, Elimination)
- Half-life and steady-state timing
- CYP enzyme interactions
- Standard dosing and adjustments

#### 5. **Clinical Guidance**
- Approved indications
- Off-label uses (with evidence)
- Dose adjustments for:
  - Renal impairment
  - Hepatic impairment
  - Elderly patients
  - Special populations

---

## Medical Fact Checker

### Basic Fact Checking

```bash
# Investigate a health topic
python run_analysis.py factcheck --subject "Vitamin D supplementation"

# With specific context
python run_analysis.py factcheck \
    --subject "Intermittent fasting" \
    --context "effects on metabolic health"

# Investigate controversial topics
python run_analysis.py factcheck \
    --subject "Gluten sensitivity" \
    --context "non-celiac gluten sensitivity prevalence"
```

### Output Files
- `{subject}_session_{timestamp}.json` - Phase-by-phase analysis
- `{subject}_output_{timestamp}.md` - Final comprehensive output
- `{subject}_summary_{timestamp}.md` - Summary of findings

---

## Advanced Usage

### Using Different LLM Providers

```bash
# Use Claude (default, recommended)
python run_analysis.py medication --subject "Metformin" --llm claude

# Use OpenAI
python run_analysis.py procedure --subject "MRI" --llm openai

# Use Ollama (local)
python run_analysis.py factcheck --subject "Coffee" --llm ollama
```

### Adjusting Timeouts

```bash
# Short timeout for simple queries (2 minutes)
python run_analysis.py medication --subject "Aspirin" --timeout 120

# Standard timeout (5 minutes - default)
python run_analysis.py medication --subject "Warfarin" --timeout 300

# Extended timeout for complex analysis (10 minutes)
python run_analysis.py medication \
    --subject "Immunosuppressant" \
    --other-meds "Multiple medications" \
    --timeout 600
```

### Custom Output Directory

```bash
# Save to specific directory
python run_analysis.py medication \
    --subject "Metformin" \
    --output-dir "reports/diabetes_meds"

# Organize by date
python run_analysis.py medication \
    --subject "Lisinopril" \
    --output-dir "outputs/$(date +%Y-%m-%d)"
```

### List Available Agents

```bash
# See all available analysis agents
python run_analysis.py --list
```

Output:
```
================================================================================
Available Medical Analysis Agents
================================================================================

📌 procedure
   Name: Medical Procedure Analyzer
   Description: Analyzes medical procedures with organ-focused reasoning

📌 medication
   Name: Medication Analyzer
   Description: Comprehensive medication analysis with interactions and recommendations

📌 factcheck
   Name: Medical Fact Checker
   Description: Independent bio-investigator for health subjects
```

---

## Python API Usage

### Medical Procedure Analyzer

```python
from medical_procedure_analyzer.medical_reasoning_agent import (
    MedicalReasoningAgent,
    MedicalInput as ProcedureInput
)

# Initialize agent
agent = MedicalReasoningAgent(
    primary_llm_provider="claude",
    enable_logging=True
)

# Create input
procedure_input = ProcedureInput(
    procedure="MRI Scanner",
    details="With gadolinium contrast",
    objectives=("comprehensive_analysis",)
)

# Run analysis
result = agent.analyze_medical_procedure(procedure_input)

# Access detailed recommendations
for organ in result.organs_analyzed:
    print(f"\n{organ.organ_name.upper()}:")
    print(f"Risk Level: {organ.risk_level}")

    # Now includes full details with rationale, evidence, timing
    for rec in organ.known_recommendations:
        print(f"  ✓ {rec}")
```

### Medication Analyzer

```python
from medical_procedure_analyzer.medication_analyzer import (
    MedicationAnalyzer,
    MedicationInput
)

# Initialize analyzer
analyzer = MedicationAnalyzer(
    primary_llm_provider="claude",
    enable_logging=True
)

# Create input
med_input = MedicationInput(
    medication_name="Warfarin",
    indication="Atrial Fibrillation",
    patient_medications=["Aspirin", "Amoxicillin"]
)

# Run comprehensive analysis
result = analyzer.analyze_medication(med_input)

# Access comprehensive data
print(f"Drug Class: {result.drug_class}")
print(f"Mechanism: {result.mechanism_of_action}")

# Check interactions
for interaction in result.drug_interactions:
    if interaction.severity.value == "severe":
        print(f"\n⚠️ SEVERE INTERACTION: {interaction.interacting_agent}")
        print(f"   Effect: {interaction.clinical_effect}")
        print(f"   Management: {interaction.management}")

# Access detailed recommendations
for rec in result.evidence_based_recommendations:
    if isinstance(rec, dict):
        print(f"\n✓ {rec['intervention']}")
        print(f"  Rationale: {rec['rationale']}")
        print(f"  Evidence: {rec['evidence_level']}")
        print(f"  Implementation: {rec['implementation']}")

# Export to file
analyzer.export_medication_analysis(result, "warfarin_analysis.json")
```

### Generate Reports

```python
from medical_procedure_analyzer.report_generator import generate_reports

# Generate all report formats
report_files = generate_reports(
    analysis_result=result,
    output_dir="reports",
    base_filename="metformin_20240101"
)

# Access generated files
print(f"JSON: {report_files['json_report']}")
print(f"Detailed MD: {report_files['detailed_markdown']}")
print(f"Summary MD: {report_files['summary_markdown']}")
```

---

## Real-World Examples

### Example 1: Pre-Procedure Planning

```bash
# Analyze MRI for a patient with kidney concerns
python run_analysis.py procedure \
    --subject "MRI Scanner" \
    --details "With gadolinium contrast, patient has CKD stage 3" \
    --timeout 600
```

**Output includes:**
- Kidney-specific risks with detailed rationale
- Pre-procedure hydration protocol with timing
- Monitoring requirements (creatinine, eGFR)
- What medications to avoid (NSAIDs, metformin timing)
- Warning signs of contrast nephropathy

### Example 2: Polypharmacy Review

```bash
# Analyze potential interactions in elderly patient
python run_analysis.py medication \
    --subject "Digoxin" \
    --indication "Heart Failure" \
    --other-meds "Furosemide" "Potassium" "Levothyroxine" "Omeprazole" \
    --timeout 600
```

**Output includes:**
- All drug-drug interactions with severity
- Electrolyte considerations (potassium effects)
- Dosing adjustments for renal function
- Monitoring parameters (digoxin levels, potassium)
- Food interactions (fiber, dairy timing)
- Warning signs of toxicity

### Example 3: New Medication Research

```bash
# Research a medication you've been prescribed
python run_analysis.py medication \
    --subject "Empagliflozin" \
    --indication "Type 2 Diabetes" \
    --other-meds "Metformin" "Lisinopril"
```

**Output includes:**
- How it works (SGLT2 inhibition mechanism)
- What to expect (glucose lowering, weight loss)
- Important warnings (ketoacidosis risk, genital infections)
- When to take it (morning, with/without food)
- What to avoid (dehydration, certain situations)
- Monitoring needed (kidney function, ketones if sick)

---

## Tips for Best Results

### 1. **Be Specific**
- Good: "Metformin" with "--indication Type 2 Diabetes"
- Better: "Metformin" with multiple --other-meds for interaction check

### 2. **Use Appropriate Timeouts**
- Simple queries: 120-300 seconds
- Complex analysis with interactions: 600 seconds
- Multiple medications: 600-900 seconds

### 3. **Review All Output Files**
- JSON for programmatic access
- Summary for quick overview
- Detailed report for comprehensive information

### 4. **Check Interaction Severity**
- 🔴 SEVERE: Requires immediate attention or contraindicated
- 🟡 MODERATE: Requires monitoring or dose adjustment
- 🟢 MINOR: Usually manageable

### 5. **Verify Information**
- All analyses include evidence levels
- Check provided references
- Consult healthcare provider for medical decisions

---

## Troubleshooting

### Issue: Analysis Timing Out

```bash
# Increase timeout
python run_analysis.py medication --subject "Medication" --timeout 900
```

### Issue: Want More Detail

```bash
# Enable verbose logging (already default)
# Check the detailed report markdown file in outputs/
```

### Issue: LLM Provider Not Working

```bash
# Try fallback provider
python run_analysis.py medication --subject "Aspirin" --llm openai

# Check API keys in environment
echo $ANTHROPIC_API_KEY
echo $OPENAI_API_KEY
```

---

## Important Disclaimers

⚠️ **Medical Disclaimer:**
- These analyses are for educational and research purposes only
- Do not constitute medical advice
- Always consult qualified healthcare professionals
- Individual circumstances vary significantly
- Medical knowledge evolves - verify with current guidelines

⚠️ **Interaction Analysis:**
- Covers common and clinically significant interactions
- May not include all possible interactions
- Individual responses vary
- Timing and severity depend on many factors
- Always inform healthcare providers of all medications

---

## Getting Help

For issues or questions:
1. Check the [README.md](README.md) for setup instructions
2. Review the [README_FOR_LLM_DEVELOPMENT.md](README_FOR_LLM_DEVELOPMENT.md) for development guidelines
3. File an issue at: https://github.com/anthropics/claude-code/issues

---

**Last Updated:** 2025-12-07
**Version:** 2.0.0 (Enhanced with Medication Analyzer)
