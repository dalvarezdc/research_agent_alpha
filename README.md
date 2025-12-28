# Medical Analysis Agents

A multi-agent medical analysis system that produces structured, evidence-based reports for:
- Medical procedures (organ-focused reasoning)
- Medications (interactions, safety, recommendations)
- Medical fact checks (multi-phase investigation)

## Key Features

- Multi-agent CLI and Python API
- LangChain implementation by default with optional legacy path
- Multi-LLM support (Claude, OpenAI, Grok, Ollama)
- Evidence-based recommendations and debunked claims
- Reasoning traces and optional audit logs
- Markdown + JSON outputs, with optional PDF generation

## Installation

Requirements: Python 3.12+ and the UV package manager.

```bash
# Create virtual environment
uv venv --python 3.12
source .venv/bin/activate

# Install dependencies
uv sync

# Optional: dev dependencies (tests)
uv sync --extra dev
```

### API Keys

```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
export OPENAI_API_KEY="your-openai-key"
```

Or create a `.env` file in the project root.

## Quick Start

```bash
# List agents
uv run python run_analysis.py --list

# Check configured LLM providers
uv run python run_analysis.py --check-llms

# List supported model identifiers
uv run python run_analysis.py --models
```

### Procedure Analyzer

```bash
uv run python run_analysis.py procedure \
  --subject "MRI Scanner" \
  --details "With gadolinium contrast"
```

### Medication Analyzer

```bash
uv run python run_analysis.py medication \
  --subject "Warfarin" \
  --indication "Atrial Fibrillation" \
  --other-meds "Aspirin" "Amoxicillin" "Simvastatin"
```

### Medical Fact Checker

```bash
uv run python run_analysis.py factcheck \
  --subject "Vitamin D supplementation" \
  --context "optimal dosing"
```

### Choose LLM / Implementation

```bash
# Pick a provider
uv run python run_analysis.py medication --subject "Metformin" --llm grok-4-1-fast

# Switch implementation
uv run python run_analysis.py medication --subject "Metformin" --implementation original

# Enable web research (LangChain only)
uv run python run_analysis.py medication --subject "Metformin" --web-search
```

### Interactive Router

```bash
uv run python router.py
```

## Output Files

All outputs are written to `outputs/` by default (override with `--output-dir`).

### Procedure Analyzer

- `{procedure}_reasoning_trace_{timestamp}.json`
- `{procedure}_analysis_result_{timestamp}.json`
- `{procedure}_practitioner_report_{timestamp}.md`
- `{procedure}_summary_report_{timestamp}.md`
- `{procedure}_cost_report_{timestamp}.json`
- Optional PDFs and `{procedure}_audit_{timestamp}.json`

### Medication Analyzer

- `{medication}_medication_analysis_{timestamp}.json`
- `{medication}_practitioner_report_{timestamp}.md`
- `{medication}_medication_summary_{timestamp}.md`
- `{medication}_medication_detailed_{timestamp}.md`
- `{medication}_cost_report_{timestamp}.json`
- Optional PDFs and `{medication}_audit_{timestamp}.json`

### Medical Fact Checker

- `{subject}_session_{timestamp}.json`
- `{subject}_practitioner_report_{timestamp}.md`
- `{subject}_patient_report_{timestamp}.md`
- `{subject}_summary_{timestamp}.md`
- `{subject}_cost_report_{timestamp}.json`
- Optional PDFs and `{subject}_audit_{timestamp}.json`

## CLI Options

Common flags:
- `--subject` subject/procedure/medication/topic
- `--llm` provider (`claude-sonnet`, `claude-opus`, `openai`, `ollama`, `grok-4-1-fast`, `grok-4-1-code`, `grok-4-1-reasoning`)
- `--implementation` `langchain` or `original`
- `--output-dir` output directory (default `outputs/`)
- `--timeout` API timeout seconds (default 300)
- `--web-search` enable web research (LangChain only)

Procedure-specific:
- `--details` extra procedure details

Medication-specific:
- `--indication` primary indication
- `--other-meds` additional medications

Fact-check-specific:
- `--context` scope for investigation

## Python API Usage

### Procedure Analyzer

```python
from medical_procedure_analyzer import MedicalReasoningAgent, MedicalInput

agent = MedicalReasoningAgent(primary_llm_provider="claude-sonnet", enable_logging=True)
result = agent.analyze_medical_procedure(
    MedicalInput(
        procedure="MRI Scanner",
        details="With gadolinium contrast",
        objectives=("risks", "post-procedure care"),
    )
)
print(result.procedure_summary)
```

### Medication Analyzer

```python
from medical_procedure_analyzer.medication_analyzer import MedicationAnalyzer, MedicationInput

analyzer = MedicationAnalyzer(primary_llm_provider="claude-sonnet", enable_logging=True)
result = analyzer.analyze_medication(
    MedicationInput(
        medication_name="Warfarin",
        indication="Atrial Fibrillation",
        patient_medications=["Aspirin", "Amoxicillin"],
    )
)
print(result.drug_class)
```

### Medical Fact Checker

```python
from medical_fact_checker import MedicalFactChecker

checker = MedicalFactChecker(primary_llm_provider="claude-sonnet")
session = checker.start_analysis("Vitamin D supplementation")
print(session.final_output)
```

## Advanced Usage

```bash
# Increase timeout for complex analysis
uv run python run_analysis.py medication --subject "Digoxin" --timeout 600

# Custom output directory
uv run python run_analysis.py factcheck --subject "Coffee" --output-dir reports/coffee

# Batch runs
for procedure in "MRI Scanner" "CT Scan" "X-Ray"; do
  uv run python run_analysis.py procedure --subject "$procedure" --details "Standard protocol"
done
```

## Troubleshooting

- `No such file or directory: pytest` -> `uv sync --extra dev`
- API key errors -> verify `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`
- Provider failures -> try `--check-llms` and switch `--llm`

## Contributing

When adding new agents:
1. Follow the existing agent structure
2. Implement consistent output formats
3. Add tests
4. Update this README

## License

Part of the medical reasoning agent research project.

---

**Educational Use Only:** This repository is for research and education. It does not provide medical advice. Always consult qualified healthcare professionals.
