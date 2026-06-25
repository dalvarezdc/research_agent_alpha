# Medical Procedure Analyzer

Organ-focused reasoning toolkit for medical procedures and medications. This is
the **original (DSPy-style)** implementation; the LangChain variants in
`langchain_agents/` are the router defaults, but both share the data types and
output structures defined here.

Routed to via the `procedure_agent` and `medication_agent` router IDs.

## What it does

- **Procedure analysis** — breaks a procedure down organ-by-organ: which organs
  are affected/at risk, evidence-based vs. investigational vs. debunked claims,
  risk levels, and peri-operative recommendations.
- **Medication analysis** — pharmacology, drug/food interactions, safety
  profile, contraindications, monitoring, and evidence-graded recommendations
  (the `MedicationAnalyzer` extends the procedure agent).

## Quick start

```python
from medical_procedure_analyzer import MedicalReasoningAgent, MedicalInput

agent = MedicalReasoningAgent(primary_llm_provider="claude-sonnet")

medical_input = MedicalInput(
    procedure="Laparoscopic cholecystectomy",
    details="Elective, standard adult patient",
    objectives=("identify risks", "affected organs", "post-procedure care"),
    patient_context="Standard adult patient",
)
output = agent.analyze_medical_procedure(medical_input)

print(output.procedure_summary)
for organ in output.organs_analyzed:
    print(organ.organ_name, organ.risk_level, organ.evidence_quality)
```

```python
from medical_procedure_analyzer.medication_analyzer import (
    MedicationAnalyzer, MedicationInput,
)

analyzer = MedicationAnalyzer(primary_llm_provider="claude-sonnet")
result = analyzer.analyze_medication(
    MedicationInput(medication_name="Metformin", indication="Type 2 diabetes")
)
print(result.drug_class, result.mechanism_of_action)
```

Both are usually invoked through `AgentOrchestrator` in `run_analysis.py`.

## Reasoning pipeline (`ReasoningStage`)

The procedure agent follows a broad → specific → critical pattern:

1. `INPUT_ANALYSIS`
2. `ORGAN_IDENTIFICATION`
3. `EVIDENCE_GATHERING`
4. `RISK_ASSESSMENT`
5. `RECOMMENDATION_SYNTHESIS`
6. `CRITICAL_EVALUATION`

Each step is captured as a `ReasoningStep` (stage, reasoning, output,
confidence, sources) and surfaced in `MedicalOutput.reasoning_trace`.

## Core data types

| Type | Purpose |
|------|---------|
| `MedicalInput` | Frozen/hashable procedure input (`procedure`, `details`, `objectives` tuple, `patient_context`). |
| `OrganAnalysis` | Per-organ result: affected/at-risk flags, `risk_level`, pathways, known/potential/debunked claims, `evidence_quality`. |
| `MedicalOutput` | Final output: summary, `organs_analyzed`, recommendations, research gaps, `confidence_score`, `reasoning_trace`, optional `practitioner_report`. |
| `MedicationInput` / `MedicationOutput` | Medication equivalents (with `Interaction`, `InteractionSeverity`, `InteractionType`). |

## Module map

| File | Responsibility |
|------|----------------|
| `medical_reasoning_agent.py` | `MedicalReasoningAgent` + core dataclasses and `ReasoningStage`. |
| `medication_analyzer.py` | `MedicationAnalyzer` (extends the reasoning agent) + medication types. |
| `input_validation.py` | `InputValidator`, `SecureMedicalInput`, `ValidationError` — sanitizes/validates user input. |
| `validation_scoring.py` | Output quality scoring: `MedicalKnowledgeValidator`, `CompletenessValidator`, `ReasoningValidator`, `QualityScorer`, and `validate_medical_output()`. |
| `dspy_schemas.py` | Pydantic schemas for structured LLM output (pharmacology, interactions, safety, references, recommendations, debunked claims). |
| `colored_logger.py` | `ColoredLogger` / `get_colored_logger()` for readable console logs. |
| `__init__.py` | Public exports (package version `2.0.0`). |

## Constructor

```python
MedicalReasoningAgent(
    primary_llm_provider="claude",     # any provider key from llm_integrations
    fallback_providers=None,
    enable_logging=True,
    enable_reference_validation=False, # store agent-level validation report
    enable_web_research=False,
)
```

`MedicationAnalyzer` shares the same constructor signature.

## Output validation & scoring

`validate_medical_output(output)` runs knowledge, completeness, and reasoning
validators, returning a `ValidationReport` with `ValidationIssue`s graded by
`SeverityLevel` and an aggregate quality score. Evidence is labelled
**strong / moderate / limited / poor**.

## Related

- `langchain_agents/` — the default LangChain implementations of these agents.
- `reference_validation/` — citation/URL validation.
- `run_analysis.py` — orchestration, report file generation, DB persistence.
- `web_research/` — optional web context when `enable_web_research=True`.

## Notes

- Output is decision-support, not medical advice; the orchestrator appends the
  standard disclaimer.
- Procedure/medication runs persist to the database (best-effort) as
  `agent_type="procedure"` / `agent_type="medication"`.
