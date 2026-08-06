# Medical Diagnostic Analyzer

A hybrid symptom-to-condition pipeline that combines **LLM clinical common
sense** (free-form extraction + differential diagnosis) with structured
Pydantic outputs and layered patient/practitioner reports. Routed to via the
`diagnostic_agent` router ID.

There is **no fixed symptom or disease list**. The model reasons over whatever
presentation the user describes (knee pain, headache, fatigue clusters, etc.).

## What it does

Given a free-text description of symptoms, it:

1. Extracts structured symptoms (positive + denied + context) in free language.
2. Builds a ranked differential with estimated relative likelihoods and severity.
3. Identifies differentiating history points and recommended exams.
4. Optionally refines the differential interactively (extra symptoms / exam results).
5. Produces a structured report and suggests the next agent to route to.

## Quick start

```python
from medical_diagnostic_analyzer.diagnostic_agent import MedicalDiagnosticAgent

agent = MedicalDiagnosticAgent(
    primary_llm_provider="claude-sonnet",
    interactive=False,   # True enables follow-up Q&A in the terminal
)
result = agent.run_diagnostic_pipeline(
    "right knee pain and swelling for two weeks after a hike"
)

print(result["report"]["most_probable"])
print(result["report"]["top_5_candidates"])
print(result["report"]["suggested_agent"])  # medication_agent | procedure_agent
```

Typically invoked through `AgentOrchestrator.run_diagnostic_analyzer(...)` in
`run_analysis.py`.

## The 5-level pipeline

`MedicalDiagnosticAgent.run_diagnostic_pipeline(user_query)`:

| Level | Step | Engine |
|-------|------|--------|
| 1 | Free-form symptom extraction → `SymptomExtraction` | LLM |
| 2 | Differential assessment → `DifferentialAssessment` | LLM (clinical common sense) |
| 3 | Differentiating questions phrased for the patient | LLM |
| 4 | Iterative update (interactive only) | Re-runs Level 1–2 with new info |
| 5 | Final report → `DiagnosticReport` + layered reports | LLM + deterministic appendix |

Returns:

```python
{
  "extraction": {...},
  "probabilities": [{"name", "probability", "severity", "rationale", "id"}, ...],
  "report": {...},
  "references": [...],
  "patient_report": "...",
  "practitioner_report": "...",
  "assessment": {...},
}
```

## Module map

| File | Responsibility |
|------|----------------|
| `diagnostic_agent.py` | `MedicalDiagnosticAgent` — orchestrates the 5-level pipeline |
| `dspy_schemas.py` | Pydantic models: extraction, differential, final report |

## Constructor

```python
MedicalDiagnosticAgent(
    primary_llm_provider="claude-sonnet",
    fallback_providers=None,           # defaults to ["openai", "ollama"]
    enable_logging=True,
    interactive=True,
    enable_web_research=False,
)
```

The LLM is obtained through `create_llm_manager(...)`, so provider aliases and
fallbacks behave consistently with the rest of the system.

## Notes

- This is decision-support, **not** a diagnosis. Reports must carry the standard
  disclaimer applied by the orchestrator.
- Likelihoods are **relative estimates** among listed candidates (normalized to
  sum to 1.0), not calibrated population posteriors.
- Diagnostic runs persist to the database (best-effort) as `agent_type="diagnostic"`.
