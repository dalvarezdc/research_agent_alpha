# Medical Diagnostic Analyzer (Diagnostic Specialist)

Router ID: **`diagnostic_agent`**. Decision-support **differential diagnosis**
from free-text symptom presentations — not a final diagnosis, and **not** the
multi-perspective fact-checker (no Mainstream / Naturist / Biohacker assembly).

A hybrid symptom-to-condition pipeline that combines **LLM clinical common
sense** (free-form extraction + differential diagnosis) with structured
Pydantic outputs and layered patient/practitioner reports.

There is **no fixed symptom or disease list**. The model reasons over whatever
presentation the user describes (knee pain, headache, fatigue clusters, etc.).

## What it does

Given a free-text description of symptoms, it:

1. Extracts structured symptoms (positive + denied + context) in free language.
2. Builds a ranked differential with **relative** likelihoods (host-normalized)
   and independent severity (1–5), including cannot-miss items.
3. Identifies differentiating history points and recommended exams.
4. Optionally (interactive CLI only) asks one clarifying question and re-runs
   the differential with new answers. API runs with `interactive=False`.
5. Produces a structured report (`most_probable` vs `most_serious`, next steps,
   references) and a follow-up hint (`medication_agent` or `procedure_agent`).
6. Emits layered patient (conclusions → reasoning) and practitioner (same +
   deterministic likelihood Statistical Appendix) reports.

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
| 3 | One patient-friendly differentiating question | LLM (interactive only) |
| 4 | Iterative update from answers | Re-runs Level 1–2 (interactive only) |
| 5 | Final report → `DiagnosticReport` + layered reports | LLM + deterministic appendix |

If Level 2 fails or returns no candidates, the agent **fail-closes** to a single
safe candidate: *Undifferentiated presentation — clinician evaluation needed*
(severity 2; exam: in-person clinical assessment).

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
