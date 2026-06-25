# Medical Diagnostic Analyzer

A hybrid symptom-to-condition pipeline that combines **LLM natural-language
understanding** with a **Naive Bayes** probabilistic engine over a local symptom
database. Routed to via the `diagnostic_agent` router ID.

## What it does

Given a free-text description of symptoms, it:

1. Extracts structured symptoms (positive + denied) using an LLM.
2. Computes posterior probabilities for candidate conditions with Naive Bayes.
3. Identifies differentiating symptoms and recommended exams.
4. Optionally refines probabilities interactively (extra symptoms / exam
   results).
5. Produces a structured report and suggests the next agent to route to.

## Quick start

```python
from medical_diagnostic_analyzer.diagnostic_agent import MedicalDiagnosticAgent

agent = MedicalDiagnosticAgent(
    primary_llm_provider="claude-sonnet",
    interactive=False,   # True enables follow-up Q&A in the terminal
)
result = agent.run_diagnostic_pipeline(
    "fatigue, weight gain, and cold intolerance for a few weeks"
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
| 1 | Symptom extraction → `SymptomExtraction` | LLM (handles vague input with a clarification question) |
| 2 | Initial probability scoring | Naive Bayes |
| 3 | Differentiating symptoms + recommended exams | Naive Bayes |
| 4 | Iterative update (interactive only) | Naive Bayes (`update_with_exam_result`) |
| 5 | Final report → `DiagnosticReport` + routing | LLM |

Returns `{"extraction": ..., "probabilities": [...], "report": ...}`.

## Module map

| File | Responsibility |
|------|----------------|
| `diagnostic_agent.py` | `MedicalDiagnosticAgent` — orchestrates the 5-level pipeline; LLM extraction + report generation. |
| `bayesian_engine.py` | `NaiveBayesDiagnosticEngine` — probability math over the symptom database. |
| `dspy_schemas.py` | Pydantic models: `SymptomExtraction`, `DiagnosticReport`. |
| `symptom_database.json` | Local knowledge base: `diseases` (with per-symptom likelihoods + priors) and `exams`. |

## `NaiveBayesDiagnosticEngine`

Loads `symptom_database.json` (override with `database_path=`). Key methods:

```python
calculate_probabilities(reported_symptoms, negative_symptoms=None) -> list[dict]
get_differentiating_symptoms(top_candidates, reported_symptoms, limit=3) -> list[str]
get_recommended_exams(top_candidates) -> list[dict]
update_with_exam_result(current_probabilities, exam_id, result: bool) -> list[dict]
```

Applies Bayes' theorem:
`P(Disease | Symptoms) ∝ P(Symptoms | Disease) · P(Disease)`.

## Constructor

```python
MedicalDiagnosticAgent(
    primary_llm_provider="claude",     # any provider key from llm_integrations
    fallback_providers=None,           # defaults to ["openai"]
    enable_logging=True,
    interactive=True,
)
```

The LLM is obtained through `create_llm_manager(...)`, so provider aliases and
fallbacks behave consistently with the rest of the system.

## Extending the knowledge base

Add entries to `symptom_database.json`:

- **diseases** — each with a `prior`, a `symptoms` map (symptom → likelihood),
  and severity metadata.
- **exams** — each with an `id`, `name`, and the conditions it differentiates.

The engine auto-discovers all symptoms from the disease definitions
(`all_symptoms`), so new symptoms only need to appear in a disease entry.

## Notes

- This is decision-support, **not** a diagnosis. Reports must carry the standard
  disclaimer applied by the orchestrator.
- Diagnostic runs persist to the database (best-effort) as `agent_type="diagnostic"`.
