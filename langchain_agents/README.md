# LangChain Agents

The **primary, default** agent implementations used by the router. Each agent
extends a shared base (`LangChainAgentBase`) that centralizes the LLM manager,
prompt rendering, auditing, web research, cost tracking, and provider-specific
prompt tuning.

## Agents

| Class | Router ID | Purpose |
|-------|-----------|---------|
| `LangChainMedicalFactChecker` | `general_agent` | Open health/evidence questions via a 5-phase, multi-perspective investigation |
| `LangChainMedicalReasoningAgent` | `procedure_agent` | Organ-by-organ medical procedure analysis |
| `LangChainMedicationAnalyzer` | `medication_agent` | Drug pharmacology, interactions, safety, monitoring |

```python
from langchain_agents import (
    LangChainMedicalFactChecker,
    LangChainMedicalReasoningAgent,
    LangChainMedicationAnalyzer,
)
```

All three default to `primary_llm_provider="claude-sonnet"` and are normally
constructed by `AgentOrchestrator` in `run_analysis.py`, not directly.

## Shared base: `LangChainAgentBase`

Configured via the `LangChainAgentConfig` dataclass:

```python
@dataclass
class LangChainAgentConfig:
    primary_llm_provider: str = "claude-sonnet"
    fallback_providers: list[str] = ["openai", "ollama"]
    enable_logging: bool = True
    enable_reference_validation: bool = False
    enable_audit: bool = True
    enable_web_research: bool = False
    web_research_providers: list[str] = ["tavily", "serpapi", "duckduckgo"]
    web_research_max_results: int = 5
```

What the base provides to every agent:

- **LLM manager** — builds `create_llm_manager(...)` and resolves an available
  provider; raises if none is available.
- **`_call_llm(system_prompt, user_prompt, audit_step=None, **kwargs)`** — the
  single entry point for model calls. Renders the prompt, calls the provider,
  accumulates `total_token_usage`, records an audit event, and mirrors the call
  to LangSmith when configured.
- **`_render_prompt(...)`** — builds a `ChatPromptTemplate` and applies provider
  overrides.
- **`_apply_provider_overrides(...)`** — when the provider name contains
  `"grok"`, appends Grok-specific quality instructions (no placeholders/`N/A`,
  require numeric detail, label limited evidence). Automatic for all agents.
- **`_parse_json(text)`** — robust JSON extraction (direct → strip code fence →
  regex), used to validate structured output against Pydantic models.
- **Web research** — when `enable_web_research=True`, lazily builds a
  `web_research.WebResearchClient` and exposes `_build_web_context(query)`.
- **Auditing** — `audit_events` list captures every prompt/response/token usage;
  surfaced as `audit.json` by the orchestrator. Optional LangSmith tracing via
  `LANGCHAIN_TRACING_V2` + `LANGCHAIN_API_KEY`.

## Fact checker — 5 phases

`LangChainMedicalFactChecker.start_analysis(subject, clarifying_info="")`:

| Phase | Method (decorated with `@track_cost`) | Output model |
|-------|----------------------------------------|--------------|
| 1 Conflict scan | `_phase1...` | `_Phase1Model` |
| 2 Evidence stress-test | `_phase2...` | `_Phase2Model` |
| 3 Synthesis & lens menu | `_phase3...` | `_Phase3Model` |
| 4 Multi-perspective output | `_phase4...` | `_Phase4PerspectivesModel` (Mainstream / Naturist / Biohacker via `ThreadPoolExecutor`, plus an assembler) |
| 5 Simplified output | `_phase5...` | — |

`PerspectiveLens` enum selects the framing (M / N / B / A). Each perspective
returns a `_PerspectiveOutput` (findings, recommendations, key insight,
citations).

## Procedure agent — 3 phases

`LangChainMedicalReasoningAgent.analyze_medical_procedure(medical_input)`:

1. **Organ identification** → `_OrganList`
2. **Organ analysis** (per organ) → `_OrganAnalysisModel`
3. **Summary** → `_ProcedureSummary`

Takes/returns the shared `MedicalInput` / `MedicalOutput` types from
`medical_procedure_analyzer`.

## Medication agent — single structured pass

`LangChainMedicationAnalyzer.analyze_medication(medication_input)` produces a
`_MedicationOutputModel` (with `_InteractionModel` entries), mapped to the shared
`MedicationOutput` type.

## Cross-cutting patterns

- **Cost tracking** — every phase method is wrapped with
  `@track_cost("Phase N: ...")`; token usage is accumulated on
  `total_token_usage`.
- **Structured output** — LLM responses are parsed with `_parse_json` then
  validated against Pydantic models; failures fall back to empty models.
- **References** — phase results populate a `references` list
  (`{"raw_citation": ...}`) that the orchestrator deduplicates and URL-validates.

## Related

- `run_analysis.py` — `AgentOrchestrator` constructs and runs these agents.
- `router.py` — maps router IDs to these classes.
- `web_research/` — the search client used when web research is enabled.
- `reference_validation/` — citation/URL validation pipeline.
