# AGENTS.md — Development Guide

This file consolidates the essential rules, patterns, and agent architecture for
anyone (human or AI assistant) working on this repository.

> **Supersedes:** `README_FOR_LLM_DEVELOPMENT.md`, `README_IMPROVEMENTS.md`,
> `REFERENCE_VALIDATION_INTEGRATION.md` (those files are kept for historical
> reference but this file is the authoritative current source).

---

## Repository map

```
run_analysis.py              # Unified CLI + AgentOrchestrator (entry point)
router.py                    # LLM-based query router → agent dispatch
llm_integrations.py          # Provider adapters: Claude, OpenAI, Ollama, Grok
cost_tracker.py              # Per-phase cost tracking (class-based CostTracker)
pdf_generator.py             # Markdown → PDF via WeasyPrint
web_research/search.py       # Tavily / SerpAPI / DuckDuckGo client

document_parser/             # File → Markdown service (PDF priority; docx/txt/md/rtf/doc)
  service.py                 # parse_document() dispatcher + backend registry
  backends/                  # one backend per format (pdf, docx, text, rtf, doc)
  __main__.py                # CLI: python -m document_parser <file>

medical_procedure_analyzer/  # Procedure agent (organ-focused reasoning)
  medical_reasoning_agent.py
  medication_analyzer.py
  dspy_schemas.py            # Pydantic schemas for structured LLM output

medical_fact_checker/        # Fact-checker agent (multi-phase investigation)
  medical_fact_checker_agent.py   # Original (DSPy-based) implementation
  run_fact_checker.py

langchain_agents/            # LangChain implementations (default for router)
  base.py                    # LangChainAgentBase — shared LLM, web, audit
  factcheck_agent.py         # ← PRIMARY: multi-perspective Phase 4
  procedure_agent.py
  medication_agent.py

reference_validation/        # Citation URL correspondence validation
  core/citation_url_correspondence_validator.py
  cache/cache_manager.py     # SQLite cache, 30-day TTL

tests/                       # pytest suite (uv run python -m pytest tests/)
examples/router_queries.md   # Ready-to-paste example queries per agent
pending.md                   # Known gaps and planned work
```

---

## The four routable agents

| Router ID | Implementation class | What it does |
|-----------|---------------------|--------------|
| `medication_agent` | `LangChainMedicationAnalyzer` | Drug pharmacology, interactions, safety, monitoring |
| `procedure_agent` | `LangChainMedicalReasoningAgent` | Organ-by-organ procedure analysis, peri-op care |
| `diagnostic_agent` | `MedicalDiagnosticAgent` | Bayesian + LLM symptom-to-condition pipeline |
| `general_agent` | `LangChainMedicalFactChecker` | Open health/evidence questions |

All four agents extend `LangChainAgentBase` and share cost tracking, audit
logging, `_parse_json`, web research, and the **layered-report helpers** below.

---

## Layered reporting (shared across all agents)

`LangChainAgentBase` provides progressive-disclosure report construction reused
by every agent, so nothing from the analysis is silently discarded:

- **Layer 1 — Conclusions** (what readers value; first)
- **Layer 2 — Reasoning** (the logic behind the conclusions)
- **Layer 3 — Statistical Appendix** (precise numbers / evidence grading; last, optional)

Helpers (`langchain_agents/base.py`):
- `_layer_plain_language(body, framing=...)` — one LLM call producing plain-language
  Layers 1-2, preserving inline `[n]` citation markers.
- `_build_statistical_appendix(sections)` — assembles Layer 3 **deterministically**
  from labelled quantitative groups (numbers are never dropped/hallucinated).
- `_build_layered_report(conclusions_and_reasoning=..., appendix=...)` — returns
  `(patient, practitioner)`; the patient variant omits Layer 3 and points to the
  practitioner report; the practitioner variant appends it.
- `_verify_no_silent_loss(output, must_survive)` — logs + audits any key item that
  fails to survive into the layered output.

Each agent emits a **patient report** (Layers 1-2) and a **practitioner report**
(Layers 1-3). Critical safety content (e.g. medication black-box warnings,
diagnostic Bayesian posteriors) is placed in the deterministic appendix so it is
guaranteed present regardless of LLM phrasing.

---

## Fact-checker pipeline (current, LangChain implementation)

```
start_analysis(subject)
│
├── Phase 1  Conflict Scan          → official vs counter-narrative + references
│            user picks: Official / Independent / Both
│
├── Phase 2  Evidence Audit         → funding bias, methodology, recency
│            user picks: Dig / Proceed  (⚠️ Dig does nothing — see pending.md)
│
├── Phase 3  Synthesis              → biological_truth, industry_bias, grey_zone
│            user picks lens: M / N / B / A
│            M = Mainstream  N = Naturist  B = Biohacker  A = Balanced (default)
│
├── Phase 4  Three Parallel Perspective Agents  (ThreadPoolExecutor max_workers=3)
│   ├── Mainstream LLM call  → _PerspectiveOutput {findings, recs, key_insight,
│   │                            citations, statistical_details}
│   ├── Naturist   LLM call  → _PerspectiveOutput
│   └── Biohacker  LLM call  → _PerspectiveOutput
│   + Assembler LLM call → merged markdown with "Your Focus" summary at top
│   Total: 4 LLM calls in Phase 4
│   NOTE: perspective findings are passed to the assembler in FULL (no [:500]
│         truncation); length is bounded at generation instead. The structured
│         perspectives are also stored in PhaseResult.content["perspectives"] so
│         Phase 5 can build a lossless layered report.
│
└── Phase 5  Layered Assembly (progressive disclosure — nothing is discarded)
    Body + structured Phase 2 / Phase 4 content passed in (references split upstream)
    Produces TWO documents via one LLM call + deterministic appendix:
      • simplified_output    → patient report:  Layer 1 Conclusions → Layer 2 Reasoning
      • practitioner_layered → practitioner:    Layers 1-2 + Layer 3 Statistical Appendix
    Layer ordering: Conclusions (what users value) → Reasoning (the logic) →
      Statistical Appendix (effect sizes / CIs / p-values / evidence grading; last).
    The Statistical Appendix is built DETERMINISTICALLY in Python from
      statistical_details + Phase 2 fields, so numbers are never dropped/hallucinated.
    A verification guard logs (and audits) any perspective key_insight that fails
      to survive into the practitioner layer.
    Lens-aware framing (clinical / nature-first / optimization / balanced)
    References re-attached verbatim after Phase 5 output
```

**Reference caching:** Every `PhaseResult.references` list is populated with
`{"raw_citation": str}` dicts. `AgentOrchestrator._collect_validated_references()`
in `run_analysis.py` deduplicates by DOI/PMID/raw text and validates URLs via
`CitationURLCorrespondenceValidator` (backed by SQLite cache with 30-day TTL).

---

## Non-negotiable rules

### 1. Medical safety
- Every report output must include the hardcoded disclaimer via `_append_hardcoded_disclaimer()` in `run_analysis.py`
- Never claim the system provides medical advice or diagnosis
- Evidence quality must be labelled (Strong / Moderate / Limited / Poor)

### 2. Cost tracking — mandatory on every agent
Every agent must:
```python
from cost_tracker import track_cost, reset_tracking, CostTracker

class MyAgent:
    def __init__(self):
        self.cost_tracker = CostTracker()   # per-instance, not global
        self.total_token_usage = TokenUsage()

    def analyze(self, ...):
        reset_tracking()           # reset module-level tracker
        self.cost_tracker.reset()  # reset per-instance tracker
        ...
        # Sync at end:
        from cost_tracker import get_cost_summary as _ms
        self.cost_tracker._phase_costs = _ms()["phases"][:]

    @track_cost("Phase N: Name")   # decorator on every phase method
    def _phaseN(self, ...):
        response, token_usage = self.llm_provider.generate_response(...)
        if token_usage:
            self.total_token_usage.add(token_usage)
```

### 3. Structured output — use Pydantic, not regex
All LLM calls that need structured data must return JSON validated against a
Pydantic model. Use `self._parse_json(response)` then `Model.model_validate(parsed)`.
Fallback to an empty model on failure. Never use bare `re.search(r'\{.*\}', ...)`.

### 4. References — collect in every phase
Every `PhaseResult` has a `references: List[Dict]` field. Populate it. Minimum
format: `{"raw_citation": "Author (Year). Title. Journal. https://doi.org/..."}`.
For the fact-checker the assembler call includes all three perspectives' citations.

### 5. Disclaimers — hardcoded, not LLM-generated
Saves ~200–300 tokens per report and guarantees legal consistency.
```python
output_complete = self._append_hardcoded_disclaimer(output_with_refs)
```
The method in `run_analysis.py` checks for existing disclaimer before appending.

### 6. System prompts — include Grok overrides for Grok providers
`LangChainAgentBase._apply_provider_overrides()` in `langchain_agents/base.py`
injects additional instructions when `"grok"` is in the provider name.
This is automatic for all agents using `_call_llm()`.

---

## Adding a new agent — checklist

- [ ] Create `langchain_agents/{name}_agent.py` extending `LangChainAgentBase`
- [ ] Add Pydantic models for each phase output
- [ ] Use `@track_cost("Phase N: Name")` on every phase method
- [ ] Populate `PhaseResult.references` in every phase
- [ ] Add `self.cost_tracker = CostTracker()` in `__init__`
- [ ] Sync cost tracker at end of main analysis method
- [ ] Add `AgentSpec` entry in `router.py:sample_agents`
- [ ] Add dispatch branch in `router.py` REPL loop
- [ ] Add `run_{name}` method to `AgentOrchestrator` in `run_analysis.py`
- [ ] Add `_save_{name}_analysis` method to `AgentOrchestrator`
- [ ] Add tests in `tests/test_langchain_agents.py`
- [ ] Add example queries in `examples/router_queries.md`
- [ ] Update `pending.md` and this file

---

## Running the system

```bash
# Interactive router (default: LangChain, grok-4.3, web search ON)
uv run python router.py

# Attach a document as context for subsequent queries
# then inside the REPL: /file /path/to/report.pdf

# Disable web research
uv run python router.py --no-web-search

# Original DSPy-based implementation
uv run python router.py --implementation original

# Direct CLI (bypass router)
uv run python run_analysis.py factcheck --subject "Vitamin D" --llm claude-sonnet
uv run python run_analysis.py procedure --subject "Laparoscopic cholecystectomy"
uv run python run_analysis.py medication --subject "Metformin" --indication "Type 2 diabetes"
uv run python run_analysis.py diagnostic --subject "fatigue, weight gain, cold intolerance"

# Tests
uv run python -m pytest tests/ -q
uv run python -m pytest tests/test_langchain_agents.py -v
```

---

## LLM providers and current model IDs

| Provider key | Model | Pricing ($/1M tokens in/out) |
|---|---|---|
| `claude-sonnet` | `claude-sonnet-4-6` | $3 / $15 |
| `claude-opus` | `claude-opus-4-8` | $5 / $25 |
| `grok-4.3` | `grok-4.3` | $1.25 / $2.50 |
| `gemini-vertex` | `gemini-3.6-flash` **(default)** | ~$1.50 / $9 |
| `gemini-vertex` | `gemini-3.5-flash` | ~$1.50 / $9 |
| `gemini-vertex` | `gemini-2.5-pro` | ~$2 / $12 |
| `claude-vertex` | `claude-sonnet-4-6` (via Vertex) | $3 / $15 |
| `openai` | `gpt-4o` | $2.50 / $10 |
| `ollama` | `llama2:13b` | local |

Default routing model: `grok-4.3` (set in `router.py:DEFAULT_ROUTING_MODEL`).

> ⚠️ Old Grok models (`grok-4-1-fast-*`, `grok-code-fast`) retire **May 15 2026**.
> They are mapped to `grok-4.3` in `create_llm_manager()` for backwards compat.

**Gemini / Vertex AI setup** — Gemini 3.x models are served exclusively on the
Vertex **global endpoint** and require REST transport. Set the following in your
`.env`:
```
VERTEX_PROJECT=your-gcp-project-id
GLOBAL_SERVICE=true
GOOGLE_APPLICATION_CREDENTIALS=/path/to/adc.json  # or use gcloud ADC
```
`GLOBAL_SERVICE=true` makes both `GeminiVertexLLM` and `ClaudeVertexLLM` target
`location=global`, `api_endpoint=aiplatform.googleapis.com`, `api_transport=rest`.
Without it the regional endpoint is used (required for older Gemini 1.x models).

---

## Reference validation — how it actually works

The system has **two separate validation paths** (see `pending.md` item 7 for the
reconciliation task):

### Path A — Orchestrator level (active, writes to reports)
`AgentOrchestrator._collect_validated_references(session)` in `run_analysis.py`:
1. Iterates all `phase_result.references` across the session
2. Deduplicates by DOI / PMID / first 100 chars of raw citation
3. For each reference, calls `CitationURLCorrespondenceValidator`:
   - Parses APA citation → extracts title, authors, year
   - Fetches URL and checks if page content matches citation metadata
   - If mismatch, searches CrossRef / SemanticScholar / OpenAlex for correct URL
4. Splits into `kept` (URL validated) and `removed` (mismatch or broken)
5. Results cached in `_reference_validation_cache` keyed by `id(session)`
6. Written into `## 📚 References` and `## 🧹 Removed References` in summary reports

### Path B — Agent level (stored but not used)
When `enable_reference_validation=True`, agents call
`self.reference_validator.validate_analysis()` and store the result in
`session.validation_report`. This result is **never read by the orchestrator**.

### Cache
`reference_validation/cache/cache_manager.py` — SQLite, TTL 30 days.
Cache location: `./cache/reference_validation.db` (relative to CWD).
Mismatch log: `reference_validation_mismatches.log` (relative to CWD — see `pending.md` item 10).

---

## Output files per agent

All files written to `outputs/` with `{subject}_{type}_{timestamp}` naming.

| Agent | Files produced |
|-------|---------------|
| Procedure | `reasoning_trace.json`, `analysis_result.json`, `practitioner_report.md+pdf` (layered), `patient_report.md+pdf` (layered), `summary_report.md+pdf`, `cost_report.json` |
| Medication | `medication_analysis.json`, `practitioner_report.md+pdf` (layered), `patient_report.md+pdf` (layered), `medication_summary.md+pdf`, `medication_detailed.md+pdf`, `cost_report.json` |
| Fact-checker | `session.json`, `practitioner_report.md+pdf` (layered), `patient_report.md+pdf` (layered), `summary.md+pdf`, `cost_report.json` |
| Diagnostic | `diagnostic.json`, `diagnostic_report.md+pdf`, `practitioner_report.md+pdf` (layered), `patient_report.md+pdf` (layered), `cost_report` (in session) |

All agents produce **layered** patient (Layers 1-2) and practitioner (Layers 1-3)
reports via the shared helpers. All agents optionally produce `audit.json` when
`agent.audit_events` is present. Procedure / medication / diagnostic now collect
`references` and route them through the orchestrator URL-validation path.

---

## Deprecated / archived docs

| File | Why kept | Do not use for |
|------|----------|---------------|
| `README_FOR_LLM_DEVELOPMENT.md` | Historical patterns reference | Current model IDs, CLI flags, architecture |
| `README_IMPROVEMENTS.md` | Historical roadmap | Current status of features (many are wrong) |
| `REFERENCE_VALIDATION_INTEGRATION.md` | Reference validation API docs | Integration status (validation_report is not read by orchestrator) |

Use `pending.md` for current gaps and this file (`AGENTS.md`) for current patterns.
