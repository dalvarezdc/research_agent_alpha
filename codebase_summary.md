# Codebase Summary (research_agent_alpha)

## What this repository is
This repo implements a small suite of “medical analysis agents” that can be run from a unified CLI.
The agents are LLM-backed (multi-provider) and generate structured artifacts (JSON + Markdown, optional PDF).

Repo conventions:
- Dependency management uses `uv` with `pyproject.toml` (see `pyproject.toml` and `uv.lock`).

Primary goals that show up throughout the code/documentation:
- Systematic, phase-based medical reasoning (procedure + medication) and critical “fact-checking” workflows.
- Transparency: reasoning traces, per-phase cost tracking, and (optionally) reference validation.
- Multi-LLM support (Claude/OpenAI/Ollama/xAI Grok) with fallbacks.

## High-level architecture

### Main CLI orchestrator
- `run_analysis.py` defines `AgentOrchestrator` and the CLI `main()`.
- It exposes three functioning analyses:
  - **Procedure analysis**: `procedure` → `medical_procedure_analyzer.MedicalReasoningAgent`
  - **Medication analysis**: `medication` → `medical_procedure_analyzer.medication_analyzer.MedicationAnalyzer`
  - **Fact-check analysis**: `factcheck` → `medical_fact_checker.MedicalFactChecker`
- It saves outputs under an output directory (default `outputs/`) as:
  - JSON results + reasoning traces
  - Markdown reports (often a summary + a “practitioner” report)
  - Optional PDFs via `pdf_generator.py`

### Shared LLM integration layer
- `llm_integrations.py` contains:
  - Provider adapters (`ClaudeLLM`, `OpenAILLM`, `OllamaLLM`, `XaiLLM`)
  - `LLMManager` (provider initialization + fallback selection)
  - `create_llm_manager()` (builds configs from string provider names)
  - `get_available_models()` and `call_model()` (single-model call interface used by routing)
- Agents generally use either:
  - `LLMManager.medical_analysis_with_fallback()` (stage-labelled “medical_analysis”) or
  - `LLMInterface.generate_response()` for direct prompt+system_prompt calls.

### Router module (optional)
- `router.py` implements `route_agent()` which uses `llm_integrations.call_model()` to select an `AgentSpec` id.
- The built-in `router.py` REPL defines 4 routing labels:
  - `medication_agent` → executes medication analysis
  - `procedure_agent` → executes procedure analysis
  - `diagnostic_agent` → executes fact-check analysis (alias; no dedicated diagnostic agent yet)
  - `general_agent` → executes fact-check analysis (alias)

### Agents

#### Procedure analyzer
- `medical_procedure_analyzer/medical_reasoning_agent.py`
- Core types: `MedicalInput`, `MedicalOutput`, `OrganAnalysis`, `ReasoningStep`, `ReasoningStage`.
- Pipeline shape:
  1. Input analysis
  2. Organ identification
  3. Evidence gathering
  4. Risk assessment
  5. Recommendation synthesis
  6. Critical evaluation (final output + practitioner report)
- Uses `cost_tracker.track_cost` decorators to print per-phase cost.
- Has some caching (e.g., `@lru_cache` in evidence gathering).

#### Medication analyzer
- `medical_procedure_analyzer/medication_analyzer.py`
- Extends `MedicalReasoningAgent` but runs a medication-specific pipeline:
  1. Pharmacology analysis
  2. Interaction analysis
  3. Safety profile
  4. Recommendations
  5. Monitoring requirements
- Attempts structured output via DSPy + Pydantic schemas in `medical_procedure_analyzer/dspy_schemas.py`.

#### Fact checker
- `medical_fact_checker/medical_fact_checker_agent.py`
- Phase-based “Independent Bio-Investigator” protocol:
  1. Conflict scan
  2. Evidence stress test
  3. Synthesis/menu
  4. Complex output generation
  5. Simplified output
- Extracts and stores references per phase (via text parsing) and can run reference validation.

### Reference validation subsystem (optional)
- `reference_validation/` implements `ReferenceValidator` with:
  - Citation parsing, URL checks, identifier verification (DOI/PMID/arXiv), scoring, caching.
- Intended to be used by agents to validate citations appended to reports (and you noted this will be integrated).

### Cost tracking
- `cost_tracker.py` provides `@track_cost` and a global accumulator.
- Token usage is accumulated by agents (sometimes) and model usage is recorded by providers.

## Current repo shape (key files)
- `README.md`, `USAGE.md`, `USAGE_EXAMPLES.md`: how to run.
- `README_FOR_LLM_DEVELOPMENT.md`: coding/architecture rules and expectations.
- `run_analysis.py`: orchestrator and output writing.
- `llm_integrations.py`: LLM providers + manager.
- `router.py`: LLM-based router (standalone module + REPL).
- `medical_procedure_analyzer/`: procedure + medication analyzers.
- `medical_fact_checker/`: fact checker agent.
- `reference_validation/`: citation validation package.
- `web_research/`: LangChain web search client (Tavily / SerpAPI / DuckDuckGo).
- `tests/`: integration-ish tests (many require API keys and live calls).

## Known sharp edges / inconsistencies
- Packaging mismatch: `pyproject.toml` defines `medical-agent = "medical_reasoning_agent:main"`, but there is no `medical_reasoning_agent.py` module in the repo root.
- Import hygiene: multiple modules modify `sys.path` at runtime to import from repo root.
- Provider naming drift: some defaults use `"claude"` while `create_llm_manager` expects strings like `"claude-sonnet"` / `"claude-opus"`.
- Large modules: `run_analysis.py`, `llm_integrations.py`, and the agent modules are long and mix concerns (I/O, prompt templates, parsing, formatting).

## Likely unused / deprecated (as of current wiring)
These files may still be valuable, but they are not currently part of the main execution path (CLI via `run_analysis.py` or the router REPL via `router.py`), or they contain broken example entrypoints.

- `medical_procedure_analyzer/report_generator.py`
  - Not used by `run_analysis.py` (orchestrator writes its own JSON/MD/PDF).
  - The `__main__` example imports `medical_reasoning_agent` (a module that does not exist), so running it directly is currently broken.
- `medical_procedure_analyzer/colored_logger.py`
  - Primarily used by the old web research module; if web research stays in `web_research/`, this is effectively unused too.
- `medical_procedure_analyzer/validation_scoring.py`
  - `validate_medical_output()` exists and is referenced in docs, but it’s not used by the orchestrator output path today.
- `check_llms.py`
  - Standalone convenience script (not imported by main flows).

---

# Potential improvements

## Performance improvements

### Reduce LLM round-trips
- Batch per-organ work when possible (e.g., request evidence + risk + recommendations in one structured response per organ, or multi-organ in one call).
- Add configurable “depth” modes (fast vs thorough) to reduce expensive phases when not needed.

### Improve caching strategy
- Cache by stable, normalized keys:
  - Current `base_name` is derived from user input; normalize and use that same normalized value for caches.
- Consider caching:
  - Organ identification results per `(procedure, details)`
  - Evidence summaries per `(procedure, details, organ)`
  - Interaction checks per `(medication, other_meds, conditions)`
- Persist caches (sqlite/json) for agent outputs similarly to reference validation’s cache.

### Parallelize safe substeps
- Evidence gathering across organs can run concurrently (threads/async), as long as the LLM provider supports it and rate-limits are respected.
- Reference validation batch checks and URL checks can be parallelized (with strict timeouts).

### Avoid unnecessary work during orchestration
- `run_analysis.py` always writes multiple artifacts; add flags to skip PDF generation or skip “practitioner” versions.

## Readability and code organization

### Eliminate `sys.path` manipulation
- Convert the repo into a clean package layout:
  - Move shared modules (`llm_integrations.py`, `cost_tracker.py`, `pdf_generator.py`, `router.py`) into a package (e.g., `medical_agent_core/`).
  - Update imports to be package-relative.
- This will also make testing/importing more reliable.

### Split large files by responsibility
- `run_analysis.py`:
  - Separate CLI parsing, orchestrator logic, and file writers.
  - Move markdown/PDF helpers into a `reporting/` module.
- `llm_integrations.py`:
  - Separate provider adapters into `llm/providers/*.py`.
  - Keep `LLMManager` + factories in `llm/manager.py`.
- Agent modules:
  - Move prompt templates into dedicated prompt files or a `prompts.py` module.
  - Move parsing/validation to dedicated helpers.

### Standardize naming and configuration
- Use one consistent set of provider identifiers end-to-end:
  - Prefer `claude-sonnet`/`claude-opus`/`openai`/`ollama`/`grok-...` everywhere.
- Centralize defaults and environment-variable mapping (models, timeouts, max tokens, temperature).

### Make outputs more structured
- Procedure analyzer currently converts complex dicts into string lists in places.
- Prefer structured JSON outputs and generate human-facing Markdown from those structures.

### Improve parsing robustness
- Several parsers extract JSON via regex like `r'\{.*\}'` / `r'\[(.*?)\]'`, which is fragile.
- Prefer:
  - “JSON only” responses with strict schemas
  - Streaming/partial parsing safeguards
  - `json.loads` with clear error surfaces and fallback logic

## Security improvements

### Prevent path traversal and unsafe filenames
- Output filenames are derived from user-supplied `--subject` / procedure / medication strings.
- Today, strings are only lowercased and spaces replaced; a subject like `../../evil` could escape `outputs/`.
- Fix by slugifying and enforcing a strict allowlist (e.g., `[a-z0-9_-]`) and/or using `pathlib.Path(output_dir) / safe_filename` and verifying it stays under `output_dir`.

### Harden network-facing validation
- `reference_validation` performs URL checks and identifier lookups.
- Add SSRF mitigations for URL checks:
  - Deny private IP ranges, localhost, and link-local addresses.
  - Enforce `http/https` only.
  - Set tight timeouts, size limits, and follow-redirect limits.
- Make “thorough” validation opt-in and clearly documented, since it increases network risk and latency.

### Avoid leaking secrets and sensitive content in logs
- Prompts, system prompts, and model responses may contain sensitive user input.
- Ensure logs never include API keys and consider redacting:
  - Patient context fields
  - Full prompts/responses unless explicitly enabled
- Use structured logging with levels; default to minimal logging.

### Safer PDF generation
- Markdown-to-HTML-to-PDF pipelines can be risky if user-provided text is embedded.
- Ensure the markdown conversion configuration doesn’t allow raw HTML injection (or explicitly sanitize HTML).

## Reliability and testability
- Add a “no-network” test mode:
  - Deterministic mock providers that implement `LLMInterface`.
  - Unit tests for parsing, file naming, and report generation.
- Separate “requires_api” tests from unit tests (you already have `requires_api` marker in `pyproject.toml`).
- Fix the console script entrypoint mismatch (either add `medical_reasoning_agent.py` with `main()`, or update `pyproject.toml` to point to `run_analysis:main` or similar).

## Suggested near-term refactor plan (small steps)
1. Fix output filename sanitization in `run_analysis.py` (security + stability).
2. Standardize provider identifiers and defaults (`llm_integrations.py` + CLI choices).
3. Replace `sys.path` hacks by packaging shared modules.
4. Introduce strict schema-first JSON generation/parsing for the highest-value phases.
5. Add mock LLM provider and true unit tests for parsing/output writing.
