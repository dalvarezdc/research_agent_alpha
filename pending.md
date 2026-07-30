# Pending Work — research_agent_alpha

> Last updated: 2026-05-08  
> Branch: fix/code-quality-review

---

## 🔴 High Impact — functional gaps visible to users

### ~~1. `diagnostic_agent` is a phantom~~ ✅ Fixed
`router.py` now dispatches `diagnostic_agent` to `orchestrator.run_diagnostic_analyzer()`,
which uses `MedicalDiagnosticAgent` (Bayesian + LLM pipeline in `medical_diagnostic_analyzer/`).
The CLI also accepts `uv run python run_analysis.py diagnostic --subject "..."`.

**Remaining gap:** ~~`MedicalDiagnosticAgent` does not use `CostTracker`, does not collect
references, and is not wired to the LangChain stack.~~ ✅ Fixed — `MedicalDiagnosticAgent`
now extends `LangChainAgentBase` (cost tracking via `@track_cost`, audit logging,
`_parse_json`, web research), collects `references`, and emits layered patient +
practitioner reports with a deterministic Bayesian-probability Statistical Appendix.

---

### 2. Web research ignored in original `MedicalFactChecker`
**Location:** `medical_fact_checker/medical_fact_checker_agent.py:106, 119`  
**Effort:** 1–2 days

`enable_web_research=True` stores the flag and does nothing. The `WebResearchClient` is never imported or called. Only the LangChain agents inject web context into prompts.

**Fix:** Import and initialise `WebResearchClient` in `MedicalFactChecker.__init__` when `enable_web_research=True`; call `_build_web_context()` at the start of `start_analysis()` and inject the result into Phase 1 and Phase 2 prompts.

---

### ~~3. Procedure and medication agents collect zero references~~ ✅ Fixed
**Location:** `langchain_agents/procedure_agent.py`, `langchain_agents/medication_agent.py`

Both agents now request APA-7 citations (with DOI/PMID/URL) in their structured
LLM calls, store them on `output.references`, and the orchestrator routes them
through `_collect_validated_references()` / `_append_references_section()` (which
now also handles objects exposing a flat `.references` list, not just
`phase_results`). Applies to the diagnostic agent as well.

---

### 4. `"Dig"` Phase 2 choice does nothing
**Location:** `medical_fact_checker_agent.py:200–204`, `langchain_agents/factcheck_agent.py:90–92`  
**Effort:** 1 day

Users are presented with "Dig / Proceed" after Phase 2. The choice is recorded but both paths run identical Phase 3 logic.

**Fix:** Either remove the "Dig" option from the interactive prompt, or implement a real "Dig" sub-phase that runs an additional mechanism-focused LLM call before Phase 3 when selected.

---

## 🟡 Moderate — silent bugs or dead code

### 5. `_append_cost_section()` fully implemented but never called
**Location:** `run_analysis.py:1019–1039`  
**Effort:** 30 minutes

The method builds a formatted cost breakdown section for markdown reports and is never invoked. Cost data goes only to a separate JSON file.

**Fix:** Call `_append_cost_section(summary, cost_summary)` in `_save_fact_check_analysis`, `_save_procedure_analysis`, and `_save_medication_analysis`.

---

### 6. DSPy signatures are dead class definitions
**Location:** `medical_fact_checker/medical_fact_checker_agent.py:63–93`  
**Effort:** 30 minutes

`ConflictScanSignature`, `EvidenceAnalysisSignature`, and `SynthesisSignature` are declared as `dspy.Signature` subclasses but never instantiated. The agent calls the LLM directly via `generate_response()`.

**Fix:** Either wire them up as proper DSPy `Predict`/`ChainOfThought` modules, or delete them.

---

### 7. Two parallel, non-communicating reference validation pipelines
**Location:** `run_analysis.py:142–204`, agent `__init__` with `enable_reference_validation=True`  
**Effort:** 1–2 days

Agent-level `enable_reference_validation` calls `self.reference_validator.validate_analysis()` and stores the result in `session.validation_report`. This result is **never read by the orchestrator** and never written to any output file. The orchestrator runs its own separate pipeline via `_collect_validated_references()`.

**Fix:** Either read `session.validation_report` in the orchestrator and include it in reports, or remove `enable_reference_validation` from agents and rely solely on the orchestrator pipeline.

---

### 8. `validate_batch(parallel=True)` silently does sequential validation
**Location:** `reference_validation/orchestrator.py:143`  
**Effort:** 1 day

```python
# parallel: Whether to validate in parallel (not implemented yet)
```

The only explicit "not implemented yet" comment in the codebase. Validation is always sequential.

**Fix:** Implement with `concurrent.futures.ThreadPoolExecutor`, mirroring the pattern in `langchain_agents/factcheck_agent.py:_phase4_generate_output`.

---

### 9. `validation_report` set on untyped field in procedure and medication agents
**Location:** `langchain_agents/procedure_agent.py:94`, `langchain_agents/medication_agent.py:98`  
**Effort:** 30 minutes

`MedicalOutput` and `MedicationOutput` dataclasses don't declare a `validation_report` field. Python allows it at runtime but it breaks typed deserialization and mypy.

**Fix:** Add `validation_report: Optional[Any] = None` to both `MedicalOutput` and `MedicationOutput`.

---

### 10. Mismatch log file hardcoded to current working directory
**Location:** `reference_validation/core/citation_url_correspondence_validator.py:98`  
**Effort:** 30 minutes

```python
handler = logging.FileHandler('reference_validation_mismatches.log')
```

Creates the log file in whatever directory the process runs from on every instantiation.

**Fix:** Make the path configurable (accept a `log_dir` parameter), default to `outputs/`.

---

### 11. SerpAPI normalization broken for string responses
**Location:** `web_research/search.py:79–83`  
**Effort:** 1 hour

`SerpAPIWrapper.run()` returns a plain string in some versions. `_normalize_results()` returns `[]` immediately for any string input (line 100). SerpAPI silently produces no results when `results()` is unavailable.

**Fix:** Parse the string response from `run()` as a fallback or log a warning and skip gracefully.

---

## 🔵 Test Coverage Gaps

### 12. `web_research/search.py` — zero tests
No tests for `_normalize_results()` edge cases (string response from SerpAPI, empty results, missing API keys, provider fallback ordering, deduplication).

### 13. `router.py` `route_agent()` — no mocked unit tests
`test_router_integration.py` requires live API keys. No mocked unit tests cover normalization, fallback-to-default behavior, the scalability hint (≥10 agents), or garbage LLM responses.

### 14. `run_analysis.py` report generators — untested
`_generate_medication_summary`, `_generate_medication_detailed_report`, `_generate_procedure_summary`, `_append_references_section`, and `_collect_validated_references` have no tests.

### 15. `citation_url_correspondence_validator.py` — no dedicated tests
The most complex component in the reference validation system (APA parsing, CrossRef/SemanticScholar/OpenAlex search, Jaccard title similarity) has no test file.

---

## 🟢 Recently delivered

### Cross-agent layered reporting, references & diagnostic migration
Extended the fact-checker's layered/lossless approach to every agent:

- **Shared helpers in `LangChainAgentBase`** (`_build_statistical_appendix`,
  `_build_layered_report`, `_verify_no_silent_loss`, `_layer_plain_language`) —
  progressive disclosure: Layer 1 Conclusions → Layer 2 Reasoning → Layer 3
  Statistical Appendix (deterministic, so numbers are never dropped/hallucinated).
  The fact-checker was refactored to consume them.
- **Procedure & medication agents** now emit layered patient + practitioner
  reports (one plain-language LLM call + deterministic appendix). Critical safety
  content (medication black-box warnings, serious adverse effects) lives in the
  appendix verbatim. Removed the lossy `[:300]`/`[:100]`/"top 3" truncations in
  the orchestrator's medication summary generator. Fixed `indication or "N/A"`.
- **References** collected by procedure / medication / diagnostic agents and
  routed through the orchestrator URL-validation path (see item 3).
- **Diagnostic agent migrated** onto `LangChainAgentBase` (see item 1): cost
  tracking, audit, robust JSON parsing, references, and a deterministic Bayesian
  Statistical Appendix so exact posteriors are guaranteed in the report.

### Layered, lossless fact-checker Phase 5 (`langchain_agents/factcheck_agent.py`)
Phase 5 no longer performs a lossy free-form simplification that stripped
statistical rigor and forced a rigid 5-section template. It now produces a
**layered, progressive-disclosure** report from the structured upstream data:
Layer 1 Conclusions → Layer 2 Reasoning → Layer 3 Statistical Appendix.

- **Patient report** (`session.final_output`): Layers 1–2 in plain language.
- **Practitioner report** (`session.practitioner_report`): Layers 1–2 + the
  full Statistical Appendix (effect sizes, CIs, p-values, evidence grading).
- The Statistical Appendix is assembled **deterministically in Python** from each
  perspective's new `_PerspectiveOutput.statistical_details` field plus Phase 2
  evidence fields — precise numbers are never dropped or hallucinated.
- The Phase 4 assembler no longer truncates perspective findings to `[:500]`;
  length is bounded at generation (~350 words per perspective) instead.
- A verification guard logs and audits any perspective `key_insight` that fails to
  survive into the practitioner layer.
- Inline `[n]` citation markers are preserved so claims stay traceable; the
  reference list is still re-attached verbatim by `start_analysis`.

**Possible follow-ups:** apply the same layering to the procedure/medication
agents; measure Flesch-Kincaid on the patient layer (ties into the existing
"Automatic reading level enforcement" item).

### Document parser service (`document_parser/`)
File → Markdown conversion service. **PDF priority** (pdfplumber + pypdf
fallback, tables → GFM); also `.docx`, `.txt`/`.md` (core) and `.rtf`/`.doc`
(optional `parsing-extras`). Best-effort `ParseResult` (never raises for
recoverable issues). Entry points: Python API (`parse_document`), CLI
(`python -m document_parser`), and REST (`POST /parse`). This is the foundational
"medical-parsing" service the `database/README.md` referenced.

**Follow-up specs that consume it:**
- ✅ Inject parsed document markdown into agent runs as context (delivered — router `/file` command).
- Populate the `patient_data` table via structured extraction from parsed markdown.
- Document Q&A / analysis over uploaded files.
- OCR / vision backend for scanned/image-only PDFs (drop-in backend).

---

## 🟤 Planned features not yet started

### Short term (1–2 months)

| Feature | Impact | Notes |
|---------|--------|-------|
| Pre-execution cost estimation | Medium | Show estimated cost before running a query. Cost tracking is post-hoc only. |
| Query result caching | Medium | Each run calls the LLM fresh. Redis or local SQLite for repeated queries. |
| Light vs. Deep analysis modes | Medium | Single fixed pipeline depth for all queries. Light = phases 1–3 only. |
| Prompt optimization / token reduction | Medium | Use cheaper models (Haiku) for non-critical stages. |
| Better error messages and progress indicators | Low | Long-running analyses give no feedback. |

### Medium term (3–6 months)

| Feature | Impact | Notes |
|---------|--------|-------|
| REST API / webhook | High | CLI only today. FastAPI backend for programmatic access. |
| Drug A vs. Drug B comparison mode | High | Run medication analyzer twice and diff the outputs. |
| Multi-model pipeline / ensemble | Medium | Noted in notes.md. `call_model()` is designed for this but no chaining logic exists. |
| Batch analysis | Medium | Same subject, multiple providers — compare outputs. |
| Medical terminology glossary | Medium | Phase 5 instructs LLM to simplify; no post-processing glossary injection. |
| Automatic reading level enforcement | Medium | Flesch-Kincaid target for patient reports — not yet measured or enforced. |

### Long term / future

| Feature | Notes |
|---------|-------|
| PubMed content-claim matching | URL accessibility is checked; whether the cited paper supports the claim is not. |
| CI/CD pipeline | No `.github/`, no test automation config beyond `pyproject.toml`. |
| Docker containerization | No `Dockerfile` or `docker-compose.yml`. |
| Real-time streaming (WebSocket) | Analysis produces output only at the end. |
| Multi-language support | English only. |
| MCP Server | Architecture discussed in old dev docs; no implementation. |
| EMR/EHR integration | Long-term enterprise feature. |
| Fine-tuned medical models | Use domain-specific models for higher accuracy. |

---

## Priority recommendation for next sessions

| Priority | Item | Effort |
|----------|------|--------|
| 1 | Fix `diagnostic_agent` — build actual diagnostic pipeline | 2–3 weeks |
| 2 | Wire `_append_cost_section()` — cost visible in reports | 30 min |
| 3 | Fix `validation_report` field declarations — type safety | 30 min |
| 4 | Add references to procedure/medication agents | 1–2 days |
| 5 | Implement `validate_batch(parallel=True)` | 1 day |
| 6 | Tests for `web_research/`, `route_agent()`, report generators | 1 week |
| 7 | Pre-execution cost estimation | 1 week |
| 8 | Light vs. Deep analysis modes | 1 week |
| 9 | REST API | 3–4 weeks |
