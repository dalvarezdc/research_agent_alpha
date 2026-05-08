# Pending Work — research_agent_alpha

> Generated: 2026-05-08  
> Branch: fix/code-quality-review

---

## 🔴 High Impact — functional gaps visible to users

### 1. `diagnostic_agent` is a phantom
**Location:** `router.py:368`

Both `diagnostic_agent` and `general_agent` silently execute `run_fact_checker()`. There is no `DiagnosticAgent` or `LangChainDiagnosticAgent` class anywhere in the codebase. A user asking "why do I have joint pain?" gets the anti-mainstream narrative fact-checker pipeline, not a symptom-to-diagnosis reasoner.

**Fix:** Build a dedicated `LangChainDiagnosticAgent` or remap `diagnostic_agent` to a meaningfully different flow (e.g., structured differential diagnosis pipeline).

---

### 2. Web research ignored in original `MedicalFactChecker`
**Location:** `medical_fact_checker/medical_fact_checker_agent.py:106, 119`

`enable_web_research=True` stores the flag and does absolutely nothing. The `WebResearchClient` is never imported or called. Only the LangChain agents inject web context into prompts. Running `--implementation original` with web search enabled silently gets no web context.

**Fix:** Import and initialise `WebResearchClient` in `MedicalFactChecker.__init__` when `enable_web_research=True`; call `_build_web_context()` (or equivalent) at the start of `start_analysis()` and inject the result into the Phase 1 and Phase 2 prompts.

---

### 3. Procedure and medication agents collect zero references
**Location:** `langchain_agents/procedure_agent.py`, `langchain_agents/medication_agent.py`

The fact-checker pipeline collects citations in every `PhaseResult.references` list. Both the procedure and medication agents return empty reference lists. The orchestrator's `_collect_validated_references()` therefore always produces an empty validated set for these agent types, and the report's `## 📚 References` section is a hardcoded placeholder note for procedure output and absent for medication output.

**Fix:** Add citation extraction to Phase 3 (procedure summary) and Phase 2 (medication interaction analysis), storing APA-formatted strings in `PhaseResult.references`.

---

### 4. `"Dig"` Phase 2 choice does nothing
**Location:** `medical_fact_checker_agent.py:200–204`, `langchain_agents/factcheck_agent.py:90–92`

Users are presented with a "Dig / Proceed" choice after Phase 2. In both implementations the choice is recorded in `phase2_result.user_choice` but neither branches on it — Phase 3 runs identically regardless. Dead UX.

**Fix:** Either remove the "Dig" option from the interactive prompt, or implement a real "Dig" sub-phase that runs an additional mechanism-focused LLM call before Phase 3 when the user selects it.

---

## 🟡 Moderate — silent bugs or dead code

### 5. `_append_cost_section()` is fully implemented but never called
**Location:** `run_analysis.py:1019–1039`

The method constructs a formatted cost breakdown section for markdown reports. It is never invoked anywhere. Cost data goes only to a separate JSON file. Reports show no cost information.

**Fix:** Call `_append_cost_section(summary, cost_summary)` on the relevant markdown output strings in `_save_fact_check_analysis`, `_save_procedure_analysis`, and `_save_medication_analysis`.

---

### 6. DSPy signatures are dead class definitions
**Location:** `medical_fact_checker/medical_fact_checker_agent.py:63–93`

`ConflictScanSignature`, `EvidenceAnalysisSignature`, and `SynthesisSignature` are declared as `dspy.Signature` subclasses but the agent bypasses DSPy entirely and calls the LLM directly via `llm_manager.get_available_provider().generate_response()`. These classes are never instantiated.

**Fix:** Either wire these up as proper DSPy `Predict`/`ChainOfThought` modules, or delete them to eliminate confusion.

---

### 7. Two parallel, non-communicating reference validation pipelines
**Location:** `run_analysis.py:142–204`, agent `__init__` with `enable_reference_validation=True`

Agent-level `enable_reference_validation` calls `self.reference_validator.validate_analysis()` and stores the result in `session.validation_report` or `output.validation_report`. This result is **never read by the orchestrator** and never written to any output file. Separately, `AgentOrchestrator._collect_validated_references()` runs its own reference collection and URL validation pipeline and writes results into summary reports. The two paths share no data.

**Fix:** Either use the agent-level validator result in the orchestrator (read from `session.validation_report`), or remove `enable_reference_validation` from agents and rely solely on the orchestrator-level pipeline.

---

### 8. `validate_batch(parallel=True)` silently does sequential validation
**Location:** `reference_validation/orchestrator.py:143`

```python
# parallel: Whether to validate in parallel (not implemented yet)
```

The only explicit "not implemented yet" comment in the codebase. The `parallel` flag is accepted but validation is always sequential. For large reference lists this is the primary performance bottleneck in the reference validation system.

**Fix:** Implement with `concurrent.futures.ThreadPoolExecutor`, mirroring the pattern already used in `langchain_agents/factcheck_agent.py:_phase4_generate_output`.

---

### 9. `validation_report` set on untyped field in procedure and medication agents
**Location:** `langchain_agents/procedure_agent.py:94`, `langchain_agents/medication_agent.py:98`

```python
output.validation_report = self.reference_validator.validate_analysis(output)
```

`MedicalOutput` and `MedicationOutput` dataclasses do not declare a `validation_report` field. Python allows setting ad-hoc attributes on dataclasses at runtime without error, but this breaks typed deserialization, mypy analysis, and any consumer expecting the typed interface.

**Fix:** Add `validation_report: Optional[Any] = None` to both `MedicalOutput` (in `medical_reasoning_agent.py`) and `MedicationOutput` (in `medication_analyzer.py`).

---

### 10. Mismatch log file hardcoded to current working directory
**Location:** `reference_validation/core/citation_url_correspondence_validator.py:98`

```python
handler = logging.FileHandler('reference_validation_mismatches.log')
```

Creates `reference_validation_mismatches.log` in whatever directory the process runs from on every instantiation. Pollutes the project root (or any CWD) unconditionally.

**Fix:** Make the path configurable (accept a `log_dir` parameter), default to `outputs/` or the OS temp directory.

---

### 11. SerpAPI normalization broken for string responses
**Location:** `web_research/search.py:79–83`

`SerpAPIWrapper.run()` returns a plain string in some versions. `_normalize_results()` returns `[]` immediately for any string input (line 100). This means SerpAPI is silently non-functional when `results()` is not available on the wrapper.

**Fix:** Parse the string response from `run()` as a fallback (it is typically a comma-separated snippet text), or log a warning and skip gracefully.

---

## 🔵 Test Coverage Gaps

### 12. `web_research/search.py` — zero tests
No tests for `_normalize_results()` edge cases (string response from SerpAPI, empty results, missing API keys, provider fallback ordering, deduplication).

### 13. `router.py` `route_agent()` — no mocked unit tests
`test_router_integration.py` requires live API keys and returns early without them. There are no mocked unit tests covering normalization, fallback-to-default behavior, the scalability hint (≥10 agents), or the case where LLM returns garbage.

### 14. `run_analysis.py` report generators — untested
`_generate_medication_summary`, `_generate_medication_detailed_report`, `_generate_procedure_summary`, `_append_references_section`, and `_collect_validated_references` have no tests.

### 15. `citation_url_correspondence_validator.py` — no dedicated tests
The most complex component in the reference validation system (APA parsing, CrossRef/SemanticScholar/OpenAlex search, title similarity via Jaccard) has no test file. It is only covered indirectly via orchestrator-level tests.

---

## 🟤 Planned features not yet started

These are listed in `README_IMPROVEMENTS.md` as unchecked items and confirmed absent from the codebase:

| Feature | Notes |
|---------|-------|
| Pre-execution cost estimation | No implementation; cost tracking is post-hoc only |
| Query result caching for repeated queries | No session-level cache; each run calls the LLM fresh |
| Multi-model pipeline / ensemble | Noted in `notes.md`; `call_model()` designed to support chains but no chaining logic exists |
| Light vs. Deep analysis modes | Single fixed pipeline depth for all queries |
| REST API / webhook | No FastAPI/Flask layer; CLI only |
| CI/CD pipeline | No `.github/`, no `Makefile`, no test automation config beyond `pyproject.toml` |
| PubMed content-claim matching | URL accessibility is checked; whether the cited paper actually supports the claim is not verified |
| Glossary / reading-level enforcement | Phase 5 instructs the LLM to simplify; no post-processing enforcement or glossary injection |

---

## Priority recommendation for next sessions

1. **Fix `diagnostic_agent`** — high visibility, affects all routed queries for symptoms/conditions
2. **Wire `_append_cost_section()`** — trivial fix, immediate user value (cost visible in reports)
3. **Fix `validation_report` field declarations** — low effort, fixes type safety
4. **Add references to procedure/medication agents** — medium effort, improves report quality
5. **Implement `validate_batch(parallel=True)`** — medium effort, significant performance improvement for large reference lists
6. **Tests for `web_research/`, `router.py`, report generators** — medium effort, important before any production use
