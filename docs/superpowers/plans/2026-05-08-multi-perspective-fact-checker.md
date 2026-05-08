# Multi-Perspective MedicalFactChecker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the LangChain fact-checker's single Phase 4 LLM call with three parallel perspective agents (Mainstream, Naturist, Biohacker) that merge into a side-by-side report, with the user's chosen lens shaping the Phase 5 simplification, while ensuring all references are preserved through caching.

**Architecture:** `langchain_agents/factcheck_agent.py` is fully rewritten to add a `PerspectiveLens` enum, three new Pydantic models, a `_call_perspective()` helper that runs in a `ThreadPoolExecutor`, and a lens-aware Phase 5. References from all three perspectives are collected into `PhaseResult.references` so the existing `run_analysis.py` aggregation logic works unchanged. Phase 5 receives body and references as separate strings so citations are never lost during simplification.

**Tech Stack:** Python 3.12, LangChain, Pydantic v2, `concurrent.futures.ThreadPoolExecutor`, existing `LangChainAgentBase._call_llm` and `_parse_json`.

**Spec:** `docs/superpowers/specs/2026-05-08-multi-perspective-fact-checker.md`

---

## Files modified

| File | What changes |
|------|-------------|
| `langchain_agents/factcheck_agent.py` | Full rewrite of Phase 4 + Phase 5; new enum + models; parallel execution |
| `tests/test_langchain_agents.py` | Add tests for new Phase 4 and Phase 5 behaviour |

No other files change. `run_analysis.py`, `medical_fact_checker_agent.py`, and `CacheManager` are untouched.

---

## Baseline: understand the current file

Before touching code, confirm the current test baseline so we know what not to break.

```bash
uv run python -m pytest tests/test_langchain_agents.py -v --tb=short
```

Expected: 3 tests pass (`test_langchain_procedure_agent`, `test_langchain_medication_agent`, `test_langchain_fact_checker`).

---

### Task 1: Add `PerspectiveLens` enum and new Pydantic models

**Files:**
- Modify: `langchain_agents/factcheck_agent.py`
- Test: `tests/test_langchain_agents.py`

- [ ] **Step 1: Write a test for the new enum and models**

  Add to `tests/test_langchain_agents.py`:

  ```python
  def test_perspective_lens_enum():
      from langchain_agents.factcheck_agent import PerspectiveLens
      assert PerspectiveLens("M") == PerspectiveLens.MAINSTREAM
      assert PerspectiveLens("N") == PerspectiveLens.NATURIST
      assert PerspectiveLens("B") == PerspectiveLens.BIOHACKER
      assert PerspectiveLens("A") == PerspectiveLens.BALANCED


  def test_perspective_output_model_validates():
      from langchain_agents.factcheck_agent import _PerspectiveOutput
      out = _PerspectiveOutput(
          findings="Test findings",
          recommendations=["Rec 1", "Rec 2"],
          key_insight="Test insight",
          citations=["Author (2024). Title. Journal. https://doi.org/10.0/x"],
      )
      assert out.key_insight == "Test insight"
      assert len(out.citations) == 1


  def test_perspective_output_model_empty_fallback():
      from langchain_agents.factcheck_agent import _PerspectiveOutput
      out = _PerspectiveOutput(
          findings="Analysis unavailable",
          recommendations=[],
          key_insight="",
          citations=[],
      )
      assert out.findings == "Analysis unavailable"
  ```

- [ ] **Step 2: Run to confirm they fail**

  ```bash
  uv run python -m pytest tests/test_langchain_agents.py::test_perspective_lens_enum -v
  ```
  Expected: `ImportError: cannot import name 'PerspectiveLens'`

- [ ] **Step 3: Add the enum and models to `factcheck_agent.py`**

  After the existing `_Phase3Model` class definition (around line 43), add:

  ```python
  class PerspectiveLens(Enum):
      """User-chosen perspective lens for Phase 4 and Phase 5."""
      MAINSTREAM = "M"   # Evidence-based medicine and clinical guidelines
      NATURIST   = "N"   # Evolutionary biology and ancestral health
      BIOHACKER  = "B"   # Optimization, cutting-edge, n=1 protocols
      BALANCED   = "A"   # All perspectives weighted equally (default)


  class _PerspectiveOutput(BaseModel):
      """Output from one perspective agent in Phase 4."""
      findings: str
      recommendations: List[str] = Field(default_factory=list)
      key_insight: str = ""
      citations: List[str] = Field(default_factory=list)


  class _Phase4PerspectivesModel(BaseModel):
      """Assembled output from all three perspective agents."""
      mainstream: _PerspectiveOutput
      naturist: _PerspectiveOutput
      biohacker: _PerspectiveOutput
  ```

  Also add `from enum import Enum` to the imports at the top of the file if not already present (check — the file currently imports from `medical_fact_checker_agent` which has `OutputType` but `Enum` may not be imported directly in `factcheck_agent.py`).

  The full updated import block for `factcheck_agent.py`:
  ```python
  from __future__ import annotations

  from concurrent.futures import ThreadPoolExecutor, as_completed
  from datetime import datetime
  from enum import Enum
  from typing import Any, Dict, List, Optional

  from pydantic import BaseModel, Field

  from cost_tracker import print_cost_summary, reset_tracking, track_cost, CostTracker
  from medical_fact_checker.medical_fact_checker_agent import (
      AnalysisPhase,
      FactCheckSession,
      OutputType,
      PhaseResult,
  )

  from .base import LangChainAgentBase, LangChainAgentConfig
  ```

- [ ] **Step 4: Run tests to confirm they pass**

  ```bash
  uv run python -m pytest tests/test_langchain_agents.py::test_perspective_lens_enum tests/test_langchain_agents.py::test_perspective_output_model_validates tests/test_langchain_agents.py::test_perspective_output_model_empty_fallback -v
  ```
  Expected: 3 PASS.

- [ ] **Step 5: Run the full existing test suite to confirm no regressions**

  ```bash
  uv run python -m pytest tests/test_langchain_agents.py -v
  ```
  Expected: all tests still pass.

---

### Task 2: Replace the Phase 3 user prompt with the new lens picker

**Files:**
- Modify: `langchain_agents/factcheck_agent.py`
- Test: `tests/test_langchain_agents.py`

The current `start_analysis` sets `phase3.user_choice = "P"` in non-interactive mode. We replace this with `PerspectiveLens.BALANCED.value` (`"A"`). The interactive prompt changes from A/B/C/D/P to M/N/B/A.

- [ ] **Step 1: Write a test for the non-interactive default lens**

  Add to `tests/test_langchain_agents.py`:

  ```python
  def test_factchecker_noninteractive_uses_balanced_lens(monkeypatch):
      """Non-interactive mode must default to PerspectiveLens.BALANCED ('A')."""
      import langchain_agents.base as lc_base
      from langchain_agents import LangChainMedicalFactChecker

      monkeypatch.setattr(lc_base, "create_llm_manager", lambda *a, **kw: _DummyManager())

      agent = LangChainMedicalFactChecker(
          primary_llm_provider="claude-sonnet",
          interactive=False,
          enable_web_research=False,
      )

      # Patch start_analysis to only check phase3 user_choice is set to "A"
      captured = {}
      original_phase4 = agent._phase4_generate_output

      def fake_phase4(subject, synthesis, lens):
          captured["lens"] = lens
          # Return minimal PhaseResult to let start_analysis complete
          from medical_fact_checker.medical_fact_checker_agent import (
              AnalysisPhase, PhaseResult,
          )
          from datetime import datetime
          return PhaseResult(
              phase=AnalysisPhase.COMPLEX_OUTPUT,
              timestamp=datetime.now(),
              content={"output": "test report"},
              references=[],
          )

      monkeypatch.setattr(agent, "_phase4_generate_output", fake_phase4)

      # Also patch phases 1-3 and 5 to return minimal results
      from medical_fact_checker.medical_fact_checker_agent import (
          AnalysisPhase, PhaseResult,
      )
      from datetime import datetime
      from langchain_agents.factcheck_agent import PerspectiveLens

      dummy_phase1 = PhaseResult(
          phase=AnalysisPhase.CONFLICT_SCAN, timestamp=datetime.now(),
          content={"official_narrative": "", "counter_narrative": "", "key_conflicts": ""},
          references=[],
      )
      dummy_phase2 = PhaseResult(
          phase=AnalysisPhase.EVIDENCE_STRESS_TEST, timestamp=datetime.now(),
          content={"industry_funded_studies": "", "independent_research": "",
                   "methodology_quality": "", "anecdotal_signals": "", "time_weighted_evidence": ""},
          references=[],
      )
      dummy_phase3 = PhaseResult(
          phase=AnalysisPhase.SYNTHESIS_MENU, timestamp=datetime.now(),
          content={"biological_truth": "", "industry_bias": "", "grey_zone": ""},
          references=[],
      )
      dummy_phase5 = PhaseResult(
          phase=AnalysisPhase.SIMPLIFIED_OUTPUT, timestamp=datetime.now(),
          content={"simplified_output": "simple"},
          references=[],
      )

      monkeypatch.setattr(agent, "_phase1_conflict_scan", lambda *a, **kw: dummy_phase1)
      monkeypatch.setattr(agent, "_phase2_evidence_stress_test", lambda *a, **kw: dummy_phase2)
      monkeypatch.setattr(agent, "_phase3_synthesis_menu", lambda *a, **kw: dummy_phase3)
      monkeypatch.setattr(agent, "_phase5_simplify_output", lambda *a, **kw: dummy_phase5)

      agent.start_analysis("test subject")

      from langchain_agents.factcheck_agent import PerspectiveLens
      assert "lens" in captured, "Phase 4 was not called"
      assert captured["lens"] == PerspectiveLens.BALANCED
  ```

- [ ] **Step 2: Run test to confirm it fails**

  ```bash
  uv run python -m pytest tests/test_langchain_agents.py::test_factchecker_noninteractive_uses_balanced_lens -v
  ```
  Expected: FAIL — `_phase4_generate_output` currently takes `(subject, synthesis, output_type)` not `(subject, synthesis, lens)`.

- [ ] **Step 3: Update `start_analysis` in `LangChainMedicalFactChecker`**

  Replace the Phase 3 → Phase 4 transition. The full updated `start_analysis` method:

  ```python
  def start_analysis(self, subject: str, clarifying_info: str = "") -> FactCheckSession:
      reset_tracking()
      self.cost_tracker.reset()
      self.current_session = FactCheckSession(subject=subject)
      self.web_context = self._build_web_context(subject)

      phase1 = self._phase1_conflict_scan(subject, clarifying_info)
      self.current_session.phase_results.append(phase1)

      if self.interactive:
          phase1.user_choice = self._prompt_user_phase1()
      else:
          phase1.user_choice = "Both"

      phase2 = self._phase2_evidence_stress_test(subject, phase1.content, phase1.user_choice)
      self.current_session.phase_results.append(phase2)

      if self.interactive:
          phase2.user_choice = self._prompt_user_phase2()
      else:
          phase2.user_choice = "Proceed"

      phase3 = self._phase3_synthesis_menu(subject, phase1.content, phase2.content)
      self.current_session.phase_results.append(phase3)

      # Pick perspective lens
      if self.interactive:
          lens_str = self._prompt_user_lens()
      else:
          lens_str = PerspectiveLens.BALANCED.value  # "A"

      try:
          lens = PerspectiveLens(lens_str)
      except ValueError:
          lens = PerspectiveLens.BALANCED
      phase3.user_choice = lens.value

      # Phase 4: three parallel perspective agents
      phase4 = self._phase4_generate_output(subject, phase3.content, lens)
      self.current_session.phase_results.append(phase4)

      # Split body from references before Phase 5
      assembled = phase4.content.get("output", "")
      ref_separator = "\n## 📚 References\n"
      if ref_separator in assembled:
          body, refs_block = assembled.split(ref_separator, 1)
      else:
          body, refs_block = assembled, ""

      # Phase 5: simplify the body only, using chosen lens for framing
      self.current_session.practitioner_report = assembled
      phase5 = self._phase5_simplify_output(body, lens=lens)
      self.current_session.phase_results.append(phase5)

      # Reattach references verbatim
      simplified = phase5.content.get("simplified_output", body)
      if refs_block:
          simplified = simplified + ref_separator + refs_block
      self.current_session.final_output = simplified

      if self.enable_reference_validation and self.reference_validator:
          self.current_session.validation_report = self.reference_validator.validate_analysis(
              self.current_session
          )

      from cost_tracker import get_cost_summary as _module_summary
      self.cost_tracker._phase_costs = _module_summary()["phases"][:]
      self.cost_tracker.print_summary()
      return self.current_session
  ```

  Also add `_prompt_user_lens` method (interactive):
  ```python
  def _prompt_user_lens(self) -> str:
      print("\nPHASE 3 COMPLETE: Synthesis")
      print("Which perspective resonates most with you?\n")
      print("  [M] Mainstream   — Clinical guidelines and established evidence")
      print("  [N] Naturist     — Evolutionary biology and ancestral health")
      print("  [B] Biohacker    — Optimization, cutting-edge, n=1 protocols")
      print("  [A] All equal    — Balanced report, no preference\n")
      while True:
          choice = input("Your choice (M/N/B/A): ").strip().upper()
          if choice in ("M", "N", "B", "A"):
              return choice
          print("Invalid choice. Enter M, N, B, or A.")
  ```

  Remove the now-unused `_prompt_user_phase3` method.

- [ ] **Step 4: Run the lens test**

  ```bash
  uv run python -m pytest tests/test_langchain_agents.py::test_factchecker_noninteractive_uses_balanced_lens -v
  ```
  Expected: PASS.

- [ ] **Step 5: Run full test suite**

  ```bash
  uv run python -m pytest tests/test_langchain_agents.py -v
  ```
  Expected: all pass.

---

### Task 3: Implement `_call_perspective()` helper

**Files:**
- Modify: `langchain_agents/factcheck_agent.py`
- Test: `tests/test_langchain_agents.py`

This is the core new method: takes a perspective name and synthesis dict, calls the LLM, returns a `_PerspectiveOutput`.

- [ ] **Step 1: Write the test**

  Add to `tests/test_langchain_agents.py`:

  ```python
  def test_call_perspective_returns_perspective_output(monkeypatch):
      """_call_perspective must return a _PerspectiveOutput even when LLM returns garbage."""
      import langchain_agents.base as lc_base
      from langchain_agents import LangChainMedicalFactChecker
      from langchain_agents.factcheck_agent import _PerspectiveOutput, PerspectiveLens

      # LLM returns a valid JSON string
      class _GoodProvider:
          def generate_response(self, prompt, system_prompt=None):
              from llm_integrations import TokenUsage
              import json
              data = {
                  "findings": "Test findings for mainstream",
                  "recommendations": ["Rec 1", "Rec 2"],
                  "key_insight": "Take statins",
                  "citations": ["Smith (2024). Title. NEJM. https://doi.org/10.1/x"],
              }
              return json.dumps(data), TokenUsage()

      class _GoodManager:
          def get_available_provider(self): return _GoodProvider()
          def get_provider_direct(self): return _GoodProvider()

      monkeypatch.setattr(lc_base, "create_llm_manager", lambda *a, **kw: _GoodManager())

      agent = LangChainMedicalFactChecker(
          primary_llm_provider="claude-sonnet",
          interactive=False,
          enable_web_research=False,
      )

      synthesis = {"biological_truth": "Sugar causes inflammation", "industry_bias": "none", "grey_zone": "none"}
      result = agent._call_perspective("mainstream", "Sugar and cancer", synthesis, PerspectiveLens.BALANCED)

      assert isinstance(result, _PerspectiveOutput)
      assert result.findings == "Test findings for mainstream"
      assert "Rec 1" in result.recommendations
      assert result.key_insight == "Take statins"
      assert len(result.citations) == 1


  def test_call_perspective_fallback_on_bad_json(monkeypatch):
      """_call_perspective must return a fallback _PerspectiveOutput when LLM returns non-JSON."""
      import langchain_agents.base as lc_base
      from langchain_agents import LangChainMedicalFactChecker
      from langchain_agents.factcheck_agent import _PerspectiveOutput, PerspectiveLens
      from llm_integrations import TokenUsage

      class _BadProvider:
          def generate_response(self, prompt, system_prompt=None):
              return "This is not JSON at all.", TokenUsage()

      class _BadManager:
          def get_available_provider(self): return _BadProvider()
          def get_provider_direct(self): return _BadProvider()

      monkeypatch.setattr(lc_base, "create_llm_manager", lambda *a, **kw: _BadManager())

      agent = LangChainMedicalFactChecker(
          primary_llm_provider="claude-sonnet",
          interactive=False,
          enable_web_research=False,
      )

      result = agent._call_perspective("naturist", "Vitamin D", {}, PerspectiveLens.NATURIST)

      assert isinstance(result, _PerspectiveOutput)
      assert "unavailable" in result.findings.lower()
      assert result.citations == []
  ```

- [ ] **Step 2: Run tests to confirm they fail**

  ```bash
  uv run python -m pytest tests/test_langchain_agents.py::test_call_perspective_returns_perspective_output tests/test_langchain_agents.py::test_call_perspective_fallback_on_bad_json -v
  ```
  Expected: `AttributeError: 'LangChainMedicalFactChecker' object has no attribute '_call_perspective'`

- [ ] **Step 3: Implement `_call_perspective` in `factcheck_agent.py`**

  Add this method to `LangChainMedicalFactChecker`:

  ```python
  def _call_perspective(
      self,
      perspective: str,
      subject: str,
      synthesis: Dict[str, Any],
      lens: "PerspectiveLens",
  ) -> "_PerspectiveOutput":
      """
      Call the LLM for one perspective (mainstream / naturist / biohacker).
      Returns _PerspectiveOutput. Falls back to empty output on any failure.
      Thread-safe: uses self._call_llm which holds the GIL during network I/O.
      """
      _SYSTEM_PROMPTS = {
          "mainstream": (
              "You are a clinical researcher writing evidence-based medical analysis. "
              "Prioritize RCTs, Cochrane reviews, FDA/EMA guidance, and GRADE-A evidence. "
              "Third-person, objective tone. Cite DOI/PMID for every claim. "
              "Return ONLY valid JSON matching the schema provided."
          ),
          "naturist": (
              "You are an evolutionary medicine researcher. Prioritize ancestral biology, "
              "circadian alignment, whole-food interventions, and small independent studies. "
              "Use evolutionary logic as a tiebreaker when RCT evidence is mixed. "
              "Cite peer-reviewed support where available. "
              "Return ONLY valid JSON matching the schema provided."
          ),
          "biohacker": (
              "You are an optimization researcher. Prioritize recent cutting-edge findings, "
              "promising n=1 protocols, quantified self data, and emerging mechanisms even "
              "with limited RCT backing. Label evidence level explicitly (e.g. 'Limited RCT', "
              "'Anecdotal', 'Mechanistic'). "
              "Return ONLY valid JSON matching the schema provided."
          ),
      }

      system_prompt = _SYSTEM_PROMPTS.get(perspective, _SYSTEM_PROMPTS["mainstream"])

      user_prompt = """Analyze {subject} from the {perspective} perspective.

  Synthesis context:
  {synthesis}

  Web research context:
  {web_context}

  Return JSON with this exact schema:
  {schema}

  Requirements:
  - findings: 3-5 paragraphs covering evidence, mechanisms, and context
  - recommendations: 3-5 concrete, actionable items
  - key_insight: single sentence capturing the most important takeaway
  - citations: 5-10 APA 7 references, each MUST include a DOI, PMID, or direct URL
  """

      try:
          response = self._call_llm(
              system_prompt,
              user_prompt,
              audit_step=f"factcheck_phase4_{perspective}",
              subject=subject,
              perspective=perspective,
              synthesis=synthesis,
              web_context=self.web_context or "None",
              schema=_PerspectiveOutput.model_json_schema(),
          )
          parsed = self._parse_json(response)
          if isinstance(parsed, dict):
              try:
                  return _PerspectiveOutput.model_validate(parsed)
              except Exception:
                  pass
      except Exception as exc:
          import logging
          logging.getLogger(__name__).warning(
              f"Perspective '{perspective}' LLM call failed: {exc}"
          )

      # Fallback: return empty but valid output
      return _PerspectiveOutput(
          findings=f"Analysis unavailable for {perspective} perspective.",
          recommendations=[],
          key_insight="",
          citations=[],
      )
  ```

- [ ] **Step 4: Run perspective tests**

  ```bash
  uv run python -m pytest tests/test_langchain_agents.py::test_call_perspective_returns_perspective_output tests/test_langchain_agents.py::test_call_perspective_fallback_on_bad_json -v
  ```
  Expected: 2 PASS.

- [ ] **Step 5: Run full test suite**

  ```bash
  uv run python -m pytest tests/test_langchain_agents.py -v
  ```

---

### Task 4: Implement the new `_phase4_generate_output` with parallel execution

**Files:**
- Modify: `langchain_agents/factcheck_agent.py`
- Test: `tests/test_langchain_agents.py`

This replaces the existing `_phase4_generate_output` with one that runs three perspective calls in parallel and then assembles the report.

- [ ] **Step 1: Write the test**

  Add to `tests/test_langchain_agents.py`:

  ```python
  def test_phase4_generates_three_perspective_report(monkeypatch):
      """Phase 4 must produce a report containing all three perspective sections."""
      import json
      import langchain_agents.base as lc_base
      from langchain_agents import LangChainMedicalFactChecker
      from langchain_agents.factcheck_agent import PerspectiveLens
      from llm_integrations import TokenUsage

      call_count = {"n": 0}

      class _CountingProvider:
          def generate_response(self, prompt, system_prompt=None):
              call_count["n"] += 1
              # Return valid JSON for perspective calls
              if "schema" in prompt and "findings" in prompt:
                  data = {
                      "findings": f"Findings for call {call_count['n']}",
                      "recommendations": ["Rec A"],
                      "key_insight": f"Insight {call_count['n']}",
                      "citations": [f"Author{call_count['n']} (2024). Title. J. https://doi.org/10.1/x"],
                  }
                  return json.dumps(data), TokenUsage()
              # Assembler call — return markdown
              return (
                  "# Report\n\n"
                  "## 🏥 Mainstream View\nMainstream findings.\n\n"
                  "## 🌿 Naturist View\nNaturist findings.\n\n"
                  "## 🚀 Biohacker View\nBiohacker findings.\n\n"
                  "## 📚 References\n[1] Author (2024). https://doi.org/10.1/x"
              ), TokenUsage()

      class _CountingManager:
          def get_available_provider(self): return _CountingProvider()
          def get_provider_direct(self): return _CountingProvider()

      monkeypatch.setattr(lc_base, "create_llm_manager", lambda *a, **kw: _CountingManager())

      agent = LangChainMedicalFactChecker(
          primary_llm_provider="claude-sonnet",
          interactive=False,
          enable_web_research=False,
      )

      from medical_fact_checker.medical_fact_checker_agent import AnalysisPhase, PhaseResult
      from datetime import datetime

      synthesis = {"biological_truth": "test", "industry_bias": "test", "grey_zone": "test"}
      result = agent._phase4_generate_output("Vitamin D", synthesis, PerspectiveLens.BALANCED)

      assert result.phase == AnalysisPhase.COMPLEX_OUTPUT
      report = result.content.get("output", "")
      assert "Mainstream" in report
      assert "Naturist" in report
      assert "Biohacker" in report
      # References should be collected in PhaseResult.references
      assert len(result.references) >= 1


  def test_phase4_references_stored_in_phase_result(monkeypatch):
      """All citations from three perspectives must be in PhaseResult.references."""
      import json
      import langchain_agents.base as lc_base
      from langchain_agents import LangChainMedicalFactChecker
      from langchain_agents.factcheck_agent import PerspectiveLens
      from llm_integrations import TokenUsage

      class _CitingProvider:
          def generate_response(self, prompt, system_prompt=None):
              if "findings" in prompt:
                  data = {
                      "findings": "findings",
                      "recommendations": [],
                      "key_insight": "insight",
                      "citations": [
                          "Smith (2024). Title. NEJM. https://doi.org/10.1/mainstream",
                          "Jones (2023). Title. Nature. https://doi.org/10.1/nature",
                      ],
                  }
                  return json.dumps(data), TokenUsage()
              return "## 📚 References\n[1] Smith https://doi.org/10.1/mainstream", TokenUsage()

      class _CitingManager:
          def get_available_provider(self): return _CitingProvider()
          def get_provider_direct(self): return _CitingProvider()

      monkeypatch.setattr(lc_base, "create_llm_manager", lambda *a, **kw: _CitingManager())

      agent = LangChainMedicalFactChecker(
          primary_llm_provider="claude-sonnet",
          interactive=False,
          enable_web_research=False,
      )

      synthesis = {"biological_truth": "test", "industry_bias": "", "grey_zone": ""}
      result = agent._phase4_generate_output("Sugar", synthesis, PerspectiveLens.MAINSTREAM)

      # Each perspective returns 2 citations → up to 6 total, deduplicated
      raw_citations = [r["raw_citation"] for r in result.references]
      assert any("mainstream" in c for c in raw_citations)
      assert any("nature" in c for c in raw_citations)
  ```

- [ ] **Step 2: Run tests to confirm they fail**

  ```bash
  uv run python -m pytest tests/test_langchain_agents.py::test_phase4_generates_three_perspective_report tests/test_langchain_agents.py::test_phase4_references_stored_in_phase_result -v
  ```
  Expected: FAIL — current Phase 4 doesn't run three perspectives.

- [ ] **Step 3: Replace `_phase4_generate_output` in `factcheck_agent.py`**

  Remove the existing `@track_cost("Phase 4: Complex Output (LangChain)")` method and replace with:

  ```python
  @track_cost("Phase 4: Multi-Perspective Output (LangChain)")
  def _phase4_generate_output(
      self,
      subject: str,
      synthesis: Dict[str, Any],
      lens: "PerspectiveLens",
  ) -> PhaseResult:
      """
      Run three perspective agents in parallel, then assemble into a single report.
      Total LLM calls: 3 parallel + 1 assembler = 4.
      """
      # ── 1. Run three perspectives in parallel ─────────────────────────────
      perspectives = ("mainstream", "naturist", "biohacker")
      results: Dict[str, _PerspectiveOutput] = {}

      with ThreadPoolExecutor(max_workers=3) as pool:
          future_map = {
              pool.submit(self._call_perspective, p, subject, synthesis, lens): p
              for p in perspectives
          }
          for future in as_completed(future_map):
              name = future_map[future]
              try:
                  results[name] = future.result()
              except Exception as exc:
                  import logging
                  logging.getLogger(__name__).warning(f"Perspective {name} failed: {exc}")
                  results[name] = _PerspectiveOutput(
                      findings="Analysis unavailable.",
                      recommendations=[],
                      key_insight="",
                      citations=[],
                  )

      mainstream = results["mainstream"]
      naturist = results["naturist"]
      biohacker = results["biohacker"]

      # ── 2. Collect and deduplicate all citations ───────────────────────────
      all_citations = mainstream.citations + naturist.citations + biohacker.citations
      seen_keys: set[str] = set()
      unique_refs: List[Dict[str, Any]] = []
      for c in all_citations:
          key = c.strip().lower()[:100]
          if key and key not in seen_keys:
              seen_keys.add(key)
              unique_refs.append({"raw_citation": c.strip()})

      # ── 3. Assemble the final report via LLM ──────────────────────────────
      lens_label = {
          "M": "Mainstream", "N": "Naturist", "B": "Biohacker", "A": "All perspectives",
      }.get(lens.value, "Balanced")

      assembler_system = (
          "You are a medical report editor. Combine the three perspective summaries "
          "into a clear, well-structured markdown report. Write concisely. "
          "Do NOT add information — only organize what is given."
      )
      assembler_user = """Combine these three perspectives on {subject} into a markdown report.

  Chosen lens: {lens_label}

  MAINSTREAM PERSPECTIVE:
  Key insight: {mainstream_insight}
  Findings: {mainstream_findings}
  Recommendations: {mainstream_recs}

  NATURIST PERSPECTIVE:
  Key insight: {naturist_insight}
  Findings: {naturist_findings}
  Recommendations: {naturist_recs}

  BIOHACKER PERSPECTIVE:
  Key insight: {biohacker_insight}
  Findings: {biohacker_findings}
  Recommendations: {biohacker_recs}

  ALL CITATIONS (include all, deduplicated):
  {all_citations}

  Required output structure (use exactly these markdown headings):

  ## 🎯 Your Focus: {lens_label} Perspective
  [2-3 sentences: the chosen perspective's key_insight + top 3 recommendations]

  ---

  ## 🏥 Mainstream Medicine View
  [findings + recommendations for mainstream]

  ## 🌿 Naturist / Evolutionary View
  [findings + recommendations for naturist]

  ## 🚀 Biohacker / Optimization View
  [findings + recommendations for biohacker]

  ## 📚 References
  [All citations, numbered, APA 7 format]
  """

      assembled = self._call_llm(
          assembler_system,
          assembler_user,
          audit_step="factcheck_phase4_assembler",
          subject=subject,
          lens_label=lens_label,
          mainstream_insight=mainstream.key_insight,
          mainstream_findings=mainstream.findings[:500],
          mainstream_recs="\n".join(f"- {r}" for r in mainstream.recommendations),
          naturist_insight=naturist.key_insight,
          naturist_findings=naturist.findings[:500],
          naturist_recs="\n".join(f"- {r}" for r in naturist.recommendations),
          biohacker_insight=biohacker.key_insight,
          biohacker_findings=biohacker.findings[:500],
          biohacker_recs="\n".join(f"- {r}" for r in biohacker.recommendations),
          all_citations="\n".join(c["raw_citation"] for c in unique_refs),
      )

      return PhaseResult(
          phase=AnalysisPhase.COMPLEX_OUTPUT,
          timestamp=datetime.now(),
          content={"output": assembled, "output_type": lens.value},
          references=unique_refs,
      )
  ```

- [ ] **Step 4: Run Phase 4 tests**

  ```bash
  uv run python -m pytest tests/test_langchain_agents.py::test_phase4_generates_three_perspective_report tests/test_langchain_agents.py::test_phase4_references_stored_in_phase_result -v
  ```
  Expected: 2 PASS.

- [ ] **Step 5: Run full test suite**

  ```bash
  uv run python -m pytest tests/test_langchain_agents.py -v
  ```

---

### Task 5: Update Phase 5 to be lens-aware and preserve references

**Files:**
- Modify: `langchain_agents/factcheck_agent.py`
- Test: `tests/test_langchain_agents.py`

Phase 5 currently has no lens parameter. We add `lens: PerspectiveLens` and update the system prompt to frame tone accordingly. The body/references split is handled in `start_analysis` (Task 2), so Phase 5 only receives the body — it never sees the references string, preventing corruption.

- [ ] **Step 1: Write a test**

  Add to `tests/test_langchain_agents.py`:

  ```python
  def test_phase5_uses_lens_framing(monkeypatch):
      """_phase5_simplify_output must pass lens context in the system prompt."""
      import langchain_agents.base as lc_base
      from langchain_agents import LangChainMedicalFactChecker
      from langchain_agents.factcheck_agent import PerspectiveLens
      from llm_integrations import TokenUsage

      captured_system = {}

      class _CapturingProvider:
          def generate_response(self, prompt, system_prompt=None):
              captured_system["system"] = system_prompt or ""
              return "Simplified content.", TokenUsage()

      class _CapturingManager:
          def get_available_provider(self): return _CapturingProvider()
          def get_provider_direct(self): return _CapturingProvider()

      monkeypatch.setattr(lc_base, "create_llm_manager", lambda *a, **kw: _CapturingManager())

      agent = LangChainMedicalFactChecker(
          primary_llm_provider="claude-sonnet",
          interactive=False,
          enable_web_research=False,
      )

      agent._phase5_simplify_output("Some complex body text.", lens=PerspectiveLens.BIOHACKER)
      assert "biohack" in captured_system["system"].lower() or "optim" in captured_system["system"].lower(), (
          f"Expected biohacker framing in system prompt, got: {captured_system['system'][:200]}"
      )


  def test_phase5_references_not_in_body(monkeypatch):
      """Phase 5 body must not contain the references section (split happens upstream)."""
      import langchain_agents.base as lc_base
      from langchain_agents import LangChainMedicalFactChecker
      from langchain_agents.factcheck_agent import PerspectiveLens
      from llm_integrations import TokenUsage

      captured_user = {}

      class _CapturingProvider:
          def generate_response(self, prompt, system_prompt=None):
              captured_user["user"] = prompt
              return "Simplified.", TokenUsage()

      class _CapturingManager:
          def get_available_provider(self): return _CapturingProvider()
          def get_provider_direct(self): return _CapturingProvider()

      monkeypatch.setattr(lc_base, "create_llm_manager", lambda *a, **kw: _CapturingManager())

      agent = LangChainMedicalFactChecker(
          primary_llm_provider="claude-sonnet",
          interactive=False,
          enable_web_research=False,
      )

      body_only = "## Key Findings\nSome findings here."
      agent._phase5_simplify_output(body_only, lens=PerspectiveLens.NATURIST)

      # References should not appear in the user prompt since body_only has no refs section
      assert "📚 References" not in captured_user["user"]
  ```

- [ ] **Step 2: Run to confirm they fail**

  ```bash
  uv run python -m pytest tests/test_langchain_agents.py::test_phase5_uses_lens_framing tests/test_langchain_agents.py::test_phase5_references_not_in_body -v
  ```
  Expected: FAIL — `_phase5_simplify_output` doesn't accept `lens` parameter.

- [ ] **Step 3: Replace `_phase5_simplify_output` in `factcheck_agent.py`**

  Remove the existing `@track_cost("Phase 5: Simplified Output (LangChain)")` method and replace with:

  ```python
  @track_cost("Phase 5: Simplified Output (LangChain)")
  def _phase5_simplify_output(
      self,
      body: str,
      lens: "PerspectiveLens" = None,
  ) -> PhaseResult:
      """
      Simplify the body text for a general audience using the chosen lens for framing.
      References are NOT passed in — they are re-attached verbatim by start_analysis.
      """
      if lens is None:
          from langchain_agents.factcheck_agent import PerspectiveLens as _L
          lens = _L.BALANCED

      _LENS_FRAMING = {
          "M": (
              "clinical, evidence-graded tone. Use 'your doctor recommends' framing. "
              "Emphasize the strength of the evidence behind each recommendation."
          ),
          "N": (
              "warm, nature-first tone. Use 'your body evolved to...' framing. "
              "Emphasize ancestral wisdom and natural approaches."
          ),
          "B": (
              "optimization mindset tone. Use 'here is your protocol' framing. "
              "Emphasize measurable outcomes, n=1 experimentation, and cutting-edge insights."
          ),
          "A": (
              "balanced, neutral tone. Cover all perspectives equally. "
              "Let the reader decide which approach suits them."
          ),
      }
      framing = _LENS_FRAMING.get(lens.value, _LENS_FRAMING["A"])

      system_prompt = (
          f"You are a medical writer simplifying content for a general audience. "
          f"Use a {framing} "
          f"Write at a 6th grade reading level. Use short sentences and common words. "
          f"Replace statistical notation (RR, OR, CI, p-values) with plain language. "
          f"Keep essential biomarkers (HbA1c, LDL, etc.) but explain them simply in parentheses."
      )

      user_prompt = """Simplify this medical content for a non-medical reader.

  Content to simplify:
  {body}

  Web research context:
  {web_context}

  Structure the output as:
  # Simplified Guide: [subject from content]

  ## Key Findings
  ## Practical Recommendations
  ## What to Watch Out For
  ## Tests or Markers to Track (if applicable)
  ## Supplements or Medications Mentioned (if applicable)

  Do NOT include a References section — that will be added separately.
  """

      response = self._call_llm(
          system_prompt,
          user_prompt,
          audit_step="factcheck_phase_5",
          body=body,
          web_context=self.web_context or "None",
      )

      return PhaseResult(
          phase=AnalysisPhase.SIMPLIFIED_OUTPUT,
          timestamp=datetime.now(),
          content={"simplified_output": response},
      )
  ```

- [ ] **Step 4: Run Phase 5 tests**

  ```bash
  uv run python -m pytest tests/test_langchain_agents.py::test_phase5_uses_lens_framing tests/test_langchain_agents.py::test_phase5_references_not_in_body -v
  ```
  Expected: 2 PASS.

- [ ] **Step 5: Run full test suite**

  ```bash
  uv run python -m pytest tests/test_langchain_agents.py -v
  ```
  Expected: all tests pass.

---

### Task 6: End-to-end integration test

**Files:**
- Test: `tests/test_langchain_agents.py`

Verify the full `start_analysis` flow produces a session with references in every phase result and a final output that contains the references section verbatim.

- [ ] **Step 1: Write the integration test**

  Add to `tests/test_langchain_agents.py`:

  ```python
  def test_factchecker_end_to_end_references_preserved(monkeypatch):
      """Full start_analysis: references must appear in final_output, not be lost."""
      import json
      import langchain_agents.base as lc_base
      from langchain_agents import LangChainMedicalFactChecker
      from llm_integrations import TokenUsage

      REFS_BLOCK = "[1] Author (2024). Title. NEJM. https://doi.org/10.1/sentinel"
      call_count = {"n": 0}

      class _FullProvider:
          def generate_response(self, prompt, system_prompt=None):
              call_count["n"] += 1
              # Perspective calls return JSON with citations
              if "findings" in prompt and "schema" in prompt:
                  data = {
                      "findings": "Findings here.",
                      "recommendations": ["Do this."],
                      "key_insight": "Important insight.",
                      "citations": [REFS_BLOCK],
                  }
                  return json.dumps(data), TokenUsage()
              # Phase 1/2/3 JSON calls
              if "official_narrative" in prompt or "Return JSON" in prompt:
                  data = {
                      "official_narrative": "Official says X.",
                      "counter_narrative": "Independent says Y.",
                      "key_conflicts": "Conflict on Z.",
                      "industry_funded_studies": "Big pharma studies.",
                      "independent_research": "Small labs show.",
                      "methodology_quality": "Good.",
                      "anecdotal_signals": "Users report.",
                      "time_weighted_evidence": "2024 shows.",
                      "biological_truth": "Truth is.",
                      "industry_bias": "Bias here.",
                      "grey_zone": "Grey area.",
                      "references": [],
                  }
                  return json.dumps(data), TokenUsage()
              # Assembler returns markdown with references section
              return (
                  f"## 🎯 Your Focus: Balanced\nKey insight.\n\n"
                  f"## 🏥 Mainstream View\nFindings.\n\n"
                  f"## 🌿 Naturist View\nFindings.\n\n"
                  f"## 🚀 Biohacker View\nFindings.\n\n"
                  f"\n## 📚 References\n{REFS_BLOCK}"
              ), TokenUsage()

      class _FullManager:
          def get_available_provider(self): return _FullProvider()
          def get_provider_direct(self): return _FullProvider()

      monkeypatch.setattr(lc_base, "create_llm_manager", lambda *a, **kw: _FullManager())

      agent = LangChainMedicalFactChecker(
          primary_llm_provider="claude-sonnet",
          interactive=False,
          enable_web_research=False,
      )

      session = agent.start_analysis("Vitamin D and bone health")

      # References must be in final_output
      assert REFS_BLOCK in session.final_output, (
          f"Sentinel reference not found in final_output.\n"
          f"final_output[:500]:\n{session.final_output[:500]}"
      )

      # Phase 4 PhaseResult must have references
      phase4 = next(
          (p for p in session.phase_results
           if p.phase.value == "complex_output"), None
      )
      assert phase4 is not None
      assert len(phase4.references) >= 1
      assert any(REFS_BLOCK in r.get("raw_citation", "") for r in phase4.references)
  ```

- [ ] **Step 2: Run the integration test**

  ```bash
  uv run python -m pytest tests/test_langchain_agents.py::test_factchecker_end_to_end_references_preserved -v --tb=short
  ```
  Expected: PASS.

- [ ] **Step 3: Run the complete test suite one final time**

  ```bash
  uv run python -m pytest tests/test_langchain_agents.py -v --tb=short
  ```
  Expected: all tests pass (3 original + all new ones).

---

## Self-review checklist

**Spec coverage:**
- [x] `PerspectiveLens` enum replacing `OutputType` in LangChain agent → Task 1
- [x] Phase 3 user prompt replaced with M/N/B/A lens picker → Task 2
- [x] `_call_perspective` helper with per-perspective system prompts → Task 3
- [x] Phase 4 parallel execution with `ThreadPoolExecutor` → Task 4
- [x] References collected into `PhaseResult.references` → Task 4
- [x] Phase 5 lens-aware simplification → Task 5
- [x] Body/references split prevents reference corruption → Task 2 (`start_analysis`) + Task 5
- [x] End-to-end reference preservation test → Task 6
- [x] Fallback behavior when perspective agent fails → Task 3

**Confirmed no placeholders.**

**Type consistency:** `_call_perspective` returns `_PerspectiveOutput`. `_phase4_generate_output` takes `lens: PerspectiveLens` in Tasks 2, 4. `_phase5_simplify_output` takes `lens: PerspectiveLens` in Tasks 2, 5. All consistent.
