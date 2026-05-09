# Multi-Perspective MedicalFactChecker Design

**Date:** 2026-05-08  
**Target implementation:** `langchain_agents/factcheck_agent.py` (LangChain only)  
**Status:** Approved

---

## Goal

Replace the single-LLM Phase 4 output with three parallel perspective agents
(Mainstream, Naturist, Biohacker) that each produce their own findings and
recommendations, then merge into a unified side-by-side report. The user picks a
lens that highlights their preferred worldview in the summary and shapes Phase 5
simplification.

---

## Pipeline (revised)

```
Phase 1  Conflict Scan          (unchanged)
Phase 2  Evidence Audit         (prompt improvements — one focused call per concern)
Phase 3  Synthesis              (strengthened prompt to collect references)
         └─ user picks lens: Mainstream / Naturist / Biohacker / Balanced
Phase 4  Three Perspective Agents (parallel, ThreadPoolExecutor max_workers=3)
         ├── Mainstream Agent   → JSON {findings, recommendations, citations, key_insight}
         ├── Naturist Agent     → JSON {findings, recommendations, citations, key_insight}
         └── Biohacker Agent    → JSON {findings, recommendations, citations, key_insight}
         → assembler merges into markdown with "Your Focus" summary at top
Phase 5  Simplification         (written from the chosen lens, references preserved)
```

---

## Data model changes

### New `PerspectiveLens` enum (replaces `OutputType`)

```python
class PerspectiveLens(Enum):
    MAINSTREAM = "M"   # Evidence-based medicine, clinical guidelines
    NATURIST   = "N"   # Evolutionary biology, ancestral health
    BIOHACKER  = "B"   # Optimization, cutting-edge, n=1 experimentation
    BALANCED   = "A"   # All perspectives weighted equally
```

`OutputType` is kept in `medical_fact_checker_agent.py` (original agent, untouched).
`LangChainMedicalFactChecker.start_analysis` uses `PerspectiveLens` internally; the
`phase3_result.user_choice` stores the enum value string (`"M"`, `"N"`, `"B"`, `"A"`).

### New Pydantic models (in `factcheck_agent.py`)

```python
class _PerspectiveOutput(BaseModel):
    findings: str          # Core findings from this perspective
    recommendations: list[str]   # 3-5 actionable recommendations
    key_insight: str       # One-sentence headline for this perspective
    citations: list[str]   # APA references with DOI/PMID/URL

class _Phase4Model(BaseModel):
    mainstream: _PerspectiveOutput
    naturist: _PerspectiveOutput
    biohacker: _PerspectiveOutput
    unified_references: list[str]  # deduplicated across all three
```

`FactCheckSession.practitioner_report` stores the assembled markdown.
`phase4_result.content["output"]` stores the same markdown (existing key, no change
to downstream consumers in `run_analysis.py`).

---

## Phase 4 execution detail

### Parallel calls

```python
with ThreadPoolExecutor(max_workers=3) as pool:
    futures = {
        "mainstream": pool.submit(self._call_perspective, "mainstream", subject, synthesis, lens),
        "naturist":   pool.submit(self._call_perspective, "naturist",   subject, synthesis, lens),
        "biohacker":  pool.submit(self._call_perspective, "biohacker",  subject, synthesis, lens),
    }
    results = {name: future.result() for name, future in futures.items()}
```

Each `_call_perspective` call is a self-contained `_call_llm` → `_parse_json` →
`_PerspectiveOutput.model_validate`. Failures fall back to an empty
`_PerspectiveOutput` with `findings="Analysis unavailable"`.

### System prompts per perspective

**Mainstream:**
> You are a clinical researcher writing evidence-based medical analysis. Prioritize
> RCTs, Cochrane reviews, FDA/EMA guidance, and GRADE-A evidence. Third-person,
> objective, cite DOI/PMID for every claim. Return ONLY valid JSON.

**Naturist:**
> You are an evolutionary medicine researcher. Prioritize ancestral biology,
> circadian alignment, whole-food interventions, and small independent studies.
> Weight evolutionary logic as a tiebreaker when RCT evidence is mixed. Cite
> peer-reviewed support where available. Return ONLY valid JSON.

**Biohacker:**
> You are an optimization researcher. Prioritize recent cutting-edge findings,
> promising n=1 protocols, quantified self data, and emerging mechanisms even
> with limited RCT backing. Label evidence level explicitly. Return ONLY valid JSON.

### Assembler

After all three perspective calls return, a single assembler prompt merges them:

```
Given three perspectives on {subject}, produce a markdown report with:
1. A "Your Focus" summary block (chosen perspective's key_insight + top 3 recs)
2. Three sections: Mainstream / Naturist / Biohacker
3. A unified References section (deduplicated)
```

The assembler uses `_call_llm` once. Total Phase 4: 4 LLM calls (3 parallel + 1 assembler).

---

## Phase 5 simplification (lens-aware)

The `_phase5_simplify_output` prompt receives the chosen lens and uses it to frame tone:

- **Mainstream:** clinical, evidence-graded, follow-your-doctor language
- **Naturist:** warm, nature-first, "your body evolved to..." framing
- **Biohacker:** optimization mindset, "here's your stack and protocol" framing
- **Balanced:** neutral, covers all angles equally

References section instruction strengthened: pass the references as a separate
field, not embedded in the content string, so they survive simplification unchanged.

---

## User interaction (between Phase 3 and 4)

```
PHASE 3 COMPLETE: Synthesis
Which perspective resonates most with you?

  [M] Mainstream   — Clinical guidelines and established evidence
  [N] Naturist     — Evolutionary biology and ancestral health
  [B] Biohacker    — Optimization, cutting-edge, n=1 protocols
  [A] All equal    — Balanced report, no preference

Your choice (M/N/B/A):
```

Non-interactive default: `"A"` (balanced).

---

## Files changed

| File | Change |
|------|--------|
| `langchain_agents/factcheck_agent.py` | Primary implementation — new models, parallel Phase 4, lens-aware Phase 5 |
| `medical_fact_checker/medical_fact_checker_agent.py` | No changes |
| `run_analysis.py` | No changes (consumes `session.final_output` and `session.practitioner_report` unchanged) |

---

## Reference caching strategy

References flow through the pipeline at two levels:

### 1. In-session accumulation (`PhaseResult.references`)

Every `PhaseResult` has a `references: List[Dict]` field. After Phase 4, all three
perspectives produce `citations: list[str]`. These must be converted to the standard
`{"raw_citation": str}` dict format and stored on the `phase4_result.references` list
so that `_collect_validated_references()` in `run_analysis.py` can pick them up
without any changes to that file.

Concretely, `_phase4_generate_output` collects:
```python
all_citations = (
    mainstream_output.citations +
    naturist_output.citations +
    biohacker_output.citations
)
# deduplicate by normalized text
seen = set()
unique = []
for c in all_citations:
    key = c.strip().lower()[:100]
    if key not in seen:
        seen.add(key)
        unique.append({"raw_citation": c.strip()})

phase4_result.references = unique
```

This means `run_analysis.py`'s `_collect_validated_references()` loop already
handles deduplication across phases by DOI/PMID/raw text — no changes needed there.

### 2. Persistent URL validation cache (`reference_validation/cache/cache_manager.py`)

The existing `CacheManager` (SQLite, TTL 30 days) already caches the
`CitationURLCorrespondenceValidator` results keyed by URL. This means the second time
the same DOI/URL is validated (across different analysis sessions), it hits the cache
instead of making network calls. No changes needed here.

### 3. Phase 5 reference preservation

The assembler call (step 4 of Phase 4) produces a markdown block with a
`## 📚 References` section containing all deduplicated citations. This section is
passed **verbatim** to Phase 5 by splitting the content:

```python
# Split content and references before passing to Phase 5
ref_separator = "\n## 📚 References\n"
if ref_separator in assembled_markdown:
    body, refs_block = assembled_markdown.split(ref_separator, 1)
else:
    body, refs_block = assembled_markdown, ""

# Phase 5 simplifies only the body; references appended unchanged after
simplified_body = self._phase5_simplify_output(body, lens=lens)
final_output = simplified_body + (ref_separator + refs_block if refs_block else "")
```

This prevents Phase 5 from corrupting or losing citations during simplification.

---

## Non-goals

- Original agent (`medical_fact_checker_agent.py`) is not touched
- No changes to Phase 1, Phase 2 structure (only prompt text improvements)
- No changes to `run_analysis.py` output saving logic
- No changes to the router or `AgentOrchestrator`
- No changes to `CacheManager` (existing SQLite cache already handles URL validation caching)
