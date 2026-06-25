# Phoenix Tracing Fix — Design Spec

**Date:** 2026-05-10  
**Status:** Approved

---

## Problem

The Arize Phoenix dashboard shows all spans with `kind: unknown` and `input`/`output`/`annotations` columns displaying `--`. Tracing is structurally active (spans are emitted and received by Phoenix) but the data inside them is either absent or keyed incorrectly.

Four root causes were identified:

### 1. Wrong OpenInference attribute key names
Phoenix renders its `input` and `output` columns from the [OpenInference semantic conventions](https://arize-ai.github.io/openinference/spec/semantic_conventions.html):
- `input.value` → Phoenix "Input" column
- `output.value` → Phoenix "Output" column
- `openinference.span.kind` → Phoenix "Kind" column (expected values: `"LLM"`, `"CHAIN"`, `"RETRIEVER"`, `"EMBEDDING"`, `"TOOL"`)

The current code emits `llm.input_messages`, `llm.output`, `llm.output.value` — none of which Phoenix maps to its visible columns. Spans appear as `kind: unknown`.

### 2. Agent `_call_llm` creates no span
Every agent phase calls `LangChainAgentBase._call_llm()` → `generate_response()` directly. No `with tracer.start_as_current_span(...)` wraps these calls. All agent LLM activity is invisible in the Phoenix call tree; only the routing-level `call_model()` span exists.

### 3. Dead cost annotation code in `cost_tracker.py`
`CostTracker.get_summary()` has an `add_span_attributes({cost.total, ...})` block after the `return` statement (lines 167–174). It never executes. Cost data is never sent to Phoenix.

### 4. Only `ClaudeLLM` calls `add_span_attributes`
`OpenAILLM`, `OllamaLLM`, and `XaiLLM` return from `generate_response()` without any `add_span_attributes` call. `ClaudeLLM` has one but uses wrong key names (see root cause 1).

---

## Approach: Surgical fix — model-agnostic span in `_call_llm`

Rather than fixing each provider's `generate_response()` individually, the span is added once in `LangChainAgentBase._call_llm()`. This method is the single aggregation point for all agent LLM calls regardless of provider; by the time `generate_response()` returns, it already yields a normalized `(response: str, token_usage: TokenUsage)` tuple with provider differences fully abstracted away.

This means:
- One code change covers Claude, OpenAI, Ollama, and Grok
- Future providers are automatically covered
- Provider classes remain pure: no observability coupling inside `llm_integrations.py`

The existing `add_span_attributes` call inside `ClaudeLLM.generate_response()` is removed (it becomes redundant and uses wrong key names).

---

## Changes

### `langchain_agents/base.py` — add span in `_call_llm`

Add `from observability import get_tracer` to imports.

Wrap the `generate_response()` call in a manual span:

```python
def _call_llm(self, system_prompt, user_prompt, audit_step=None, **kwargs):
    system_text, user_text = self._render_prompt(system_prompt, user_prompt, **kwargs)

    tracer = get_tracer()
    model_name = getattr(getattr(self, "llm_provider", None), "config", None)
    model_name = getattr(model_name, "model", "unknown") if model_name else "unknown"

    with tracer.start_as_current_span("llm.call") as span:
        span.set_attribute("openinference.span.kind", "LLM")
        span.set_attribute("input.value", user_text[:2000])
        span.set_attribute("llm.model_name", model_name)

        response, token_usage = self.llm_provider.generate_response(
            prompt=user_text, system_prompt=system_text
        )

        span.set_attribute("output.value", (response or "")[:2000])
        if token_usage:
            span.set_attribute("llm.token_count.prompt", token_usage.input_tokens)
            span.set_attribute("llm.token_count.completion", token_usage.output_tokens)
            span.set_attribute("llm.token_count.total", token_usage.total_tokens)

    # ... existing audit + token accumulation code unchanged
```

### `llm_integrations.py` — fix `call_model()` span attributes and remove `ClaudeLLM` duplicate

**`call_model()` span** (lines 1055–1069): rename attributes to OpenInference conventions:
- `llm.input_messages` → `input.value`
- `llm.output.value` → `output.value`
- `llm.token_count.prompt/completion/total` — keep (already correct)
- Add `openinference.span.kind = "LLM"`

**`ClaudeLLM.generate_response()`** (lines 277–290): remove the `add_span_attributes` block entirely. The `_call_llm` span replaces it for agent calls; the `call_model` span replaces it for routing calls. Also remove the top-level `from observability import add_span_attributes` import at line 16 — after the `ClaudeLLM` block is removed, it becomes unused dead code.

### `cost_tracker.py` — fix dead code in `get_summary()`

Move `add_span_attributes(...)` to before the `return` statement so cost annotations are actually emitted.

### `router.py` — fix `router.session` span attributes

- Add `openinference.span.kind = "CHAIN"` to the `router.session` span
- Rename `query` attribute → `input.value` (so Phoenix shows the user's query in the Input column)
- Rename `output.files_count` — keep as-is (it's a custom attribute, not an OpenInference column, which is fine)

---

## Files modified

| File | What changes |
|---|---|
| `langchain_agents/base.py` | Add `get_tracer` import; add span in `_call_llm` |
| `llm_integrations.py` | Fix `call_model()` span attribute names; remove `ClaudeLLM` duplicate `add_span_attributes` block; remove unused `add_span_attributes` import |
| `cost_tracker.py` | Move `add_span_attributes` before `return` in `get_summary()` |
| `router.py` | Add `openinference.span.kind`; rename `query` → `input.value` on session span |

---

## What Phoenix will show after the fix

| Column | Before | After |
|---|---|---|
| Kind | `unknown` | `LLM` (agent calls), `CHAIN` (session), `LLM` (routing call) |
| Input | `--` | User prompt text (truncated to 2000 chars) |
| Output | `--` | LLM response text (truncated to 2000 chars) |
| Annotations | `--` | Cost total, duration, phase count (from `cost_tracker`) |
| Token counts | missing | `llm.token_count.prompt/completion/total` on each LLM span |

---

## Out of scope

- Phase-level `CHAIN` spans (one span per agent phase) — deferred to a follow-on
- Fixing `OllamaLLM`, `OpenAILLM`, `XaiLLM` `add_span_attributes` calls directly — handled by removing the pattern from the provider layer entirely
- Reference validation observability
- LangSmith tracing changes

---

## Testing

The existing test suite (`uv run python -m pytest tests/ -q`) must continue to pass. No new Phoenix-specific integration tests are added (Phoenix requires a running server; unit tests mock the tracer). Existing tests that mock `generate_response` will exercise the new span code via the `_call_llm` wrapper.
