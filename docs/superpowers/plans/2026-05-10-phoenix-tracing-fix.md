# Phoenix Tracing Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix Arize Phoenix dashboard showing `kind: unknown` and `--` for input/output/annotations by using OpenInference semantic conventions and adding a model-agnostic span in `_call_llm`.

**Architecture:** A single span is added in `LangChainAgentBase._call_llm()` — the one method all four LLM providers flow through — using OpenInference-standard attribute keys (`input.value`, `output.value`, `openinference.span.kind`). The duplicate `add_span_attributes` in `ClaudeLLM` is removed. Dead cost annotation code in `cost_tracker.py` is fixed. The `router.session` span gets the correct kind and input attribute.

**Tech Stack:** Python, OpenTelemetry, `openinference-instrumentation-langchain`, Arize Phoenix

---

### Task 1: Fix `cost_tracker.py` dead code

**Files:**
- Modify: `cost_tracker.py:154-174`
- Test: `tests/test_cost_tracker.py` (existing file)

The `add_span_attributes(...)` block in `CostTracker.get_summary()` currently sits after a `return` statement and never executes. Move it before the return.

- [ ] **Step 1: Read the existing test file**

Run: `cat tests/test_cost_tracker.py`

Check if there's an existing test for `get_summary()`. If not, we'll add one.

- [ ] **Step 2: Write a failing test**

In `tests/test_cost_tracker.py`, add (or replace the existing `get_summary` test with):

```python
from unittest.mock import patch, MagicMock

def test_get_summary_emits_span_attributes():
    """Cost summary must call add_span_attributes before returning."""
    from cost_tracker import CostTracker
    tracker = CostTracker()
    # Simulate a phase being tracked
    tracker._phase_costs = [
        {"phase": "Phase 1", "cost": 0.05, "duration": 1.2, "model": "grok-4.3"}
    ]
    with patch("cost_tracker.add_span_attributes") as mock_attrs:
        result = tracker.get_summary()
    # Must have been called (not dead code)
    mock_attrs.assert_called_once()
    call_kwargs = mock_attrs.call_args[0][0]
    assert "cost.total" in call_kwargs
    assert call_kwargs["cost.total"] == 0.05
    assert "cost.phases_count" in call_kwargs
    assert call_kwargs["cost.phases_count"] == 1
    # Return value must still be the dict
    assert result["total_cost"] == 0.05
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_cost_tracker.py::test_get_summary_emits_span_attributes -v`

Expected: FAIL — `assert_called_once()` raises `AssertionError` because `add_span_attributes` is never called (it's after the `return`).

- [ ] **Step 4: Fix `cost_tracker.py`**

In `cost_tracker.py`, the current `get_summary` method looks like:

```python
def get_summary(self) -> Dict:
    """Return summary of all tracked costs."""
    total_cost = sum(p["cost"] for p in self._phase_costs)
    total_duration = sum(p["duration"] for p in self._phase_costs)
    return {
        "total_cost": total_cost,
        "total_duration": total_duration,
        "phases": self._phase_costs,
        "most_expensive": sorted(
            self._phase_costs, key=lambda x: x["cost"], reverse=True
        )[:3],
    }

    # Phoenix observability: cost annotations
    add_span_attributes(
        {
            "cost.total": total_cost,
            "cost.duration": total_duration,
            "cost.phases_count": len(self._phase_costs),
        }
    )
```

Replace the entire method body so the `add_span_attributes` call happens before the `return`:

```python
def get_summary(self) -> Dict:
    """Return summary of all tracked costs."""
    total_cost = sum(p["cost"] for p in self._phase_costs)
    total_duration = sum(p["duration"] for p in self._phase_costs)
    # Phoenix observability: cost annotations on the active span
    add_span_attributes(
        {
            "cost.total": total_cost,
            "cost.duration": total_duration,
            "cost.phases_count": len(self._phase_costs),
        }
    )
    return {
        "total_cost": total_cost,
        "total_duration": total_duration,
        "phases": self._phase_costs,
        "most_expensive": sorted(
            self._phase_costs, key=lambda x: x["cost"], reverse=True
        )[:3],
    }
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_cost_tracker.py::test_get_summary_emits_span_attributes -v`

Expected: PASS

- [ ] **Step 6: Run full test suite**

Run: `uv run python -m pytest tests/ -q`

Expected: all previously passing tests still pass.

- [ ] **Step 7: Commit**

```bash
git add cost_tracker.py tests/test_cost_tracker.py
git commit -m "fix(observability): move cost add_span_attributes before return in get_summary"
```

---

### Task 2: Fix `llm_integrations.py` — rename `call_model()` span attributes and remove `ClaudeLLM` duplicate

**Files:**
- Modify: `llm_integrations.py` (two locations: `call_model()` span ~line 1055, `ClaudeLLM.generate_response()` ~line 277)
- Test: `tests/test_llm_integrations.py` (existing)

Two changes:
1. In `call_model()` rename `llm.input_messages` → `input.value`, `llm.output.value` → `output.value`, add `openinference.span.kind = "LLM"`.
2. In `ClaudeLLM.generate_response()` remove the `add_span_attributes({...})` block (lines ~277–290).
3. Remove the now-unused top-level `from observability import add_span_attributes` import.

- [ ] **Step 1: Write a failing test for `call_model()` span attributes**

In `tests/test_llm_integrations.py`, add:

```python
from unittest.mock import patch, MagicMock, call
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

def _make_test_tracer():
    """Create an in-memory tracer for span attribute assertions."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test"), exporter

def test_call_model_span_uses_openinference_keys():
    """call_model() must emit input.value, output.value, openinference.span.kind."""
    import llm_integrations as li

    tracer, exporter = _make_test_tracer()

    mock_provider = MagicMock()
    mock_provider.generate_response.return_value = ("the response", MagicMock(
        input_tokens=10, output_tokens=20, total_tokens=30
    ))

    with patch.object(li, "_get_tracer", return_value=tracer), \
         patch.object(li, "create_llm_manager", return_value=mock_provider):
        result = li.call_model(
            messages=[{"role": "user", "content": "hello"}],
            model_name="grok-4.3",
        )

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes)
    assert attrs.get("openinference.span.kind") == "LLM"
    assert "input.value" in attrs
    assert "output.value" in attrs
    assert "llm.input_messages" not in attrs   # old key must be gone
    assert "llm.output.value" not in attrs     # old key must be gone
    assert result == "the response"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_llm_integrations.py::test_call_model_span_uses_openinference_keys -v`

Expected: FAIL — `attrs.get("openinference.span.kind")` returns `None`, and `llm.input_messages` key is present.

- [ ] **Step 3: Fix `call_model()` span attributes in `llm_integrations.py`**

Find the span block inside `call_model()` (around line 1053). Replace:

```python
    tracer = _get_tracer()
    with tracer.start_as_current_span("llm.call") as span:
        span.set_attribute("llm.model_name", model_name)
        span.set_attribute("llm.provider", provider_name)
        span.set_attribute("llm.input_messages", str(messages)[:2000])
        try:
            response_text, token_usage = llm_provider.generate_response(
                prompt=user_prompt, system_prompt=system_prompt
            )
            span.set_attribute("llm.output.value", response_text[:2000])
            if token_usage:
                span.set_attribute("llm.token_count.prompt", token_usage.input_tokens)
                span.set_attribute(
                    "llm.token_count.completion", token_usage.output_tokens
                )
                span.set_attribute("llm.token_count.total", token_usage.total_tokens)
            return response_text
        except Exception as e:
            span.record_exception(e)
            raise RuntimeError(f"LLM call failed for model {model_name}: {e}")
```

With:

```python
    tracer = _get_tracer()
    with tracer.start_as_current_span("llm.call") as span:
        span.set_attribute("openinference.span.kind", "LLM")
        span.set_attribute("llm.model_name", model_name)
        span.set_attribute("llm.provider", provider_name)
        span.set_attribute("input.value", str(messages)[:2000])
        try:
            response_text, token_usage = llm_provider.generate_response(
                prompt=user_prompt, system_prompt=system_prompt
            )
            span.set_attribute("output.value", response_text[:2000])
            if token_usage:
                span.set_attribute("llm.token_count.prompt", token_usage.input_tokens)
                span.set_attribute(
                    "llm.token_count.completion", token_usage.output_tokens
                )
                span.set_attribute("llm.token_count.total", token_usage.total_tokens)
            return response_text
        except Exception as e:
            span.record_exception(e)
            raise RuntimeError(f"LLM call failed for model {model_name}: {e}")
```

- [ ] **Step 4: Remove the `add_span_attributes` block from `ClaudeLLM.generate_response()`**

Find and remove this block (around lines 277–290 inside `ClaudeLLM.generate_response`):

```python
            # Phoenix observability: output + cost annotations
            add_span_attributes(
                {
                    "llm.output": response.content or "",
                    "llm.tokens.input": token_usage.input_tokens,
                    "llm.tokens.output": token_usage.output_tokens,
                    "llm.tokens.total": token_usage.total_tokens,
                }
            )
```

Delete those lines entirely (including the comment).

- [ ] **Step 5: Remove the unused `add_span_attributes` import**

At the top of `llm_integrations.py` (around line 16), find and remove:

```python
from observability import add_span_attributes
```

- [ ] **Step 6: Run tests**

Run: `uv run python -m pytest tests/test_llm_integrations.py -v`

Expected: all passing, including the new test.

Run full suite: `uv run python -m pytest tests/ -q`

Expected: all previously passing tests still pass.

- [ ] **Step 7: Commit**

```bash
git add llm_integrations.py tests/test_llm_integrations.py
git commit -m "fix(observability): use OpenInference keys in call_model span, remove ClaudeLLM duplicate"
```

---

### Task 3: Add model-agnostic span in `LangChainAgentBase._call_llm`

**Files:**
- Modify: `langchain_agents/base.py`
- Test: `tests/test_langchain_agents.py` (existing)

Add `from observability import get_tracer` to imports. Wrap `generate_response()` in `_call_llm` with a manual span using OpenInference keys.

- [ ] **Step 1: Write a failing test**

In `tests/test_langchain_agents.py`, add:

```python
from unittest.mock import patch, MagicMock
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

def _make_tracer_with_exporter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test"), exporter

def test_call_llm_emits_openinference_span():
    """_call_llm must emit a span with input.value, output.value, openinference.span.kind=LLM."""
    from langchain_agents.base import LangChainAgentBase

    tracer, exporter = _make_tracer_with_exporter()

    mock_provider = MagicMock()
    mock_provider.config.model = "grok-4.3"
    mock_provider.generate_response.return_value = (
        "the answer",
        MagicMock(input_tokens=5, output_tokens=10, total_tokens=15),
    )

    agent = LangChainAgentBase.__new__(LangChainAgentBase)
    agent.llm_provider = mock_provider
    agent.enable_audit = False
    agent.total_token_usage = MagicMock()
    agent.total_token_usage.add = MagicMock()

    with patch("langchain_agents.base.get_tracer", return_value=tracer):
        result = agent._call_llm(
            system_prompt="You are helpful.",
            user_prompt="What is 2+2?",
        )

    assert result == "the answer"
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes)
    assert attrs.get("openinference.span.kind") == "LLM"
    assert "input.value" in attrs
    assert "output.value" in attrs
    assert attrs.get("output.value") == "the answer"
    assert attrs.get("llm.model_name") == "grok-4.3"
    assert attrs.get("llm.token_count.prompt") == 5
    assert attrs.get("llm.token_count.completion") == 10
    assert attrs.get("llm.token_count.total") == 15
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_langchain_agents.py::test_call_llm_emits_openinference_span -v`

Expected: FAIL — `ImportError: cannot import name 'get_tracer' from 'langchain_agents.base'` or `AssertionError` because no spans are emitted.

- [ ] **Step 3: Add `get_tracer` import to `langchain_agents/base.py`**

At the top of `langchain_agents/base.py`, after the existing imports, add:

```python
from observability import get_tracer
```

- [ ] **Step 4: Wrap `generate_response()` in `_call_llm` with a span**

The current `_call_llm` method in `langchain_agents/base.py` (lines 141–176) looks like:

```python
    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        audit_step: str | None = None,
        **kwargs: Any,
    ) -> str:
        system_text, user_text = self._render_prompt(
            system_prompt, user_prompt, **kwargs
        )
        response, token_usage = self.llm_provider.generate_response(
            prompt=user_text, system_prompt=system_text
        )
        if token_usage:
            self.total_token_usage.add(token_usage)
        if self.enable_audit:
            self.audit_events.append(
                {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "step": audit_step,
                    "system_prompt": system_text,
                    "user_prompt": user_text,
                    "response": response,
                    "token_usage": (
                        {
                            "input_tokens": token_usage.input_tokens,
                            "output_tokens": token_usage.output_tokens,
                            "total_tokens": token_usage.total_tokens,
                        }
                        if token_usage
                        else None
                    ),
                }
            )
            self._record_langsmith(audit_step, system_text, user_text, response)
        return response
```

Replace with:

```python
    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        audit_step: str | None = None,
        **kwargs: Any,
    ) -> str:
        system_text, user_text = self._render_prompt(
            system_prompt, user_prompt, **kwargs
        )
        model_name = getattr(
            getattr(self.llm_provider, "config", None), "model", "unknown"
        )
        tracer = get_tracer()
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
        if token_usage:
            self.total_token_usage.add(token_usage)
        if self.enable_audit:
            self.audit_events.append(
                {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "step": audit_step,
                    "system_prompt": system_text,
                    "user_prompt": user_text,
                    "response": response,
                    "token_usage": (
                        {
                            "input_tokens": token_usage.input_tokens,
                            "output_tokens": token_usage.output_tokens,
                            "total_tokens": token_usage.total_tokens,
                        }
                        if token_usage
                        else None
                    ),
                }
            )
            self._record_langsmith(audit_step, system_text, user_text, response)
        return response
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_langchain_agents.py::test_call_llm_emits_openinference_span -v`

Expected: PASS

- [ ] **Step 6: Run full test suite**

Run: `uv run python -m pytest tests/ -q`

Expected: all previously passing tests still pass.

- [ ] **Step 7: Commit**

```bash
git add langchain_agents/base.py tests/test_langchain_agents.py
git commit -m "feat(observability): add model-agnostic OpenInference span in _call_llm"
```

---

### Task 4: Fix `router.py` session span attributes

**Files:**
- Modify: `router.py` (around lines 342–346)
- Test: No automated test (router REPL is interactive); verify by inspection.

Add `openinference.span.kind = "CHAIN"` to the `router.session` span and rename the `query` attribute to `input.value`.

- [ ] **Step 1: Find the session span block in `router.py`**

The block (around lines 342–346) currently looks like:

```python
            tracer = get_tracer()
            with tracer.start_as_current_span("router.session") as session_span:
                session_span.set_attribute("query", query)
                session_span.set_attribute("model", selected_model)
                session_span.set_attribute("implementation", implementation)
                session_span.set_attribute("web_search_enabled", web_search_enabled)
```

- [ ] **Step 2: Update span attributes**

Replace the block with:

```python
            tracer = get_tracer()
            with tracer.start_as_current_span("router.session") as session_span:
                session_span.set_attribute("openinference.span.kind", "CHAIN")
                session_span.set_attribute("input.value", query)
                session_span.set_attribute("model", selected_model)
                session_span.set_attribute("implementation", implementation)
                session_span.set_attribute("web_search_enabled", web_search_enabled)
```

- [ ] **Step 3: Run full test suite**

Run: `uv run python -m pytest tests/ -q`

Expected: all previously passing tests still pass.

- [ ] **Step 4: Commit**

```bash
git add router.py
git commit -m "fix(observability): set CHAIN span kind and input.value on router.session span"
```
