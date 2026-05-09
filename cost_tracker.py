#!/usr/bin/env python3
"""
Cost tracking for LLM phases.

Provides both a class-based CostTracker (preferred, isolated per session)
and module-level functions (backwards-compatible wrappers around a default instance).
"""

from functools import wraps
from datetime import datetime
from typing import Dict, List, Optional

from observability import add_span_attributes


# Model pricing (price per 1M tokens)
PRICING = {
    # ── Claude models — current ───────────────────────────────────────────────
    # Provider aliases
    "claude-sonnet": {
        "input": 3.00,
        "output": 15.00,
        "cache_read": 0.30,
        "cache_write": 3.75,
    },
    "claude-opus": {
        "input": 5.00,
        "output": 25.00,
        "cache_read": 0.50,
        "cache_write": 6.25,
    },
    # claude-sonnet-4-6
    "claude-sonnet-4-6": {
        "input": 3.00,
        "output": 15.00,
        "cache_read": 0.30,
        "cache_write": 3.75,
    },
    # claude-opus-4-7
    "claude-opus-4-7": {
        "input": 5.00,
        "output": 25.00,
        "cache_read": 0.50,
        "cache_write": 6.25,
    },
    # claude-haiku-4-5
    "claude-haiku-4-5": {
        "input": 1.00,
        "output": 5.00,
        "cache_read": 0.10,
        "cache_write": 1.25,
    },
    "claude-haiku-4-5-20251001": {
        "input": 1.00,
        "output": 5.00,
        "cache_read": 0.10,
        "cache_write": 1.25,
    },
    # ── Claude models — legacy (still active, not deprecated) ─────────────────
    "claude-sonnet-4": {
        "input": 3.00,
        "output": 15.00,
        "cache_read": 0.30,
        "cache_write": 3.75,
    },
    "claude-sonnet-4-5-20250929": {
        "input": 3.00,
        "output": 15.00,
        "cache_read": 0.30,
        "cache_write": 3.75,
    },
    "claude-opus-4-5-20251101": {
        "input": 5.00,
        "output": 25.00,
        "cache_read": 0.50,
        "cache_write": 6.25,
    },
    "claude-haiku": {
        "input": 0.80,
        "output": 4.00,
        "cache_read": 0.08,
        "cache_write": 1.00,
    },
    # ── OpenAI models ─────────────────────────────────────────────────────────
    "openai": {"input": 2.50, "output": 10.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4-turbo-preview": {"input": 10.00, "output": 30.00},
    # ── xAI Grok models — current ─────────────────────────────────────────────
    # grok-4.3: $1.25/1M input, $2.50/1M output
    "grok-4.3": {"input": 1.25, "output": 2.50},
    # ── xAI Grok models — legacy (retiring May 15 2026) ──────────────────────
    "grok-4-1-fast": {"input": 0.20, "output": 0.50},
    "grok-4-1-code": {"input": 0.20, "output": 1.50},
    "grok-4-1-reasoning": {"input": 0.20, "output": 0.50},
    "grok-4-1-fast-reasoning-latest": {"input": 0.20, "output": 0.50},
    "grok-4-1-fast-non-reasoning-latest": {"input": 0.20, "output": 0.50},
    "grok-4-fast-reasoning": {"input": 0.20, "output": 0.50},
    "grok-4-fast-non-reasoning": {"input": 0.20, "output": 0.50},
    "grok-code-fast": {"input": 0.20, "output": 1.50},
    "grok-4-0709": {"input": 3.00, "output": 15.00},
    # Default
    "default": {"input": 3.00, "output": 15.00},
}


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "claude-sonnet-4",
    cache_read: int = 0,
    cache_write: int = 0,
) -> float:
    """Calculate cost from token counts."""
    pricing = PRICING.get(model, PRICING["default"])
    cost = (input_tokens / 1_000_000) * pricing["input"]
    cost += (output_tokens / 1_000_000) * pricing["output"]
    cost += (cache_read / 1_000_000) * pricing.get("cache_read", 0)
    cost += (cache_write / 1_000_000) * pricing.get("cache_write", 0)
    return cost


class CostTracker:
    """
    Per-session cost tracker. Instantiate one per analysis run to avoid
    cross-session state pollution from the module-level global tracker.

    Usage:
        tracker = CostTracker()
        tracker.reset()   # clear at start of analysis

        @tracker.track_phase("Phase 1: Analysis")
        def _phase1(self, ...):
            ...

        summary = tracker.get_summary()
    """

    def __init__(self) -> None:
        self._phase_costs: List[Dict] = []
        self._current_phase_models: List[str] = []

    def reset(self) -> None:
        """Clear all tracked costs. Call at the start of a new analysis."""
        self._phase_costs = []
        self._current_phase_models = []

    def record_model_usage(self, model_name: str) -> None:
        """Record that a specific model was used in the current phase."""
        if model_name and model_name not in self._current_phase_models:
            self._current_phase_models.append(model_name)

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

    def print_summary(self) -> None:
        """Print cost summary to console."""
        summary = self.get_summary()
        print("\n" + "=" * 60)
        print("💰 COST SUMMARY")
        print("=" * 60)
        print(f"Total Cost: ${summary['total_cost']:.4f}")
        print(f"Total Duration: {summary['total_duration']:.1f}s")
        print("\nPhases:")
        for p in summary["phases"]:
            pct = (
                (p["cost"] / summary["total_cost"] * 100)
                if summary["total_cost"] > 0
                else 0
            )
            print(f"  {p['phase']}: ${p['cost']:.4f} ({pct:.1f}%)")
        print("=" * 60 + "\n")

    def track_phase(self, phase_name: str):
        """
        Decorator factory that wraps a bound method to record cost for a named phase.

        NOTE: Because Python decorators are evaluated at class-definition time,
        this must be applied as an instance method decorator at call time, or
        used via the module-level track_cost() which delegates to _default_tracker.

        The decorated function must be a method on an object that has
        a `total_token_usage` attribute (TokenUsage instance).
        """

        def decorator(func):
            @wraps(func)
            def wrapper(agent_self, *args, **kwargs):
                self._current_phase_models = []
                start = datetime.now()

                tu = getattr(agent_self, "total_token_usage", None)
                start_input = getattr(tu, "input_tokens", 0)
                start_output = getattr(tu, "output_tokens", 0)
                start_cache_read = getattr(tu, "cache_read_tokens", 0)
                start_cache_write = getattr(tu, "cache_write_tokens", 0)

                result = func(agent_self, *args, **kwargs)

                duration = (datetime.now() - start).total_seconds()

                tu = getattr(agent_self, "total_token_usage", None)
                if tu is not None:
                    phase_input = getattr(tu, "input_tokens", 0) - start_input
                    phase_output = getattr(tu, "output_tokens", 0) - start_output
                    phase_cache_read = (
                        getattr(tu, "cache_read_tokens", 0) - start_cache_read
                    )
                    phase_cache_write = (
                        getattr(tu, "cache_write_tokens", 0) - start_cache_write
                    )

                    model = getattr(agent_self, "primary_llm", "claude-sonnet-4")
                    cost = calculate_cost(
                        phase_input,
                        phase_output,
                        model,
                        phase_cache_read,
                        phase_cache_write,
                    )
                    models_used = (
                        list(set(self._current_phase_models))
                        if self._current_phase_models
                        else [model]
                    )

                    self._phase_costs.append(
                        {
                            "phase": phase_name,
                            "cost": cost,
                            "duration": duration,
                            "input_tokens": phase_input,
                            "output_tokens": phase_output,
                            "models_used": models_used,
                        }
                    )
                    print(
                        f"  💰 {phase_name}: ${cost:.4f} ({duration:.1f}s)"
                        f" [{', '.join(models_used)}]"
                    )

                return result

            return wrapper

        return decorator


# ── Backwards-compatible module-level API ─────────────────────────────────────
# All existing code that imports track_cost / reset_tracking / etc. continues
# to work without any changes. They delegate to a single shared default tracker.

_default_tracker = CostTracker()


def track_cost(phase_name: str):
    """
    Module-level decorator (backwards compat). Delegates to _default_tracker.

    Usage:
        @track_cost("Phase 1: Evidence Gathering")
        def _phase1_evidence(self, ...):
            pass
    """
    return _default_tracker.track_phase(phase_name)


def get_cost_summary() -> Dict:
    """Module-level getter (backwards compat)."""
    return _default_tracker.get_summary()


def print_cost_summary() -> None:
    """Module-level printer (backwards compat)."""
    _default_tracker.print_summary()


def record_model_usage(model_name: str) -> None:
    """Module-level recorder (backwards compat)."""
    _default_tracker.record_model_usage(model_name)


def reset_tracking() -> None:
    """Module-level reset (backwards compat)."""
    _default_tracker.reset()
