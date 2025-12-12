#!/usr/bin/env python3
"""
Simple cost tracking for LLM phases
Works with existing TokenUsage tracking in medical agents
"""

from functools import wraps
from datetime import datetime
from typing import Dict, List

# Model pricing (price per 1M tokens)
PRICING = {
    # Claude models (provider names)
    "claude-sonnet": {"input": 3.00, "output": 15.00, "cache_read": 0.30, "cache_write": 3.75},
    "claude-opus": {"input": 15.00, "output": 75.00, "cache_read": 1.50, "cache_write": 18.75},
    # Claude models (full model names)
    "claude-sonnet-4": {"input": 3.00, "output": 15.00, "cache_read": 0.30, "cache_write": 3.75},
    "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00, "cache_read": 0.30, "cache_write": 3.75},
    "claude-opus-4-5-20251101": {"input": 15.00, "output": 75.00, "cache_read": 1.50, "cache_write": 18.75},
    "claude-haiku": {"input": 0.80, "output": 4.00, "cache_read": 0.08, "cache_write": 1.00},
    # OpenAI models
    "openai": {"input": 3.00, "output": 15.00},
    "gpt-4-turbo-preview": {"input": 3.00, "output": 15.00},
    # xAI Grok models (provider names as used in CLI)
    "grok-4-1-fast": {"input": 0.20, "output": 0.50},
    "grok-4-1-code": {"input": 0.20, "output": 1.50},
    "grok-4-1-reasoning": {"input": 0.20, "output": 0.50},
    # xAI Grok models (full model names)
    "grok-4-1-fast-reasoning-latest": {"input": 0.20, "output": 0.50},
    "grok-4-1-fast-non-reasoning-latest": {"input": 0.20, "output": 0.50},
    "grok-4-fast-reasoning": {"input": 0.20, "output": 0.50},
    "grok-4-fast-non-reasoning": {"input": 0.20, "output": 0.50},
    "grok-code-fast": {"input": 0.20, "output": 1.50},
    "grok-4": {"input": 3.00, "output": 15.00},
    # Default
    "default": {"input": 3.00, "output": 15.00},
}


def calculate_cost(input_tokens: int, output_tokens: int, model: str = "claude-sonnet-4",
                  cache_read: int = 0, cache_write: int = 0) -> float:
    """Calculate cost from token counts"""
    pricing = PRICING.get(model, PRICING["default"])

    cost = (input_tokens / 1_000_000) * pricing["input"]
    cost += (output_tokens / 1_000_000) * pricing["output"]
    cost += (cache_read / 1_000_000) * pricing.get("cache_read", 0)
    cost += (cache_write / 1_000_000) * pricing.get("cache_write", 0)

    return cost


# Global cost tracking
_phase_costs: List[Dict] = []
_current_phase_models: List[str] = []  # Track models used in current phase


def track_cost(phase_name: str):
    """
    Decorator to track cost of a phase

    Usage:
        @track_cost("Phase 1: Evidence Gathering")
        def _phase1_evidence(self, ...):
            # your code
            pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            global _current_phase_models
            start = datetime.now()

            # Reset phase models tracking
            _current_phase_models = []

            # Capture token state before
            if hasattr(self, 'total_token_usage'):
                start_input = self.total_token_usage.input_tokens
                start_output = self.total_token_usage.output_tokens
                start_cache_read = getattr(self.total_token_usage, 'cache_read_tokens', 0)
                start_cache_write = getattr(self.total_token_usage, 'cache_write_tokens', 0)
            else:
                start_input = start_output = start_cache_read = start_cache_write = 0

            # Run the function
            result = func(self, *args, **kwargs)

            duration = (datetime.now() - start).total_seconds()

            # Calculate tokens used in this phase
            if hasattr(self, 'total_token_usage'):
                phase_input = self.total_token_usage.input_tokens - start_input
                phase_output = self.total_token_usage.output_tokens - start_output
                phase_cache_read = getattr(self.total_token_usage, 'cache_read_tokens', 0) - start_cache_read
                phase_cache_write = getattr(self.total_token_usage, 'cache_write_tokens', 0) - start_cache_write

                model = getattr(self, 'primary_llm', 'claude-sonnet-4')

                cost = calculate_cost(
                    phase_input,
                    phase_output,
                    model,
                    phase_cache_read,
                    phase_cache_write
                )

                # Get unique models used in this phase
                models_used = list(set(_current_phase_models)) if _current_phase_models else [model]

                _phase_costs.append({
                    "phase": phase_name,
                    "cost": cost,
                    "duration": duration,
                    "input_tokens": phase_input,
                    "output_tokens": phase_output,
                    "models_used": models_used,
                })

                print(f"  💰 {phase_name}: ${cost:.4f} ({duration:.1f}s) [{', '.join(models_used)}]")

            return result
        return wrapper
    return decorator


def get_cost_summary() -> Dict:
    """Get summary of all tracked costs"""
    total_cost = sum(p['cost'] for p in _phase_costs)
    total_duration = sum(p['duration'] for p in _phase_costs)

    return {
        "total_cost": total_cost,
        "total_duration": total_duration,
        "phases": _phase_costs,
        "most_expensive": sorted(_phase_costs, key=lambda x: x['cost'], reverse=True)[:3]
    }


def print_cost_summary():
    """Print cost summary to console"""
    summary = get_cost_summary()

    print("\n" + "=" * 60)
    print("💰 COST SUMMARY")
    print("=" * 60)
    print(f"Total Cost: ${summary['total_cost']:.4f}")
    print(f"Total Duration: {summary['total_duration']:.1f}s")
    print("\nPhases:")
    for p in summary['phases']:
        pct = (p['cost'] / summary['total_cost'] * 100) if summary['total_cost'] > 0 else 0
        print(f"  {p['phase']}: ${p['cost']:.4f} ({pct:.1f}%)")
    print("=" * 60 + "\n")


def record_model_usage(model_name: str):
    """Record that a specific model was used in the current phase"""
    global _current_phase_models
    if model_name and model_name not in _current_phase_models:
        _current_phase_models.append(model_name)


def reset_tracking():
    """Clear tracked costs (call at start of new analysis)"""
    global _phase_costs, _current_phase_models
    _phase_costs = []
    _current_phase_models = []
