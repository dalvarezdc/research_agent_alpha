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
    "claude-sonnet-4": {"input": 3.00, "output": 15.00, "cache_read": 0.30, "cache_write": 3.75},
    "claude-haiku": {"input": 0.80, "output": 4.00, "cache_read": 0.08, "cache_write": 1.00},
    "openai": {"input": 3.00, "output": 15.00},
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
            start = datetime.now()

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

                _phase_costs.append({
                    "phase": phase_name,
                    "cost": cost,
                    "duration": duration,
                    "input_tokens": phase_input,
                    "output_tokens": phase_output,
                })

                print(f"  💰 {phase_name}: ${cost:.4f} ({duration:.1f}s)")

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


def reset_tracking():
    """Clear tracked costs (call at start of new analysis)"""
    global _phase_costs
    _phase_costs = []
