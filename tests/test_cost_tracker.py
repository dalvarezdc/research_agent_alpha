"""Unit tests for the class-based CostTracker."""

import pytest
from unittest.mock import patch, MagicMock
from cost_tracker import CostTracker, calculate_cost


def test_tracker_starts_empty():
    tracker = CostTracker()
    summary = tracker.get_summary()
    assert summary["total_cost"] == 0.0
    assert summary["phases"] == []


def test_tracker_reset_clears_data():
    tracker = CostTracker()
    tracker._phase_costs.append(
        {
            "phase": "p1",
            "cost": 1.0,
            "duration": 1.0,
            "input_tokens": 100,
            "output_tokens": 50,
            "models_used": ["m"],
        }
    )
    tracker.reset()
    assert tracker.get_summary()["total_cost"] == 0.0


def test_two_trackers_are_independent():
    t1 = CostTracker()
    t2 = CostTracker()
    t1._phase_costs.append(
        {
            "phase": "p1",
            "cost": 5.0,
            "duration": 1.0,
            "input_tokens": 100,
            "output_tokens": 50,
            "models_used": ["m"],
        }
    )
    assert t2.get_summary()["total_cost"] == 0.0


def test_get_summary_computes_totals():
    tracker = CostTracker()
    tracker._phase_costs.append(
        {
            "phase": "p1",
            "cost": 1.5,
            "duration": 2.0,
            "input_tokens": 100,
            "output_tokens": 50,
            "models_used": ["m"],
        }
    )
    tracker._phase_costs.append(
        {
            "phase": "p2",
            "cost": 0.5,
            "duration": 1.0,
            "input_tokens": 50,
            "output_tokens": 25,
            "models_used": ["m"],
        }
    )
    summary = tracker.get_summary()
    assert summary["total_cost"] == pytest.approx(2.0)
    assert summary["total_duration"] == pytest.approx(3.0)
    assert len(summary["phases"]) == 2


def test_module_level_reset_tracking_and_get_cost_summary():
    """Module-level functions still work (backward compat)."""
    from cost_tracker import reset_tracking, get_cost_summary

    reset_tracking()
    summary = get_cost_summary()
    assert summary["total_cost"] == 0.0


def test_record_model_usage_registers_model():
    tracker = CostTracker()
    tracker.record_model_usage("gpt-4o")
    assert "gpt-4o" in tracker._current_phase_models


def test_record_model_usage_no_duplicates():
    tracker = CostTracker()
    tracker.record_model_usage("gpt-4o")
    tracker.record_model_usage("gpt-4o")
    assert tracker._current_phase_models.count("gpt-4o") == 1


def test_reset_tracking_isolates_sessions():
    """Calling reset_tracking between sessions must clear previous data."""
    from cost_tracker import reset_tracking, get_cost_summary, _default_tracker

    reset_tracking()
    _default_tracker._phase_costs.append(
        {
            "phase": "old session",
            "cost": 99.0,
            "duration": 1.0,
            "input_tokens": 1,
            "output_tokens": 1,
            "models_used": ["m"],
        }
    )
    reset_tracking()
    assert get_cost_summary()["total_cost"] == 0.0, (
        "reset_tracking must clear previous session data"
    )


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
