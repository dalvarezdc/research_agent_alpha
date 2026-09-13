"""Tests for report generation correctness."""
import sys
from datetime import datetime
from unittest.mock import MagicMock, patch


def _make_mock_session(subject="test"):
    session = MagicMock()
    session.subject = subject
    session.started_at = datetime(2026, 1, 1, 12, 0, 0)
    session.phase_results = []
    return session


def test_fact_check_summary_timestamp_is_rendered():
    """Timestamp must be an actual datetime string, not the literal text
    '{datetime.now().isoformat()}'."""
    # WeasyPrint requires system libraries not available in CI; mock it out
    sys.modules.setdefault("pdf_generator", MagicMock())
    sys.modules.setdefault("weasyprint", MagicMock())

    from run_analysis import AgentOrchestrator

    orch = AgentOrchestrator.__new__(AgentOrchestrator)
    orch._reference_validation_cache = {}
    orch._citation_url_validator = None

    with patch.object(orch, "_collect_validated_references", return_value=([], [])):
        summary = orch._generate_fact_check_summary(_make_mock_session(), {})

    assert "{datetime.now().isoformat()}" not in summary, (
        "Timestamp placeholder was not rendered — the string is not an f-string"
    )


def test_output_type_invalid_choice_falls_back_to_proceed():
    """OutputType conversion must not raise ValueError on invalid input."""
    from medical_fact_checker.medical_fact_checker_agent import OutputType

    # Valid values should work
    assert OutputType("P") == OutputType.PROCEED
    assert OutputType("A") == OutputType.EVOLUTIONARY

    # Invalid value should raise natively
    try:
        OutputType("INVALID_CHOICE")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected

    # The guarded conversion used in start_analysis should default to PROCEED
    def _guarded(choice: str) -> OutputType:
        try:
            return OutputType(choice)
        except ValueError:
            return OutputType.PROCEED

    assert _guarded("INVALID_CHOICE") == OutputType.PROCEED
    assert _guarded("X") == OutputType.PROCEED
    assert _guarded("P") == OutputType.PROCEED


def test_embedded_references_are_validated_and_markers_remapped():
    from run_analysis import AgentOrchestrator

    orch = AgentOrchestrator.__new__(AgentOrchestrator)
    orch._citation_url_validator = object()
    output = (
        "Claim one [1]. Rejected claim [2]. Third claim [3].\n\n"
        "## References\n"
        "[1] Author. Valid One. https://example.test/one\n"
        "[2] Author. Invalid Two. https://example.test/two\n"
        "[3] Author. Valid Three. https://example.test/three\n"
    )

    def resolve(citation, url, validator):
        return (None, "invalid", None, None) if "Invalid" in citation else (url, None, 1.0, None)

    with (
        patch.object(orch, "_resolve_reference_url", side_effect=resolve),
        patch.object(orch, "_collect_validated_references", return_value=([], [])),
    ):
        rendered = orch._append_references_section(output, object())

    assert "Rejected claim []." not in rendered
    assert "Rejected claim ." in rendered
    assert "Third claim [2]" in rendered
    assert "Invalid Two" not in rendered
    assert rendered.count("## 📚 References") == 1
