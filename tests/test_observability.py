"""Tests for observability module (Phoenix + OTEL setup)."""
from unittest.mock import MagicMock, patch


def test_get_tracer_returns_tracer():
    """get_tracer() returns an OpenTelemetry tracer instance."""
    from observability import get_tracer
    tracer = get_tracer()
    assert hasattr(tracer, "start_as_current_span")


def test_setup_phoenix_returns_url_on_success():
    """setup_phoenix() returns the Phoenix UI URL when successful."""
    import observability
    observability._tracer_provider = None
    observability._phoenix_app = None

    mock_app = MagicMock()
    mock_app.url = "http://localhost:6006"

    with patch("observability._launch_phoenix", return_value=mock_app), \
         patch("observability.LangChainInstrumentor") as mock_instrumentor, \
         patch("observability.OTLPSpanExporter"), \
         patch("observability.trace.set_tracer_provider"):
        mock_instrumentor.return_value.instrument = MagicMock()
        url = observability.setup_phoenix()
        assert url == "http://localhost:6006"


def test_setup_phoenix_returns_none_on_failure():
    """setup_phoenix() returns None if Phoenix cannot start."""
    import observability
    observability._tracer_provider = None
    observability._phoenix_app = None

    with patch("observability._launch_phoenix", side_effect=Exception("port in use")):
        url = observability.setup_phoenix()
        assert url is None


def test_setup_phoenix_is_idempotent():
    """Calling setup_phoenix() twice returns same URL without re-initializing."""
    import observability
    observability._tracer_provider = None
    observability._phoenix_app = None

    mock_app = MagicMock()
    mock_app.url = "http://localhost:6006"

    with patch("observability._launch_phoenix", return_value=mock_app) as mock_launch, \
         patch("observability.LangChainInstrumentor"), \
         patch("observability.OTLPSpanExporter"), \
         patch("observability.trace.set_tracer_provider"):
        observability.setup_phoenix()
        observability.setup_phoenix()  # second call
        assert mock_launch.call_count == 1
