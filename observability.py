"""
Observability module — Arize Phoenix + OpenTelemetry tracing.

Call setup_phoenix() once at process startup (router.py:main).
Call get_tracer() anywhere you need a manual span.
"""

from __future__ import annotations

import logging
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from openinference.instrumentation.langchain import LangChainInstrumentor

# Logging imports for Phoenix log ingestion
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter

logger = logging.getLogger(__name__)

_tracer_provider: Optional[TracerProvider] = None
_logger_provider: Optional[LoggerProvider] = None
_phoenix_app = None


def _launch_phoenix():
    """Start the Phoenix server. Separated for testability."""
    import phoenix as px

    return px.launch_app()


def setup_phoenix() -> Optional[str]:
    """
    Start local Arize Phoenix server and configure OTEL tracer.

    Called once at router startup. Safe to call multiple times (idempotent).

    Returns:
        Phoenix UI URL (e.g. "http://localhost:6006") or None on failure.
    """
    global _tracer_provider, _phoenix_app

    if _tracer_provider is not None:
        # Already initialized — return existing URL
        return getattr(_phoenix_app, "url", "http://localhost:6006")

    try:
        # 1. Start Phoenix server
        _phoenix_app = _launch_phoenix()
        phoenix_url = getattr(_phoenix_app, "url", "http://localhost:6006")

        # 2. Set up OTLP exporter pointing at Phoenix
        from opentelemetry.sdk.resources import Resource

        resource = Resource(attributes={"service.name": "research-agent-alpha"})
        _tracer_provider = TracerProvider(resource=resource)

        otlp_exporter = OTLPSpanExporter(
            endpoint="http://localhost:6006/v1/traces",
        )
        _tracer_provider.add_span_processor(SimpleSpanProcessor(otlp_exporter))
        trace.set_tracer_provider(_tracer_provider)

        # 3. Auto-instrument LangChain
        LangChainInstrumentor().instrument(tracer_provider=_tracer_provider)

        # 4. Configure logging to Phoenix via OTLP
        global _logger_provider
        _logger_provider = LoggerProvider(resource=resource)
        log_exporter = OTLPLogExporter(endpoint="http://localhost:6006/v1/logs")
        _logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
        logging_handler = LoggingHandler(
            level=logging.INFO, logger_provider=_logger_provider
        )
        logging.getLogger().addHandler(logging_handler)
        logging.getLogger().setLevel(logging.INFO)

        logger.info("Phoenix tracing + logging active at %s", phoenix_url)
        print(f"  Phoenix UI: {phoenix_url}")
        return phoenix_url

    except Exception as exc:
        logger.warning("Phoenix setup failed (tracing disabled): %s", exc)
        return None


def get_tracer() -> trace.Tracer:
    """Return the project-level OpenTelemetry tracer."""
    return trace.get_tracer("research-agent-alpha")


def add_span_attributes(attributes: dict):
    """Add custom attributes to the current active span (safe no-op if no span)."""
    try:
        span = trace.get_current_span()
        if span and span.is_recording():
            for key, value in attributes.items():
                if value is not None:
                    span.set_attribute(
                        str(key), str(value)[:500]
                    )  # Truncate long values
    except Exception:
        pass  # Never break execution on observability
