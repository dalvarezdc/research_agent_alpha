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

logger = logging.getLogger(__name__)

_tracer_provider: Optional[TracerProvider] = None
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

        logger.info("Phoenix tracing active at %s", phoenix_url)
        print(f"  Phoenix UI: {phoenix_url}")
        return phoenix_url

    except Exception as exc:
        logger.warning("Phoenix setup failed (tracing disabled): %s", exc)
        return None


def get_tracer() -> trace.Tracer:
    """Return the project-level OpenTelemetry tracer."""
    return trace.get_tracer("research-agent-alpha")
