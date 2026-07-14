"""Dispatcher: resolve a file to its backend and produce a :class:`ParseResult`.

``parse_document`` is the single public entry point. It never raises for
recoverable problems (unsupported extension, missing optional dependency, or an
unexpected library error mid-parse) — those become a FAILED/PARTIAL result with
warnings. It raises ``FileNotFoundError`` only when the path does not exist.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .backends import (
    DocBackend,
    DocxBackend,
    ParserBackend,
    PdfBackend,
    RtfBackend,
    TextBackend,
)
from .models import DocumentMetadata, ParseResult, ParseStatus

logger = logging.getLogger(__name__)

# Instantiate one backend per format. Order is irrelevant; lookup is by extension.
_BACKENDS: tuple[ParserBackend, ...] = (
    PdfBackend(),
    DocxBackend(),
    TextBackend(),
    RtfBackend(),
    DocBackend(),
)

# Build the extension → backend registry.
_REGISTRY: dict[str, ParserBackend] = {}
for _backend in _BACKENDS:
    for _ext in _backend.extensions:
        _REGISTRY[_ext.lower()] = _backend


def supported_extensions() -> list[str]:
    """Return all registered file extensions (sorted)."""
    return sorted(_REGISTRY.keys())


def _resolve_extension(path: Path, content_type: str | None) -> str:
    """Determine the lookup extension from an explicit override or the path."""
    if content_type:
        override = content_type.strip().lower()
        return override if override.startswith(".") else f".{override}"
    return path.suffix.lower()


def _failed(path: Path, ext: str, message: str, backend: str = "none") -> ParseResult:
    """Build a FAILED result carrying a single document-level warning."""
    result = ParseResult(
        markdown="",
        status=ParseStatus.FAILED,
        metadata=DocumentMetadata(
            source_path=str(path),
            file_format=ext.lstrip("."),
            backend=backend,
        ),
    )
    result.add_warning(message)
    return result


def parse_document(
    path: str | Path,
    *,
    content_type: str | None = None,
) -> ParseResult:
    """Convert a document at ``path`` into structured markdown.

    Args:
        path: Path to the document.
        content_type: Optional format override (e.g. "pdf" or ".pdf") used when
            the extension is missing or misleading.

    Returns:
        A :class:`ParseResult`. Recoverable problems are reported via
        ``status`` + ``warnings`` rather than exceptions.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
    """
    file_path = Path(path)
    if not file_path.exists():
        message = f"File not found: {file_path}"
        raise FileNotFoundError(message)

    ext = _resolve_extension(file_path, content_type)
    backend = _REGISTRY.get(ext)

    if backend is None:
        return _failed(
            file_path,
            ext,
            f"Unsupported format '{ext or '(no extension)'}'. "
            f"Supported: {', '.join(supported_extensions())}.",
        )

    if not backend.is_available():
        dep = backend.required_import or "an optional dependency"
        return _failed(
            file_path,
            ext,
            f"'{ext}' support requires '{dep}'. "
            f"Install it (e.g. `uv sync --extra parsing-extras`) to enable this format.",
            backend=backend.name,
        )

    try:
        return backend.parse(file_path)
    except Exception as exc:  # noqa: BLE001 - best-effort; never crash a batch
        logger.warning("Backend %s failed on %s: %s", backend.name, file_path, exc)
        return _failed(
            file_path,
            ext,
            f"Failed to parse document: {exc}",
            backend=backend.name,
        )
