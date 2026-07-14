"""Document parser service: convert PDF / Word / text documents to markdown.

Self-contained and dependency-guarded. PDF is the priority format; ``.docx``,
``.txt``/``.md`` are core; ``.rtf`` and legacy ``.doc`` are optional (install the
``parsing-extras`` extra). The service always returns a :class:`ParseResult` and
never raises for recoverable problems.

Public surface
--------------
- parse_document(path, *, content_type=None) -> ParseResult
- supported_extensions() -> list[str]
- Models: ParseResult, ParseStatus, ParseWarning, DocumentMetadata
"""

from __future__ import annotations

from .models import (
    DocumentMetadata,
    ParseResult,
    ParseStatus,
    ParseWarning,
)
from .service import parse_document, supported_extensions

__all__ = [
    "DocumentMetadata",
    "ParseResult",
    "ParseStatus",
    "ParseWarning",
    "parse_document",
    "supported_extensions",
]
