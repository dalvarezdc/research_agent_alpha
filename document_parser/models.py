"""Result and metadata models for the document parser service.

The service always returns a :class:`ParseResult`; it never raises for
recoverable problems (unsupported format, missing optional dependency, a page
with no extractable text). Only genuinely invalid input (a path that does not
exist) raises, and that happens in the dispatcher, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ParseStatus(str, Enum):
    """Outcome of a parse attempt.

    SUCCESS  -- content extracted with no recoverable problems.
    PARTIAL  -- some content extracted, but parts were skipped (e.g. an
                image-only page in an otherwise text PDF). See ``warnings``.
    FAILED   -- no usable content could be extracted (unsupported format,
                missing dependency, corrupt file, or a fully image-only doc).
    """

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass
class ParseWarning:
    """A single recoverable problem encountered during parsing.

    ``location`` is a human-readable scope hint such as "page 3" or
    "table on page 1"; ``None`` for document-level warnings.
    """

    message: str
    location: str | None = None

    def __str__(self) -> str:
        if self.location:
            return f"[{self.location}] {self.message}"
        return self.message


@dataclass
class DocumentMetadata:
    """Descriptive metadata about a parsed document."""

    source_path: str
    file_format: str  # normalized extension, e.g. "pdf", "docx"
    backend: str  # name of the backend that produced the result
    page_count: int | None = None  # PDFs (and where knowable)
    char_count: int = 0  # length of the produced markdown


@dataclass
class ParseResult:
    """The value object returned by :func:`document_parser.parse_document`."""

    markdown: str
    status: ParseStatus
    metadata: DocumentMetadata
    warnings: list[ParseWarning] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when at least some content was extracted (SUCCESS or PARTIAL)."""
        return self.status in (ParseStatus.SUCCESS, ParseStatus.PARTIAL)

    def add_warning(self, message: str, location: str | None = None) -> None:
        """Append a recoverable-problem warning."""
        self.warnings.append(ParseWarning(message=message, location=location))
