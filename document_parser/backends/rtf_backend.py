"""Rich Text Format (.rtf) backend using striprtf (optional dependency).

Install with the ``parsing-extras`` extra. When striprtf is not installed the
dispatcher reports a FAILED result with an install hint rather than crashing.
"""

from __future__ import annotations

from pathlib import Path

from ..models import DocumentMetadata, ParseResult, ParseStatus
from .base import ParserBackend, split_paragraphs

try:
    from striprtf.striprtf import rtf_to_text  # type: ignore

    _STRIPRTF_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    rtf_to_text = None  # type: ignore
    _STRIPRTF_AVAILABLE = False


class RtfBackend(ParserBackend):
    name = "rtf"
    extensions = (".rtf",)
    required_import = "striprtf"

    def is_available(self) -> bool:
        return _STRIPRTF_AVAILABLE

    def parse(self, path: Path) -> ParseResult:
        raw = path.read_text(encoding="utf-8", errors="replace")
        text = rtf_to_text(raw) or ""
        markdown = "\n\n".join(split_paragraphs(text)).strip()

        status = ParseStatus.SUCCESS if markdown else ParseStatus.FAILED
        result = ParseResult(
            markdown=markdown,
            status=status,
            metadata=DocumentMetadata(
                source_path=str(path),
                file_format="rtf",
                backend=self.name,
                char_count=len(markdown),
            ),
        )
        if status is ParseStatus.FAILED:
            result.add_warning("Document contained no text.")
        return result
