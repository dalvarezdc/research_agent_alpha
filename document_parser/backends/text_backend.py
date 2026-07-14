"""Plain-text and markdown backend.

``.md`` files are returned verbatim (already markdown). ``.txt`` files are split
into paragraphs so the output is clean markdown. Core backend — no optional deps.
"""

from __future__ import annotations

from pathlib import Path

from ..models import DocumentMetadata, ParseResult, ParseStatus
from .base import ParserBackend, split_paragraphs


class TextBackend(ParserBackend):
    name = "text"
    extensions = (".txt", ".md", ".markdown")
    required_import = None

    def parse(self, path: Path) -> ParseResult:
        raw = path.read_text(encoding="utf-8", errors="replace")
        suffix = path.suffix.lower()

        if suffix in (".md", ".markdown"):
            markdown = raw.strip()
        else:
            markdown = "\n\n".join(split_paragraphs(raw))

        status = ParseStatus.SUCCESS if markdown.strip() else ParseStatus.FAILED
        result = ParseResult(
            markdown=markdown,
            status=status,
            metadata=DocumentMetadata(
                source_path=str(path),
                file_format=suffix.lstrip("."),
                backend=self.name,
                char_count=len(markdown),
            ),
        )
        if status is ParseStatus.FAILED:
            result.add_warning("Document contained no text.")
        return result
