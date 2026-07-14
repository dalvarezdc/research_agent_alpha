"""Legacy Microsoft Word (.doc) backend — best effort, optional.

The old binary ``.doc`` format has no pure-Python reader. This backend attempts
extraction via ``textract`` if installed (which shells out to ``antiword`` /
``catdoc``). When unavailable, the dispatcher returns a FAILED result with an
install hint. Kept isolated so the awkward dependency never affects other formats.
"""

from __future__ import annotations

from pathlib import Path

from ..models import DocumentMetadata, ParseResult, ParseStatus
from .base import ParserBackend, split_paragraphs

try:
    import textract  # type: ignore

    _TEXTRACT_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    textract = None  # type: ignore
    _TEXTRACT_AVAILABLE = False


class DocBackend(ParserBackend):
    name = "doc"
    extensions = (".doc",)
    required_import = "textract"

    def is_available(self) -> bool:
        return _TEXTRACT_AVAILABLE

    def parse(self, path: Path) -> ParseResult:
        raw = textract.process(str(path))
        text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
        markdown = "\n\n".join(split_paragraphs(text)).strip()

        status = ParseStatus.SUCCESS if markdown else ParseStatus.FAILED
        result = ParseResult(
            markdown=markdown,
            status=status,
            metadata=DocumentMetadata(
                source_path=str(path),
                file_format="doc",
                backend=self.name,
                char_count=len(markdown),
            ),
        )
        if status is ParseStatus.FAILED:
            result.add_warning("Document contained no text.")
        return result
