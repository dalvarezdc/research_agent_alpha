"""PDF backend (priority format).

Primary engine is pdfplumber (extracts both text and tables). When a page yields
no text via pdfplumber, pypdf is tried as a fallback; if that also yields
nothing the page is recorded as a warning (likely scanned / image-only) and
parsing continues. Overall status:

* SUCCESS  -- every page produced text.
* PARTIAL  -- some pages produced text, some did not.
* FAILED   -- no page produced any text.
"""

from __future__ import annotations

from pathlib import Path

from ..models import DocumentMetadata, ParseResult, ParseStatus
from .base import (
    ParserBackend,
    looks_like_heading,
    render_table,
)

try:
    import pdfplumber  # type: ignore

    _PDFPLUMBER_AVAILABLE = True
except ImportError:  # pragma: no cover - pdfplumber is a base dependency
    pdfplumber = None  # type: ignore
    _PDFPLUMBER_AVAILABLE = False

try:
    from pypdf import PdfReader  # type: ignore

    _PYPDF_AVAILABLE = True
except ImportError:  # pragma: no cover - pypdf is a base dependency
    PdfReader = None  # type: ignore
    _PYPDF_AVAILABLE = False


class PdfBackend(ParserBackend):
    name = "pdf"
    extensions = (".pdf",)
    required_import = "pdfplumber"

    def is_available(self) -> bool:
        return _PDFPLUMBER_AVAILABLE

    def parse(self, path: Path) -> ParseResult:
        blocks: list[str] = []
        pages_with_text = 0
        page_count = 0
        warnings: list[tuple[str, str]] = []

        with pdfplumber.open(str(path)) as pdf:
            page_count = len(pdf.pages)
            for index, page in enumerate(pdf.pages, start=1):
                page_blocks, had_text = self._render_page(page, index, path)
                if had_text:
                    pages_with_text += 1
                else:
                    warnings.append(
                        (f"page {index}", "No extractable text (possibly scanned/image-only)."),
                    )
                if page_blocks:
                    blocks.append(f"<!-- page {index} -->")
                    blocks.extend(page_blocks)

        markdown = "\n\n".join(blocks).strip()

        if pages_with_text == 0:
            status = ParseStatus.FAILED
        elif pages_with_text < page_count:
            status = ParseStatus.PARTIAL
        else:
            status = ParseStatus.SUCCESS

        result = ParseResult(
            markdown=markdown,
            status=status,
            metadata=DocumentMetadata(
                source_path=str(path),
                file_format="pdf",
                backend=self.name,
                page_count=page_count,
                char_count=len(markdown),
            ),
        )
        for location, message in warnings:
            result.add_warning(message, location=location)
        if status is ParseStatus.FAILED:
            result.add_warning(
                "No text could be extracted from any page.", location=None,
            )
        return result

    def _render_page(
        self, page: pdfplumber.page.Page, index: int, path: Path,
    ) -> tuple[list[str], bool]:
        """Return (markdown blocks for the page, whether any text was found)."""
        blocks: list[str] = []

        # Tables first — they are the highest-value structured content.
        try:
            tables = page.extract_tables() or []
        except Exception:  # noqa: BLE001 - table extraction is best-effort
            tables = []
        for table in tables:
            rendered = render_table(table)
            if rendered:
                blocks.append(rendered)

        text = page.extract_text() or ""
        if not text.strip():
            text = self._pypdf_fallback(path, index)

        had_text = bool(text.strip())
        blocks.extend(self._render_text_blocks(text))
        return blocks, had_text or bool(tables)

    @staticmethod
    def _render_text_blocks(text: str) -> list[str]:
        """Convert extracted page text into heading/paragraph markdown blocks.

        PDF ``extract_text`` separates visual lines with single newlines, so we
        evaluate each line for the heading heuristic and group consecutive
        non-heading lines into paragraphs.
        """
        blocks: list[str] = []
        current: list[str] = []

        def flush() -> None:
            if current:
                blocks.append(" ".join(current))
                current.clear()

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                flush()
            elif looks_like_heading(line):
                flush()
                blocks.append(f"## {line}")
            else:
                current.append(line)
        flush()
        return blocks

    @staticmethod
    def _pypdf_fallback(path: Path, index: int) -> str:
        """Try pypdf for a single page (1-based ``index``) when pdfplumber fails."""
        if not _PYPDF_AVAILABLE:
            return ""
        try:
            reader = PdfReader(str(path))
            if index - 1 < len(reader.pages):
                return reader.pages[index - 1].extract_text() or ""
        except Exception:  # noqa: BLE001 - fallback is best-effort
            return ""
        return ""
