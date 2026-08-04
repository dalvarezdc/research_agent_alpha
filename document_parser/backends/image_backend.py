"""Image backend: parse image files (.png, .jpg, .jpeg, .webp, .gif) into Markdown."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..models import DocumentMetadata, ParseResult, ParseStatus
from .base import ParserBackend

if TYPE_CHECKING:
    from collections.abc import Sequence


class ImageBackend(ParserBackend):
    """Parses image files into structured markdown."""

    name = "image"
    extensions: Sequence[str] = (".png", ".jpg", ".jpeg", ".webp", ".gif")
    required_import = "PIL"

    def is_available(self) -> bool:
        """Return True (always available)."""
        return True

    def parse(self, path: Path) -> ParseResult:
        file_name = path.name
        file_size = path.stat().st_size if path.exists() else 0
        ext = path.suffix.lower().lstrip(".")

        markdown_lines = [
            f"# Attached Image: {file_name}",
            "",
            f"- **File Format**: `{ext.upper()}`",
            f"- **File Size**: `{file_size:,} bytes`",
            "",
            "### Clinical Image Context",
            f"Medical image file attached: `{file_name}`.",
        ]

        # Best-effort OCR if PIL & pytesseract are installed
        try:
            from PIL import Image
            import pytesseract
            img = Image.open(path)
            ocr_text = pytesseract.image_to_string(img).strip()
            if ocr_text:
                markdown_lines.extend([
                    "",
                    "### Extracted OCR Text Content",
                    "```text",
                    ocr_text,
                    "```"
                ])
        except Exception:
            pass

        markdown = "\n".join(markdown_lines)

        return ParseResult(
            markdown=markdown,
            status=ParseStatus.SUCCESS,
            metadata=DocumentMetadata(
                source_path=str(path),
                file_format=ext,
                page_count=1,
                char_count=len(markdown),
                backend=self.name,
            ),
        )
