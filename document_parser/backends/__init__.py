"""Format-specific parser backends."""

from __future__ import annotations

from .base import ParserBackend
from .doc_backend import DocBackend
from .docx_backend import DocxBackend
from .pdf_backend import PdfBackend
from .rtf_backend import RtfBackend
from .text_backend import TextBackend

__all__ = [
    "DocBackend",
    "DocxBackend",
    "ParserBackend",
    "PdfBackend",
    "RtfBackend",
    "TextBackend",
]
