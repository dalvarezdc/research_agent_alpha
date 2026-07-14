"""Backend interface and shared markdown helpers.

Each supported format has one backend module implementing :class:`ParserBackend`.
Shared rendering logic (GFM tables, lists, conservative heading detection,
paragraph splitting) lives here so backends do not duplicate it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

from ..models import ParseResult


class ParserBackend(ABC):
    """A format-specific document-to-markdown converter.

    Subclasses set ``name``/``extensions`` and, when they depend on an optional
    third-party library, ``required_import`` (the module name) plus an
    availability flag checked by :meth:`is_available`.
    """

    name: str = "base"
    extensions: tuple[str, ...] = ()
    required_import: str | None = None

    def is_available(self) -> bool:
        """Return True if this backend's dependencies are importable.

        Core backends (no optional dep) return True. Backends with an optional
        dependency override this to reflect whether the import succeeded.
        """
        return True

    @abstractmethod
    def parse(self, path: Path) -> ParseResult:
        """Convert ``path`` into a :class:`ParseResult`. Must not raise for
        recoverable problems — record them as warnings and adjust status."""
        raise NotImplementedError


# ── Shared markdown rendering helpers ────────────────────────────────────────


def render_table(rows: Sequence[Sequence[str | None]]) -> str:
    """Render a 2D grid of cells as a GitHub-Flavored Markdown table.

    The first row is treated as the header. ``None`` cells become empty
    strings; pipe characters are escaped. Returns "" for an empty grid.
    """
    cleaned = [
        [_escape_cell(cell) for cell in row]
        for row in rows
        if row is not None
    ]
    if not cleaned:
        return ""

    width = max(len(row) for row in cleaned)
    # Pad short rows so every row has the same column count.
    for row in cleaned:
        row.extend([""] * (width - len(row)))

    header = cleaned[0]
    body = cleaned[1:]

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def _escape_cell(cell: str | None) -> str:
    """Normalize a single table cell for markdown output."""
    if cell is None:
        return ""
    # Collapse internal newlines/whitespace; escape pipes.
    text = " ".join(str(cell).split())
    return text.replace("|", "\\|")


def render_list(items: Sequence[str], *, ordered: bool = False) -> str:
    """Render items as a markdown bullet or numbered list."""
    out = []
    for idx, item in enumerate(items, start=1):
        text = " ".join(str(item).split())
        if not text:
            continue
        prefix = f"{idx}." if ordered else "-"
        out.append(f"{prefix} {text}")
    return "\n".join(out)


def split_paragraphs(text: str) -> list[str]:
    """Split raw extracted text into paragraphs on blank lines.

    Consecutive non-blank lines are joined into one paragraph; runs of blank
    lines act as separators. Trailing/leading whitespace is trimmed.
    """
    paragraphs: list[str] = []
    current: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line:
            current.append(line)
        elif current:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return paragraphs


def looks_like_heading(line: str) -> bool:
    """Conservative heuristic: is this short line likely a heading?

    True when the line is short, has no terminal sentence punctuation, and is
    either ALL-CAPS (letters) or title-ish. Kept deliberately conservative to
    avoid turning ordinary short sentences into headings.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return False
    if stripped[-1] in ".:;,!?":
        return False
    word_count = len(stripped.split())
    if word_count == 0 or word_count > 10:
        return False
    letters = [c for c in stripped if c.isalpha()]
    if not letters:
        return False
    return all(c.isupper() for c in letters)
