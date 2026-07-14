# Document Parser Service — Design

> Date: 2026-07-14
> Status: Approved (brainstorming) → implementation
> Scope: **Parser core only.** Downstream integrations (agent context injection,
> `PatientData` population, document Q&A) are separate follow-up specs.

## Purpose

Provide a self-contained service that reads documents (PDF priority; also Word
`.docx`, plain `.txt`/`.md`, `.rtf`, and legacy `.doc`) and converts them into
structured GitHub-Flavored Markdown (headings, tables, lists). This is the
foundational "medical-parsing" service the `database/README.md` references as the
future populator of the `patient_data` table.

The service is consumed later by three separate specs:
1. Feeding parsed markdown into agents as context.
2. Populating the `PatientData` DB table via structured extraction.
3. Enabling document Q&A / analysis.

None of those three are implemented here.

## Architecture (Approach A — registry of format backends)

Self-contained package mirroring `web_research/` and `database/`:

```
document_parser/
  __init__.py          # public API surface
  models.py            # ParseResult, ParseWarning, ParseStatus, DocumentMetadata
  service.py           # parse_document() dispatcher + backend registry
  backends/
    __init__.py
    base.py            # ParserBackend ABC + shared markdown helpers
    pdf_backend.py     # pdfplumber (tables) + pypdf fallback   [PRIORITY]
    docx_backend.py    # python-docx
    text_backend.py    # .txt/.md passthrough
    rtf_backend.py     # striprtf (optional dep)
    doc_backend.py     # legacy .doc (optional dep)
  __main__.py          # CLI: python -m document_parser <file>
  README.md
```

### Public Python API

```python
from document_parser import parse_document, ParseResult, ParseStatus

result: ParseResult = parse_document("bloodwork.pdf")
result.markdown     # str  — converted GFM markdown
result.status       # ParseStatus: SUCCESS | PARTIAL | FAILED
result.warnings     # list[ParseWarning] — page/section-scoped notes
result.metadata     # DocumentMetadata: format, page/para count, backend, char count
```

`parse_document(path: str | Path, *, content_type: str | None = None) -> ParseResult`

- Resolves extension (or explicit `content_type` override) → backend via registry.
- Always returns a `ParseResult`; never raises for recoverable issues.
- Raises `FileNotFoundError` only when the path does not exist.

### `ParserBackend` interface (`backends/base.py`)

```python
class ParserBackend(ABC):
    name: str
    extensions: tuple[str, ...]        # e.g. (".pdf",)
    required_import: str | None        # optional dep module name, or None if core

    def is_available(self) -> bool     # True if optional deps import
    def parse(self, path: Path) -> ParseResult
```

Shared markdown helpers (GFM table rendering, list rendering, conservative
heading detection, paragraph splitting) live in `base.py` so backends don't
duplicate logic.

### Registry & dispatch (`service.py`)

1. Resolve extension/override → backend. Unknown → `ParseResult(FAILED, warning="unsupported format '.xyz'")`.
2. Backend optional dep missing (`is_available()` False) → `ParseResult(FAILED, warning="'.rtf' support requires 'striprtf'; install ...")`. No crash.
3. Else call `backend.parse(path)` inside a catch-all that converts unexpected
   library exceptions into `status=FAILED` + warning (one bad file never crashes a batch).

Each backend wraps its library import in `try/except ImportError` at module load
(sets an availability flag), following `web_research/search.py`.

## PDF backend (priority)

- **Primary: `pdfplumber`.** Per page:
  - `extract_tables()` → GFM tables (critical for lab-result grids).
  - `extract_text()` → paragraphs (blank-line split).
  - Conservative heading heuristic: short lines without terminal punctuation
    followed by blank line, or ALL-CAPS lines → `##`. Kept conservative to avoid noise.
  - Page boundary markers (`<!-- page N -->`), always emitted.
- **Fallback: `pypdf`.** If `pdfplumber` yields no text for a page, try `pypdf`;
  if still empty, emit a per-page `ParseWarning` (likely scanned/image-only) and continue.
- **Status:** `PARTIAL` if some pages had text and some did not; `FAILED` if zero
  text anywhere; `SUCCESS` otherwise.

## Other backends

- **docx** (`python-docx`, core): Word heading styles → `#`/`##`; paragraphs →
  text; `doc.tables` → GFM tables; numbered/bulleted lists → `1.`/`-`.
- **text/md** (stdlib, core): `.md` returned as-is; `.txt` wrapped as paragraphs.
- **rtf** (`striprtf`, optional): strip to plain text → basic paragraphs.
- **doc** (optional): best-effort via available tooling (`textract`/`antiword`);
  else `FAILED` with install hint.

## Entry points

- **CLI** (`__main__.py`): `uv run python -m document_parser <file> [-o out.md]`.
  Markdown to stdout or `-o` file; warnings to stderr; exit code 0 success,
  1 partial/failed. Uses `pathlib`.
- **FastAPI** (`api.py`): `POST /parse`, `multipart/form-data` upload. Save to
  `tempfile` (cleaned in `finally`), call `parse_document`, return
  `{status, markdown, warnings, metadata}`. Size limit (default 25 MB) → HTTP 413.
  Adds `python-multipart` dependency.

## Dependencies (`pyproject.toml`)

- Core: `pdfplumber`, `pypdf`, `python-docx`, `python-multipart`.
- Optional extra `parsing-extras`: `striprtf` (+ legacy-doc tooling if feasible).

## Error handling

Result-object model throughout. `ParseStatus` = `SUCCESS | PARTIAL | FAILED`.
Warnings carry page/section context. Only `FileNotFoundError` propagates from
`parse_document`. Catch-all around `backend.parse()` guarantees best-effort semantics.

## Testing (`tests/test_document_parser.py`)

Fixtures generated in-test where possible (docx via python-docx, PDF via
reportlab or a tiny committed fixture, txt/md/rtf inline). Cases:

- Text PDF with a table → table present in markdown.
- Multi-page PDF; page markers present.
- Image-only/empty PDF → `PARTIAL`/`FAILED` + warning.
- docx headings/tables/lists.
- txt/md passthrough.
- Unsupported extension → `FAILED`.
- Missing optional dep (monkeypatch availability flag) → `FAILED` with install hint.
- Nonexistent path → `FileNotFoundError`.
- CLI smoke test (`__main__`).
- FastAPI `/parse` via `TestClient` uploading a fixture.

Coverage target: match the DB layer discipline (~99% on the new package).

## Docs & tracking

- `document_parser/README.md` (module map, API, formats table, optional-deps note).
- Update `AGENTS.md` repository map.
- Update `pending.md`: mark the foundational medical-parsing service delivered;
  note the three downstream specs as follow-ups.

## Explicitly out of scope (future specs)

OCR / vision extraction; `PatientData` population; agent-context injection;
document Q&A; layout-aware/multi-column fidelity.
