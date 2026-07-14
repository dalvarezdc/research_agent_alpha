# Document Parser — File → Markdown Service

Reads documents and converts them into structured GitHub-Flavored Markdown
(headings, tables, lists). **PDF is the priority format.** Self-contained and
dependency-guarded, mirroring the `web_research/` and `database/` service
packages.

This is the foundational "medical-parsing" service that the three downstream
features build on (agent-context injection, `PatientData` population, document
Q&A) — those are separate specs and **not** included here.

## Design principle: best-effort, never crash

The service **always returns a `ParseResult`**. Recoverable problems
(unsupported format, missing optional dependency, an image-only PDF page, a
corrupt file) are reported via `status` + `warnings`, not exceptions. Only a
genuinely missing path raises (`FileNotFoundError`). One bad file never crashes a
batch.

```
parse_document(path) → resolve extension → backend.parse() → ParseResult
                            │ unknown ext / missing dep / error
                            └────────────────────────────────→ FAILED result + warning
```

## Quick start

```python
from document_parser import parse_document, ParseStatus

result = parse_document("bloodwork.pdf")
print(result.status)        # ParseStatus.SUCCESS | PARTIAL | FAILED
print(result.markdown)      # converted GFM markdown
for w in result.warnings:   # e.g. "[page 3] No extractable text ..."
    print(w)
print(result.metadata.page_count, result.metadata.backend)
```

## Supported formats

| Extension | Backend | Dependency | Availability |
|-----------|---------|------------|--------------|
| `.pdf` | `PdfBackend` | `pdfplumber` (+ `pypdf` fallback) | **core** (priority) |
| `.docx` | `DocxBackend` | `python-docx` | **core** |
| `.txt` | `TextBackend` | stdlib | **core** |
| `.md` / `.markdown` | `TextBackend` | stdlib | **core** (passthrough) |
| `.rtf` | `RtfBackend` | `striprtf` | optional (`parsing-extras`) |
| `.doc` | `DocBackend` | `textract` (+ system tools) | optional |

Enable the optional formats:

```bash
uv sync --extra parsing-extras
```

## PDF handling (priority)

- **Primary:** `pdfplumber` — extracts text **and tables** (lab-result grids
  become GFM tables). Conservative heading heuristic promotes short ALL-CAPS
  lines to `##`. Page boundaries marked with `<!-- page N -->`.
- **Fallback:** `pypdf` — used per page when pdfplumber finds no text.
- **Status:** `SUCCESS` (all pages had text), `PARTIAL` (some pages empty —
  likely scanned/image-only, each flagged as a warning), `FAILED` (no text
  anywhere).

> OCR / vision extraction for scanned PDFs is intentionally out of scope; the
> backend registry makes adding an OCR backend later a drop-in change.

## Entry points

### Python API
```python
from document_parser import parse_document, supported_extensions
parse_document(path, *, content_type=None)   # content_type overrides the extension
supported_extensions()                        # ['.doc', '.docx', '.markdown', ...]
```

### CLI
```bash
uv run python -m document_parser report.pdf              # markdown to stdout
uv run python -m document_parser report.pdf -o out.md    # write to file
uv run python -m document_parser file.data --format pdf  # override format
```
Exit code: `0` on SUCCESS, `1` on PARTIAL/FAILED. Warnings go to stderr.

### REST API
```
POST /parse   (multipart/form-data, field name "file")
```
Returns `{filename, status, markdown, warnings, metadata}`. Uploads over
`MAX_PARSE_UPLOAD_BYTES` (default 25 MB, env-configurable) return HTTP 413.
Unparseable files return HTTP 200 with `status: "failed"` for a consistent shape.

## Module map

| File | Responsibility |
|------|----------------|
| `models.py` | `ParseResult`, `ParseStatus`, `ParseWarning`, `DocumentMetadata` |
| `service.py` | `parse_document()` dispatcher + extension→backend registry |
| `backends/base.py` | `ParserBackend` ABC + shared markdown helpers (tables/lists/headings) |
| `backends/pdf_backend.py` | PDF (pdfplumber + pypdf fallback) |
| `backends/docx_backend.py` | Word `.docx` (python-docx) |
| `backends/text_backend.py` | `.txt` / `.md` |
| `backends/rtf_backend.py` | `.rtf` (striprtf, optional) |
| `backends/doc_backend.py` | legacy `.doc` (textract, optional) |
| `__main__.py` | CLI |

## Testing

```bash
uv run python -m pytest tests/test_document_parser.py -v
```

Fixtures are generated in-test (docx via python-docx, tiny hand-built PDFs with
no system-library dependency, text/rtf inline).

## Out of scope (future specs)

OCR / vision extraction · `PatientData` population · agent-context injection ·
document Q&A · layout-aware/multi-column fidelity.
