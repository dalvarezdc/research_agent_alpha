# Medical Analysis Agents

A multi-agent medical analysis system that produces structured, evidence-based
reports for procedures, medications, diagnoses, and general health questions.
Attach patient documents (PDFs, Word files, lab reports) to ground the analysis
in real clinical data.

---

## Table of Contents

1. [What it does](#what-it-does)
2. [Installation](#installation)
3. [Quick start](#quick-start)
4. [Interactive router](#interactive-router)
   - [Starting a session](#starting-a-session)
   - [Attaching a document](#attaching-a-document)
   - [Router commands](#router-commands)
5. [Direct CLI](#direct-cli)
   - [Procedure analyzer](#procedure-analyzer)
   - [Medication analyzer](#medication-analyzer)
   - [Fact checker](#fact-checker)
   - [Diagnostic analyzer](#diagnostic-analyzer)
6. [REST API](#rest-api)
7. [Document parsing](#document-parsing)
8. [Output files](#output-files)
9. [Configuration](#configuration)
10. [Python API](#python-api)
11. [Observability](#observability)
12. [Troubleshooting](#troubleshooting)

---

## What it does

Four specialized agents, each addressing a different clinical question type:

| Agent | Use for | Example query |
|-------|---------|---------------|
| **Procedure** | Surgeries, scans, interventions — organ-by-organ risk and peri-op care | "Laparoscopic cholecystectomy" |
| **Medication** | Drug pharmacology, interactions, dosing, safety, monitoring | "Warfarin in a patient on Aspirin" |
| **Diagnostic** | Symptom-to-condition pipeline (Bayesian + LLM) | "fatigue, weight gain, cold intolerance" |
| **Fact checker** | Open evidence questions, multi-perspective investigation | "Vitamin D supplementation — optimal dosing" |

The interactive router automatically picks the right agent for each query. You
can also drive any agent directly from the CLI or REST API.

---

## Installation

Requirements: **Python 3.12+** and the **[uv](https://docs.astral.sh/uv/)** package manager.

```bash
# Clone and enter the repository
git clone <repo-url>
cd research_agent_alpha

# Install core dependencies
uv sync

# Install optional PDF/Word parsing formats (.rtf, legacy .doc)
uv sync --extra parsing-extras

# Install dev dependencies (tests, linting)
uv sync --extra dev
```

### API keys

Set at least one provider key. The default model is `grok-4.3` (xAI):

```bash
# .env or shell
XAI_API_KEY="your-xai-key"           # grok-4.3 (default)
ANTHROPIC_API_KEY="your-key"          # Claude models
OPENAI_API_KEY="your-key"             # GPT-4o
```

Optional integrations:

```bash
TAVILY_API_KEY="your-key"             # Tavily web search (best quality)
SERPAPI_API_KEY="your-key"            # SerpAPI web search (fallback)
# DuckDuckGo is the free fallback — no key required
```

Vertex AI (Claude/Gemini on GCP):

```bash
VERTEX_PROJECT="your-gcp-project-id"
VERTEX_LOCATION="us-east5"            # optional, defaults to us-east5
```

Create a `.env` file in the project root — it is loaded automatically.

---

## Quick start

```bash
# Check which LLM providers are configured
uv run python router.py --check-llms

# List available model identifiers
uv run python router.py --models

# Start the interactive router (recommended entry point)
uv run python router.py
```

---

## Interactive router

The router is the recommended way to use the system. It accepts natural-language
queries, picks the right agent automatically, and lets you attach documents for
grounded analysis.

### Starting a session

```bash
uv run python router.py
```

At startup you select an LLM model (or press Enter for the default), then type
queries in a loop.

```
Medical Multi-Agent Router
==================================================
Available LLM models by supplier:

  xAI:
    1. grok-4.3 (default)

  Anthropic:
    2. claude-sonnet
    3. claude-opus

Select a model (1-3) or press Enter for default [grok-4.3]:

Using model: grok-4.3
Implementation: langchain
Web research: enabled

Commands:
  - Type a query to route and execute it
  - '/models' to list available models
  - '/model <number>' to change model
  - '/impl <original|langchain>' to change implementation
  - '/web <on|off>' to toggle web research (on by default)
  - '/file <path>' to attach a document as context (PDF/Word/txt/md/rtf)
  - '/file' to show attachment status, '/file clear' to remove
  - 'quit' or 'exit' to stop

Enter query:
```

Type any medical question — the router decides which agent to use:

```
Enter query: What are the main risks of laparoscopic cholecystectomy?
→ Routing query...
→ Routed to: procedure_agent (Medical Procedure Specialist)
→ Executing Medical Procedure Specialist...
```

### Attaching a document

Use `/file` to parse a document and attach its content as context for your
queries. The document **stays attached** (sticky) until you explicitly clear it
— you can ask multiple questions against the same file.

```
Enter query: /file /path/to/patient_labs.pdf
✓ Attached patient_labs.pdf (pdf, 2 pages, 3,840 chars). Stays attached until '/file clear'.

Enter query: Is the TSH level in this report a concern?
→ Routed to: diagnostic_agent ...

Enter query: What medications interact with levothyroxine?
→ Routed to: medication_agent ...

Enter query: /file clear
→ Document context cleared.
```

The document content is injected alongside web research context in every agent
prompt. Without a document, the system behaves exactly as before — attaching one
is always optional.

**Supported file types for `/file`:**

| Format | Extension | Availability |
|--------|-----------|--------------|
| PDF | `.pdf` | always available (priority) |
| Word | `.docx` | always available |
| Markdown | `.md`, `.markdown` | always available |
| Plain text | `.txt` | always available |
| Rich text | `.rtf` | `uv sync --extra parsing-extras` |
| Legacy Word | `.doc` | `uv sync --extra parsing-extras` |

**Truncation:** documents over 100,000 characters are truncated. You will see a
notification showing the original size, how many characters were dropped, and the
percentage:

```
⚠️  Document is 148,320 chars — exceeds the 100,000-char context limit.
    Truncated to 100,000 chars (dropped 48,320 chars, ~32.6% of the document).
    Only the first 100,000 chars will be used as context.
```

**Partial documents:** if some PDF pages have no extractable text (e.g.
scanned/image-only), the document is still attached with a per-page warning.
Pure-image PDFs cannot be read without OCR — install `parsing-extras` and an
OCR backend, or use a text-layer PDF.

### Router commands

| Command | Effect |
|---------|--------|
| `/file <path>` | Parse and attach a document as query context |
| `/file` | Show current attachment status (attached / none, size) |
| `/file clear` | Drop the attached document |
| `/model <n>` | Switch to model number `n` from the menu |
| `/models` | List available models |
| `/impl original\|langchain` | Switch agent implementation |
| `/web on\|off` | Toggle web research (default: on) |
| `quit` / `exit` | Exit the router |

### Router flags

```bash
# Disable web research globally
uv run python router.py --no-web-search

# Use the original DSPy-based implementation
uv run python router.py --implementation original

# Check provider configuration and exit
uv run python router.py --check-llms

# List model identifiers and exit
uv run python router.py --models
```

---

## Direct CLI

Bypass the router and run a specific agent directly.

### Procedure analyzer

Organ-focused analysis of a medical procedure — risks, peri-op care, affected
organs, evidence-based and investigational recommendations.

```bash
uv run python run_analysis.py procedure \
  --subject "Laparoscopic cholecystectomy" \
  --details "Elective, ASA II patient"
```

With options:
```bash
uv run python run_analysis.py procedure \
  --subject "MRI with gadolinium contrast" \
  --llm claude-sonnet \
  --web-search \
  --output-dir reports/mri
```

### Medication analyzer

Drug pharmacology, interactions, adverse effects, monitoring, debunked claims.

```bash
uv run python run_analysis.py medication \
  --subject "Warfarin" \
  --indication "Atrial Fibrillation" \
  --other-meds "Aspirin" "Amoxicillin" "Simvastatin"
```

```bash
uv run python run_analysis.py medication \
  --subject "Metformin" \
  --indication "Type 2 Diabetes" \
  --llm grok-4.3 \
  --web-search
```

### Fact checker

Five-phase investigation (conflict scan → evidence audit → synthesis →
multi-perspective output → simplified patient version). Produces mainstream,
naturist, and biohacker perspectives, then a balanced synthesis.

```bash
uv run python run_analysis.py factcheck \
  --subject "Vitamin D supplementation" \
  --context "optimal dosing for adults"
```

```bash
uv run python run_analysis.py factcheck \
  --subject "Statin therapy" \
  --llm claude-opus \
  --web-search \
  --implementation langchain
```

### Diagnostic analyzer

Bayesian + LLM five-level symptom-to-condition pipeline. Provide symptoms as
the subject.

```bash
uv run python run_analysis.py diagnostic \
  --subject "fatigue, weight gain, cold intolerance, dry skin"
```

### Common CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--llm` | `grok-4.3` | Provider: `claude-sonnet`, `claude-opus`, `openai`, `grok-4.3`, `ollama` |
| `--implementation` | `langchain` | `langchain` or `original` (DSPy) |
| `--web-search` | off | Enable web research (LangChain only) |
| `--output-dir` | `outputs/` | Where to write output files |
| `--timeout` | `300` | API timeout in seconds |

---

## REST API

Start the API server:

```bash
uv run python api.py
# or
uv run medical-api --host 0.0.0.0 --port 8000
```

### Parse a document

```bash
# Upload a file and receive markdown + metadata
curl -X POST http://localhost:8000/parse \
  -F "file=@/path/to/report.pdf"
```

Response:

```json
{
  "filename": "report.pdf",
  "status": "success",
  "markdown": "## MEDICAL REPORT\n\nPatient shows...",
  "warnings": [],
  "metadata": {
    "file_format": "pdf",
    "backend": "pdf",
    "page_count": 3,
    "char_count": 4210
  }
}
```

The endpoint accepts PDF, Word, txt, md, and rtf. Files over 25 MB (configurable
via `MAX_PARSE_UPLOAD_BYTES`) return HTTP 413. An unparseable file returns HTTP
200 with `status: "failed"` and a `warnings` list explaining why.

### Route a query (no execution)

```bash
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the risks of Warfarin?"}'
```

### Run a full analysis (synchronous)

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Vitamin D supplementation",
    "model": "grok-4.3",
    "web_search": true
  }'
```

### Run a full analysis (async / background job)

```bash
# Start job
curl -X POST http://localhost:8000/analyze/async \
  -H "Content-Type: application/json" \
  -d '{"query": "Metformin interactions", "model": "grok-4.3"}'
# → {"job_id": "abc-123", "status": "pending", "check_status_url": "/jobs/abc-123"}

# Poll for result
curl http://localhost:8000/jobs/abc-123
```

### Other endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service health check |
| `GET` | `/agents` | List routable agents |
| `GET` | `/models` | List available LLM models |
| `GET` | `/outputs/{file}` | Serve a generated output file |

---

## Document parsing

The `document_parser` package is a standalone service for converting documents
to structured GFM markdown. Use it independently of the router:

```python
from document_parser import parse_document, ParseStatus

result = parse_document("bloodwork.pdf")

if result.ok:                          # SUCCESS or PARTIAL
    print(result.markdown)
    for w in result.warnings:          # per-page warnings if PARTIAL
        print(f"Warning: {w}")
else:
    print(f"Failed: {result.warnings[0]}")

print(result.metadata.page_count)     # int | None
print(result.metadata.char_count)     # int
```

CLI:

```bash
# Print markdown to stdout
uv run python -m document_parser report.pdf

# Write to file
uv run python -m document_parser report.pdf -o report.md

# Override format when extension is wrong
uv run python -m document_parser mystery_file --format pdf
```

PDF features: table extraction → GFM tables, heading detection, page markers
(`<!-- page N -->`), per-page pdfplumber + pypdf fallback. A `PARTIAL` status
means some pages had no extractable text (likely scanned) — the rest is still
usable.

See [`document_parser/README.md`](document_parser/README.md) for full API
reference, module map, and backend details.

---

## Output files

All outputs go to `outputs/` (override with `--output-dir`). File names follow
`{subject}_{type}_{timestamp}` pattern.

### Procedure analyzer

| File | Contents |
|------|----------|
| `*_reasoning_trace.json` | Per-organ LLM reasoning steps |
| `*_analysis_result.json` | Structured result object |
| `*_practitioner_report.md` + `.pdf` | Full technical report |
| `*_summary_report.md` + `.pdf` | Executive summary |
| `*_cost_report.json` | Token usage and cost per phase |
| `*_audit.json` | Every LLM prompt + response (if audit enabled) |

### Medication analyzer

| File | Contents |
|------|----------|
| `*_medication_analysis.json` | Structured analysis |
| `*_practitioner_report.md` + `.pdf` | Clinical detail report |
| `*_medication_summary.md` + `.pdf` | Brief summary |
| `*_medication_detailed.md` + `.pdf` | Extended detail |
| `*_cost_report.json` | Token/cost breakdown |

### Fact checker

| File | Contents |
|------|----------|
| `*_session.json` | Full multi-phase session |
| `*_practitioner_report.md` + `.pdf` | All perspectives, full citations |
| `*_patient_report.md` + `.pdf` | Simplified patient version |
| `*_summary.md` + `.pdf` | Balanced synthesis |
| `*_cost_report.json` | Token/cost breakdown |

---

## Configuration

All configuration is via environment variables or a `.env` / `.env.dev` file in
the project root (loaded automatically).

| Variable | Default | Description |
|----------|---------|-------------|
| `XAI_API_KEY` | — | xAI (Grok) API key |
| `ANTHROPIC_API_KEY` | — | Anthropic (Claude) API key |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `VERTEX_PROJECT` | — | GCP project ID for Vertex AI |
| `VERTEX_LOCATION` | `us-east5` | GCP region for Vertex AI |
| `TAVILY_API_KEY` | — | Tavily web search (best quality) |
| `SERPAPI_API_KEY` | — | SerpAPI web search (fallback) |
| `DATABASE_URL` | `sqlite:///data/app.db` | SQLAlchemy connection URL |
| `DB_PERSISTENCE_ENABLED` | `true` | Set `false` to disable DB writes entirely |
| `APP_ENV` | `local` | `local`/`dev` → developer user; other → no auth user |
| `MAX_PARSE_UPLOAD_BYTES` | `26214400` | Max API upload size for `/parse` (25 MB) |
| `LANGCHAIN_TRACING_V2` | — | Set `true` to enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | — | LangSmith API key |
| `LANGCHAIN_PROJECT` | `research-agent-alpha` | LangSmith project name |

---

## Python API

### Using the orchestrator directly

```python
from run_analysis import AgentOrchestrator

orch = AgentOrchestrator(output_dir="outputs")

# Procedure
result, files = orch.run_procedure_analyzer(
    procedure="Laparoscopic cholecystectomy",
    details="Elective",
    llm_provider="grok-4.3",
    enable_web_research=True,
)

# Medication
result, files = orch.run_medication_analyzer(
    medication="Warfarin",
    indication="Atrial Fibrillation",
    other_medications=["Aspirin"],
    llm_provider="claude-sonnet",
)

# Fact checker
session, files = orch.run_fact_checker(
    subject="Vitamin D supplementation",
    context="optimal dosing",
    llm_provider="grok-4.3",
    enable_web_research=True,
)

# Diagnostic
result, files = orch.run_diagnostic_analyzer(
    query="fatigue, cold intolerance, weight gain",
    llm_provider="claude-sonnet",
    interactive=False,
)
```

All four methods accept an optional `document_context: str` parameter to provide
pre-parsed document content as grounding:

```python
from document_parser import parse_document

parsed = parse_document("patient_labs.pdf")
if parsed.ok:
    result, files = orch.run_medication_analyzer(
        medication="Levothyroxine",
        document_context=parsed.markdown,
    )
```

### Agents directly

```python
from langchain_agents.factcheck_agent import LangChainMedicalFactChecker
from langchain_agents.medication_agent import LangChainMedicationAnalyzer
from langchain_agents.procedure_agent import LangChainMedicalReasoningAgent
```

---

## Observability

LLM request tracing via [Arize Phoenix](https://phoenix.arize.com/) is
always-on when running the router. The URL is printed at startup:

```
Tracing (Phoenix): http://localhost:6006
```

Open that URL to see full traces including routing decisions, agent phases, token
counts, and cost attribution per span.

| Span | Name | Key attributes |
|------|------|----------------|
| Full session | `router.session` | `query`, `routed_to`, `model`, `document_context_attached` |
| Router LLM call | `llm.call` | `llm.model_name`, `llm.provider`, token counts |
| Agent phases | auto-instrumented | model, prompt, completion, latency |

To keep Phoenix running across router sessions:

```bash
# Terminal 1 — Phoenix server
uv run python -m phoenix.server.main serve

# Terminal 2 — Router
uv run python router.py
```

Phoenix fails silently if the port is unavailable — the router continues normally.

---

## Troubleshooting

**`No such file or directory: pytest`**
```bash
uv sync --extra dev
```

**API key errors / no provider configured**
```bash
uv run python router.py --check-llms
```
Set the missing key in `.env` and re-run.

**Vertex AI fails with missing VERTEX_PROJECT**

The router catches this and falls back to `grok-4.3` automatically. Set
`VERTEX_PROJECT` in `.env.dev` to use Vertex models.

**`/file` says "Could not parse" for a PDF**

- Is the PDF text-based or scanned? Scanned PDFs have no text layer — the parser
  cannot read them without OCR.
- Try the CLI to see detailed warnings: `uv run python -m document_parser file.pdf`
- For `.rtf` or `.doc` files: `uv sync --extra parsing-extras`

**`/file` truncation warning**

Your document is over 100,000 characters. Only the first 100,000 are used as
context. Consider splitting the document or using the most relevant section.

**PDF generation fails**

WeasyPrint requires system libraries (Pango/GObject). PDF output is optional —
markdown files are always produced regardless. See
[WeasyPrint installation](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html).

**Web research returns no results**

DuckDuckGo is the free fallback and works without a key. For higher-quality
results set `TAVILY_API_KEY` or `SERPAPI_API_KEY`.

**DB writes fail silently**

DB persistence is best-effort — a failure never blocks a run. To disable it
entirely: `DB_PERSISTENCE_ENABLED=false`. To inspect the DB:
```python
import database
database.init_db()
with database.session_scope() as s:
    print(database.list_reports(s))
```

---

## Testing

```bash
# Full suite
uv run python -m pytest tests/ -q

# Specific modules
uv run python -m pytest tests/test_database.py -v
uv run python -m pytest tests/test_document_parser.py -v
uv run python -m pytest tests/test_router_document_parser.py -v
```

---

## License

Part of the medical reasoning agent research project.

---

**Educational Use Only:** This system is for research and education. It does not
provide medical advice or diagnosis. Always consult qualified healthcare
professionals for clinical decisions.
