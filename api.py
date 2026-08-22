"""
FastAPI REST API service for the medical multi-agent system.
Exposes query routing, synchronous analysis, and asynchronous background jobs.
"""

import errno
import os
import sys
import uuid
import json
import logging
import tempfile
import threading
import traceback
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterator

from fastapi import FastAPI, BackgroundTasks, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Load environment variables: .env.dev first (dev-specific), then .env (base)
try:
    from dotenv import load_dotenv as _load_dotenv
    import pathlib as _pathlib

    _repo_root = _pathlib.Path(__file__).parent
    _load_dotenv(_repo_root / ".env.dev", override=False)  # dev-specific vars
    _load_dotenv(_repo_root / ".env", override=False)       # base vars (don't overwrite)
except ImportError:
    pass  # python-dotenv not installed, rely on shell environment

# Import database persistence
from database.session import ensure_initialized, session_scope
from database import repository

# Import agent router functions and models
from router import route_agent, sample_agents, DEFAULT_ROUTING_MODEL
from run_analysis import AgentOrchestrator
from llm_integrations import get_available_models, create_llm_manager
from document_parser import parse_document
from medical_report_categorizer import (
    categorize_medical_markdown,
    classify_patient_description,
)
import app_config

# Maximum upload size for document parsing (bytes). Default 25 MB; override via env.
MAX_PARSE_UPLOAD_BYTES = int(os.getenv("MAX_PARSE_UPLOAD_BYTES", str(25 * 1024 * 1024)))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("medical_api")

# Load UI-managed secrets/webhooks and apply API keys into the process environment
# (after dotenv so .env remains the base; stored keys override for local UI setup).
try:
    app_config.load_and_apply()
except Exception as _cfg_err:  # noqa: BLE001 — never block API startup on config I/O
    logger.warning("Could not load app_config: %s", _cfg_err)

app = FastAPI(
    title="Medical Multi-Agent API",
    description=(
        "REST API service for classifying medical queries and invoking specialized "
        "medical agents (Medication, Procedure, Diagnostic, and Fact Checker)."
    ),
    version="0.1.0"
)

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure outputs directory exists and serve files statically
os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# In-memory thread-safe storage for asynchronous jobs
jobs_lock = threading.Lock()
jobs: Dict[str, Dict[str, Any]] = {}


class RouteRequest(BaseModel):
    query: str = Field(..., description="The query to be routed to a specialized agent.")
    model: str = Field(DEFAULT_ROUTING_MODEL, description="LLM model identifier to use for routing.")


class AnalyzeRequest(BaseModel):
    query: str = Field(..., description="The medical query/subject to analyze.")
    model: str = Field(DEFAULT_ROUTING_MODEL, description="LLM model to use for analysis.")
    implementation: str = Field("langchain", description="Agent implementation to use ('original' or 'langchain').")
    web_search: bool = Field(True, description="Whether to enable web search for the agent.")
    timeout: int = Field(300, description="LLM API timeout in seconds.")
    agent_id: Optional[str] = Field(None, description="Target agent ID override (medication_agent, procedure_agent, diagnostic_agent, general_agent).")
    patient_id: Optional[str] = Field(None, description="Linked patient ID for grouping conversations.")
    context_job_ids: Optional[List[str]] = Field(None, description="List of prior conversation/job IDs to include as context.")
    # Optional pre-built markdown describing patient/document/intake context sent with the query.
    # When provided, written to outputs as context_report.md; otherwise a minimal report is generated.
    context_report: Optional[str] = Field(
        None,
        description="Markdown report of the clinical context assembled for the agent.",
    )


class IntakeChatMessage(BaseModel):
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message text content")


class IntakeChatRequest(BaseModel):
    messages: List[IntakeChatMessage] = Field(..., description="Intake conversation history so far.")
    model: Optional[str] = Field(DEFAULT_ROUTING_MODEL, description="LLM model to use for intake assistant.")
    document_context: Optional[str] = Field(None, description="Optional background document context.")


class IntakeSummarizeRequest(BaseModel):
    messages: List[IntakeChatMessage] = Field(..., description="Full intake conversation history to summarize.")
    document_context: Optional[str] = Field(None, description="Optional background document context.")
    model: Optional[str] = Field(DEFAULT_ROUTING_MODEL, description="LLM model to use for prompt synthesis.")



class RegenerateRequest(BaseModel):
    agent_id: Optional[str] = Field(None, description="Agent ID override (medication_agent, procedure_agent, diagnostic_agent, general_agent).")
    model: str = Field(DEFAULT_ROUTING_MODEL, description="LLM model to use.")
    implementation: str = Field("langchain", description="Agent implementation.")
    web_search: bool = Field(True, description="Enable web search.")
    timeout: int = Field(300, description="Timeout in seconds.")


class SlackNotifyRequest(BaseModel):
    webhook_url: Optional[str] = Field(
        None, description="Slack Incoming Webhook URL (optional if webhook_id is set)"
    )
    webhook_id: Optional[str] = Field(
        None, description="ID of a saved Slack webhook from /config/slack-webhooks"
    )
    job_ids: List[str] = Field(..., description="List of task/job IDs to include in the notification")


class SlackWebhookCreate(BaseModel):
    name: str = Field("Slack Webhook", description="Friendly label for this webhook")
    url: str = Field(..., description="Slack Incoming Webhook URL (https://...)")


class SlackWebhookUpdate(BaseModel):
    name: Optional[str] = Field(None, description="Friendly label for this webhook")
    url: Optional[str] = Field(None, description="Slack Incoming Webhook URL (https://...)")


class ApiKeyUpsert(BaseModel):
    env_var: str = Field(..., description="Environment variable name, e.g. GROK_API_KEY")
    value: str = Field(..., description="Secret value / API key / project id")


class PatientCreate(BaseModel):
    name: str = Field(..., description="Patient full name")
    age: Optional[int] = Field(None, description="Patient age")
    gender: Optional[str] = Field(None, description="Patient gender")
    primary_condition: Optional[str] = Field(None, description="Primary condition")
    contact_email: Optional[str] = Field(None, description="Contact email")
    contact_phone: Optional[str] = Field(None, description="Contact phone")
    metadata_json: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Key-value metadata dictionary")
    clinical_data: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Structured clinical data tables")


class PatientUpdate(BaseModel):
    name: Optional[str] = Field(None, description="Patient full name")
    age: Optional[int] = Field(None, description="Patient age")
    gender: Optional[str] = Field(None, description="Patient gender")
    primary_condition: Optional[str] = Field(None, description="Primary condition")
    contact_email: Optional[str] = Field(None, description="Contact email")
    contact_phone: Optional[str] = Field(None, description="Contact phone")
    metadata_json: Optional[Dict[str, Any]] = Field(None, description="Key-value metadata dictionary")
    clinical_data: Optional[Dict[str, Any]] = Field(None, description="Structured clinical data tables")


class PatientClassifyRequest(BaseModel):
    text: str = Field(..., description="Patient free-form description or clinical summary")
    model: str = Field(DEFAULT_ROUTING_MODEL, description="LLM model identifier to perform classification")


def _patient_to_dict(patient) -> Dict[str, Any]:
    return {
        "id": patient.id,
        "name": patient.name,
        "age": patient.age,
        "gender": patient.gender,
        "primary_condition": patient.primary_condition,
        "contact_email": patient.contact_email,
        "contact_phone": patient.contact_phone,
        "metadata_json": patient.metadata_json or {},
        "clinical_data": patient.clinical_data or {},
        "created_at": patient.created_at.isoformat() if getattr(patient, "created_at", None) else None,
        "updated_at": patient.updated_at.isoformat() if getattr(patient, "updated_at", None) else None,
    }




def load_json_result(agent_id: str, files: Dict[str, str]) -> Dict[str, Any]:
    """
    Helper function to load the primary JSON results generated by each agent.
    """
    key_map = {
        "diagnostic_agent": "json_session",
        "procedure_agent": "result",
        "general_agent": "session",
        "medication_agent": "result"
    }
    key = key_map.get(agent_id)
    if not key or key not in files:
        return {}

    file_path = files[key]
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON result from {file_path}: {e}")
            return {"error": f"Failed to parse JSON file: {e}"}
    return {"warning": f"JSON result file not found at {file_path}"}


def _persist_conversation_update(job_id: str, **fields: Any) -> None:
    """Best-effort write of conversation fields to the durable store."""
    try:
        ensure_initialized()
        with session_scope() as session:
            repository.update_conversation(session, job_id, **fields)
    except Exception as e:
        logger.warning(f"Failed to update conversation {job_id}: {e}")


def _job_to_public_dict(job: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize an in-memory job for API responses (ISO dates + has_docs)."""
    out = dict(job)
    for key in ("created_at", "updated_at"):
        val = out.get(key)
        if isinstance(val, datetime):
            out[key] = val.isoformat()
    files = out.get("files") or {}
    out["has_docs"] = repository._conversation_has_docs(files) if files else False
    return out


def _slug_for_filename(text: str, max_len: int = 48) -> str:
    """Make a filesystem-safe short slug from a query string.

    Uses only the first non-empty line (before any attached context blocks)
    so patient/document appendices do not bloat the filename.
    """
    import re

    first_line = ""
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("---"):
            if first_line:
                break
            continue
        first_line = stripped
        break

    slug = re.sub(r"[^\w\s-]", "", first_line[:120], flags=re.UNICODE)
    slug = re.sub(r"[-\s]+", "_", slug).strip("_").lower()
    if not slug:
        slug = "analysis"
    return slug[:max_len]


def write_context_report_md(
    *,
    query: str,
    model: str,
    agent_id: str,
    web_search: bool,
    implementation: str = "langchain",
    context_report: Optional[str] = None,
    job_id: Optional[str] = None,
    output_dir: str = "outputs",
) -> str:
    """
    Write a small markdown artifact documenting the exact context sent to the agent.

    If the client supplied a pre-built ``context_report``, it is used as the body
    and enriched with final routing metadata. Otherwise a minimal report is built
    from the full query string (which already includes any patient/document text).

    Returns the relative path of the written file.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = _slug_for_filename(query)
    rel_path = f"{output_dir}/{base}_context_report_{timestamp}.md"

    header_lines = [
        "# Agent Context Report",
        "",
        "Small audit of the clinical context and final prompt payload delivered to the specialized agent.",
        "",
        "## Run metadata",
        "",
        f"- **Generated:** {datetime.now().isoformat(timespec='seconds')}",
        f"- **Job ID:** `{job_id or 'n/a'}`",
        f"- **Model:** `{model}`",
        f"- **Implementation:** `{implementation}`",
        f"- **Agent:** `{agent_id}`",
        f"- **Web search:** {'enabled' if web_search else 'disabled'}",
        f"- **Query length:** {len(query or '')} characters",
        "",
    ]

    if context_report and context_report.strip():
        body = context_report.strip()
        # Avoid duplicating a top-level title if the client already included one.
        if body.lstrip().startswith("#"):
            # Client report is authoritative for composition details; append run metadata first.
            content = "\n".join(header_lines) + "\n---\n\n" + body + "\n"
        else:
            content = "\n".join(header_lines) + "\n" + body + "\n"
    else:
        content = "\n".join(header_lines) + "\n".join(
            [
                "## Final prompt sent to agent",
                "",
                "The following text is the full `query` / subject string passed into the agent pipeline",
                "(including any patient context, attached document text, and intake-chat synthesis).",
                "",
                "```text",
                (query or "").rstrip() or "(empty)",
                "```",
                "",
            ]
        )

    with open(rel_path, "w", encoding="utf-8") as fh:
        fh.write(content)

    logger.info("Wrote context report: %s", rel_path)
    return rel_path


def execute_analysis_sync(
    query: str,
    model: str,
    implementation: str,
    web_search: bool,
    timeout: int,
    agent_id_override: Optional[str] = None,
    job_id: Optional[str] = None,
    context_report: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Runs the full routing and execution pipeline synchronously.

    When ``job_id`` is provided, the Report row is keyed with the same id so
    conversation delete can purge report + files in one shot.
    """
    # Map model name to provider name
    available_models_dict = get_available_models()
    llm_provider = available_models_dict.get(model, model)

    # 1. Route or select agent
    if agent_id_override:
        routed_agent_id = agent_id_override
        logger.info(f"Using explicit agent override: '{routed_agent_id}' for query '{query}'")
    else:
        logger.info(f"Routing query: '{query}' using model '{model}'")
        routed_agent_id = route_agent(
            query,
            sample_agents,
            default_agent_id="general_agent",
            model=model
        )
        logger.info(f"Routed query to agent: {routed_agent_id}")

    # 1b. Write small context audit report (what is being sent to the agent)
    context_path = write_context_report_md(
        query=query,
        model=model,
        agent_id=routed_agent_id,
        web_search=web_search,
        implementation=implementation,
        context_report=context_report,
        job_id=job_id,
        output_dir="outputs",
    )

    # 2. Run the specialized agent via AgentOrchestrator
    orchestrator = AgentOrchestrator(output_dir="outputs")
    files: Dict[str, Any] = {"context_report": context_path}

    if routed_agent_id == "medication_agent":
        _, agent_files = orchestrator.run_medication_analyzer(
            medication=query,
            indication=None,
            other_medications=None,
            llm_provider=llm_provider,
            timeout=timeout,
            implementation=implementation,
            enable_web_research=web_search,
        )
        files.update(agent_files or {})
    elif routed_agent_id == "procedure_agent":
        _, agent_files = orchestrator.run_procedure_analyzer(
            procedure=query,
            details="API Request",
            llm_provider=llm_provider,
            timeout=timeout,
            implementation=implementation,
            enable_web_research=web_search,
        )
        files.update(agent_files or {})
    elif routed_agent_id == "diagnostic_agent":
        _, agent_files = orchestrator.run_diagnostic_analyzer(
            query=query,
            llm_provider=llm_provider,
            timeout=timeout,
            interactive=False,
        )
        files.update(agent_files or {})
    elif routed_agent_id == "general_agent":
        _, agent_files = orchestrator.run_fact_checker(
            subject=query,
            context="",
            llm_provider=llm_provider,
            timeout=timeout,
            implementation=implementation,
            enable_web_research=web_search,
        )
        files.update(agent_files or {})
    else:
        raise ValueError(f"Unknown routed agent: {routed_agent_id}")

    # Ensure context report key survives agent file merges
    files["context_report"] = context_path

    # Load result data from file
    loaded_data = load_json_result(routed_agent_id, files)

    # Persist report to DB (best-effort). Use job_id as report PK when available.
    report_id = None
    try:
        ensure_initialized()
        with session_scope() as session:
            report = repository.persist_report(
                session=session,
                agent_type=routed_agent_id,
                subject_text=query,
                files=files,
                llm_provider=llm_provider,
                implementation=implementation,
                report_id=job_id,
            )
            report_id = report.id
    except Exception as e:
        logger.warning(f"Failed to persist report to database: {e}")

    return {
        "agent_id": routed_agent_id,
        "files": files,
        "result": loaded_data,
        "report_id": report_id,
    }


class _NonBrokenStream:
    """Wrap stdout/stderr so CLI ``print()`` cannot raise BrokenPipeError.

    Background analysis jobs run under uvicorn/FastAPI where the client pipe
    may already be closed; orchestrator code still uses print for CLI banners.
    """

    __slots__ = ("_stream",)

    def __init__(self, stream: Any) -> None:
        self._stream = stream

    def write(self, data: Any) -> int:
        try:
            return self._stream.write(data)
        except BrokenPipeError:
            return len(data) if isinstance(data, (str, bytes, bytearray)) else 0
        except OSError as exc:
            if getattr(exc, "errno", None) in (errno.EPIPE, 32):
                return len(data) if isinstance(data, (str, bytes, bytearray)) else 0
            raise

    def flush(self) -> None:
        try:
            self._stream.flush()
        except BrokenPipeError:
            pass
        except OSError as exc:
            if getattr(exc, "errno", None) not in (errno.EPIPE, 32):
                raise

    def isatty(self) -> bool:
        try:
            return bool(self._stream.isatty())
        except Exception:
            return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


@contextmanager
def _shield_stdio_from_broken_pipe() -> Iterator[None]:
    """Temporarily wrap process stdio so print() never aborts background work."""
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _NonBrokenStream(old_out)  # type: ignore[assignment]
    sys.stderr = _NonBrokenStream(old_err)  # type: ignore[assignment]
    try:
        yield
    finally:
        sys.stdout = old_out
        sys.stderr = old_err


def run_background_job(
    job_id: str,
    query: str,
    model: str,
    implementation: str,
    web_search: bool,
    timeout: int,
    agent_id_override: Optional[str] = None,
    context_report: Optional[str] = None,
):
    """
    Worker task for executing an analysis asynchronously in the background.
    """
    logger.info(f"Starting background job: {job_id}")
    with jobs_lock:
        jobs[job_id]["status"] = JobStatus.RUNNING
        jobs[job_id]["updated_at"] = datetime.now()
    _persist_conversation_update(job_id, status=JobStatus.RUNNING)

    try:
        # Orchestrator / agents still use print() for CLI progress. Under uvicorn
        # a closed client pipe raises BrokenPipeError and would fail the job.
        with _shield_stdio_from_broken_pipe():
            data = execute_analysis_sync(
                query=query,
                model=model,
                implementation=implementation,
                web_search=web_search,
                timeout=timeout,
                agent_id_override=agent_id_override,
                job_id=job_id,
                context_report=context_report,
            )
        with jobs_lock:
            jobs[job_id]["status"] = JobStatus.COMPLETED
            jobs[job_id]["agent_id"] = data["agent_id"]
            jobs[job_id]["files"] = data["files"]
            jobs[job_id]["result"] = data["result"]
            jobs[job_id]["report_id"] = data.get("report_id")
            jobs[job_id]["updated_at"] = datetime.now()
        _persist_conversation_update(
            job_id,
            status=JobStatus.COMPLETED,
            agent_id=data["agent_id"],
            files=data["files"],
            result=data["result"] if isinstance(data.get("result"), dict) else None,
            report_id=data.get("report_id") or job_id,
        )
        logger.info(f"Successfully completed background job: {job_id}")
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Failed background job: {job_id}. Error: {e}")
        with jobs_lock:
            jobs[job_id]["status"] = JobStatus.FAILED
            jobs[job_id]["error"] = f"{e}\n{tb}"
            jobs[job_id]["updated_at"] = datetime.now()
        _persist_conversation_update(
            job_id,
            status=JobStatus.FAILED,
            error=f"{e}\n{tb}",
        )



@app.get("/api-info")
def api_info_endpoint():
    """
    Metadata endpoint for Medical Multi-Agent API.
    Provides API metadata and links to interactive docs (/docs) and health check.
    """
    return {
        "title": app.title,
        "version": app.version,
        "description": app.description,
        "documentation": "/docs",
        "health": "/health",
        "endpoints": {
            "GET /health": "Health check status",
            "GET /agents": "List available specialized agents",
            "POST /route": "Route a medical query to an agent",
            "POST /analyze": "Run synchronous medical analysis",
            "POST /analyze/async": "Submit asynchronous medical analysis job",
            "GET /jobs": "List all background jobs and tasks",
            "GET /jobs/{job_id}": "Check status of an async job",
            "POST /slack/notify": "Send task notifications & descriptions to a Slack webhook",
            "POST /parse": "Parse uploaded PDF/Word document to markdown",
            "GET /config": "UI-safe app configuration (webhooks + masked API keys)",
            "GET/POST/PUT/DELETE /config/slack-webhooks": "Manage saved Slack webhooks",
            "GET/PUT/DELETE /config/api-keys": "Manage LLM provider API keys",
        }
    }


@app.get("/health")
def health_endpoint():
    """
    Status endpoint. Returns status, time, and default routing model.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "default_model": DEFAULT_ROUTING_MODEL
    }


@app.get("/agents")
def list_agents_endpoint():
    """
    List all routable medical agents and their specs.
    """
    return [
        {
            "id": agent.id,
            "name": agent.name,
            "description": agent.description,
            "routing_notes": agent.routing_notes
        }
        for agent in sample_agents
    ]


@app.get("/models")
def list_models_endpoint():
    """
    List available models and their mapped providers.
    """
    try:
        return get_available_models()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving models: {e}"
        )


@app.post("/route")
def route_query_endpoint(req: RouteRequest):
    """
    Classify a medical query and return the selected agent without running the analysis.
    """
    try:
        routed_agent_id = route_agent(
            req.query,
            sample_agents,
            default_agent_id="general_agent",
            model=req.model
        )
        agent = next((a for a in sample_agents if a.id == routed_agent_id), None)
        return {
            "query": req.query,
            "agent_id": routed_agent_id,
            "agent_name": agent.name if agent else "Unknown Agent"
        }
    except Exception as e:
        logger.error(f"Error during query routing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Routing failure: {e}"
        )


@app.post("/intake/chat")
@app.post("/intake/chat/")
def intake_chat_endpoint(req: IntakeChatRequest):
    """
    Interactive intake assistant endpoint. Evaluates query history (+ optional document context)
    and returns concise, focused clinical clarifying questions to enrich the prompt.
    """
    try:
        model_name = req.model or DEFAULT_ROUTING_MODEL
        llm_mgr = create_llm_manager(primary_provider=model_name)
        provider = llm_mgr.get_provider_direct() or llm_mgr.get_available_provider()
        if not provider:
            raise RuntimeError("No LLM provider available for intake assistant.")

        system_prompt = (
            "You are an expert clinical intake assistant. Your sole role is to help the user refine and enrich "
            "their medical query before it is submitted to specialized clinical AI agents for full research.\n"
            "Rules:\n"
            "1. Focus strictly on asking 1 to 2 high-yield, relevant clinical clarifying questions (e.g. patient demographics, "
            "symptom duration, specific drug dosages, medical history/comorbidities, contraindications, or primary objective).\n"
            "2. If selected patient context or an attached clinical document is provided, treat it as known background—"
            "do not re-ask details already present there; use them to ask more targeted follow-ups.\n"
            "3. If the user's information is already thorough and complete, state clearly: 'Your query is detailed and ready for analysis run! Feel free to press Start Analysis Run or provide any final details.'\n"
            "4. Keep your response brief, professional, and directly actionable (2-4 sentences max).\n"
            "5. Do NOT attempt to perform the full medical analysis yourself or give diagnostic recommendations—only ask clarifying questions."
        )

        formatted_convo = ""
        for msg in req.messages:
            role_label = "User" if msg.role == "user" else "Assistant"
            formatted_convo += f"{role_label}: {msg.content}\n\n"

        if req.document_context and req.document_context.strip():
            formatted_convo += (
                "--- CLINICAL CONTEXT (selected patient and/or attached document) ---\n"
                f"{req.document_context.strip()}\n\n"
            )

        prompt = f"Below is the current intake conversation transcript:\n\n{formatted_convo}Please review the clinical prompt above and respond with clarifying questions or confirmation."

        res = provider.generate_response(prompt, system_prompt=system_prompt)
        reply_text = res[0] if isinstance(res, (tuple, list)) else str(res)
        return {
            "role": "assistant",
            "content": reply_text.strip()
        }
    except Exception as e:
        logger.error(f"Error in intake chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Intake chat failure: {e}"
        )


@app.post("/intake/summarize")
@app.post("/intake/summarize/")
def intake_summarize_endpoint(req: IntakeSummarizeRequest):
    """
    Synthesizes a multi-turn intake chat transcript (+ optional document context)
    into a unified, rich clinical query prompt ready for agent analysis.
    """
    try:
        model_name = req.model or DEFAULT_ROUTING_MODEL
        llm_mgr = create_llm_manager(primary_provider=model_name)
        provider = llm_mgr.get_provider_direct() or llm_mgr.get_available_provider()
        if not provider:
            raise RuntimeError("No LLM provider available for prompt synthesis.")

        system_prompt = (
            "You are a medical prompt synthesizer. Your task is to synthesize the provided intake chat conversation "
            "and clinical context into a single, cohesive, comprehensive clinical query prompt.\n"
            "Instructions:\n"
            "- Combine all user goals, patient details, symptoms, dosages, questions, and clarifying details into one structured prompt.\n"
            "- Omit chat preamble, greetings, and conversational filler.\n"
            "- Output ONLY the final synthesized clinical prompt, clear and ready for specialized medical analysis."
        )

        formatted_convo = ""
        for msg in req.messages:
            role_label = "User" if msg.role == "user" else "Assistant"
            formatted_convo += f"{role_label}: {msg.content}\n\n"

        if req.document_context and req.document_context.strip():
            formatted_convo += (
                "--- CLINICAL CONTEXT (selected patient and/or attached document) ---\n"
                f"{req.document_context.strip()}\n\n"
            )

        prompt = f"Synthesize the following intake transcript into a single clinical query prompt:\n\n{formatted_convo}"

        res = provider.generate_response(prompt, system_prompt=system_prompt)
        summary_text = res[0] if isinstance(res, (tuple, list)) else str(res)
        return {
            "summary": summary_text.strip()
        }
    except Exception as e:
        logger.error(f"Error in intake summarize: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Intake summarize failure: {e}"
        )


@app.post("/analyze")
def analyze_query_sync_endpoint(req: AnalyzeRequest):
    """
    Synchronously route and analyze a medical query. Blocks until analysis completes.
    """
    try:
        with _shield_stdio_from_broken_pipe():
            res = execute_analysis_sync(
                query=req.query,
                model=req.model,
                implementation=req.implementation,
                web_search=req.web_search,
                timeout=req.timeout,
                agent_id_override=req.agent_id,
                context_report=req.context_report,
            )
        return {
            "status": "success",
            "query": req.query,
            "agent_id": res["agent_id"],
            "files": res["files"],
            "result": res["result"]
        }
    except Exception as e:
        logger.error(f"Error during synchronous analysis execution: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis execution failure: {e}"
        )


@app.post("/analyze/async", status_code=status.HTTP_202_ACCEPTED)
def analyze_query_async_endpoint(req: AnalyzeRequest, background_tasks: BackgroundTasks):
    """
    Asynchronously route and analyze a medical query. Returns a job ID to poll for status.
    Conversation is persisted immediately so history survives restarts.
    """
    job_id = str(uuid.uuid4())
    now = datetime.now()

    job_record = {
        "id": job_id,
        "query": req.query,
        "agent_id": req.agent_id,
        "status": JobStatus.PENDING,
        "model": req.model,
        "implementation": req.implementation,
        "patient_id": req.patient_id,
        "created_at": now,
        "updated_at": now,
        "error": None,
        "files": None,
        "result": None,
        "report_id": None,
        "parent_job_id": None,
        "has_docs": False,
    }

    with jobs_lock:
        jobs[job_id] = job_record

    # Durable cache — create conversation row before work starts.
    try:
        ensure_initialized()
        with session_scope() as session:
            repository.create_conversation(
                session,
                conversation_id=job_id,
                query=req.query,
                agent_id=req.agent_id,
                status=JobStatus.PENDING,
                model=req.model,
                implementation=req.implementation,
                patient_id=req.patient_id,
            )
    except Exception as e:
        logger.warning(f"Failed to persist conversation {job_id}: {e}")

    background_tasks.add_task(
        run_background_job,
        job_id=job_id,
        query=req.query,
        model=req.model,
        implementation=req.implementation,
        web_search=req.web_search,
        timeout=req.timeout,
        agent_id_override=req.agent_id,
        context_report=req.context_report,
    )

    return {
        "job_id": job_id,
        "status": JobStatus.PENDING,
        "check_status_url": f"/jobs/{job_id}"
    }


@app.get("/jobs")
def list_jobs_endpoint():
    """
    List all conversations/jobs sorted by creation date descending.

    Merges live in-memory jobs with durable DB conversations. Existing Report
    rows without a Conversation are backfilled once so prior runs appear.
    """
    merged: Dict[str, Dict[str, Any]] = {}

    # Durable store first (survives restarts).
    try:
        ensure_initialized()
        with session_scope() as session:
            repository.backfill_conversations_from_reports(session)
            for conv in repository.list_conversations(session, limit=300):
                merged[conv.id] = repository.conversation_to_job_dict(conv)
    except Exception as e:
        logger.warning(f"Failed to load conversations from DB: {e}")

    # Overlay in-memory jobs (fresher status for running work).
    with jobs_lock:
        for jid, job in jobs.items():
            merged[jid] = _job_to_public_dict(job)

    job_list = list(merged.values())

    def _sort_key(j: Dict[str, Any]):
        created = j.get("created_at") or ""
        if isinstance(created, datetime):
            return created.isoformat()
        return str(created)

    job_list.sort(key=_sort_key, reverse=True)
    return job_list


@app.get("/jobs/{job_id}")
def get_job_status_endpoint(job_id: str):
    """
    Retrieve status and result of a background job / cached conversation.
    """
    with jobs_lock:
        job = jobs.get(job_id)

    if job:
        return _job_to_public_dict(job)

    # Fall back to durable conversation cache.
    try:
        ensure_initialized()
        with session_scope() as session:
            conv = repository.get_conversation(session, job_id)
            if conv:
                return repository.conversation_to_job_dict(conv)
            # Last resort: report row (pre-conversation era).
            rep = repository.get_report(session, job_id)
            if rep:
                files = {rf.file_type: rf.file_path for rf in rep.files}
                return {
                    "id": rep.id,
                    "query": rep.subject_text,
                    "agent_id": rep.agent_type,
                    "status": JobStatus.COMPLETED,
                    "files": files or None,
                    "result": None,
                    "report_id": rep.id,
                    "has_docs": repository._conversation_has_docs(files),
                    "created_at": rep.created_at.isoformat() if rep.created_at else None,
                    "updated_at": rep.created_at.isoformat() if rep.created_at else None,
                }
    except Exception as e:
        logger.warning(f"Failed to load conversation {job_id} from DB: {e}")

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Job not found"
    )


@app.delete("/jobs/{job_id}")
def delete_job_endpoint(job_id: str):
    """
    Delete a conversation with one action: in-memory job, DB conversation,
    linked report rows, and all associated on-disk report files.
    """
    removed_files: list[str] = []

    with jobs_lock:
        job = jobs.pop(job_id, None)

    if job:
        for fpath in (job.get("files") or {}).values():
            if fpath and os.path.exists(fpath):
                try:
                    os.remove(fpath)
                    removed_files.append(fpath)
                except Exception as e:
                    logger.warning(f"Could not remove file {fpath}: {e}")

    db_found = False
    try:
        ensure_initialized()
        with session_scope() as session:
            conv = repository.get_conversation(session, job_id)
            rep = repository.get_report(session, job_id)
            if conv is not None or rep is not None:
                db_found = True
            removed_files.extend(
                repository.delete_conversation_and_artifacts(session, job_id)
            )
            # Orphan report with no conversation row (legacy).
            if rep is not None and (conv is None or conv.report_id != job_id):
                removed_files.extend(
                    repository.delete_report_and_artifacts(session, job_id)
                )
    except Exception as e:
        logger.warning(f"Error purging conversation/report {job_id}: {e}")

    if job is None and not db_found:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found."
        )

    return {
        "status": "success",
        "job_id": job_id,
        "removed_files": list(dict.fromkeys(removed_files)),
        "message": (
            f"Successfully deleted conversation {job_id} "
            "and cleared associated reports & cache."
        ),
    }


@app.post("/jobs/{job_id}/regenerate", status_code=status.HTTP_202_ACCEPTED)
def regenerate_job_endpoint(job_id: str, req: RegenerateRequest, background_tasks: BackgroundTasks):
    """
    Regenerate report for an existing conversation query using a chosen agent and model.
    """
    with jobs_lock:
        old_job = jobs.get(job_id)

    query = old_job.get("query") if old_job else None
    if not query:
        # Fallback: durable conversation, then report.
        try:
            ensure_initialized()
            with session_scope() as session:
                conv = repository.get_conversation(session, job_id)
                if conv:
                    query = conv.query
                else:
                    rep = repository.get_report(session, job_id)
                    if rep:
                        query = rep.subject_text
        except Exception:
            pass

    if not query:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation query for job {job_id} not found."
        )

    new_job_id = str(uuid.uuid4())
    now = datetime.now()

    with jobs_lock:
        jobs[new_job_id] = {
            "id": new_job_id,
            "query": query,
            "agent_id": req.agent_id,
            "status": JobStatus.PENDING,
            "model": req.model,
            "implementation": req.implementation,
            "created_at": now,
            "updated_at": now,
            "error": None,
            "files": None,
            "result": None,
            "parent_job_id": job_id,
            "report_id": None,
            "has_docs": False,
        }

    try:
        ensure_initialized()
        with session_scope() as session:
            repository.create_conversation(
                session,
                conversation_id=new_job_id,
                query=query,
                agent_id=req.agent_id,
                status=JobStatus.PENDING,
                model=req.model,
                implementation=req.implementation,
                parent_job_id=job_id,
            )
    except Exception as e:
        logger.warning(f"Failed to persist regenerate conversation {new_job_id}: {e}")

    background_tasks.add_task(
        run_background_job,
        job_id=new_job_id,
        query=query,
        model=req.model,
        implementation=req.implementation,
        web_search=req.web_search,
        timeout=req.timeout,
        agent_id_override=req.agent_id
    )

    return {
        "job_id": new_job_id,
        "parent_job_id": job_id,
        "status": JobStatus.PENDING,
        "check_status_url": f"/jobs/{new_job_id}",
        "message": f"Regenerating report for '{query}' using agent '{req.agent_id or 'auto-routed'}'"
    }


# ── Patient Management Endpoints ───────────────────────────────────────────────


@app.get("/patients")
def list_patients_endpoint():
    """List all patient records with demographics, metadata, and clinical tables."""
    ensure_initialized()
    with session_scope() as session:
        patients = repository.list_patients(session)
        return [_patient_to_dict(p) for p in patients]


@app.post("/patients", status_code=status.HTTP_201_CREATED)
def create_patient_endpoint(req: PatientCreate):
    """Create a new patient record with custom key-value metadata and clinical data tables."""
    ensure_initialized()
    with session_scope() as session:
        patient = repository.create_patient(
            session,
            name=req.name,
            age=req.age,
            gender=req.gender,
            primary_condition=req.primary_condition,
            contact_email=req.contact_email,
            contact_phone=req.contact_phone,
            metadata_json=req.metadata_json,
            clinical_data=req.clinical_data,
        )
        return _patient_to_dict(patient)


@app.get("/patients/{patient_id}")
def get_patient_endpoint(patient_id: str):
    """Retrieve details, metadata, and clinical tables for a specific patient."""
    ensure_initialized()
    with session_scope() as session:
        patient = repository.get_patient(session, patient_id)
        if not patient:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Patient {patient_id} not found"
            )
        return _patient_to_dict(patient)


@app.put("/patients/{patient_id}")
def update_patient_endpoint(patient_id: str, req: PatientUpdate):
    """Update patient information, clinical tables, and metadata."""
    ensure_initialized()
    with session_scope() as session:
        updated = repository.update_patient(
            session,
            patient_id,
            name=req.name,
            age=req.age,
            gender=req.gender,
            primary_condition=req.primary_condition,
            contact_email=req.contact_email,
            contact_phone=req.contact_phone,
            metadata_json=req.metadata_json,
            clinical_data=req.clinical_data,
        )
        if not updated:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Patient {patient_id} not found"
            )
        return _patient_to_dict(updated)


@app.delete("/patients/{patient_id}")
def delete_patient_endpoint(patient_id: str):
    """Delete a patient record."""
    ensure_initialized()
    with session_scope() as session:
        success = repository.delete_patient(session, patient_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Patient {patient_id} not found"
            )
        return {"status": "success", "message": f"Patient {patient_id} deleted successfully."}


@app.post("/patients/parse-report")
async def parse_patient_report_endpoint(
    file: UploadFile = File(...),
    model: str = DEFAULT_ROUTING_MODEL
):
    """Parse an uploaded patient document (PDF/Word/Image) and categorize findings into Organ System tables."""
    contents = await file.read()
    if len(contents) > MAX_PARSE_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum size of {MAX_PARSE_UPLOAD_BYTES} bytes."
        )

    suffix = Path(file.filename or "").suffix
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        # 1. Extract markdown from file
        parse_result = parse_document(tmp_path)
        markdown_text = parse_result.markdown or ""

        # 2. Categorize markdown text into organ system categories using LLM
        categorized_data = categorize_medical_markdown(markdown_text, model_name=model)

    except Exception as e:
        logger.error(f"Error parsing patient report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Patient report parsing failure: {e}"
        )
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

    return {
        "filename": file.filename,
        "status": parse_result.status.value,
        "markdown": markdown_text,
        "categorized_data": categorized_data
    }


@app.post("/patients/classify-text")
def classify_patient_text_endpoint(req: PatientClassifyRequest):
    """
    Classify a free-form patient clinical narrative/description using LLM.
    Extracts demographics, custom metadata tags, and categorized organ system findings.
    """
    if not req.text or not req.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Patient description text cannot be empty."
        )

    try:
        classification = classify_patient_description(req.text, model_name=req.model)
        return {
            "status": "success",
            "classification": classification
        }
    except Exception as e:
        logger.error(f"Error classifying patient description: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification failure: {e}"
        )




# ── Configuration (Slack webhooks + LLM API keys) ───────────────────────────
# Dual-register with/without trailing slash so POSTs never fall through to the
# frontend static handler (which only allows GET/HEAD → confusing 405s).


@app.get("/config")
@app.get("/config/")
def get_config_endpoint():
    """Return UI-safe app configuration (webhooks + masked API key status)."""
    return app_config.get_public_config()


@app.get("/config/slack-webhooks")
@app.get("/config/slack-webhooks/")
def list_slack_webhooks_endpoint():
    """List saved Slack incoming webhooks."""
    return {"webhooks": app_config.list_slack_webhooks()}


@app.post("/config/slack-webhooks", status_code=status.HTTP_201_CREATED)
@app.post("/config/slack-webhooks/", status_code=status.HTTP_201_CREATED)
def create_slack_webhook_endpoint(req: SlackWebhookCreate):
    """Add a named Slack incoming webhook."""
    try:
        entry = app_config.add_slack_webhook(req.name, req.url)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    return entry


@app.put("/config/slack-webhooks/{webhook_id}")
@app.put("/config/slack-webhooks/{webhook_id}/")
def update_slack_webhook_endpoint(webhook_id: str, req: SlackWebhookUpdate):
    """Update a saved Slack webhook name and/or URL."""
    if req.name is None and req.url is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide at least one of: name, url",
        )
    try:
        return app_config.update_slack_webhook(webhook_id, name=req.name, url=req.url)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook not found: {webhook_id}",
        ) from None
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e


@app.delete("/config/slack-webhooks/{webhook_id}")
@app.delete("/config/slack-webhooks/{webhook_id}/")
def delete_slack_webhook_endpoint(webhook_id: str):
    """Remove a saved Slack webhook."""
    if not app_config.delete_slack_webhook(webhook_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook not found: {webhook_id}",
        )
    return {"status": "deleted", "id": webhook_id}


@app.get("/config/api-keys")
@app.get("/config/api-keys/")
def list_api_keys_endpoint():
    """List known LLM credential slots with configured status (values masked)."""
    return {"api_keys": app_config.list_api_key_status()}


@app.put("/config/api-keys")
@app.put("/config/api-keys/")
def upsert_api_key_endpoint(req: ApiKeyUpsert):
    """Create or update an LLM provider API key / credential.

    The value is stored in the local app config file and applied to the
    process environment so subsequent LLM calls can use the provider.
    """
    try:
        return app_config.set_api_key(req.env_var, req.value)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e


@app.delete("/config/api-keys/{env_var}")
@app.delete("/config/api-keys/{env_var}/")
def delete_api_key_endpoint(env_var: str):
    """Remove a UI-managed API key (does not clear keys that only exist in the shell/.env)."""
    if not app_config.delete_api_key(env_var):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No UI-managed key for '{env_var}'. "
                "Keys set only via environment variables cannot be deleted here."
            ),
        )
    return {"status": "deleted", "env_var": env_var, "api_keys": app_config.list_api_key_status()}


@app.post("/slack/notify")
def send_slack_notification_endpoint(req: SlackNotifyRequest):
    """
    Send selected task descriptions and reports to a Slack Webhook URL.
    Task descriptions are included as formatted text snippet attachments.

    Provide either ``webhook_url`` directly or ``webhook_id`` of a saved webhook
    from the configuration menu.
    """
    webhook_url = (req.webhook_url or "").strip()
    if req.webhook_id and not webhook_url:
        saved = app_config.get_slack_webhook(req.webhook_id)
        if not saved:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Saved Slack webhook not found: {req.webhook_id}",
            )
        webhook_url = saved["url"]

    if not webhook_url.startswith("https://"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid Slack Webhook URL. Must start with 'https://' (or pass a valid webhook_id).",
        )
    if not req.job_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No job IDs specified for notification."
        )

    selected_jobs = []
    with jobs_lock:
        for jid in req.job_ids:
            if jid in jobs:
                selected_jobs.append(jobs[jid])

    if not selected_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="None of the specified job IDs were found."
        )

    # Build Slack payload with block kit and text snippet attachments
    attachments = []
    for job in selected_jobs:
        job_id = job.get("id", "N/A")
        query = job.get("query", "Unknown query")
        agent_id = job.get("agent_id", "Unassigned")
        job_status = job.get("status", "unknown")

        # Get description or result text
        result_data = job.get("result") or {}
        description = ""
        if isinstance(result_data, dict):
            if "summary" in result_data:
                description = str(result_data["summary"])
            elif "detail" in result_data:
                description = str(result_data["detail"])
            elif "simplified_output" in result_data:
                description = str(result_data["simplified_output"])
            else:
                description = json.dumps(result_data, indent=2)
        elif isinstance(result_data, str):
            description = result_data

        if not description:
            description = f"Status: {job_status}. Query: {query}"

        # Truncate description if extremely long for Slack attachment limits (~3000 chars)
        if len(description) > 3000:
            description = description[:3000] + "\n... (truncated)"

        status_emoji = "✅" if job_status == "completed" else ("❌" if job_status == "failed" else "⏳")

        attachment = {
            "color": "#36a64f" if job_status == "completed" else "#e01e5a",
            "title": f"{status_emoji} Task: {query}",
            "text": f"```\n{description}\n```",
            "fields": [
                {"title": "Job ID", "value": job_id, "short": True},
                {"title": "Agent", "value": str(agent_id), "short": True},
                {"title": "Status", "value": str(job_status).upper(), "short": True}
            ]
        }
        attachments.append(attachment)

    slack_payload = {
        "text": f"🏥 *Medical Intelligence - {len(selected_jobs)} Task Report(s)*",
        "attachments": attachments
    }

    # Send POST request to webhook
    try:
        import urllib.request
        req_data = json.dumps(slack_payload).encode("utf-8")
        request = urllib.request.Request(
            webhook_url,
            data=req_data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            resp_body = response.read().decode("utf-8")
            logger.info(f"Slack webhook response: {response.status} {resp_body}")
    except Exception as e:
        logger.error(f"Error sending Slack notification: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to post notification to Slack webhook: {e}"
        )

    return {
        "status": "success",
        "sent_count": len(selected_jobs),
        "message": f"Successfully sent {len(selected_jobs)} task notification(s) to Slack."
    }


@app.post("/parse")
async def parse_document_endpoint(file: UploadFile = File(...)):
    """Parse an uploaded document (PDF/Word/text/rtf) into markdown.

    Accepts a multipart file upload and returns the converted markdown along
    with status, warnings, and metadata. The parser is best-effort: an
    unparseable file returns a 200 with ``status: "failed"`` and warnings
    rather than an HTTP error, so callers get a consistent result shape.
    """
    contents = await file.read()
    if len(contents) > MAX_PARSE_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File exceeds maximum size of {MAX_PARSE_UPLOAD_BYTES} bytes."
            ),
        )

    # Preserve the original extension so the parser can pick the right backend.
    suffix = Path(file.filename or "").suffix
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        result = parse_document(tmp_path)
    except Exception as e:  # noqa: BLE001 - surface as HTTP 500 with detail
        logger.error(f"Error during document parsing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document parsing failure: {e}",
        )
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

    return {
        "filename": file.filename,
        "status": result.status.value,
        "markdown": result.markdown,
        "warnings": [str(w) for w in result.warnings],
        "metadata": {
            "file_format": result.metadata.file_format,
            "backend": result.metadata.backend,
            "page_count": result.metadata.page_count,
            "char_count": result.metadata.char_count,
        },
    }


# Serve the Web UI with GET-only handlers.
# IMPORTANT: do NOT use ``app.mount("/", StaticFiles(...))`` — a catch-all Mount
# intercepts unmatched POST/PUT/DELETE and returns 405 Method Not Allowed, which
# masked missing/mismatched API routes (e.g. trailing-slash variants).
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)


def _safe_frontend_file(relative: str) -> Optional[Path]:
    """Resolve a path under FRONTEND_DIR, or None if missing / path-escape."""
    if not relative or relative.startswith("/") or ".." in Path(relative).parts:
        return None
    candidate = (FRONTEND_DIR / relative).resolve()
    try:
        candidate.relative_to(FRONTEND_DIR.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


@app.get("/")
def serve_frontend_index():
    """Serve the SPA entry point with no-cache headers."""
    index = FRONTEND_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404, detail="Frontend index.html not found")
    return FileResponse(index, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/{asset_path:path}")
def serve_frontend_asset(asset_path: str):
    """Serve static frontend assets (GET only) with no-cache headers."""
    file_path = _safe_frontend_file(asset_path)
    if file_path is not None:
        return FileResponse(file_path, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    # SPA-style fallback for unknown GET paths
    index = FRONTEND_DIR / "index.html"
    if index.is_file():
        return FileResponse(index, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    raise HTTPException(status_code=404, detail="Not found")


def main():
    """Entry point for medical-api console script."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Medical Multi-Agent API Service")
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to (default: 8080)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    args = parser.parse_args()

    uvicorn.run("api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
