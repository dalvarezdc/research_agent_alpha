"""
FastAPI REST API service for the medical multi-agent system.
Exposes query routing, synchronous analysis, and asynchronous background jobs.
"""

import os
import sys
import uuid
import json
import logging
import tempfile
import threading
import traceback
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, BackgroundTasks, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
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
from llm_integrations import get_available_models
from document_parser import parse_document
from medical_report_categorizer import categorize_medical_markdown

# Maximum upload size for document parsing (bytes). Default 25 MB; override via env.
MAX_PARSE_UPLOAD_BYTES = int(os.getenv("MAX_PARSE_UPLOAD_BYTES", str(25 * 1024 * 1024)))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("medical_api")

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


class RegenerateRequest(BaseModel):
    agent_id: Optional[str] = Field(None, description="Agent ID override (medication_agent, procedure_agent, diagnostic_agent, general_agent).")
    model: str = Field(DEFAULT_ROUTING_MODEL, description="LLM model to use.")
    implementation: str = Field("langchain", description="Agent implementation.")
    web_search: bool = Field(True, description="Enable web search.")
    timeout: int = Field(300, description="Timeout in seconds.")


class SlackNotifyRequest(BaseModel):
    webhook_url: str = Field(..., description="Slack Incoming Webhook URL")
    job_ids: List[str] = Field(..., description="List of task/job IDs to include in the notification")


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


def execute_analysis_sync(
    query: str,
    model: str,
    implementation: str,
    web_search: bool,
    timeout: int,
    agent_id_override: Optional[str] = None
) -> Dict[str, Any]:
    """
    Runs the full routing and execution pipeline synchronously.
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

    # 2. Run the specialized agent via AgentOrchestrator
    orchestrator = AgentOrchestrator(output_dir="outputs")
    files = {}

    if routed_agent_id == "medication_agent":
        _, files = orchestrator.run_medication_analyzer(
            medication=query,
            indication=None,
            other_medications=None,
            llm_provider=llm_provider,
            timeout=timeout,
            implementation=implementation,
            enable_web_research=web_search,
        )
    elif routed_agent_id == "procedure_agent":
        _, files = orchestrator.run_procedure_analyzer(
            procedure=query,
            details="API Request",
            llm_provider=llm_provider,
            timeout=timeout,
            implementation=implementation,
            enable_web_research=web_search,
        )
    elif routed_agent_id == "diagnostic_agent":
        _, files = orchestrator.run_diagnostic_analyzer(
            query=query,
            llm_provider=llm_provider,
            timeout=timeout,
            interactive=False,
        )
    elif routed_agent_id == "general_agent":
        _, files = orchestrator.run_fact_checker(
            subject=query,
            context="",
            llm_provider=llm_provider,
            timeout=timeout,
            implementation=implementation,
            enable_web_research=web_search,
        )
    else:
        raise ValueError(f"Unknown routed agent: {routed_agent_id}")

    # Load result data from file
    loaded_data = load_json_result(routed_agent_id, files)

    # Persist report to DB (best-effort)
    try:
        ensure_initialized()
        with session_scope() as session:
            repository.persist_report(
                session=session,
                agent_type=routed_agent_id,
                subject_text=query,
                files=files,
                llm_provider=llm_provider,
                implementation=implementation,
            )
    except Exception as e:
        logger.warning(f"Failed to persist report to database: {e}")

    return {
        "agent_id": routed_agent_id,
        "files": files,
        "result": loaded_data
    }


def run_background_job(
    job_id: str,
    query: str,
    model: str,
    implementation: str,
    web_search: bool,
    timeout: int,
    agent_id_override: Optional[str] = None
):
    """
    Worker task for executing an analysis asynchronously in the background.
    """
    logger.info(f"Starting background job: {job_id}")
    with jobs_lock:
        jobs[job_id]["status"] = JobStatus.RUNNING
        jobs[job_id]["updated_at"] = datetime.now()

    try:
        data = execute_analysis_sync(
            query=query,
            model=model,
            implementation=implementation,
            web_search=web_search,
            timeout=timeout,
            agent_id_override=agent_id_override
        )
        with jobs_lock:
            jobs[job_id]["status"] = JobStatus.COMPLETED
            jobs[job_id]["agent_id"] = data["agent_id"]
            jobs[job_id]["files"] = data["files"]
            jobs[job_id]["result"] = data["result"]
            jobs[job_id]["updated_at"] = datetime.now()
        logger.info(f"Successfully completed background job: {job_id}")
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Failed background job: {job_id}. Error: {e}")
        with jobs_lock:
            jobs[job_id]["status"] = JobStatus.FAILED
            jobs[job_id]["error"] = f"{e}\n{tb}"
            jobs[job_id]["updated_at"] = datetime.now()



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


@app.post("/analyze")
def analyze_query_sync_endpoint(req: AnalyzeRequest):
    """
    Synchronously route and analyze a medical query. Blocks until analysis completes.
    """
    try:
        res = execute_analysis_sync(
            query=req.query,
            model=req.model,
            implementation=req.implementation,
            web_search=req.web_search,
            timeout=req.timeout
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
    """
    job_id = str(uuid.uuid4())
    now = datetime.now()

    with jobs_lock:
        jobs[job_id] = {
            "id": job_id,
            "query": req.query,
            "agent_id": None,
            "status": JobStatus.PENDING,
            "created_at": now,
            "updated_at": now,
            "error": None,
            "files": None,
            "result": None
        }

    background_tasks.add_task(
        run_background_job,
        job_id=job_id,
        query=req.query,
        model=req.model,
        implementation=req.implementation,
        web_search=req.web_search,
        timeout=req.timeout
    )

    return {
        "job_id": job_id,
        "status": JobStatus.PENDING,
        "check_status_url": f"/jobs/{job_id}"
    }


@app.get("/jobs")
def list_jobs_endpoint():
    """
    List all background jobs and tasks sorted by creation date descending.
    """
    with jobs_lock:
        job_list = list(jobs.values())

    # Sort newest first
    job_list.sort(key=lambda j: j.get("created_at", datetime.min), reverse=True)
    return job_list


@app.get("/jobs/{job_id}")
def get_job_status_endpoint(job_id: str):
    """
    Retrieve status and result of a background job.
    """
    with jobs_lock:
        job = jobs.get(job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )

    return job


@app.delete("/jobs/{job_id}")
def delete_job_endpoint(job_id: str):
    """
    Delete a conversation/job, purge its output files from disk, and remove database records & cache.
    """
    removed_files = []
    with jobs_lock:
        job = jobs.pop(job_id, None)

    if job:
        files = job.get("files") or {}
        for ftype, fpath in files.items():
            if fpath and os.path.exists(fpath):
                try:
                    os.remove(fpath)
                    removed_files.append(fpath)
                except Exception as e:
                    logger.warning(f"Could not remove file {fpath}: {e}")

    # Check and delete from database as well
    try:
        ensure_initialized()
        with session_scope() as session:
            db_removed = repository.delete_report_and_artifacts(session, job_id)
            removed_files.extend(db_removed)
    except Exception as e:
        logger.warning(f"Error purging database report {job_id}: {e}")

    if not job and not removed_files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found."
        )

    return {
        "status": "success",
        "job_id": job_id,
        "removed_files": list(set(removed_files)),
        "message": f"Successfully deleted conversation {job_id} and cleared associated cache."
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
        # Fallback: check database for report subject text
        try:
            ensure_initialized()
            with session_scope() as session:
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
            "created_at": now,
            "updated_at": now,
            "error": None,
            "files": None,
            "result": None,
            "parent_job_id": job_id
        }

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




@app.post("/slack/notify")
def send_slack_notification_endpoint(req: SlackNotifyRequest):
    """
    Send selected task descriptions and reports to a Slack Webhook URL.
    Task descriptions are included as formatted text snippet attachments.
    """
    if not req.webhook_url.startswith("https://"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid Slack Webhook URL. Must start with 'https://'."
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
            req.webhook_url,
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


# Ensure frontend directory exists and serve static Web UI
os.makedirs("frontend", exist_ok=True)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


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
