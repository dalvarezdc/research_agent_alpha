import unittest
from unittest.mock import patch, ANY

import pytest
from fastapi.testclient import TestClient

from api import app, JobStatus


client = TestClient(app)


def test_health_endpoint():
    """Test the GET /health status check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "default_model" in data


def test_agents_endpoint():
    """Test the GET /agents specification endpoint."""
    response = client.get("/agents")
    assert response.status_code == 200
    agents = response.json()
    assert len(agents) == 4
    
    agent_ids = [agent["id"] for agent in agents]
    assert "medication_agent" in agent_ids
    assert "procedure_agent" in agent_ids
    assert "diagnostic_agent" in agent_ids
    assert "general_agent" in agent_ids
    
    # Check medication_agent details
    med_agent = next(a for a in agents if a["id"] == "medication_agent")
    assert med_agent["name"] == "Medication Specialist"
    assert "dosages" in med_agent["description"]


def test_models_endpoint():
    """Test the GET /models endpoint to retrieve model list."""
    response = client.get("/models")
    assert response.status_code == 200
    models = response.json()
    assert "grok-4.3" in models
    assert "gpt-4o" in models


@patch("api.route_agent")
def test_route_endpoint(mock_route):
    """Test query routing classification endpoint with mock router."""
    mock_route.return_value = "medication_agent"
    
    response = client.post("/route", json={"query": "Is Metformin safe?"})
    assert response.status_code == 200
    data = response.json()
    
    assert data["query"] == "Is Metformin safe?"
    assert data["agent_id"] == "medication_agent"
    assert data["agent_name"] == "Medication Specialist"
    
    mock_route.assert_called_once_with(
        "Is Metformin safe?",
        ANY,
        default_agent_id="general_agent",
        model=ANY
    )


@patch("api.execute_analysis_sync")
def test_analyze_sync_endpoint(mock_execute):
    """Test synchronous query analysis execution with mock execution."""
    mock_execute.return_value = {
        "agent_id": "medication_agent",
        "files": {"result": "outputs/test_med.json"},
        "result": {"status": "mocked_success", "detail": "Metformin analysis"}
    }
    
    payload = {
        "query": "Is Metformin safe?",
        "model": "grok-4.3",
        "implementation": "langchain",
        "web_search": True,
        "timeout": 300
    }
    
    response = client.post("/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "success"
    assert data["agent_id"] == "medication_agent"
    assert data["result"]["status"] == "mocked_success"
    assert data["files"]["result"] == "outputs/test_med.json"
    
    mock_execute.assert_called_once_with(
        query="Is Metformin safe?",
        model="grok-4.3",
        implementation="langchain",
        web_search=True,
        timeout=300
    )


@patch("api.execute_analysis_sync")
def test_analyze_async_job_flow(mock_execute):
    """Test asynchronous background job flow (queue, execute, poll)."""
    mock_execute.return_value = {
        "agent_id": "procedure_agent",
        "files": {"result": "outputs/test_proc.json"},
        "result": {"status": "mocked_procedure_success"}
    }
    
    payload = {
        "query": "How is appendectomy performed?",
        "model": "grok-4.3",
        "implementation": "langchain",
        "web_search": False,
        "timeout": 150
    }
    
    # 1. Enqueue background job
    response = client.post("/analyze/async", json=payload)
    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    assert data["status"] == JobStatus.PENDING
    job_id = data["job_id"]
    
    # 2. Check job status - in TestClient, background tasks run synchronously when response returns
    # So by the time we check the endpoint, the job should already be completed!
    status_response = client.get(f"/jobs/{job_id}")
    assert status_response.status_code == 200
    job_data = status_response.json()
    
    assert job_data["id"] == job_id
    assert job_data["status"] == JobStatus.COMPLETED
    assert job_data["agent_id"] == "procedure_agent"
    assert job_data["result"]["status"] == "mocked_procedure_success"
    
    # 3. Test missing job check
    missing_response = client.get("/jobs/non-existent-job-uuid-1234")
    assert missing_response.status_code == 404


# ── /parse document upload endpoint ──────────────────────────────────────────


def test_parse_endpoint_txt():
    """Upload a plain-text file and get markdown back."""
    response = client.post(
        "/parse",
        files={"file": ("note.txt", b"Para one\n\nPara two", "text/plain")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "Para one" in data["markdown"]
    assert data["metadata"]["file_format"] == "txt"
    assert data["metadata"]["backend"] == "text"


def test_parse_endpoint_md_passthrough():
    response = client.post(
        "/parse",
        files={"file": ("doc.md", b"# Title\n\nBody", "text/markdown")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["markdown"] == "# Title\n\nBody"


def test_parse_endpoint_unsupported_format_returns_failed_status():
    """Unsupported files return 200 with status 'failed' (best-effort shape)."""
    response = client.post(
        "/parse",
        files={"file": ("data.xyz", b"whatever", "application/octet-stream")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "failed"
    assert any("Unsupported format" in w for w in data["warnings"])


def test_parse_endpoint_rejects_oversize_upload(monkeypatch):
    import api

    monkeypatch.setattr(api, "MAX_PARSE_UPLOAD_BYTES", 5)
    response = client.post(
        "/parse",
        files={"file": ("big.txt", b"way too large payload", "text/plain")},
    )
    assert response.status_code == 413
