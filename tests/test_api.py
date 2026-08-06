import unittest
from unittest.mock import patch, ANY, MagicMock

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
    assert "claude-sonnet-4-6" in models


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
        timeout=300,
        agent_id_override=None,
        context_report=None,
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


def test_list_jobs_endpoint():
    """Test GET /jobs to list all jobs."""
    response = client.get("/jobs")
    assert response.status_code == 200
    jobs_list = response.json()
    assert isinstance(jobs_list, list)


@patch("urllib.request.urlopen")
def test_slack_notify_endpoint_success(mock_urlopen):
    """Test POST /slack/notify sends formatted task description attachments."""
    from unittest.mock import MagicMock
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.read.return_value = b"ok"
    mock_urlopen.return_value.__enter__.return_value = mock_response

    # 1. Create a job first
    with patch("api.execute_analysis_sync") as mock_exec:
        mock_exec.return_value = {
            "agent_id": "medication_agent",
            "files": {},
            "result": {"summary": "Metformin analysis summary text"}
        }
        res = client.post("/analyze/async", json={
            "query": "Is Metformin safe?",
            "model": "grok-4.3"
        })
        job_id = res.json()["job_id"]

    # 2. Dispatch Slack notification
    notify_res = client.post("/slack/notify", json={
        "webhook_url": "https://hooks.slack.com/services/T000/B000/XXXX",
        "job_ids": [job_id]
    })

    assert notify_res.status_code == 200
    data = notify_res.json()
    assert data["status"] == "success"
    assert data["sent_count"] == 1
    assert mock_urlopen.called


def test_slack_notify_invalid_url():
    """Test error handling for non-https webhook URL."""
    res = client.post("/slack/notify", json={
        "webhook_url": "http://invalid-url.com",
        "job_ids": ["job-123"]
    })
    assert res.status_code == 400
    assert "Invalid Slack Webhook URL" in res.json()["detail"]


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


@patch("api.create_llm_manager")
def test_intake_chat_endpoint(mock_llm_factory):
    mock_mgr = MagicMock()
    mock_provider = MagicMock()
    mock_provider.generate_response.return_value = ("Could you clarify patient age and duration of symptoms?", None)
    mock_mgr.get_provider_direct.return_value = mock_provider
    mock_llm_factory.return_value = mock_mgr

    payload = {
        "messages": [
            {"role": "user", "content": "I have headache and mild fever"}
        ],
        "model": "grok-4.5",
        "document_context": "Patient lab report: WBC 11.2"
    }

    response = client.post("/intake/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["role"] == "assistant"
    assert "patient age" in data["content"]


@patch("api.create_llm_manager")
def test_intake_summarize_endpoint(mock_llm_factory):
    mock_mgr = MagicMock()
    mock_provider = MagicMock()
    mock_provider.generate_response.return_value = ("Clinical Query: 45yo male with 3-day history of headache and mild fever (100.4F). WBC 11.2.", None)
    mock_mgr.get_provider_direct.return_value = mock_provider
    mock_llm_factory.return_value = mock_mgr

    payload = {
        "messages": [
            {"role": "user", "content": "I have headache and mild fever"},
            {"role": "assistant", "content": "What is the duration?"},
            {"role": "user", "content": "3 days"}
        ],
        "model": "grok-4.5"
    }

    response = client.post("/intake/summarize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "45yo male" in data["summary"]


# ── /config endpoints (Slack webhooks + LLM API keys) ────────────────────────


@pytest.fixture()
def isolated_app_config(tmp_path, monkeypatch):
    """Isolate UI config storage for API tests."""
    import app_config

    path = tmp_path / "api_app_config.json"
    app_config.set_config_path(path)
    for entry in app_config.KNOWN_API_KEY_VARS:
        monkeypatch.delenv(entry["env_var"], raising=False)
    app_config._managed_env_vars = set()  # noqa: SLF001
    yield path
    app_config.set_config_path(app_config._DEFAULT_CONFIG_PATH)  # noqa: SLF001
    app_config._managed_env_vars = set()  # noqa: SLF001


def test_get_config_endpoint(isolated_app_config):
    res = client.get("/config")
    assert res.status_code == 200
    data = res.json()
    assert "slack_webhooks" in data
    assert "api_keys" in data
    assert isinstance(data["api_keys"], list)
    env_vars = {k["env_var"] for k in data["api_keys"]}
    assert "GROK_API_KEY" in env_vars
    assert "ANTHROPIC_API_KEY" in env_vars


def test_slack_webhook_crud_via_api(isolated_app_config):
    create = client.post(
        "/config/slack-webhooks",
        json={"name": "Team", "url": "https://hooks.slack.com/services/T/B/C"},
    )
    assert create.status_code == 201
    wh = create.json()
    assert wh["id"]
    assert wh["name"] == "Team"

    listed = client.get("/config/slack-webhooks")
    assert listed.status_code == 200
    assert len(listed.json()["webhooks"]) == 1

    updated = client.put(
        f"/config/slack-webhooks/{wh['id']}",
        json={"name": "Renamed"},
    )
    assert updated.status_code == 200
    assert updated.json()["name"] == "Renamed"

    deleted = client.delete(f"/config/slack-webhooks/{wh['id']}")
    assert deleted.status_code == 200
    assert client.get("/config/slack-webhooks").json()["webhooks"] == []


def test_api_key_upsert_and_clear(isolated_app_config):
    put = client.put(
        "/config/api-keys",
        json={"env_var": "GROK_API_KEY", "value": "xai-api-test-key-abcdef"},
    )
    assert put.status_code == 200
    body = put.json()
    assert body["configured"] is True
    assert body["source"] == "config"
    assert "xai-api-test-key-abcdef" not in str(body)

    # Full secret must not appear on GET /config
    cfg = client.get("/config").json()
    assert "xai-api-test-key-abcdef" not in str(cfg)

    cleared = client.delete("/config/api-keys/GROK_API_KEY")
    assert cleared.status_code == 200
    statuses = cleared.json()["api_keys"]
    grok = next(s for s in statuses if s["env_var"] == "GROK_API_KEY")
    assert grok["configured"] is False


def test_api_key_unknown_var_rejected(isolated_app_config):
    res = client.put(
        "/config/api-keys",
        json={"env_var": "FAKE_KEY", "value": "nope"},
    )
    assert res.status_code == 400


@patch("urllib.request.urlopen")
def test_slack_notify_with_saved_webhook_id(mock_urlopen, isolated_app_config):
    """POST /slack/notify accepts webhook_id from saved config."""
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.read.return_value = b"ok"
    mock_urlopen.return_value.__enter__.return_value = mock_response

    wh = client.post(
        "/config/slack-webhooks",
        json={"name": "Saved", "url": "https://hooks.slack.com/services/SAVED/B/C"},
    ).json()

    with patch("api.execute_analysis_sync") as mock_exec:
        mock_exec.return_value = {
            "agent_id": "general_agent",
            "files": {},
            "result": {"summary": "Fact check summary"},
        }
        job_id = client.post(
            "/analyze/async",
            json={"query": "Vitamin D evidence?", "model": "grok-4.3"},
        ).json()["job_id"]

    notify = client.post(
        "/slack/notify",
        json={"webhook_id": wh["id"], "job_ids": [job_id]},
    )
    assert notify.status_code == 200
    assert notify.json()["sent_count"] == 1
    assert mock_urlopen.called
