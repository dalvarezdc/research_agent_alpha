"""Unit tests for Patient CRUD and Conversation Deletion/Regeneration endpoints."""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from api import app, jobs, jobs_lock
from database.session import reset_initialized_flag, session_scope, init_db
from database.models import Base
from database.config import get_engine

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_test_db():
    """Ensure database schema is reset for each test run."""
    engine = get_engine()
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    reset_initialized_flag()
    init_db(seed=True)
    yield


def test_patient_crud_flow():
    """Test full Patient CRUD lifecycle via API endpoints."""
    # 1. Create Patient
    create_payload = {
        "name": "Jane Doe",
        "age": 45,
        "gender": "Female",
        "primary_condition": "Hypertension",
        "contact_email": "jane.doe@example.com",
        "contact_phone": "+15551234567",
        "metadata_json": {
            "Insurance": "BlueCross 9876",
            "Allergies": "Penicillin"
        },
        "clinical_data": {
            "vitals": {"bp": "120/80", "hr": 72}
        }
    }
    response = client.post("/patients", json=create_payload)
    assert response.status_code == 201
    patient = response.json()
    patient_id = patient["id"]
    assert patient["name"] == "Jane Doe"
    assert patient["age"] == 45
    assert patient["metadata_json"]["Allergies"] == "Penicillin"

    # 2. List Patients
    list_res = client.get("/patients")
    assert list_res.status_code == 200
    patients_list = list_res.json()
    assert len(patients_list) >= 1
    assert any(p["id"] == patient_id for p in patients_list)

    # 3. Get Patient by ID
    get_res = client.get(f"/patients/{patient_id}")
    assert get_res.status_code == 200
    assert get_res.json()["name"] == "Jane Doe"

    # 4. Update Patient
    update_payload = {
        "age": 46,
        "primary_condition": "Hypertension & Type 2 Diabetes",
        "metadata_json": {
            "Insurance": "BlueCross 9876",
            "Allergies": "Penicillin, Sulfa",
            "Emergency Contact": "John Doe (Husband)"
        }
    }
    put_res = client.put(f"/patients/{patient_id}", json=update_payload)
    assert put_res.status_code == 200
    updated = put_res.json()
    assert updated["age"] == 46
    assert updated["primary_condition"] == "Hypertension & Type 2 Diabetes"
    assert updated["metadata_json"]["Allergies"] == "Penicillin, Sulfa"

    # 5. Delete Patient
    del_res = client.delete(f"/patients/{patient_id}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "success"

    # Verify 404 on get
    get_again = client.get(f"/patients/{patient_id}")
    assert get_again.status_code == 404


def test_delete_and_regenerate_job_endpoints():
    """Test DELETE /jobs/{job_id} and POST /jobs/{job_id}/regenerate endpoints."""
    # 1. Create a dummy job in memory
    job_id = "test-job-uuid-123"
    with jobs_lock:
        jobs[job_id] = {
            "id": job_id,
            "query": "Metformin side effects",
            "agent_id": "medication_agent",
            "status": "completed",
            "files": {},
            "result": {"summary": "Metformin is well tolerated."}
        }

    # 2. Test Regenerate Endpoint
    regen_res = client.post(f"/jobs/{job_id}/regenerate", json={
        "agent_id": "procedure_agent",
        "model": "grok-4.3"
    })
    assert regen_res.status_code == 202
    regen_data = regen_res.json()
    new_job_id = regen_data["job_id"]
    assert regen_data["parent_job_id"] == job_id
    assert new_job_id in jobs

    # 3. Test Delete Job Endpoint
    del_job_res = client.delete(f"/jobs/{job_id}")
    assert del_job_res.status_code == 200
    assert job_id not in jobs


@patch("api.categorize_medical_markdown")
@patch("api.parse_document")
def test_parse_patient_report_endpoint(mock_parse_doc, mock_categorize):
    """Test POST /patients/parse-report parses document and categorizes metrics."""
    mock_parse = MagicMock()
    mock_parse.status.value = "success"
    mock_parse.markdown = "## Clinical Lab Results\nALT: 45 U/L\nHbA1c: 5.8%\nVitamin D: 18 ng/mL"
    mock_parse_doc.return_value = mock_parse

    mock_categorize.return_value = {
        "heart": [],
        "liver": [{"marker": "ALT", "value": "45 U/L", "reference_range": "7-56 U/L", "status": "Normal", "notes": ""}],
        "pancreas": [{"marker": "HbA1c", "value": "5.8%", "reference_range": "<5.7%", "status": "High", "notes": ""}],
        "nutrients": [{"marker": "Vitamin D", "value": "18 ng/mL", "reference_range": "30-100", "status": "Low", "notes": ""}],
        "overall_health": [],
        "medications": [],
        "summary": "Sample parsed labs"
    }

    files = {"file": ("report.txt", b"ALT: 45 U/L\nHbA1c: 5.8%\nVitamin D: 18 ng/mL", "text/plain")}
    response = client.post("/patients/parse-report", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "report.txt"
    assert "categorized_data" in data
    assert len(data["categorized_data"]["liver"]) == 1
    assert data["categorized_data"]["liver"][0]["marker"] == "ALT"
    assert data["categorized_data"]["pancreas"][0]["marker"] == "HbA1c"


@patch("api.execute_analysis_sync")
def test_agent_override_in_analyze_async(mock_exec):
    """Test POST /analyze/async correctly respects explicit agent_id override."""
    mock_exec.return_value = {
        "agent_id": "general_agent",
        "files": {"patient_report": "outputs/patient_report.md"},
        "result": {"summary": "Fact check results"}
    }

    payload = {
        "query": "cortisol can cause cancer",
        "model": "grok-4.5",
        "agent_id": "general_agent",
        "web_search": True
    }
    res = client.post("/analyze/async", json=payload)
    assert res.status_code == 202
    job_data = res.json()
    job_id = job_data["job_id"]
    
    assert job_id in jobs
    assert jobs[job_id]["agent_id"] == "general_agent"


