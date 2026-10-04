"""Saved chat persistence, migration and context isolation regression tests."""
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from api import app, jobs, jobs_lock
from database.config import reset_engine_cache
from database.session import init_db, reset_initialized_flag, session_scope
from database.models import SavedChat, Conversation
from database import repository, chats


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/chats.db")
    reset_engine_cache()
    reset_initialized_flag()
    init_db()
    with jobs_lock:
        jobs.clear()
    yield TestClient(app)
    with jobs_lock:
        jobs.clear()
    reset_engine_cache()
    reset_initialized_flag()


def create(client, name=None):
    patient = client.post('/patients', json={'name': name}).json() if name else None
    chat = client.post('/chats', json={'patient_id': patient['id'] if patient else None}).json()
    return patient, chat


def test_persistence_and_general_scope(client):
    patient, chat = create(client, 'Jane')
    _, general = create(client)
    assert client.patch(f"/chats/{chat['id']}", json={'draft': 'clinical draft', 'title': 'Follow up'}).status_code == 200
    attachment = client.post(f"/chats/{chat['id']}/attachments", json={'filename': 'lab.txt', 'markdown': 'Jane lab marker'}).json()
    with patch('api.intake_chat_endpoint', return_value={'content': 'Which symptoms?'}) as intake:
        sent = client.post(f"/chats/{chat['id']}/messages", json={'id': 'message-one', 'content': 'Fatigue'})
        assert sent.status_code == 200
        assert 'Jane lab marker' in intake.call_args.args[0].document_context
        client.post(f"/chats/{chat['id']}/messages", json={'id': 'message-one', 'content': 'Fatigue'})
        assert intake.call_count == 1
    assert len(client.get(f"/chats/{chat['id']}").json()['messages']) == 2
    assert [c['id'] for c in client.get('/chats').json()] == [general['id']]
    assert [c['id'] for c in client.get('/chats', params={'patient_id': patient['id']}).json()] == [chat['id']]
    with session_scope() as session:
        assert chats.background(session, chats.get_chat(session, general['id'])) == ''
    client.delete(f"/chats/{chat['id']}/attachments/{attachment['id']}")
    assert client.get(f"/chats/{chat['id']}").json()['attachments'] == []


def test_sources_and_analysis_context(client):
    patient, chat = create(client, 'Jane')
    _, general = create(client)
    with session_scope() as session:
        repository.create_conversation(session, conversation_id='jane-job', query='Jane evidence', patient_id=patient['id'], chat_id=chat['id'])
    assert client.patch(f"/chats/{general['id']}", json={'source_ids': ['jane-job']}).status_code == 400
    second = client.post('/chats', json={'patient_id': patient['id']}).json()
    assert client.patch(f"/chats/{second['id']}", json={'source_ids': ['jane-job']}).status_code == 200
    with patch('api.run_background_job') as run:
        response = client.post('/analyze/async', json={'chat_id': general['id'], 'patient_id': patient['id'], 'query': 'Vitamin D'})
        assert response.status_code == 202
        assert 'Jane' not in run.call_args.kwargs['query']
        assert 'Vitamin D' in run.call_args.kwargs['context_report']
    saved = client.get(f"/chats/{general['id']}").json()
    assert saved['runs'][0]['patient_id'] is None
    assert saved['messages'][0]['content'] == 'Vitamin D'


def test_failed_send_retry_and_message_ownership(client):
    _, chat = create(client)
    _, other = create(client)
    with patch('api.intake_chat_endpoint', side_effect=RuntimeError('offline')):
        with pytest.raises(RuntimeError):
            client.post(f"/chats/{chat['id']}/messages", json={'id': 'retry-id', 'content': 'Question'})
    assert client.get(f"/chats/{chat['id']}").json()['messages'][0]['status'] == 'failed'
    assert client.post(f"/chats/{other['id']}/messages", json={'id': 'retry-id', 'content': 'Question'}).status_code == 409
    with patch('api.intake_chat_endpoint', return_value={'content': 'Reply'}):
        assert client.post(f"/chats/{chat['id']}/messages", json={'id': 'retry-id', 'content': 'Question'}).status_code == 200
    assert len(client.get(f"/chats/{chat['id']}").json()['messages']) == 2


def test_history_migration_regeneration_and_reassignment(client):
    patient, _ = create(client, 'Jane')
    with session_scope() as session:
        repository.create_conversation(session, conversation_id='root', query='Original', patient_id=patient['id'], status='completed')
        repository.create_conversation(session, conversation_id='child', query='Regenerated', parent_job_id='root', status='completed')
        repository.create_conversation(session, conversation_id='unlinked', query='Jane mentioned but unlinked', status='completed')
        chats.migrate_history(session)
        root = session.get(Conversation, 'root')
        child = session.get(Conversation, 'child')
        assert root.chat_id == child.chat_id
        assert child.patient_id == patient['id']
        imported_id = session.get(Conversation, 'unlinked').chat_id
        assert session.get(SavedChat, imported_id).patient_id is None
        count = len(list(session.scalars(select(SavedChat))))
        chats.migrate_history(session)
        assert len(list(session.scalars(select(SavedChat)))) == count
    assert client.patch(f'/chats/{imported_id}', json={'patient_id': patient['id']}).status_code == 200
    with patch('api.run_background_job'):
        response = client.post('/jobs/root/regenerate', json={})
        assert response.status_code == 202
    with session_scope() as session:
        regenerated = session.get(Conversation, response.json()['job_id'])
        assert regenerated.patient_id == patient['id']
        assert regenerated.chat_id == session.get(Conversation, 'root').chat_id


def test_upgrade_database_without_chat_column(client):
    from database.config import get_engine
    from sqlalchemy import text, inspect
    with get_engine().begin() as connection:
        connection.execute(text('DROP TABLE conversations'))
        connection.execute(text('CREATE TABLE conversations (id VARCHAR(36) PRIMARY KEY, query TEXT NOT NULL, agent_id VARCHAR(64), status VARCHAR(32), model VARCHAR(128), implementation VARCHAR(64), error TEXT, files JSON, result JSON, parent_job_id VARCHAR(36), patient_id VARCHAR(36), report_id VARCHAR(36), created_at DATETIME, updated_at DATETIME)'))
        connection.execute(text("INSERT INTO conversations (id, query, status, created_at, updated_at) VALUES ('legacy', 'Old query', 'completed', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"))
    init_db()
    init_db()
    assert 'chat_id' in {c['name'] for c in inspect(get_engine()).get_columns('conversations')}
    with session_scope() as session:
        job = session.get(Conversation, 'legacy')
        assert job.chat_id
        assert session.get(SavedChat, job.chat_id).imported
        assert len(list(session.scalars(select(SavedChat)))) == 1


def test_pending_reply_is_bound_to_origin_and_duplicate_blocked(client):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    _, first = create(client)
    _, second = create(client)
    entered, release = threading.Event(), threading.Event()
    def delayed(req):
        entered.set()
        assert release.wait(5)
        return {'content': 'Only in first chat'}
    with patch('api.intake_chat_endpoint', side_effect=delayed), ThreadPoolExecutor() as pool:
        future = pool.submit(client.post, f"/chats/{first['id']}/messages", json={'id': 'pending', 'content': 'First chat'})
        assert entered.wait(5)
        try:
            assert client.get(f"/chats/{second['id']}").json()['messages'] == []
            assert client.post(f"/chats/{first['id']}/messages", json={'content': 'Duplicate'}).status_code == 409
        finally:
            release.set()
        assert future.result().status_code == 200
    assert client.get(f"/chats/{second['id']}").json()['messages'] == []
    assert client.get(f"/chats/{first['id']}").json()['messages'][-1]['content'] == 'Only in first chat'


def test_patient_edits_and_removed_sources_change_future_context(client):
    patient, chat = create(client, 'Jane')
    other = client.post('/chats', json={'patient_id': patient['id']}).json()
    with session_scope() as session:
        repository.create_conversation(session, conversation_id='source', query='Explicit old evidence', patient_id=patient['id'], chat_id=other['id'])
    client.patch(f"/chats/{chat['id']}", json={'source_ids': ['source']})
    client.put(f"/patients/{patient['id']}", json={'clinical_data': {'overall_health': [{'marker': 'Updated marker', 'value': 0}]}})
    with session_scope() as session:
        context = chats.background(session, chats.get_chat(session, chat['id']))
        assert 'Updated marker' in context and 'Explicit old evidence' in context
    client.patch(f"/chats/{chat['id']}", json={'source_ids': []})
    with session_scope() as session:
        assert 'Explicit old evidence' not in chats.background(session, chats.get_chat(session, chat['id']))
    assert client.delete(f"/patients/{patient['id']}").status_code == 200
    with session_scope() as session:
        assert session.get(Conversation, 'source') is not None
        assert session.get(SavedChat, chat['id']).patient_id is None


def test_saved_synthesis_retains_complete_context(client):
    _, chat = create(client)
    with patch('api.intake_chat_endpoint', return_value={'content': 'Clarifying reply'}):
        client.post(f"/chats/{chat['id']}/messages", json={'content': 'Original detail'})
    with patch('api.intake_summarize_endpoint', return_value={'summary': 'Short synthesis'}), patch('api.run_background_job') as run:
        assert client.post('/analyze/async', json={'chat_id': chat['id'], 'query': 'Extra detail'}).status_code == 202
        query = run.call_args.kwargs['query']
        assert all(value in query for value in ('Short synthesis', 'Original detail', 'Extra detail'))
        assert query in run.call_args.kwargs['context_report']
