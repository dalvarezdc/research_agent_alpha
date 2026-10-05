"""Durable chat storage and workspace-scoped context assembly."""
import json
from pathlib import Path
from sqlalchemy import select
from .models import SavedChat, ChatMessage, ChatAttachment, Conversation, Patient


def migrate_history(session):
    jobs = list(session.scalars(select(Conversation)))
    by_id = {job.id: job for job in jobs}
    for job in jobs:
        if job.chat_id:
            continue
        root, seen = job, set()
        while root.parent_job_id in by_id and root.id not in seen:
            seen.add(root.id)
            root = by_id[root.parent_job_id]
        if not root.chat_id:
            chat = SavedChat(title=root.query[:80] or "Imported analysis", patient_id=root.patient_id,
                             imported=True, created_at=root.created_at)
            session.add(chat)
            session.flush()
            root.chat_id = chat.id
        job.chat_id = root.chat_id
        job.patient_id = root.patient_id
    session.flush()


def get_chat(session, chat_id):
    chat = session.get(SavedChat, chat_id)
    if chat is None:
        raise ValueError("Chat not found")
    return chat


def messages(session, chat_id):
    return list(session.scalars(select(ChatMessage).where(ChatMessage.chat_id == chat_id)
                               .order_by(ChatMessage.created_at, ChatMessage.id)))


def serialize(session, chat, detail=True):
    result = {key: getattr(chat, key) for key in
              ("id", "patient_id", "title", "draft", "source_ids", "imported")}
    result["updated_at"] = chat.updated_at.isoformat()
    if detail:
        result["messages"] = [{"id": m.id, "role": m.role, "content": m.content, "status": m.status}
                              for m in messages(session, chat.id)]
        result["attachments"] = [{"id": a.id, "filename": a.filename, "markdown": a.markdown}
                                 for a in session.scalars(select(ChatAttachment).where(ChatAttachment.chat_id == chat.id))]
        from .repository import conversation_to_job_dict
        result["runs"] = [conversation_to_job_dict(j) for j in session.scalars(
            select(Conversation).where(Conversation.chat_id == chat.id).order_by(Conversation.created_at.desc()))]
    return result


def validate_sources(session, chat, ids):
    for source_id in ids:
        job = session.get(Conversation, source_id) or session.get(SavedChat, source_id)
        origin = job.id if isinstance(job, SavedChat) else getattr(job, "chat_id", None)
        if job is None or job.patient_id != chat.patient_id or origin == chat.id:
            raise ValueError("Sources must belong to another chat in this workspace")


def background(session, chat):
    parts = []
    if chat.patient_id:
        patient = session.get(Patient, chat.patient_id)
        if patient is None:
            raise ValueError("Patient no longer exists")
        record = {k: getattr(patient, k) for k in
                  ("name", "age", "gender", "primary_condition", "metadata_json", "clinical_data")}
        parts.append("--- PATIENT RECORD ---\n" + json.dumps(record, ensure_ascii=False))
    for attachment in session.scalars(select(ChatAttachment).where(ChatAttachment.chat_id == chat.id)):
        parts.append(f"--- ATTACHMENT: {attachment.filename} ---\n{attachment.markdown}")
    validate_sources(session, chat, chat.source_ids or [])
    for source_id in chat.source_ids or []:
        job = session.get(Conversation, source_id)
        if job:
            parts.append(f"--- SELECTED ANALYSIS {job.id} ---\n" + analysis_text(job))
        else:
            source = get_chat(session, source_id)
            parts.append(f"--- SELECTED CHAT {source.title} ---\n" + transcript(session, source))
    return "\n\n".join(parts)


def analysis_text(job):
    """Read this run's own report, never a browser's most recently viewed report."""
    root = Path("outputs").resolve()
    for key in ("patient_report", "summary", "markdown_report"):
        filename = (job.files or {}).get(key)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.is_relative_to(root) and path.is_file() and path.suffix == ".md":
            return job.query + "\n\n" + path.read_text(encoding="utf-8")
    return job.query + "\n\n" + json.dumps(job.result or {}, ensure_ascii=False)


def transcript(session, chat):
    return "\n\n".join(f"{m.role.upper()}: {m.content}" for m in messages(session, chat.id)
                        if m.status == "complete")
