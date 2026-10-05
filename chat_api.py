"""Saved medical chat API. All source context is resolved within its workspace."""
import threading
import uuid
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select, update
from database import chats
from database.models import SavedChat, ChatMessage, ChatAttachment, Patient, Conversation
from database.session import ensure_initialized, session_scope

router = APIRouter(prefix="/chats")
_send_lock = threading.Lock()
_busy = set()


class ChatCreate(BaseModel):
    patient_id: Optional[str] = None


class ChatUpdate(BaseModel):
    title: Optional[str] = None
    draft: Optional[str] = None
    source_ids: Optional[list[str]] = None
    patient_id: Optional[str] = None


class AttachmentCreate(BaseModel):
    filename: str
    markdown: str


class MessageCreate(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str = Field(min_length=1)
    model: Optional[str] = None


def require_chat(session, chat_id):
    try:
        return chats.get_chat(session, chat_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("")
def list_chats(patient_id: Optional[str] = None):
    ensure_initialized()
    with session_scope() as session:
        chats.migrate_history(session)
        return [chats.serialize(session, c, False) for c in session.scalars(
            select(SavedChat).where(SavedChat.patient_id == patient_id).order_by(SavedChat.updated_at.desc()))]


@router.post("", status_code=201)
def create_chat(req: ChatCreate):
    ensure_initialized()
    with session_scope() as session:
        if req.patient_id and not session.get(Patient, req.patient_id):
            raise HTTPException(404, "Patient not found")
        chat = SavedChat(patient_id=req.patient_id)
        session.add(chat)
        session.flush()
        return chats.serialize(session, chat)


@router.get("/{chat_id}")
def read_chat(chat_id: str):
    ensure_initialized()
    with session_scope() as session:
        return chats.serialize(session, require_chat(session, chat_id))


@router.patch("/{chat_id}")
def update_chat(chat_id: str, req: ChatUpdate):
    ensure_initialized()
    with session_scope() as session:
        chat = require_chat(session, chat_id)
        if "patient_id" in req.model_fields_set and req.patient_id != chat.patient_id:
            if not chat.imported:
                raise HTTPException(400, "Only imported history can be reassigned")
            if chats.messages(session, chat.id) or session.scalar(select(Conversation.id).where(
                    Conversation.chat_id == chat.id, Conversation.status.in_(["pending", "running"]))):
                raise HTTPException(409, "Imported history with new chat messages or running analyses cannot be moved")
            if req.patient_id and not session.get(Patient, req.patient_id):
                raise HTTPException(404, "Patient not found")
            chat.patient_id = req.patient_id
            chat.source_ids = []
            moved_ids = {chat.id, *session.scalars(select(Conversation.id).where(Conversation.chat_id == chat.id))}
            for other in session.scalars(select(SavedChat)):
                if moved_ids.intersection(other.source_ids or []):
                    other.source_ids = [source for source in other.source_ids if source not in moved_ids]
            session.execute(update(Conversation).where(Conversation.chat_id == chat.id).values(patient_id=req.patient_id))
        if req.title is not None:
            chat.title = req.title.strip()[:255] or "New chat"
        if req.draft is not None:
            chat.draft = req.draft
        if req.source_ids is not None:
            try:
                chats.validate_sources(session, chat, req.source_ids)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            chat.source_ids = list(dict.fromkeys(req.source_ids))
        session.flush()
        return chats.serialize(session, chat)


@router.post("/{chat_id}/attachments", status_code=201)
def add_attachment(chat_id: str, req: AttachmentCreate):
    ensure_initialized()
    with session_scope() as session:
        chat = require_chat(session, chat_id)
        chat.updated_at = datetime.now(timezone.utc)
        item = ChatAttachment(chat_id=chat_id, **req.model_dump())
        session.add(item)
        session.flush()
        return {"id": item.id, "filename": item.filename}


@router.delete("/{chat_id}/attachments/{attachment_id}")
def remove_attachment(chat_id: str, attachment_id: str):
    ensure_initialized()
    with session_scope() as session:
        item = session.get(ChatAttachment, attachment_id)
        if not item or item.chat_id != chat_id:
            raise HTTPException(404, "Attachment not found")
        session.delete(item)
        return {"status": "removed"}


@router.post("/{chat_id}/messages")
def send_message(chat_id: str, req: MessageCreate):
    ensure_initialized()
    with _send_lock:
        if chat_id in _busy:
            raise HTTPException(409, "A reply is already pending in this chat")
        _busy.add(chat_id)
    try:
        with session_scope() as session:
            chat = require_chat(session, chat_id)
            existing = session.get(ChatMessage, req.id)
            if existing and (existing.chat_id != chat_id or existing.content != req.content):
                raise HTTPException(409, "Message identifier already used")
            if existing and existing.status == "complete":
                return chats.serialize(session, chat)
            if not existing:
                existing = ChatMessage(id=req.id, chat_id=chat_id, role="user", content=req.content, status="pending")
                session.add(existing)
            existing.status = "pending"
            chat.draft = ""
            chat.updated_at = datetime.now(timezone.utc)
            if chat.title == "New chat":
                chat.title = req.content.strip()[:80]
            session.flush()
            history = [{"role": m.role, "content": m.content} for m in chats.messages(session, chat_id)
                       if m.status == "complete" or m.id == req.id]
            context = chats.background(session, chat)
        from api import intake_chat_endpoint, IntakeChatRequest
        result = intake_chat_endpoint(IntakeChatRequest(messages=history, model=req.model, document_context=context))
        with session_scope() as session:
            session.get(ChatMessage, req.id).status = "complete"
            session.add(ChatMessage(chat_id=chat_id, role="assistant", content=result["content"]))
            session.flush()
            return chats.serialize(session, require_chat(session, chat_id))
    except Exception:
        with session_scope() as session:
            message = session.get(ChatMessage, req.id)
            if message and message.chat_id == chat_id and message.status == "pending":
                message.status = "failed"
        raise
    finally:
        with _send_lock:
            _busy.discard(chat_id)
