"""
API route para el chatbot asistente (LLM + RAG).

Endpoints:
  POST   /api/v1/nlp/assistant/chat    → Enviar mensaje (JSON o SSE si stream=true)
  GET    /api/v1/nlp/assistant/history → Historial de conversaciones del tenant
  DELETE /api/v1/nlp/assistant/history → Borrar historial del tenant
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_current_tenant, require_editor, require_viewer
from app.core.config import settings
from app.core.rate_limit import limiter
from app.database import get_db
from app.models.assistant import AssistantConversation
from app.models.tenant import Tenant
from app.models.user import User

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Schemas ──────────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    stream: bool = False


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    sources: list = []
    tokens: dict = {}


class ChatHistoryItem(BaseModel):
    conversation_id: str
    created_at: Any
    messages: list


# ──────────────────────────────────────────────────────────────────────────────
# POST /assistant/chat
# ──────────────────────────────────────────────────────────────────────────────


def _get_or_create_conversation(
    db: Session, tenant_id: str, user_id: Optional[str], conversation_id: Optional[str]
) -> AssistantConversation:
    if conversation_id:
        conv = (
            db.query(AssistantConversation)
            .filter(
                AssistantConversation.id == conversation_id,
                AssistantConversation.tenant_id == tenant_id,
            )
            .first()
        )
        if conv:
            return conv
    conv = AssistantConversation(
        tenant_id=tenant_id, user_id=user_id, messages=[]
    )
    db.add(conv)
    db.flush()
    return conv


def _append_message(db: Session, conv: AssistantConversation, role: str, content: str):
    messages = list(conv.messages or [])
    messages.append({"role": role, "content": content, "ts": datetime.now(timezone.utc).isoformat()})
    conv.messages = messages
    db.commit()


@router.post("/assistant/chat")
@limiter.limit(settings.RATE_LIMIT_LLM)
async def assistant_chat(
    request: Request,
    req: ChatRequest,
    user: User = Depends(require_editor),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """
    Envía un mensaje al chatbot. Si `stream=true` responde en SSE,
    si no, devuelve JSON {answer, sources, tokens, conversation_id}.
    """
    from app.services.llm_gateway import get_llm_gateway
    from app.services.rag_service import get_rag_service

    gateway = get_llm_gateway()
    rag = get_rag_service()

    conv = _get_or_create_conversation(
        db, tenant.id, user.id, req.conversation_id
    )
    _append_message(db, conv, "user", req.message)

    rag_data = rag.retrieve_context(db, tenant.id, req.message)

    messages: List[Dict[str, str]] = []
    for msg in (conv.messages or [])[-8:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    async def _sse_stream():
        """SSE: {delta} chunks + cierre con {sources, tokens, conversation_id}."""
        t0 = time.perf_counter()
        accumulated_chunks: List[str] = []
        usage_payload: Dict[str, int] = {"input": 0, "output": 0}
        try:
            async for chunk in gateway.stream(
                messages, system=rag.SYSTEM_PROMPT + "\n\nContexto:\n" + rag_data["context"]
            ):
                if "delta" in chunk:
                    accumulated_chunks.append(chunk["delta"])
                    yield f"data: {json.dumps({'delta': chunk['delta']}, ensure_ascii=False)}\n\n"
                elif "usage" in chunk:
                    usage_payload = chunk["usage"]
                    full_answer = "".join(accumulated_chunks).strip()
                    _append_message(db, conv, "assistant", full_answer)
                    try:
                        gateway.record_usage(
                            db=db,
                            tenant_id=tenant.id,
                            usage=usage_payload,
                            latency_ms=(time.perf_counter() - t0) * 1000,
                            tags={"domain": "assistant", "stream": True},
                        )
                        db.commit()
                    except Exception as exc:
                        logger.warning("LLM streaming usage recording failed (non-critical): %s", exc)
                    yield (
                        f"data: {json.dumps({'sources': rag_data['sources'], 'tokens': usage_payload, 'conversation_id': conv.id}, ensure_ascii=False)}\n\n"
                    )
            yield "data: [DONE]\n\n"
        except Exception as exc:
            logger.error("Streaming LLM error: %s", exc)
            yield f"data: {json.dumps({'error': str(exc)}, ensure_ascii=False)}\n\n"

    if req.stream:
        return StreamingResponse(
            _sse_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # No-stream
    result = await gateway.complete(
        messages, system=rag.SYSTEM_PROMPT + "\n\nContexto:\n" + rag_data["context"]
    )
    _append_message(db, conv, "assistant", result["content"])

    try:
        gateway.record_usage(
            db=db, tenant_id=tenant.id, usage=result["usage"],
            latency_ms=result.get("latency_ms", 0),
            tags={"domain": "assistant"},
        )
        db.commit()
    except Exception as exc:
        logger.warning("LLM usage recording failed (non-critical): %s", exc)

    return ChatResponse(
        conversation_id=conv.id,
        answer=result["content"],
        sources=rag_data["sources"],
        tokens=result["usage"],
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /assistant/history
# ──────────────────────────────────────────────────────────────────────────────


@router.get("/assistant/history")
def get_assistant_history(
    limit: int = Query(20, ge=1, le=100),
    _user: User = Depends(require_viewer),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """
    Devuelve el historial de conversaciones del tenant, ordenado por actualización.
    """
    rows = (
        db.query(AssistantConversation)
        .filter(AssistantConversation.tenant_id == tenant.id)
        .order_by(AssistantConversation.updated_at.desc())
        .limit(limit)
        .all()
    )
    return [
        ChatHistoryItem(
            conversation_id=c.id,
            created_at=c.created_at,
            messages=c.messages or [],
        )
        for c in rows
    ]


# ──────────────────────────────────────────────────────────────────────────────
# DELETE /assistant/history
# ──────────────────────────────────────────────────────────────────────────────


@router.delete("/assistant/history")
def clear_assistant_history(
    _user: User = Depends(require_editor),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """Borra el historial de conversaciones del tenant."""
    db.query(AssistantConversation).filter(
        AssistantConversation.tenant_id == tenant.id
    ).delete()
    db.commit()
    return {"message": "Historial del asistente borrado."}
