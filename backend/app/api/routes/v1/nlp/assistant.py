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


def _synthesize_rag_fallback(question: str, structured: Dict[str, Any]) -> str:
    """Construye una respuesta determinista basada únicamente en el contexto RAG del tenant."""
    q = question.lower()
    models = structured.get("modelos") or []
    endpoints = structured.get("model_endpoints") or []
    datasets = structured.get("datasets") or []
    drift = structured.get("drift_reciente") or []
    costs = structured.get("costes_recientes") or []

    lines: List[str] = ["Resumen en vivo de tu plataforma PraxisML (RAG local):"]

    if any(k in q for k in ("endpoint", "serving", "inferencia", "latencia", "canary", "tráfico", "trafico")) or endpoints:
        if endpoints:
            ep_parts = [
                f"- `{e['slug']}` ({e['status']}): {e['request_count']} reqs, {e['error_count']} errores, {e['avg_latency_ms']} ms avg"
                + (f", Canary {e['traffic_split_pct']}%" if e.get("traffic_split_pct") else "")
                for e in endpoints[:5]
            ]
            lines.append("Model Serving Endpoints:\n" + "\n".join(ep_parts))
        else:
            lines.append("Model Serving Endpoints: no hay endpoints desplegados actualmente.")

    if models:
        prod = [m for m in models if (m.get("stage") or "").lower() == "production"]
        nlp_m = [m for m in models if m.get("domain") == "nlp"]
        m_parts = []
        for m in models[:6]:
            acc = m.get("accuracy")
            f1 = m.get("f1")
            metric_str = ""
            if acc is not None:
                metric_str += f" · acc={float(acc):.3f}"
            if f1 is not None:
                metric_str += f" · f1={float(f1):.3f}"
            m_parts.append(
                f"- **{m['name']}** [{m.get('stage') or 'Staging'}] ({m.get('domain') or 'tabular'}/{m.get('algorithm') or 'ml'}){metric_str}"
            )
        lines.append(
            f"Modelos registrados ({len(models)} total · {len(prod)} en Production · {len(nlp_m)} NLP):\n"
            + "\n".join(m_parts)
        )

    if datasets:
        dvc_cnt = sum(1 for d in datasets if d.get("is_dvc_tracked"))
        lines.append(f"Datasets: {len(datasets)} registrados ({dvc_cnt} versionados con DVC).")

    if any(k in q for k in ("cost", "gasto", "finops", "precio", "mes")) or costs:
        total_cost = sum(float(c.get("amount") or 0.0) for c in costs)
        lines.append(f"FinOps reciente: ${total_cost:.4f} USD en {len(costs)} registros.")

    if any(k in q for k in ("drift", "anomal", "deriva")) or drift:
        anomalies = sum(1 for d in drift if d.get("is_anomaly"))
        lines.append(
            f"Drift monitoring: {len(drift)} chequeos recientes ({anomalies} anomalías detectadas)."
            if drift
            else "Drift monitoring: sin anomalías ni eventos críticos recientes."
        )

    return "\n\n".join(lines)


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
    from app.domains.nlp.llm_gateway import get_llm_gateway
    from app.domains.nlp.rag_service import get_rag_service

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
            logger.warning("External LLM unavailable for streaming (%s); using local RAG summary", exc)
            fallback_answer = _synthesize_rag_fallback(req.message, rag_data.get("structured") or {})
            usage_payload = {
                "input": len(req.message.split()),
                "output": len(fallback_answer.split()),
            }
            _append_message(db, conv, "assistant", fallback_answer)
            try:
                gateway.record_usage(
                    db=db,
                    tenant_id=tenant.id,
                    usage=usage_payload,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    tags={"domain": "assistant", "mode": "rag_local", "stream": True},
                )
                db.commit()
            except Exception:
                pass
            yield f"data: {json.dumps({'delta': fallback_answer}, ensure_ascii=False)}\n\n"
            yield (
                f"data: {json.dumps({'sources': rag_data['sources'], 'tokens': usage_payload, 'conversation_id': conv.id}, ensure_ascii=False)}\n\n"
            )
            yield "data: [DONE]\n\n"

    if req.stream:
        return StreamingResponse(
            _sse_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # No-stream
    t0 = time.perf_counter()
    try:
        result = await gateway.complete(
            messages, system=rag.SYSTEM_PROMPT + "\n\nContexto:\n" + rag_data["context"]
        )
        mode_tag = "llm_gateway"
    except Exception as exc:
        logger.warning("External LLM unavailable (%s); using local RAG summary", exc)
        fallback_answer = _synthesize_rag_fallback(req.message, rag_data.get("structured") or {})
        result = {
            "content": fallback_answer,
            "usage": {
                "input": len(req.message.split()),
                "output": len(fallback_answer.split()),
            },
            "latency_ms": (time.perf_counter() - t0) * 1000,
        }
        mode_tag = "rag_local"

    _append_message(db, conv, "assistant", result["content"])

    try:
        gateway.record_usage(
            db=db, tenant_id=tenant.id, usage=result["usage"],
            latency_ms=result.get("latency_ms", 0),
            tags={"domain": "assistant", "mode": mode_tag},
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
