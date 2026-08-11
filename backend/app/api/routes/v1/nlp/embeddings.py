"""
API route para embeddings y búsqueda semántica (pgvector).

Endpoints:
  POST /api/v1/nlp/embed   → Generar embeddings de texto libre
  GET  /api/v1/nlp/search  → Búsqueda semántica sobre modelos/datasets del tenant
"""

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import get_current_tenant, require_viewer
from app.core.config import settings
from app.database import get_db
from app.models.tenant import Tenant
from app.models.user import User

logger = logging.getLogger(__name__)
router = APIRouter()


class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=128)


@router.post("/embed")
def embed_texts(
    req: EmbedRequest,
    _user: User = Depends(require_viewer),
    _tenant: Tenant = Depends(get_current_tenant),
):
    """
    Genera embeddings de una lista de textos con el modelo configurado.

    Requiere rol **viewer** o superior.
    """
    from app.services.embedding_service import get_embedding_service

    service = get_embedding_service()
    try:
        embeddings = service.embed_texts(req.texts)
    except Exception as exc:
        logger.error("Error generando embeddings: %s", exc)
        raise HTTPException(status_code=500, detail=f"Error generando embeddings: {exc}")

    return {
        "dim": settings.EMBEDDING_DIM,
        "model": settings.EMBEDDING_MODEL_NAME,
        "embeddings": embeddings.tolist(),
    }


@router.get("/search")
def semantic_search(
    q: str = Query(..., min_length=1, description="Consulta en lenguaje natural"),
    index: str = Query("models", description="models | datasets"),
    limit: int = Query(20, ge=1, le=100),
    _user: User = Depends(require_viewer),
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """
    Búsqueda semántica sobre modelos o datasets del tenant.

    Requiere rol **viewer** o superior.
    """
    if index not in ("models", "datasets"):
        raise HTTPException(
            status_code=400, detail="index debe ser 'models' o 'datasets'."
        )

    from app.services.embedding_service import get_embedding_service

    service = get_embedding_service()
    results = service.search(db, tenant.id, q, top_k=limit, index=index)
    return {"query": q, "index": index, "results": results}
