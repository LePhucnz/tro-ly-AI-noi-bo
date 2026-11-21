"""
api module: FastAPI routers and request/response schemas.
Bao gồm:
- Chat / RAG routes (api_router)
- Document Ingestion routes (ingest_router)
- CV Screening routes (cv_router)
"""

from api.cv_routes import router as cv_router
from api.ingest_routes import router as ingest_router
from api.routes import router as api_router
from api.schemas import (
    BatchCVScreeningResponse,
    ChatNonStreamingResponse,
    ChatRequest,
    ClearSessionResponse,
    HealthCheckResponse,
    ScreenTextRequest,
    SessionHistoryResponse,
    SessionListResponse,
)

__all__ = [
    "api_router",
    "ingest_router",
    "cv_router",
    "ChatRequest",
    "ChatNonStreamingResponse",
    "SessionHistoryResponse",
    "SessionListResponse",
    "ClearSessionResponse",
    "HealthCheckResponse",
    "ScreenTextRequest",
    "BatchCVScreeningResponse",
]
