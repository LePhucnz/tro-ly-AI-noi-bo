"""
api/routes.py
FastAPI Router tiếp nhận request, tích hợp Server-Sent Events (SSE) streaming
thông qua sse-starlette, cung cấp endpoint quản lý phiên và lịch sử hội thoại.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator
from fastapi import APIRouter, HTTPException, Query, Request, status
from sse_starlette.sse import EventSourceResponse

from api.schemas import (
    ChatNonStreamingResponse,
    ChatRequest,
    ClearSessionResponse,
    HealthCheckResponse,
    SessionHistoryResponse,
    SessionListResponse,
    SessionMessageSchema,
)
from services.memory import get_memory_manager
from services.rag_coordinator import RAGCoordinator
from services.rag_core_interface import get_rag_core

logger = logging.getLogger("api_routes")
router = APIRouter(prefix="/api/v1", tags=["RAG Assistant & Streaming"])


@router.post(
    "/chat/stream",
    summary="SSE Streaming Chat Endpoint",
    description="Stream câu trả lời của AI và danh sách tài liệu tham khảo theo thời gian thực (SSE)",
    response_class=EventSourceResponse,
)
async def chat_stream_endpoint(
    request: Request,
    payload: ChatRequest,
) -> EventSourceResponse:
    """
    Endpoint chính xử lý Streaming phản hồi theo thời gian thực.
    - Nhận: session_id, query, strategy (similarity/mmr/threshold), k, threshold.
    - Trả về: SSE EventSourceResponse đẩy từng token chữ cái ra client mượt mà như ChatGPT.
    """
    # 1. Kiểm tra tính hợp lệ của query
    clean_query = payload.query.strip()
    if not clean_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Nội dung câu hỏi (query) không được để trống.",
        )

    clean_session_id = payload.session_id.strip()
    if not clean_session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Mã phiên (session_id) không được để trống.",
        )

    coordinator = RAGCoordinator()

    async def event_generator() -> AsyncIterator[dict[str, Any]]:
        """Bọc luồng stream và lắng nghe sự kiện client ngắt kết nối."""
        try:
            async for sse_event in coordinator.stream_chat(
                session_id=clean_session_id,
                query=clean_query,
                strategy=payload.strategy,
                k=payload.k,
                threshold=payload.threshold,
            ):
                # Kiểm tra nếu client đã ngắt kết nối đột ngột
                if await request.is_disconnected():
                    logger.warning("Client đã ngắt kết nối giữa chừng cho session: %s", clean_session_id)
                    break

                yield sse_event

        except Exception as exc:
            logger.exception("Lỗi bất thường trong luồng event_generator: %s", exc)
            yield {
                "event": "error",
                "data": f'{{"error": "Lỗi luồng stream: {str(exc)}"}}',
            }

    return EventSourceResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Chống buffer proxy của NGINX
        },
    )


@router.post(
    "/chat",
    response_model=ChatNonStreamingResponse,
    summary="Non-Streaming Chat Endpoint",
    description="Nhận câu trả lời hoàn chỉnh cùng nguồn tham khảo theo phương thức HTTP đồng bộ",
)
async def chat_sync_endpoint(payload: ChatRequest) -> ChatNonStreamingResponse:
    """Endpoint đồng bộ dành cho các client không hỗ trợ SSE streaming."""
    clean_query = payload.query.strip()
    if not clean_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Nội dung câu hỏi (query) không được để trống.",
        )

    clean_session_id = payload.session_id.strip()
    if not clean_session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Mã phiên (session_id) không được để trống.",
        )

    coordinator = RAGCoordinator()
    result = await coordinator.get_response_sync(
        session_id=clean_session_id,
        query=clean_query,
        strategy=payload.strategy,
        k=payload.k,
        threshold=payload.threshold,
    )
    return ChatNonStreamingResponse(**result)


@router.get(
    "/sessions",
    response_model=SessionListResponse,
    summary="Danh sách phiên hội thoại",
    description="Lấy danh sách các session_id hiện đang được lưu trong MemoryManager",
)
async def list_sessions() -> SessionListResponse:
    """Trả về danh sách tất cả các phiên làm việc đang hoạt động."""
    memory = get_memory_manager()
    sessions = memory.list_active_sessions()
    return SessionListResponse(
        active_sessions=sessions,
        total_sessions=len(sessions),
    )


@router.get(
    "/sessions/{session_id}/history",
    response_model=SessionHistoryResponse,
    summary="Lịch sử hội thoại của một phiên",
    description="Truy xuất toàn bộ danh sách tin nhắn và nguồn tham khảo của một session_id",
)
async def get_session_history(session_id: str) -> SessionHistoryResponse:
    """Lấy chi tiết lịch sử tin nhắn của session cụ thể."""
    memory = get_memory_manager()
    history = memory.get_history(session_id=session_id)
    return SessionHistoryResponse(
        session_id=session_id,
        total_messages=len(history),
        messages=[
            SessionMessageSchema(
                role=m.role,
                content=m.content,
                timestamp=m.timestamp,
                citations=m.citations,
            )
            for m in history
        ],
    )


@router.delete(
    "/sessions/{session_id}",
    response_model=ClearSessionResponse,
    summary="Xóa phiên hội thoại",
    description="Xóa sạch lịch sử trao đổi của session_id khỏi bộ nhớ",
)
async def clear_session(session_id: str) -> ClearSessionResponse:
    """Xóa bỏ phiên làm việc khỏi bộ nhớ RAM."""
    memory = get_memory_manager()
    cleared = memory.clear_session(session_id)
    return ClearSessionResponse(
        session_id=session_id,
        cleared=cleared,
        message="Đã xóa lịch sử phiên thành công." if cleared else "Phiên không tồn tại hoặc đã được xóa trước đó.",
    )


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Kiểm tra trạng thái hệ thống",
)
async def health_check() -> HealthCheckResponse:
    """Kiểm tra tình trạng hoạt động của API và RAG backend."""
    memory = get_memory_manager()
    rag_core = get_rag_core()
    return HealthCheckResponse(
        status="healthy",
        active_sessions_count=len(memory.list_active_sessions()),
        rag_core=rag_core.__class__.__name__,
    )

