"""
api/schemas.py
Pydantic Schemas định nghĩa các kiểu dữ liệu vào/ra cho FastAPI.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from services.rag_core_interface import RetrieverStrategy


class ChatRequest(BaseModel):
    """Yêu cầu gửi từ client đến hệ thống RAG."""
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Mã định danh phiên làm việc (session_id) để quản lý lịch sử hội thoại riêng biệt",
        examples=["session-marketing-01", "user-123-rag"],
    )
    query: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Nội dung câu hỏi hoặc yêu cầu cần trợ giúp từ RAG",
        examples=["Quy định làm việc từ xa của công ty là gì?"],
    )
    strategy: RetrieverStrategy = Field(
        default=RetrieverStrategy.similarity,
        description="Chiến lược tìm kiếm: similarity (tương đồng), mmr (đa dạng nguồn), threshold (ngưỡng tối thiểu)",
    )
    k: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Số lượng tài liệu tham khảo tối đa cần trích xuất",
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Ngưỡng điểm tương đồng tối thiểu (áp dụng khi chọn chiến lược threshold)",
    )


class ChatNonStreamingResponse(BaseModel):
    """Phản hồi đồng bộ hoàn chỉnh cho client không dùng SSE."""
    session_id: str
    answer: str
    citations: List[Dict[str, Any]]
    strategy: str


class SessionMessageSchema(BaseModel):
    """Chi tiết từng tin nhắn trong lịch sử phiên."""
    role: str
    content: str
    timestamp: float
    citations: Optional[List[Dict[str, Any]]] = None


class SessionHistoryResponse(BaseModel):
    """Danh sách lịch sử hội thoại của một phiên làm việc."""
    session_id: str
    total_messages: int
    messages: List[SessionMessageSchema]


class SessionListResponse(BaseModel):
    """Danh sách các session_id hiện hữu."""
    active_sessions: List[str]
    total_sessions: int


class ClearSessionResponse(BaseModel):
    """Kết quả xóa lịch sử phiên."""
    session_id: str
    cleared: bool
    message: str


class HealthCheckResponse(BaseModel):
    """Trạng thái hoạt động của hệ thống."""
    status: str
    active_sessions_count: int
    rag_core: str

