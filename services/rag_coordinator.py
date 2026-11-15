"""
services/rag_coordinator.py
Điều phối chu trình hội thoại RAG:
1. Quản lý lịch sử qua MemoryManager.
2. Tìm kiếm tài liệu bằng RAG Core (theo chiến lược similarity/mmr/threshold).
3. Đóng gói và phát SSE sự kiện: Citations -> Token Stream -> Done.
4. Tự động lưu trữ câu trả lời hoàn chỉnh vào bộ nhớ phiên.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Dict, Optional
from services.memory import MemoryManager, get_memory_manager
from services.output_parser import RAGOutputParser, SourceCitation
from services.rag_core_interface import (
    BaseRAGCore,
    RetrieverStrategy,
    get_rag_core,
)

logger = logging.getLogger("rag_coordinator")


class RAGCoordinator:
    """
    Bộ điều phối chính (Orchestrator) kết nối API, MemoryManager, OutputParser và RAG Core.
    """

    def __init__(
        self,
        memory_manager: Optional[MemoryManager] = None,
        rag_core: Optional[BaseRAGCore] = None,
    ):
        self.memory = memory_manager or get_memory_manager()
        self.rag_core = rag_core or get_rag_core()

    async def stream_chat(
        self,
        session_id: str,
        query: str,
        strategy: RetrieverStrategy = RetrieverStrategy.similarity,
        k: int = 4,
        threshold: float = 0.5,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Async generator sinh ra các sự kiện SSE có định dạng:
        - data: {"type": "citations", "citations": [...]}
        - data: {"type": "token", "content": "..."}
        - data: {"type": "done", "session_id": "...", ...}
        - data: {"type": "error", "message": "..."}
        """
        try:
            # 1. Ghi nhận tin nhắn người dùng vào Memory
            self.memory.add_user_message(session_id=session_id, content=query)
            history = self.memory.get_history(session_id)

            # 2. Truy vấn tài liệu tham khảo theo chiến lược được chọn
            citations: list[SourceCitation] = await self.rag_core.retrieve_documents(
                query=query,
                strategy=strategy,
                k=k,
                threshold=threshold,
            )

            # 3. Phát sự kiện Citations để frontend nhận danh sách nguồn trước hoặc song song
            yield {
                "event": "citations",
                "data": json.dumps(RAGOutputParser.format_citations_event(citations), ensure_ascii=False),
            }

            # 4. Stream từng token trả lời từ LLM
            full_answer_parts: list[str] = []
            token_count = 0

            async for token in self.rag_core.stream_answer(
                query=query,
                history=history,
                citations=citations,
            ):
                full_answer_parts.append(token)
                token_count += 1
                yield {
                    "event": "token",
                    "data": json.dumps(RAGOutputParser.format_token_event(token), ensure_ascii=False),
                }

            # 5. Lưu câu trả lời hoàn chỉnh cùng citations vào MemoryManager
            complete_answer = "".join(full_answer_parts)
            citation_dicts = [c.to_dict() for c in citations]
            self.memory.add_assistant_message(
                session_id=session_id,
                content=complete_answer,
                citations=citation_dicts,
            )

            # 6. Phát sự kiện hoàn tất (Done Event)
            yield {
                "event": "done",
                "data": json.dumps(
                    RAGOutputParser.format_done_event(
                        session_id=session_id,
                        total_tokens=token_count,
                        strategy_used=strategy.value,
                    ),
                    ensure_ascii=False,
                ),
            }

        except Exception as exc:
            logger.exception("Lỗi trong quá trình điều phối RAG: %s", exc)
            yield {
                "event": "error",
                "data": json.dumps(
                    RAGOutputParser.format_error_event(
                        error_msg=f"Đã xảy ra lỗi khi xử lý câu hỏi: {str(exc)}",
                        error_code="INTERNAL_SERVER_ERROR",
                    ),
                    ensure_ascii=False,
                ),
            }

    async def get_response_sync(
        self,
        session_id: str,
        query: str,
        strategy: RetrieverStrategy = RetrieverStrategy.similarity,
        k: int = 4,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Phương thức lấy phản hồi dạng non-streaming (dành cho sync client)."""
        self.memory.add_user_message(session_id=session_id, content=query)
        history = self.memory.get_history(session_id)

        citations = await self.rag_core.retrieve_documents(
            query=query,
            strategy=strategy,
            k=k,
            threshold=threshold,
        )

        full_answer_parts = []
        async for token in self.rag_core.stream_answer(query=query, history=history, citations=citations):
            full_answer_parts.append(token)

        complete_answer = "".join(full_answer_parts)
        citation_dicts = [c.to_dict() for c in citations]

        self.memory.add_assistant_message(
            session_id=session_id,
            content=complete_answer,
            citations=citation_dicts,
        )

        return {
            "session_id": session_id,
            "answer": complete_answer,
            "citations": citation_dicts,
            "strategy": strategy.value,
        }

