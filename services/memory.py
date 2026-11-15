"""
services/memory.py
Quản lý bộ nhớ lịch sử hội thoại (MemoryManager) riêng biệt theo từng phiên làm việc (session_id).
Đảm bảo thread-safety, giới hạn kích thước cửa sổ hội thoại (sliding window),
và định dạng lịch sử linh hoạt cho RAG Chain / LLM Prompt.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Mô hình dữ liệu cho từng lượt hội thoại."""
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: float = Field(default_factory=time.time)
    citations: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi sang dict định dạng chuẩn."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "citations": self.citations or [],
        }


class MemoryManager:
    """
    Quản lý bộ nhớ hội thoại theo từng session_id.
    - Phân tách dữ liệu giữa các phiên làm việc độc lập.
    - Hỗ trợ sliding window (giới hạn số lượng tin nhắn gần nhất) để tối ưu context length.
    - Thread-safe khi có nhiều request đồng thời.
    """

    def __init__(self, max_history_messages: int = 30):
        self._max_history = max_history_messages
        self._storage: Dict[str, List[ChatMessage]] = {}
        self._lock = threading.Lock()

    def add_user_message(self, session_id: str, content: str) -> ChatMessage:
        """Thêm tin nhắn người dùng vào session."""
        msg = ChatMessage(role="user", content=content.strip())
        with self._lock:
            if session_id not in self._storage:
                self._storage[session_id] = []
            self._storage[session_id].append(msg)
            self._trim_session(session_id)
        return msg

    def add_assistant_message(
        self,
        session_id: str,
        content: str,
        citations: Optional[List[Dict[str, Any]]] = None,
    ) -> ChatMessage:
        """Thêm câu trả lời của trợ lý kèm nguồn tham khảo."""
        msg = ChatMessage(
            role="assistant",
            content=content.strip(),
            citations=citations or [],
        )
        with self._lock:
            if session_id not in self._storage:
                self._storage[session_id] = []
            self._storage[session_id].append(msg)
            self._trim_session(session_id)
        return msg

    def get_history(self, session_id: str) -> List[ChatMessage]:
        """Lấy danh sách các tin nhắn trong session."""
        with self._lock:
            return list(self._storage.get(session_id, []))

    def get_history_as_dicts(self, session_id: str) -> List[Dict[str, Any]]:
        """Lấy lịch sử hội thoại dưới dạng danh sách từ điển."""
        history = self.get_history(session_id)
        return [msg.to_dict() for msg in history]

    def format_history_for_prompt(
        self,
        session_id: str,
        max_messages: Optional[int] = None,
    ) -> str:
        """
        Định dạng lịch sử hội thoại thành chuỗi văn bản cho Prompt RAG/LLM.
        Ví dụ:
        User: Quy định làm việc từ xa là gì?
        Assistant: Quy định làm việc từ xa gồm...
        """
        history = self.get_history(session_id)
        if max_messages:
            history = history[-max_messages:]

        formatted_lines: List[str] = []
        for msg in history:
            role_label = "User" if msg.role == "user" else "Assistant"
            formatted_lines.append(f"{role_label}: {msg.content}")

        return "\n".join(formatted_lines)

    def clear_session(self, session_id: str) -> bool:
        """Xóa toàn bộ lịch sử của một session_id."""
        with self._lock:
            if session_id in self._storage:
                del self._storage[session_id]
                return True
            return False

    def list_active_sessions(self) -> List[str]:
        """Liệt kê tất cả các session_id hiện có trong bộ nhớ."""
        with self._lock:
            return list(self._storage.keys())

    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Lấy thông tin tóm tắt của phiên làm việc."""
        history = self.get_history(session_id)
        return {
            "session_id": session_id,
            "total_messages": len(history),
            "last_active": history[-1].timestamp if history else None,
        }

    def _trim_session(self, session_id: str) -> None:
        """Cắt bớt các tin nhắn cũ nếu vượt quá ngưỡng cấu hình."""
        if len(self._storage[session_id]) > self._max_history:
            self._storage[session_id] = self._storage[session_id][-self._max_history:]


# Global singleton instance for app-wide sharing
_memory_manager_instance: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Lấy hoặc khởi tạo instance singleton của MemoryManager."""
    global _memory_manager_instance
    if _memory_manager_instance is None:
        _memory_manager_instance = MemoryManager()
    return _memory_manager_instance

