"""
tests_rag/test_memory.py
Kiểm thử đơn vị cho MemoryManager.
"""

import pytest
from services.memory import MemoryManager


def test_memory_isolation_between_sessions():
    manager = MemoryManager()
    manager.add_user_message("session_1", "Câu hỏi từ phiên 1")
    manager.add_assistant_message("session_1", "Trả lời phiên 1", citations=[{"source": "doc1.pdf"}])

    manager.add_user_message("session_2", "Câu hỏi từ phiên 2")

    hist1 = manager.get_history("session_1")
    hist2 = manager.get_history("session_2")

    assert len(hist1) == 2
    assert len(hist2) == 1
    assert hist1[0].content == "Câu hỏi từ phiên 1"
    assert hist1[1].citations == [{"source": "doc1.pdf"}]
    assert hist2[0].content == "Câu hỏi từ phiên 2"


def test_memory_sliding_window_trim():
    manager = MemoryManager(max_history_messages=4)
    for i in range(10):
        manager.add_user_message("session_trim", f"Message {i}")

    history = manager.get_history("session_trim")
    assert len(history) == 4
    assert history[0].content == "Message 6"
    assert history[-1].content == "Message 9"


def test_memory_clear_session():
    manager = MemoryManager()
    manager.add_user_message("session_del", "Hello")
    assert "session_del" in manager.list_active_sessions()

    cleared = manager.clear_session("session_del")
    assert cleared is True
    assert "session_del" not in manager.list_active_sessions()
    assert len(manager.get_history("session_del")) == 0


def test_format_history_for_prompt():
    manager = MemoryManager()
    manager.add_user_message("session_fmt", "Xin chào")
    manager.add_assistant_message("session_fmt", "Chào bạn, tôi có thể giúp gì?")

    formatted = manager.format_history_for_prompt("session_fmt")
    assert "User: Xin chào" in formatted
    assert "Assistant: Chào bạn, tôi có thể giúp gì?" in formatted

