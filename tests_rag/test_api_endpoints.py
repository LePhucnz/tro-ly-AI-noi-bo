"""
tests_rag/test_api_endpoints.py
Kiểm thử tích hợp các Endpoint FastAPI bao gồm SSE streaming và quản lý phiên.
"""

import json
import pytest
from starlette.testclient import TestClient
from server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "rag_core" in data


def test_chat_sync_endpoint(client):
    payload = {
        "session_id": "test_api_session",
        "query": "Quy chế bảo mật thông tin nội bộ",
        "strategy": "similarity",
        "k": 2,
    }
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "test_api_session"
    assert "answer" in data
    assert len(data["answer"]) > 0
    assert len(data["citations"]) > 0


def test_chat_stream_sse_endpoint(client):
    payload = {
        "session_id": "test_sse_session",
        "query": "Quy định làm việc từ xa WFH?",
        "strategy": "mmr",
        "k": 3,
    }
    response = client.post(
        "/api/v1/chat/stream",
        json=payload,
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    lines = response.text.split("\r\n")
    events = [l for l in lines if l.startswith("event:")]
    data_lines = [l for l in lines if l.startswith("data:")]

    assert len(events) > 0
    assert len(data_lines) > 0
    # Đảm bảo có event citations hoặc token
    assert any("token" in e or "citations" in e for e in events)


def test_session_management_endpoints(client):
    sess_id = "test_management_session"
    # Gửi tin nhắn để tạo session
    client.post(
        "/api/v1/chat",
        json={"session_id": sess_id, "query": "Tin nhắn khởi tạo", "strategy": "similarity"},
    )

    # 1. Kiểm tra session có trong danh sách
    res_list = client.get("/api/v1/sessions")
    assert res_list.status_code == 200
    assert sess_id in res_list.json()["active_sessions"]

    # 2. Lấy lịch sử session
    res_hist = client.get(f"/api/v1/sessions/{sess_id}/history")
    assert res_hist.status_code == 200
    assert res_hist.json()["total_messages"] >= 2

    # 3. Xóa session
    res_del = client.delete(f"/api/v1/sessions/{sess_id}")
    assert res_del.status_code == 200
    assert res_del.json()["cleared"] is True


def test_validation_errors(client):
    # Query rỗng -> 400 Bad Request
    res_empty_query = client.post(
        "/api/v1/chat",
        json={"session_id": "valid_sess", "query": "   ", "strategy": "similarity"},
    )
    assert res_empty_query.status_code == 400

    # Session ID rỗng -> 400 Bad Request
    res_empty_sess = client.post(
        "/api/v1/chat",
        json={"session_id": "   ", "query": "Câu hỏi hợp lệ", "strategy": "similarity"},
    )
    assert res_empty_sess.status_code == 400

    # Sai strategy -> 422 Unprocessable Entity
    res_invalid_strategy = client.post(
        "/api/v1/chat",
        json={"session_id": "s1", "query": "hello", "strategy": "invalid_strat"},
    )
    assert res_invalid_strategy.status_code == 422

