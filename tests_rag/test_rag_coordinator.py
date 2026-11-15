"""
tests_rag/test_rag_coordinator.py
Kiểm thử bộ điều phối RAGCoordinator và các chiến lược retriever.
"""

import json
import pytest
from services.memory import MemoryManager
from services.rag_coordinator import RAGCoordinator
from services.rag_core_interface import DefaultInternalRAGCore, RetrieverStrategy


@pytest.mark.asyncio
async def test_retriever_strategies():
    core = DefaultInternalRAGCore()

    # 1. Similarity
    docs_sim = await core.retrieve_documents("làm việc từ xa", strategy=RetrieverStrategy.similarity, k=2)
    assert len(docs_sim) <= 2
    assert any("lam_viec_tu_xa" in d.source for d in docs_sim)

    # 2. Threshold
    docs_thresh = await core.retrieve_documents("làm việc từ xa", strategy=RetrieverStrategy.threshold, threshold=0.85)
    assert all(d.score >= 0.85 for d in docs_thresh)

    # 3. MMR
    docs_mmr = await core.retrieve_documents("bảo mật mật khẩu quy trình", strategy=RetrieverStrategy.mmr, k=3)
    sources = [d.source for d in docs_mmr]
    # Nguồn không bị trùng lặp
    assert len(sources) == len(set(sources))


@pytest.mark.asyncio
async def test_coordinator_stream_chat():
    mem = MemoryManager()
    coordinator = RAGCoordinator(memory_manager=mem)

    session_id = "test_coord_session"
    events = []

    async for event in coordinator.stream_chat(
        session_id=session_id,
        query="Quy định WFH làm việc từ xa như thế nào?",
        strategy=RetrieverStrategy.similarity,
    ):
        events.append(event)

    # Kiểm tra các sự kiện SSE phát ra
    event_names = [e["event"] for e in events]
    assert "citations" in event_names
    assert "token" in event_names
    assert "done" in event_names

    # Kiểm tra citations
    citation_event = next(e for e in events if e["event"] == "citations")
    citation_data = json.loads(citation_event["data"])
    assert len(citation_data["citations"]) > 0

    # Kiểm tra lịch sử trong memory đã được lưu
    history = mem.get_history(session_id)
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[1].role == "assistant"
    assert len(history[1].citations) > 0

