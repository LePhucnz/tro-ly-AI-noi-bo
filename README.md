# 🤖 Trợ lý AI Nội bộ (Internal Enterprise RAG Assistant)

Hệ thống Trợ lý Trí tuệ Nhân tạo tra cứu tài liệu nội bộ tích hợp **RAG (Retrieval-Augmented Generation)**, quản lý bộ nhớ hội thoại theo phiên (`session_id`), truyền phát câu trả lời thời gian thực qua **Server-Sent Events (SSE)** và giao diện tương tác **Streamlit**.

---

## 🏗️ Kiến trúc Hệ thống

```
                                    +-----------------------+
                                    |  Streamlit Frontend   |
                                    |       (app.py)        |
                                    +-----------+-----------+
                                                |
                                                | HTTP / SSE (EventSource)
                                                v
+---------------------------------------------------------------------------------------+
| FastAPI Backend Service (server.py / main.py)                                          |
|                                                                                       |
|   [POST /api/v1/chat/stream] ----> [EventSourceResponse (sse-starlette)]              |
|                                                |                                      |
|                                                v                                      |
|                                      +--------------------+                           |
|                                      |   RAGCoordinator   |                           |
|                                      +---------+----------+                           |
|                                                |                                      |
|                 +------------------------------+------------------------------+       |
|                 |                                                             |       |
|                 v                                                             v       |
|       +-------------------+                                         +-----------------+
|       |   MemoryManager   |                                         | RAG OutputParser|
|       | (Session History) |                                         |  (Citations)    |
|       +-------------------+                                         +-----------------+
|                                                |                                      |
|                                                v                                      |
|                                      +--------------------+                           |
|                                      |  RAG Core Interface|                           |
|                                      |   (BaseRAGCore)    |                           |
|                                      +---------+----------+                           |
+------------------------------------------------|--------------------------------------+
                                                 v
                                    +-----------------------+
                                    |    Dev 1 RAG Core     |
                                    | (Vector DB + LLM RAG) |
                                    +-----------------------+
```

---

## 📂 Cấu trúc Thư mục

```text
├── api/
│   ├── __init__.py
│   ├── routes.py                 # FastAPI Router, SSE streaming endpoint (/chat/stream)
│   └── schemas.py                # Pydantic schemas (ChatRequest, Response, Strategies)
├── services/
│   ├── __init__.py
│   ├── memory.py                 # MemoryManager: Quản lý lịch sử hội thoại theo session_id
│   ├── output_parser.py          # Output Parser: Bóc tách câu trả lời & Source Citations
│   ├── rag_coordinator.py        # Orchestrator: Điều phối Memory, RAG Core & SSE stream
│   └── rag_core_interface.py     # Giao diện chuẩn BaseRAGCore & DefaultInternalRAGCore
├── tests_rag/
│   ├── test_api_endpoints.py     # Kiểm thử FastAPI endpoints & SSE streaming
│   ├── test_memory.py            # Kiểm thử MemoryManager & sliding window
│   ├── test_output_parser.py     # Kiểm thử Output Parser & trích xuất citations
│   └── test_rag_coordinator.py   # Kiểm thử điều phối & 3 chiến lược retriever
├── app.py                        # Giao diện ứng dụng Streamlit trực quan
├── server.py                     # Khởi tạo FastAPI app, CORS, Middleware
├── main.py                       # Entrypoint FastAPI
├── requirements.txt              # Danh sách thư viện phụ thuộc
├── pytest.ini                    # Cấu hình test tự động
└── AI_DEVELOPMENT_GUIDELINES.md  # Kịch bản & quy tắc bắt buộc dành cho AI khi code
```

---

## 🚀 Hướng dẫn Cài đặt & Chạy Ứng dụng

### 1. Chuẩn bị môi trường Python

```bash
# Tạo môi trường ảo
python3 -m venv .venv
source .venv/bin/activate

# Cài đặt thư viện phụ thuộc
pip install -r requirements.txt
```

### 2. Khởi chạy FastAPI Backend

Backend chạy tại cổng `8000`:
```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```
- **Tài liệu Swagger OpenAPI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Kiểm tra sức khỏe hệ thống**: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)

### 3. Khởi chạy Streamlit Frontend

Mở một terminal mới:
```bash
streamlit run app.py --server.port 8501
```
Truy cập trình duyệt tại: [http://localhost:8501](http://localhost:8501)

### 4. Chạy Bộ Kiểm thử (Unit & Integration Tests)

```bash
pytest -c pytest.ini tests_rag/ -v
```

---

## 🎯 Các Tính năng Nổi bật

### 1. Quản lý Bộ nhớ Hội thoại (`MemoryManager`)
- **Tách biệt theo phiên (`session_id`)**: Đảm bảo an toàn dữ liệu giữa các người dùng hoặc các phiên chat khác nhau.
- **Sliding Window**: Tự động cắt tỉa lịch sử cũ khi vượt ngưỡng để tránh tràn context length của LLM.
- **Thread-safe**: Khóa an toàn (`threading.Lock`) bảo vệ dữ liệu khi có nhiều request đồng thời.
- **Quản lý phiên qua API**: Xem danh sách session, truy xuất lịch sử, xóa phiên làm việc.

### 2. Phản hồi Thời gian thực (SSE Streaming - `sse-starlette`)
- Truyền phát từng token chữ cái ra client mượt mà như ChatGPT.
- **Giao thức sự kiện SSE rõ ràng**:
  - `event: citations`: Đẩy danh sách tài liệu tham khảo ngay khi Vector DB trích xuất xong.
  - `event: token`: Đẩy từng token câu trả lời từ LLM.
  - `event: done`: Báo hiệu kết thúc lượt sinh dữ liệu.
  - `event: error`: Thông báo lỗi an toàn, không ngắt đột ngột kết nối.

### 3. Bộ bóc tách đầu ra (`OutputParser`)
- Bóc tách độc lập phần **Câu trả lời chính** và **Nguồn tài liệu tham khảo (Source Citations)**.
- Trích xuất đầy đủ: Tên tệp (`source`), Số trang/mục (`page`), Điểm tương đồng (`score`), và Đoạn trích dẫn (`snippet`).

### 4. 3 Chiến lược Retriever Linh hoạt
1. **`similarity`**: Lấy top $K$ tài liệu có điểm tương đồng Cosine cao nhất.
2. **`mmr` (Maximal Marginal Relevance)**: Cân bằng giữa độ liên quan và tính đa dạng, loại bỏ trùng lặp nguồn tin.
3. **`threshold`**: Chỉ giữ lại các tài liệu có độ tương đồng đạt hoặc vượt ngưỡng quy định (ngăn ngừa ảo giác AI).

---

## 🔌 Hướng dẫn dành cho Dev 1: Tích hợp RAG Core thực tế

Dev 1 chỉ cần kế thừa lớp `BaseRAGCore` trong `services/rag_core_interface.py`:

```python
from services.rag_core_interface import BaseRAGCore, register_rag_core, RetrieverStrategy
from services.output_parser import SourceCitation
from services.memory import ChatMessage
from typing import AsyncIterator, List

class Dev1ProductionRAGCore(BaseRAGCore):
    def __init__(self, vector_store, llm_chain):
        self.vector_store = vector_store
        self.llm_chain = llm_chain

    async def retrieve_documents(
        self,
        query: str,
        strategy: RetrieverStrategy = RetrieverStrategy.similarity,
        k: int = 4,
        threshold: float = 0.5,
    ) -> List[SourceCitation]:
        # Tích hợp Vector DB thực tế (Chroma, Qdrant, FAISS...)
        # Trả về danh sách SourceCitation
        ...

    async def stream_answer(
        self,
        query: str,
        history: List[ChatMessage],
        citations: List[SourceCitation],
    ) -> AsyncIterator[str]:
        # Tích hợp LLM Streaming (LangChain, OpenAI, Gemini, Ollama...)
        # yield từng token chữ cái
        async for chunk in self.llm_chain.astream(...):
            yield chunk

# Đăng ký instance vào hệ thống:
register_rag_core(Dev1ProductionRAGCore(...))
```
Toàn bộ tầng API, Memory và Streamlit Frontend sẽ tự động sử dụng RAG Core mới mà không cần chỉnh sửa một dòng mã nào!
