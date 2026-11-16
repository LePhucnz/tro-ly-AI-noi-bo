# 🤖 Trợ lý AI Nội bộ (Internal Enterprise RAG Assistant)

Hệ thống Trợ lý Trí tuệ Nhân tạo tra cứu tài liệu nội bộ doanh nghiệp theo mô hình **Clean Architecture**, tích hợp **RAG (Retrieval-Augmented Generation)**, **ChromaDB Persistent Storage**, **Document Ingestion Pipeline** chống trùng lặp, quản lý bộ nhớ hội thoại (`session_id`), truyền phát câu trả lời thời gian thực qua **Server-Sent Events (SSE)** và giao diện tương tác **Streamlit**.

---

## 🏗️ Kiến trúc Tổng thể Hệ thống (Clean Architecture)

```
                                    +-----------------------+
                                    |  Streamlit Frontend   |
                                    |       (app.py)        |
                                    +-----------+-----------+
                                                |
                                                | HTTP REST / SSE (EventSource)
                                                v
+-----------------------------------------------------------------------------------------------+
| FastAPI API Layer (server.py / main.py / api/)                                                |
|                                                                                               |
|  [POST /api/v1/chat/stream]      [POST /api/v1/chat]          [POST /api/v1/documents/upload] |
|               |                           |                                  |                |
|               v                           v                                  v                |
|       +---------------------------------------+                   +-----------------------+   |
|       |            RAGCoordinator             |                   |  DocumentIngestion    |   |
|       | (Điều phối Stream, Session & Memory)  |                   |       Pipeline        |   |
|       +-------------------+-------------------+                   +-----------+-----------+   |
|                           |                                                   |               |
+---------------------------|---------------------------------------------------|---------------+
                            |                                                   |
                            v                                                   v
+-----------------------------------------------+   +-------------------------------------------+
| TẦNG RAG CORE (rag/)                          |   | TẦNG XỬ LÝ TÀI LIỆU (documents/)          |
|                                               |   |                                           |
|  +-----------------------------------------+  |   |  +-------------------------------------+  |
|  | CleanRAGCore (implements BaseRAGCore)   |  |   |  | Loader Factory (.pdf/.docx/.txt/    |  |
|  | - stream_answer() / generate_answer()   |  |   |  |                .csv/.md)            |  |
|  +--------------------+--------------------+  |   |  +------------------+------------------+  |
|                       |                       |   |                     |                     |
|  +--------------------+--------------------+  |   |  +------------------v------------------+  |
|  | AntiHallucinationPrompt                 |  |   |  | RecursiveCharacterTextSplitter      |  |
|  | - Groundedness, Refusal, Citations      |  |   |  | (chunk_size=1000, overlap=200)      |  |
|  +--------------------+--------------------+  |   |  +------------------+------------------+  |
|                       |                       |   |                     |                     |
|  +--------------------+--------------------+  |   |  +------------------v------------------+  |
|  | AdvancedRetriever                       |  |   |  | DocumentHasher (SHA-256)            |  |
|  | - 1. similarity                        |  |   |  | - File & Deterministic Chunk IDs    |  |
|  | - 2. mmr (Maximal Marginal Relevance)  |  |   |  | - Deduplication Registry            |  |
|  | - 3. similarity_score_threshold         |  |   |  +------------------+------------------+  |
|  +--------------------+--------------------+  |   +---------------------|---------------------+
|                       |                                                 |
|                       v                                                 v
|  +---------------------------------------------------------------------------------+          |
|  | ChromaVectorStoreManager (ChromaDB Persistent Client tại ./data/chroma_db)      |          |
|  | - Embeddings: Sentence-Transformers (all-MiniLM-L6-v2)                          |          |
|  | - Idempotent Upsert & Deduplication: Ngăn chặn 100% nạp lặp dữ liệu vector     |          |
|  +---------------------------------------------------------------------------------+          |
+-----------------------------------------------------------------------------------------------+
```

---

## 📂 Cấu trúc Thư mục Chi tiết

```text
├── core/                                # Cấu hình tập trung & Ngoại lệ phân cấp
│   ├── __init__.py
│   ├── config.py                        # Pydantic Settings đọc biến môi trường từ .env
│   └── exceptions.py                    # Domain Exceptions phân cấp & FastAPI Exception Handlers
│
├── documents/                           # Pipeline nạp & tiền xử lý tài liệu
│   ├── __init__.py
│   ├── hasher.py                        # Băm SHA-256 tệp/chunk, sinh deterministic ID & chống trùng lặp
│   ├── loader_factory.py                # Factory tự động đọc các định dạng: .pdf, .docx, .txt, .csv, .md
│   ├── splitter.py                      # RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
│   └── ingestion.py                     # DocumentIngestionPipeline trọn vẹn (load -> hash -> split -> deduplicate)
│
├── rag/                                 # Vector Store, Retriever & RAG Engine
│   ├── __init__.py
│   ├── vector_store.py                  # ChromaDB Persistent Storage Client & Embedding Manager
│   ├── retriever.py                     # Advanced Retriever (similarity, mmr, similarity_score_threshold)
│   ├── prompts.py                       # Anti-Hallucination Prompt Template & Quy chuẩn trích dẫn
│   └── engine.py                        # CleanRAGCore thực thi BaseRAGCore (Streaming SSE & Sync)
│
├── api/                                 # Tầng giao tiếp HTTP & Server-Sent Events
│   ├── __init__.py
│   ├── routes.py                        # Router Hỏi - Đáp: /chat, /chat/stream, /sessions
│   ├── ingest_routes.py                 # Router Quản lý Tài liệu: /documents/upload, /stats, /reset
│   └── schemas.py                       # Pydantic Schemas xác thực dữ liệu đầu vào/ra
│
├── services/                            # Tầng nghiệp vụ điều phối phiên & bộ nhớ
│   ├── __init__.py
│   ├── memory.py                        # MemoryManager: Quản lý lịch sử hội thoại, Sliding Window
│   ├── output_parser.py                 # Bóc tách câu trả lời và danh sách Source Citations
│   ├── rag_coordinator.py               # Bộ điều phối chính kết nối Memory, RAG Core & SSE
│   └── rag_core_interface.py            # Interface chuẩn BaseRAGCore & DefaultInternalRAGCore fallback
│
├── tests_rag/                           # Bộ kiểm thử tự động (29 bài test)
│   ├── test_core_rag_ingestion.py       # Kiểm thử Ingestion, Hasher, Splitter, ChromaDB, Retriever, Anti-Hallucination
│   ├── test_api_endpoints.py            # Kiểm thử các API Endpoints FastAPI & SSE Streaming
│   ├── test_rag_coordinator.py          # Kiểm thử bộ điều phối RAGCoordinator
│   ├── test_memory.py                   # Kiểm thử quản lý bộ nhớ phiên & sliding window
│   └── test_output_parser.py            # Kiểm thử trích xuất citations và định dạng SSE
│
├── app.py                               # Giao diện người dùng Streamlit trực quan
├── server.py                            # Khởi tạo FastAPI App, CORS, Lifespan & Routers
├── main.py                              # Điểm khởi chạy FastAPI Backend
├── .env                                 # Biến môi trường đang kích hoạt cục bộ
├── .env.example                         # Tệp cấu hình mẫu cho dự án
├── requirements.txt                     # Danh sách thư viện phụ thuộc
├── pytest.ini                           # Cấu hình Pytest
└── AI_DEVELOPMENT_GUIDELINES.md         # Quy chuẩn phát triển phần mềm bắt buộc
```

---

## 🌟 Các Tính năng Cốt lõi

### 1. Ingestion Pipeline & Chống trùng lặp (SHA-256)
- **Loader Factory**: Tự động nhận diện định dạng tệp và bóc tách dữ liệu sạch từ `.pdf`, `.docx`, `.txt`, `.csv`, `.md`.
- **Recursive Text Splitter**: Phân mảnh văn bản tối ưu ngữ cảnh (`chunk_size=1000`, `chunk_overlap=200`).
- **Chống nạp trùng lặp tuyệt đối**:
  - Tính mã băm SHA-256 nhị phân của tệp gốc.
  - Sinh **Chunk ID tiền định (Deterministic ID)**: `chk_sha256(file_hash + chunk_index + chunk_content_hash)`.
  - Kiểm tra đối chiếu trước khi nạp vào Vector Store. Nếu tệp hoặc đoạn đã tồn tại, hệ thống tự động bỏ qua (idempotent), không gây phình to database.

### 2. ChromaDB Persistent Storage & Advanced Retriever
- **Lưu trữ Cục bộ (Local Persistence)**: Lưu trữ toàn bộ vector index và metadata vào thư mục `./data/chroma_db`, sẵn sàng phục vụ mà không phụ thuộc cloud database ngoài.
- **Local Embeddings**: Tích hợp mô hình `sentence-transformers/all-MiniLM-L6-v2` chạy cục bộ (CPU/GPU), có cơ chế dự phòng `DeterministicHashEmbeddings` offline.
- **3 Chiến lược Tìm kiếm Linh hoạt**:
  1. `similarity`: Tìm kiếm Cosine Similarity chuẩn, lấy top $K$ tài liệu liên quan nhất.
  2. `mmr` *(Maximal Marginal Relevance)*: Tối ưu độ đa dạng, loại bỏ các đoạn trích lặp ý và gom nhặt thông tin từ nhiều tài liệu khác nhau.
  3. `similarity_score_threshold`: Lọc bỏ dứt khoát các đoạn trích có điểm tương đồng dưới ngưỡng quy định.

### 3. Anti-Hallucination & Trích dẫn Nguồn Minh bạch (Source Citations)
- **Groundedness**: Ép LLM chỉ trả lời dựa trên thông tin có trong ngữ cảnh tài liệu nội bộ được cung cấp.
- **Honest Refusal**: Nếu không tìm thấy thông tin hoặc điểm tương đồng thấp, hệ thống từ chối lịch sự:
  > *"Xin lỗi, tôi đã kiểm tra kỹ toàn bộ tài liệu nội bộ được cung cấp nhưng không tìm thấy thông tin phù hợp..."*
- **Source Citations**: Mọi khẳng định đều đính kèm nguồn minh bạch: tên tệp (`source`), số trang (`page`), điểm tin cậy (`score`), và đoạn trích dẫn gốc (`snippet`).

### 4. Quản lý Bộ nhớ & Server-Sent Events (SSE)
- Quản lý phiên làm việc đa người dùng tách biệt qua `session_id`.
- Bộ nhớ Sliding Window tự động cắt tỉa tin nhắn cũ để bảo toàn giới hạn context của LLM.
- Truyền phát câu trả lời từng token thời gian thực dạng streaming mượt mà như ChatGPT.

---

## 🚀 Hướng dẫn Cài đặt & Khởi chạy

### 1. Chuẩn bị Môi trường & Cài đặt Thư viện

```bash
# Khởi tạo môi trường ảo
python3 -m venv .venv
source .venv/bin/activate

# Cài đặt toàn bộ dependencies
pip install -r requirements.txt

# Thiết lập file môi trường (nếu chưa có)
cp .env.example .env
```

### 2. Khởi chạy FastAPI Backend Service

```bash
# Chạy trực tiếp qua script main.py (khuyến nghị):
python3 main.py

# Hoặc chạy bằng uvicorn:
uvicorn server:app --host 0.0.0.0 --port 8000
```

- **Swagger UI Interactive Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc Documentation**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Root Health Check**: [http://localhost:8000/](http://localhost:8000/)

### 3. Khởi chạy Streamlit Frontend

Mở một cửa sổ terminal mới:
```bash
source .venv/bin/activate
streamlit run app.py --server.port 8501
```
Truy cập giao diện tương tác tại: [http://localhost:8501](http://localhost:8501)

### 4. Chạy Toàn bộ Bộ Kiểm thử Tự động (29 Tests)

```bash
pytest -c pytest.ini tests_rag/ -v
```

---

## 📡 Tài liệu API Mẫu (cURL Examples)

### 1. Nạp Tài liệu vào ChromaDB (`POST /api/v1/documents/upload`)
Hỗ trợ các định dạng `.pdf`, `.docx`, `.txt`, `.csv`, `.md`:
```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@./docs_sample/Quy_che_WFH_2024.pdf"
```

### 2. Xem Thống kê Vector Store (`GET /api/v1/documents/stats`)
```bash
curl http://localhost:8000/api/v1/documents/stats
```
Phản hồi mẫu:
```json
{
  "collection_name": "internal_knowledge_base",
  "persist_directory": "/home/noissi/tro-ly-AI-noi-bo/data/chroma_db",
  "total_chunks_count": 42,
  "embedding_provider": "huggingface",
  "supported_file_types": [".pdf", ".docx", ".txt", ".csv", ".md"]
}
```

### 3. Hỏi - Đáp Streaming SSE (`POST /api/v1/chat/stream`)
```bash
curl -N -X POST "http://localhost:8000/api/v1/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user-session-01",
    "query": "Quy định làm việc từ xa tối đa mấy ngày một tuần?",
    "strategy": "mmr",
    "k": 3,
    "threshold": 0.5
  }'
```

### 4. Hỏi - Đáp Đồng bộ (`POST /api/v1/chat`)
```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user-session-01",
    "query": "Chính sách bảo mật mật khẩu công ty như thế nào?",
    "strategy": "similarity",
    "k": 4
  }'
```

### 5. Xóa Dữ liệu Vector Database (`POST /api/v1/documents/reset`)
```bash
curl -X POST "http://localhost:8000/api/v1/documents/reset"
```

---

## ⚙️ Biến Môi trường (.env)

Tất cả các tham số có thể điều chỉnh linh hoạt trong tệp [.env](file:///home/noissi/tro-ly-AI-noi-bo/.env):

| Biến Môi trường | Giá trị Mặc định | Ý nghĩa |
| :--- | :--- | :--- |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | Đường dẫn thư mục lưu trữ persistent của ChromaDB |
| `CHROMA_COLLECTION_NAME` | `internal_knowledge_base` | Tên collection mặc định |
| `EMBEDDING_PROVIDER` | `huggingface` | Nhà cung cấp embedding (`huggingface` hoặc `openai`) |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Mô hình biểu diễn vector ngữ nghĩa |
| `CHUNK_SIZE` | `1000` | Số ký tự tối đa của một chunk |
| `CHUNK_OVERLAP` | `200` | Số ký tự gối đầu giữa 2 chunks liền kề |
| `DEFAULT_RETRIEVER_STRATEGY` | `similarity` | Chiến lược mặc định (`similarity`, `mmr`, `similarity_score_threshold`) |
| `DEFAULT_TOP_K` | `4` | Số lượng tài liệu trích xuất tối đa |
| `DEFAULT_SCORE_THRESHOLD` | `0.5` | Ngưỡng điểm tương đồng tối thiểu |
| `MMR_LAMBDA_MULT` | `0.7` | Hệ số cân bằng giữa Relevance và Diversity của MMR |
| `LLM_PROVIDER` | `mock` | Nhà cung cấp LLM (`mock`, `openai`, `gemini`) |
| `LLM_TEMPERATURE` | `0.0` | Độ sáng tạo (0.0 chống ảo giác tối đa) |
