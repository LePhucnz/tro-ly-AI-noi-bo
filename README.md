# 🤖 Trợ lý AI Nội bộ & Sàng lọc CV Tuyển dụng (Enterprise AI Assistant & CV Screener)

Hệ thống Trí tuệ Nhân tạo Doanh nghiệp toàn diện theo mô hình **Clean Architecture**, tích hợp 2 phân hệ cốt lõi:
1. **Hỏi - Đáp Tri thức Nội bộ (RAG Engine)**: Sử dụng **ChromaDB Persistent Storage**, **Multilingual Embedding (Tiếng Việt)**, **Hybrid Search (BM25 + Dense Vector)**, **Neighbor Chunk Expansion (Mở rộng ngữ cảnh liên hoàn theo Điều luật)**, **Document Ingestion Pipeline** chống nạp trùng lặp (SHA-256), bộ nhớ hội thoại (`session_id`), streaming thời gian thực qua **Server-Sent Events (SSE)** và giao diện chat **Streamlit phong cách Google Gemini**.
2. **Sàng lọc CV Tuyển dụng Thông minh (CV Screening Core & HR Dashboard)**: Tự động trích xuất hồ sơ ứng viên từ đa định dạng (`.pdf`, `.docx`, `.txt`, `.md`), đối chiếu đa tiêu chí có trọng số, **chống ảo giác tuyệt đối (Anti-Hallucination Guard)** bằng cách chỉ công nhận khi có bằng chứng trích xuất trực tiếp từ CV, hỗ trợ sàng lọc theo lô (**Batch Processing**), tự động xếp hạng ứng viên, sinh bản tóm tắt dành cho HR (**Executive Summary**) và gợi ý câu hỏi phỏng vấn có trọng tâm.

---

## 🏗️ Kiến trúc Tổng thể Hệ thống (Clean Architecture)

```
                                    +-----------------------------------------------+
                                    |         Streamlit Frontend (app.py)           |
                                    |  [💬 Hỏi đáp Gemini]    [📋 Sàng lọc CV HR]   |
                                    +-----------------------+-----------------------+
                                                            |
                                                            | HTTP REST / SSE (EventSource)
                                                            v
+-----------------------------------------------------------------------------------------------------------------------+
| FastAPI API Layer (server.py / main.py / api/)                                                                        |
|                                                                                                                       |
|  [POST /api/v1/chat/stream]     [POST /api/v1/documents/upload]     [POST /api/v1/cv/screen] (Batch)                 |
|  [POST /api/v1/chat]            [GET /api/v1/documents/stats]       [POST /api/v1/cv/screen-text]                    |
|  [GET /api/v1/sessions]         [POST /api/v1/documents/reset]      [GET /api/v1/cv/sample-criteria]                 |
|               |                                |                                   |                                  |
+---------------|--------------------------------|-----------------------------------|----------------------------------+
                |                                |                                   |
                v                                v                                   v
+-------------------------------+   +---------------------------+   +---------------------------------------------------+
| TẦNG RAG CHAT & COORDINATOR   |   | TẦNG XỬ LÝ TÀI LIỆU       |   | TẦNG SÀNG LỌC CV TUYỂN DỤNG (CV Screening Core)  |
| (services/ & rag/)            |   | (documents/)              |   | (services/cv_screener.py & core/cv_schemas.py)   |
|                               |   |                           |   |                                                   |
| +---------------------------+ |   | +-----------------------+ |   | +-----------------------------------------------+ |
| | RAGCoordinator            | |   | | Loader Factory        | |   | | CVPreprocessor                                | |
| | - MemoryManager (Sessions)| |   | | (.pdf/.docx/.txt/     | |   | | - Đọc tệp/text, chuẩn hóa & phân loại mục     | |
| | - OutputParser (Citations)| |   | |  .csv/.md)            | |   | +-----------------------+-----------------------+ |
| +-------------+-------------+ |   | +-----------+-----------+ |   |                         |                         |
|               |               |   |             |             |   |                         v                         |
|               v               |   |             v             |   | +-----------------------------------------------+ |
| +---------------------------+ |   | +-----------------------+ |   | | CVExtractor                                   | |
| | CleanRAGCore              | |   | | TextCleaner           | |   | | - Bóc tách Info, Skills, Exp, Edu             | |
| | - AntiHallucinationPrompt | |   | | - Xóa Header/Footer   | |   | +-----------------------+-----------------------+ |
| | - LLM Streaming (Gemini)  | |   | +-----------+-----------+ |   |                         |                         |
| +-------------+-------------+ |   |             |             |   |                         v                         |
|               |               |   |             v             |   | +-----------------------------------------------+ |
|               v               |   | +-----------------------+ |   | | CVScorer (Anti-Hallucination Guard)           | |
| +---------------------------+ |   | | Splitter (1000/200)   | |   | | - Đối chiếu JobCriteria, tính điểm trọng số   | |
| | AdvancedRetriever         | |   | | - SHA-256 Hasher      | |   | | - Bắt buộc bằng chứng trích xuất từ CV        | |
| | (Similarity/Hybrid/MMR)   | |   | | - Deduplication       | |   | +-----------------------+-----------------------+ |
| +-------------+-------------+ |   | +-----------+-----------+ |   |                         |                         |
|               |               |   |             |             |   |                         v                         |
|               v               |   |             v             |   | +-----------------------------------------------+ |
| +---------------------------+ |   | +-----------------------+ |   | | CVSummarizer                                  | |
| | ChromaVectorStoreManager  |<----+ | IngestionPipeline     | |   | | - Sinh tóm tắt HR, điểm mạnh, điểm thiếu      | |
| | (Persistent tại ./data)   | |   | (Nạp vào ChromaDB)      | |   | | - Gợi ý câu hỏi phỏng vấn đào sâu             | |
| +---------------------------+ |   +---------------------------+   | +-----------------------------------------------+ |
+-------------------------------+                                   +---------------------------------------------------+
```

---

## 📂 Cấu trúc Thư mục Chi tiết

```text
├── core/                                # Cấu hình tập trung & Schemas dữ liệu chuẩn
│   ├── __init__.py
│   ├── config.py                        # Pydantic Settings đọc biến môi trường từ .env
│   ├── cv_schemas.py                    # Schemas: JobCriteria, Profile, CriterionEvaluation, CVScreeningResult
│   └── exceptions.py                    # Phân cấp Domain Exceptions: RAG & CV Screening
│
├── documents/                           # Pipeline nạp, băm mã & tiền xử lý tài liệu
│   ├── __init__.py
│   ├── cv_preprocessor.py               # Tiền xử lý, chuẩn hóa nội dung hồ sơ CV (.pdf, .docx, .txt, .md)
│   ├── hasher.py                        # Băm SHA-256 tệp/chunk, sinh deterministic ID & chống nạp trùng lặp
│   ├── loader_factory.py                # Factory tự động đọc các định dạng: .pdf, .docx, .txt, .csv, .md
│   ├── text_cleaner.py                  # Loại bỏ header/footer lặp trang PDF, chuẩn hóa khoảng trắng
│   ├── splitter.py                      # RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
│   └── ingestion.py                     # DocumentIngestionPipeline (clean -> hash -> split -> deduplicate)
│
├── rag/                                 # Vector Store, Retriever & RAG Engine
│   ├── __init__.py
│   ├── vector_store.py                  # ChromaDB Persistent Storage & Multilingual Embedding Manager
│   ├── retriever.py                     # Advanced Retriever (similarity, hybrid, mmr, threshold)
│   ├── prompts.py                       # Anti-Hallucination Prompt & Professional Synthesis Formatting
│   └── engine.py                        # CleanRAGCore thực thi BaseRAGCore (LLM Streaming SSE & Sync)
│
├── api/                                 # Tầng giao tiếp HTTP & Server-Sent Events (FastAPI)
│   ├── __init__.py
│   ├── routes.py                        # Router Hỏi - Đáp: /chat, /chat/stream, /sessions
│   ├── ingest_routes.py                 # Router Quản lý Tài liệu: /documents/upload, /stats, /reset
│   ├── cv_routes.py                     # Router Sàng lọc CV: /cv/screen (Batch), /cv/screen-text, /cv/sample-criteria
│   └── schemas.py                       # Pydantic Schemas xác thực dữ liệu đầu vào/ra
│
├── services/                            # Tầng nghiệp vụ điều phối phiên & logic nghiệp vụ
│   ├── __init__.py
│   ├── cv_screener.py                   # CV Screening Core: CVExtractor, CVScorer, CVSummarizer, CVScreeningService
│   ├── memory.py                        # MemoryManager: Quản lý lịch sử hội thoại, Sliding Window
│   ├── output_parser.py                 # Bóc tách câu trả lời và danh sách Source Citations
│   ├── rag_coordinator.py               # Bộ điều phối chính kết nối Memory, RAG Core & SSE
│   └── rag_core_interface.py            # Interface chuẩn BaseRAGCore & DefaultInternalRAGCore fallback
│
├── tests_rag/                           # Bộ kiểm thử tự động toàn diện (47 bài test đạt 100%)
│   ├── test_cv_api.py                   # Kiểm thử các API Endpoints Sàng lọc CV (Batch ranking, upload, validation)
│   ├── test_cv_screening.py             # Kiểm thử phân hệ Sàng lọc CV, Chống ảo giác & Báo cáo HR
│   ├── test_core_rag_ingestion.py       # Kiểm thử Ingestion, Hasher, Splitter, ChromaDB, Retriever, Anti-Hallucination
│   ├── test_api_endpoints.py            # Kiểm thử các API Endpoints RAG Chat & SSE Streaming
│   ├── test_rag_coordinator.py          # Kiểm thử bộ điều phối RAGCoordinator
│   ├── test_memory.py                   # Kiểm thử quản lý bộ nhớ phiên & sliding window
│   └── test_output_parser.py            # Kiểm thử trích xuất citations và định dạng SSE
│
├── app.py                               # Giao diện người dùng Streamlit (2 Chế độ: RAG Chat Gemini & HR CV Dashboard)
├── server.py                            # Khởi tạo FastAPI App, CORS, Lifespan & Mount Routers
├── main.py                              # Điểm khởi chạy FastAPI Backend
├── .env                                 # Biến môi trường đang kích hoạt cục bộ
├── .env.example                         # Tệp cấu hình mẫu cho dự án
├── requirements.txt                     # Danh sách thư viện phụ thuộc
├── pytest.ini                           # Cấu hình Pytest tự động
├── .gitignore                           # Cấu hình loại trừ tệp nhạy cảm và tạm thời
└── AI_DEVELOPMENT_GUIDELINES.md         # Quy chuẩn phát triển phần mềm bắt buộc dành cho AI
```

---

## 🌟 Các Tính năng Nổi bật

### 1. Phân hệ Hỏi - Đáp Tri thức Nội bộ (RAG Engine)
- **Ingestion Pipeline chống nạp trùng lặp**: Tự động băm mã SHA-256, kiểm tra deduplication và sinh mã Chunk ID tiền định trước khi lưu vào ChromaDB.
- **Multilingual Embedding (Tiếng Việt)**: Sử dụng `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` cho độ chính xác ngữ nghĩa vượt trội.
- **Hybrid Search & Mở rộng ngữ cảnh Điều luật**: Kết hợp Dense Vector + BM25 Okapi cùng cơ chế tự động ghép nối các Điều luật liền kề khi tra cứu văn bản quy định.
- **Anti-Hallucination Guardrails**: Ép LLM chỉ trả lời dựa trên tài liệu cung cấp; từ chối lịch sự và trung thực nếu không có thông tin; đính kèm trích dẫn nguồn (**Source Citations**) minh bạch.
- **Streaming SSE**: Đẩy từng token chữ cái ra client mượt mà như ChatGPT qua giao thức Server-Sent Events.

### 2. Phân hệ Sàng lọc CV Tuyển dụng (CV Screening Core)
- **Đọc đa định dạng**: Đọc sạch sẽ `.pdf`, `.docx`, `.txt`, `.md` qua `DocumentLoaderFactory`.
- **Trích xuất thông tin có cấu trúc**: Bóc tách họ tên, email, số điện thoại, địa điểm, liên kết GitHub/LinkedIn, kinh nghiệm làm việc, học vấn và danh mục kỹ năng.
- **Chấm điểm theo trọng số & Chống ảo giác (Anti-Hallucination Guard)**:
  - Bắt buộc phải có bằng chứng thực tế trích xuất trực tiếp từ CV.
  - Nếu kỹ năng hoặc kinh nghiệm không xuất hiện trong CV, hệ thống chấm điểm **0.0** và ghi rõ `evidence: "Không tìm thấy thông tin trong CV"`, **tuyệt đối không tự suy đoán**.
  - Tính điểm tổng thể theo bình quân gia quyền của từng tiêu chí và phân loại mức độ phù hợp (`STRONG_FIT`, `POTENTIAL_FIT`, `BORDERLINE`, `NOT_FIT`).
- **Sàng lọc theo lô (Batch Processing) & Xếp hạng Ứng viên**:
  - Hỗ trợ tải lên nhiều CV cùng lúc cho một vị trí tuyển dụng.
  - Tự động sắp xếp ứng viên theo điểm số từ cao xuống thấp, xác định ngay ứng viên tiềm năng nhất (`top_candidate`).
- **Báo cáo chuyên nghiệp cho HR & Gợi ý Câu hỏi Phỏng vấn**:
  - Tự động tổng hợp điểm mạnh nổi bật kèm trích dẫn và điểm còn thiếu/chưa rõ.
  - Soạn thảo bản tóm tắt nhận định và khuyến nghị tuyển dụng (**Executive Summary**).
  - Tự động sinh danh sách câu hỏi phỏng vấn có trọng tâm giúp HR đào sâu năng lực chuyên môn và làm rõ các điểm nghi vấn trong CV.

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
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

- **Swagger UI Interactive Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc Documentation**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Root Health Check**: [http://localhost:8000/](http://localhost:8000/)

### 3. Khởi chạy Streamlit Frontend

Mở một terminal mới:
```bash
source .venv/bin/activate
streamlit run app.py --server.port 8501
```
Truy cập giao diện tương tác tại: [http://localhost:8501](http://localhost:8501)
> 💡 *Sử dụng menu bên thanh điều hướng Sidebar để chuyển đổi linh hoạt giữa:*
> - **Chế độ 1**: `💬 Hỏi - Đáp Tri thức Nội bộ`
> - **Chế độ 2**: `📋 Sàng lọc CV Tuyển dụng (HR)`

### 4. Chạy Toàn bộ Bộ Kiểm thử Tự động (47 Tests đạt 100%)

```bash
pytest -c pytest.ini tests_rag/ -v
```

---

## 📡 Tài liệu API Mẫu (cURL Examples)

### 1. Sàng lọc Hồ sơ CV Tuyển dụng Hàng loạt (`POST /api/v1/cv/screen`)
```bash
curl -X POST "http://localhost:8000/api/v1/cv/screen" \
  -F 'criteria={"job_title": "Senior Backend Developer", "required_skills": ["Python", "FastAPI"], "passing_score": 70.0}' \
  -F "files=@./docs_sample/CV_Nguyen_Van_A.pdf" \
  -F "files=@./docs_sample/CV_Tran_Thi_B.docx"
```

### 2. Sàng lọc CV từ Chuỗi Văn bản (`POST /api/v1/cv/screen-text`)
```bash
curl -X POST "http://localhost:8000/api/v1/cv/screen-text" \
  -H "Content-Type: application/json" \
  -d '{
    "cv_text": "NGUYỄN VĂN AN\nEmail: an.nguyen@example.com\nKỸ NĂNG: Python, FastAPI, Docker\nKINH NGHIỆM: 4 năm phát triển Backend...",
    "criteria": {
      "job_title": "Senior Python Developer",
      "required_skills": ["Python", "FastAPI"],
      "passing_score": 70.0
    },
    "file_name": "CV_Nguyen_Van_An.txt"
  }'
```

### 3. Lấy Danh sách Tiêu chí Tuyển dụng Mẫu (`GET /api/v1/cv/sample-criteria`)
```bash
curl http://localhost:8000/api/v1/cv/sample-criteria
```

### 4. Nạp Tài liệu vào Vector DB (`POST /api/v1/documents/upload`)
```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@./docs_sample/Quy_che_noi_bo_quan_tri.pdf"
```

### 5. Hỏi - Đáp Streaming SSE (`POST /api/v1/chat/stream`)
```bash
curl -N -X POST "http://localhost:8000/api/v1/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user-session-01",
    "query": "Quy định làm việc từ xa WFH của công ty như thế nào?",
    "strategy": "hybrid",
    "k": 6,
    "threshold": 0.35
  }'
```

### 6. Xem Thống kê Vector Store (`GET /api/v1/documents/stats`)
```bash
curl http://localhost:8000/api/v1/documents/stats
```

---

## ⚙️ Bảng Biến Môi trường (.env)

Tất cả các tham số có thể điều chỉnh linh hoạt trong tệp [.env](file:///home/nguyentrunghieu/Màn%20hình%20nền/tro-ly-AI-noi-bo/.env):

| Biến Môi trường | Giá trị Mặc định | Ý nghĩa |
| :--- | :--- | :--- |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | Thư mục lưu trữ dữ liệu ChromaDB |
| `CHROMA_COLLECTION_NAME` | `internal_knowledge_base` | Tên collection vector mặc định |
| `EMBEDDING_PROVIDER` | `huggingface` | Nhà cung cấp embedding (`huggingface` hoặc `openai`) |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Mô hình vector đa ngữ hỗ trợ tối ưu tiếng Việt |
| `CHUNK_SIZE` | `1000` | Số ký tự tối đa của một đoạn văn bản |
| `CHUNK_OVERLAP` | `200` | Số ký tự gối đầu giữa 2 đoạn văn bản liền kề |
| `DEFAULT_RETRIEVER_STRATEGY` | `similarity` | Chiến lược mặc định (`similarity`, `hybrid`, `mmr`, `similarity_score_threshold`) |
| `DEFAULT_TOP_K` | `4` | Số lượng tài liệu trích xuất mặc định (trên UI là 6) |
| `DEFAULT_SCORE_THRESHOLD` | `0.5` | Ngưỡng điểm tương đồng tối thiểu |
| `MMR_LAMBDA_MULT` | `0.7` | Hệ số cân bằng giữa Relevance và Diversity của MMR |
| `LLM_PROVIDER` | `openai` | Nhà cung cấp LLM (`openai`, `gemini`, hoặc `mock`) |
| `LLM_MODEL_NAME` | `google/gemini-2.5-flash` | Tên mô hình ngôn ngữ lớn sinh câu trả lời |
| `LLM_TEMPERATURE` | `0.0` | Độ sáng tạo (0.0 chống ảo giác tối đa) |
| `OPENAI_API_KEY` | *(Khóa API)* | API key truy cập LLM |
| `OPENAI_BASE_URL` | `https://openrouter.ai/api/v1` | Endpoint API tương thích OpenAI |
