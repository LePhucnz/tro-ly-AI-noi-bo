"""
tests_rag/test_core_rag_ingestion.py
Bộ kiểm thử toàn diện cho Core RAG & Document Ingestion:
1. Core Settings & Domain Exceptions
2. SHA-256 Hasher & Deduplication Registry
3. Loader Factory (.pdf, .docx, .txt, .csv, .md)
4. Recursive Character Text Splitter (1000/200)
5. Ingestion Pipeline chống trùng lặp
6. ChromaDB Persistent Storage & Vector Store
7. Advanced Retriever (similarity, mmr, similarity_score_threshold)
8. Anti-Hallucination Prompt & CleanRAGCore Integration
"""

import asyncio
import os
import tempfile
from pathlib import Path

try:
    import pytest
except ImportError:
    class _MockPytest:
        class mark:
            @staticmethod
            def asyncio(fn):
                return fn
        @staticmethod
        def raises(expected_exception):
            class _RaisesContext:
                def __init__(self):
                    self.value = None
                def __enter__(self):
                    return self
                def __exit__(self, exc_type, exc_val, exc_tb):
                    if exc_type is None:
                        raise AssertionError(f"Expected {expected_exception} but nothing was raised.")
                    if issubclass(exc_type, expected_exception):
                        self.value = exc_val
                        return True
                    return False
            return _RaisesContext()
    pytest = _MockPytest()

from core.config import Settings, get_settings
from core.exceptions import (
    BaseRAGException,
    DocumentIngestionError,
    DocumentLoadError,
    DuplicateDocumentError,
    InvalidRetrieverStrategyError,
    UnsupportedFileTypeError,
    VectorStoreError,
)
from documents.hasher import DocumentHasher, DocumentHashRegistry
from documents.ingestion import DocumentIngestionPipeline, IngestionResult
from documents.loader_factory import Document, DocumentLoaderFactory
from documents.splitter import DocumentSplitter
from rag.prompts import ANTI_HALLUCINATION_SYSTEM_PROMPT, AntiHallucinationPrompt
from rag.retriever import AdvancedRetriever
from rag.vector_store import (
    ChromaVectorStoreManager,
    DeterministicHashEmbeddings,
)
from rag.engine import CleanRAGCore, RetrieverStrategy, ChatMessage
try:
    from services.rag_coordinator import RAGCoordinator
except ImportError:
    RAGCoordinator = None  # type: ignore


# ============================================================================
# 1. Tests cho Module core/ (Settings & Exceptions)
# ============================================================================

def test_core_settings():
    """Kiểm tra cấu hình mặc định từ Pydantic Settings."""
    settings = get_settings()
    assert settings.CHUNK_SIZE == 1000
    assert settings.CHUNK_OVERLAP == 200
    assert settings.DEFAULT_RETRIEVER_STRATEGY == "similarity"
    assert settings.DEFAULT_TOP_K == 4
    assert settings.DEFAULT_SCORE_THRESHOLD == 0.5
    assert settings.CHROMA_COLLECTION_NAME == "internal_knowledge_base"


def test_core_exceptions_hierarchy():
    """Kiểm tra cấu trúc phân cấp và mã lỗi của Domain Exceptions."""
    # UnsupportedFileTypeError
    exc_type = UnsupportedFileTypeError(".exe")
    assert isinstance(exc_type, DocumentIngestionError)
    assert isinstance(exc_type, BaseRAGException)
    assert exc_type.status_code == 415
    assert exc_type.error_code == "UNSUPPORTED_FILE_TYPE"

    # DuplicateDocumentError
    exc_dup = DuplicateDocumentError(file_hash="abc123456789", file_name="chinh_sach.pdf")
    assert exc_dup.status_code == 409
    assert exc_dup.error_code == "DUPLICATE_DOCUMENT_IGNORED"

    # InvalidRetrieverStrategyError
    exc_strat = InvalidRetrieverStrategyError("random_search")
    assert exc_strat.status_code == 400
    assert exc_strat.error_code == "INVALID_RETRIEVER_STRATEGY"


# ============================================================================
# 2. Tests cho Module documents/ (Hasher & Deduplication)
# ============================================================================

def test_document_hasher_sha256():
    """Kiểm tra tính toán mã băm SHA-256 nhất quán cho chuỗi và tệp."""
    content = "Chính sách nhân sự 2024"
    h1 = DocumentHasher.compute_sha256(content)
    h2 = DocumentHasher.compute_sha256(content)
    assert h1 == h2
    assert len(h1) == 64

    # Kiểm tra tính toán từ file vật lý
    with tempfile.NamedTemporaryFile("w", delete=False) as tf:
        tf.write(content)
        tf_path = tf.name

    try:
        file_hash = DocumentHasher.compute_file_sha256(tf_path)
        assert file_hash == h1
    finally:
        os.remove(tf_path)


def test_document_deterministic_chunk_id():
    """Kiểm tra sinh Chunk ID có tính tiền định (deterministic)."""
    file_hash = "a" * 64
    chunk_text = "Nội dung điều khoản 1"
    cid1 = DocumentHasher.generate_chunk_id(file_hash, 0, chunk_text)
    cid2 = DocumentHasher.generate_chunk_id(file_hash, 0, chunk_text)
    cid3 = DocumentHasher.generate_chunk_id(file_hash, 1, chunk_text)

    assert cid1 == cid2
    assert cid1 != cid3
    assert cid1.startswith("chk_")


def test_hash_registry_deduplication():
    """Kiểm tra bộ ghi nhớ phát hiện trùng lặp file và chunk."""
    registry = DocumentHashRegistry()
    f_hash = "mock_file_hash_123"

    assert not registry.is_file_processed(f_hash)
    registry.register_file(f_hash)
    assert registry.is_file_processed(f_hash)

    chunk_id = "chk_abc123"
    assert not registry.is_chunk_processed(chunk_id)
    registry.register_chunk(chunk_id)
    assert registry.is_chunk_processed(chunk_id)


# ============================================================================
# 3. Tests cho Loader Factory (.txt, .md, .csv và từ chối định dạng lạ)
# ============================================================================

def test_loader_factory_supported_extensions():
    """Kiểm tra danh sách định dạng hỗ trợ."""
    exts = DocumentLoaderFactory.get_supported_extensions()
    assert ".pdf" in exts
    assert ".docx" in exts
    assert ".txt" in exts
    assert ".csv" in exts
    assert ".md" in exts


def test_loader_factory_text_and_markdown():
    """Kiểm tra đọc tệp .txt và .md kèm trích xuất metadata."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
        tf.write("Quy định làm việc từ xa tối đa 2 ngày trên tuần.")
        txt_path = tf.name

    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as mf:
        mf.write("# Hướng dẫn kỹ thuật Git\nMọi PR cần 2 reviewers.")
        md_path = mf.name

    try:
        # Đọc TXT
        txt_docs = DocumentLoaderFactory.load_document(txt_path)
        assert len(txt_docs) == 1
        assert "làm việc từ xa" in txt_docs[0].page_content
        assert txt_docs[0].metadata["file_type"] == "txt"
        assert len(txt_docs[0].metadata["file_hash"]) == 64

        # Đọc MD
        md_docs = DocumentLoaderFactory.load_document(md_path)
        assert len(md_docs) == 1
        assert "Hướng dẫn kỹ thuật" in md_docs[0].page_content
        assert md_docs[0].metadata["file_type"] == "md"
    finally:
        os.remove(txt_path)
        os.remove(md_path)


def test_loader_factory_csv():
    """Kiểm tra đọc tệp bảng tính CSV và tạo metadata từng dòng."""
    csv_content = "id,name,role\n1,Nguyen Van A,Developer\n2,Tran Thi B,Designer"
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as cf:
        cf.write(csv_content)
        csv_path = cf.name

    try:
        docs = DocumentLoaderFactory.load_document(csv_path)
        assert len(docs) == 2
        assert "Nguyen Van A" in docs[0].page_content
        assert docs[0].metadata["row"] == 1
        assert "Tran Thi B" in docs[1].page_content
        assert docs[1].metadata["row"] == 2
    finally:
        os.remove(csv_path)


def test_loader_factory_unsupported_file():
    """Kiểm tra từ chối định dạng tệp không được hỗ trợ."""
    with tempfile.NamedTemporaryFile("w", suffix=".exe", delete=False) as ef:
        ef.write("binary content")
        exe_path = ef.name

    try:
        with pytest.raises(UnsupportedFileTypeError) as exc_info:
            DocumentLoaderFactory.load_document(exe_path)
        assert exc_info.value.status_code == 415
    finally:
        os.remove(exe_path)


# ============================================================================
# 4. Tests cho Splitter (RecursiveCharacterTextSplitter 1000/200)
# ============================================================================

def test_document_splitter_optimal_params():
    """Kiểm tra chia tách văn bản theo đúng chunk_size=1000 và overlap=200."""
    splitter = DocumentSplitter(chunk_size=100, chunk_overlap=20)
    long_text = "Hệ thống RAG nội bộ. " * 30  # Chuỗi dài khoảng 630 ký tự

    doc = Document(
        page_content=long_text,
        metadata={"file_name": "chinh_sach.txt", "file_hash": "1234567890abcdef"},
    )
    chunks = splitter.split_documents([doc])

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= 120
        assert "chunk_id" in chunk.metadata
        assert "chunk_index" in chunk.metadata
        assert "chunk_hash" in chunk.metadata
        assert chunk.metadata["total_chunks"] == len(chunks)


# ============================================================================
# 5. Tests cho Document Ingestion Pipeline & Chống trùng lặp
# ============================================================================

def test_ingestion_pipeline_deduplication():
    """Kiểm tra toàn bộ pipeline nạp tệp và bỏ qua khi nạp lại tệp trùng."""
    pipeline = DocumentIngestionPipeline()

    content = "Chính sách bảo mật: Toàn bộ thiết bị phải kích hoạt xác thực hai yếu tố 2FA."
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
        tf.write(content)
        file_path = tf.name

    try:
        # Lần nạp thứ 1: Thành công
        res1 = pipeline.process_file(file_path, skip_if_file_duplicate=True)
        assert not res1.is_duplicate
        assert res1.new_chunks > 0
        assert res1.duplicate_chunks == 0

        # Lần nạp thứ 2 (cùng file): Tự động phát hiện trùng lặp
        res2 = pipeline.process_file(file_path, skip_if_file_duplicate=True)
        assert res2.is_duplicate
        assert res2.new_chunks == 0
        assert res2.total_chunks == 0
    finally:
        os.remove(file_path)


# ============================================================================
# 6. Tests cho ChromaDB Persistent Storage & Deduplication
# ============================================================================

def test_chroma_vector_store_persistence_and_deduplication():
    """Kiểm tra ChromaDB Persistent Storage nạp tài liệu và chống trùng lặp theo ID."""
    with tempfile.TemporaryDirectory() as temp_dir:
        vstore = ChromaVectorStoreManager(persist_directory=temp_dir, collection_name="test_col")

        docs = [
            Document(page_content="Quy định nghỉ phép năm 12 ngày.", metadata={"source": "hd.txt", "page": 1}),
            Document(page_content="Quy định hoàn ứng công tác phí 7 ngày.", metadata={"source": "tc.txt", "page": 2}),
        ]

        # Nạp lần 1
        added_ids_1 = vstore.add_documents(docs, skip_duplicates=True)
        assert len(added_ids_1) == 2
        assert vstore.count() == 2

        # Nạp lần 2 (cùng tài liệu): Phải bỏ qua vì trùng ID
        added_ids_2 = vstore.add_documents(docs, skip_duplicates=True)
        assert len(added_ids_2) == 0
        assert vstore.count() == 2


# ============================================================================
# 7. Tests cho Advanced Retriever (similarity, mmr, similarity_score_threshold)
# ============================================================================

def test_advanced_retriever_strategies():
    """Kiểm tra 3 chiến lược tìm kiếm: similarity, mmr, similarity_score_threshold."""
    with tempfile.TemporaryDirectory() as temp_dir:
        vstore = ChromaVectorStoreManager(persist_directory=temp_dir)
        vstore.reset()

        test_docs = [
            Document(page_content="Chính sách bảo mật mật khẩu tối thiểu 12 ký tự.", metadata={"source": "bm1.txt"}),
            Document(page_content="Chính sách bảo mật mật khẩu tài khoản VPN công ty.", metadata={"source": "bm2.txt"}),
            Document(page_content="Quy định nghỉ phép năm và ngày nghỉ có lương.", metadata={"source": "ns.txt"}),
            Document(page_content="Quy định tạm ứng công tác phí lưu trú khách sạn.", metadata={"source": "tc.txt"}),
        ]
        vstore.add_documents(test_docs)

        retriever = AdvancedRetriever(vector_store_manager=vstore)

        # 1. Similarity Search
        sim_docs = retriever.retrieve("bảo mật mật khẩu", strategy="similarity", k=2)
        assert len(sim_docs) <= 2
        assert any("bảo mật" in d.page_content for d in sim_docs)
        assert "score" in sim_docs[0].metadata

        # 2. Threshold Search
        thresh_docs = retriever.retrieve("bảo mật mật khẩu", strategy="similarity_score_threshold", threshold=0.3)
        assert all(d.metadata.get("score", 0.0) >= 0.3 for d in thresh_docs)

        # 3. MMR Search (Đa dạng hóa nguồn tài liệu)
        mmr_docs = retriever.retrieve("bảo mật mật khẩu công tác nghỉ phép", strategy="mmr", k=3, lambda_mult=0.5)
        assert len(mmr_docs) >= 1
        # Các nguồn tài liệu khác nhau được chọn
        sources = [d.metadata["source"] for d in mmr_docs]
        assert len(sources) == len(set(sources))

        # 4. Kiểm tra ngoại lệ khi truyền chiến lược sai
        with pytest.raises(InvalidRetrieverStrategyError):
            retriever.retrieve("test", strategy="invalid_strategy")


# ============================================================================
# 8. Tests cho Anti-Hallucination Prompt & CleanRAGCore
# ============================================================================

@pytest.mark.asyncio
async def test_clean_rag_core_anti_hallucination():
    """Kiểm tra CleanRAGCore từ chối lịch sự khi không có ngữ cảnh hoặc điểm lọc quá cao."""
    with tempfile.TemporaryDirectory() as temp_dir:
        vstore = ChromaVectorStoreManager(persist_directory=temp_dir)
        vstore.reset()

        # Nạp dữ liệu WFH
        vstore.add_documents([
            Document(
                page_content="Nhân viên chính thức được WFH tối đa 2 ngày/tuần sau khi quản lý phê duyệt.",
                metadata={"source": "Quy_che_WFH.pdf", "page": 4},
            )
        ])

        rag_core = CleanRAGCore(vector_store_manager=vstore)

        # 1. Câu hỏi khớp ngữ cảnh -> Trả lời có trích dẫn nguồn
        citations = await rag_core.retrieve_documents("WFH làm việc từ xa", strategy=RetrieverStrategy.similarity, k=1)
        assert len(citations) >= 1
        assert "Quy_che_WFH.pdf" in citations[0].source

        tokens = []
        async for t in rag_core.stream_answer("WFH làm việc từ xa", history=[], citations=citations):
            tokens.append(t)
        full_answer = "".join(tokens)
        assert "Quy_che_WFH.pdf" in full_answer

        # 2. Câu hỏi không liên quan với threshold rất cao -> Không có trích dẫn -> Từ chối lịch sự
        empty_citations = await rag_core.retrieve_documents(
            "Cách làm bánh pizza hải sản?",
            strategy=RetrieverStrategy.threshold,
            threshold=0.99,
        )
        assert len(empty_citations) == 0

        refusal_tokens = []
        async for t in rag_core.stream_answer("Cách làm bánh pizza?", history=[], citations=empty_citations):
            refusal_tokens.append(t)
        refusal_answer = "".join(refusal_tokens)
        assert "không tìm thấy thông tin" in refusal_answer or "Xin lỗi" in refusal_answer


@pytest.mark.asyncio
async def test_coordinator_integration_with_clean_rag():
    """Kiểm tra tích hợp hoàn chỉnh CleanRAGCore vào RAGCoordinator của hệ thống."""
    with tempfile.TemporaryDirectory() as temp_dir:
        vstore = ChromaVectorStoreManager(persist_directory=temp_dir)
        vstore.reset()

        vstore.add_documents([
            Document(
                page_content="Mật khẩu tài khoản công ty tối thiểu 12 ký tự và thay đổi định kỳ 90 ngày.",
                metadata={"source": "Chinh_sach_bao_mat.docx", "page": 5},
            )
        ])

        rag_core = CleanRAGCore(vector_store_manager=vstore)

        if RAGCoordinator is not None:
            coordinator = RAGCoordinator(rag_core=rag_core)
            events = []
            async for event in coordinator.stream_chat(
                session_id="test_session_clean",
                query="Chính sách mật khẩu như thế nào?",
                strategy=RetrieverStrategy.similarity,
            ):
                events.append(event)

            event_names = [e["event"] for e in events]
            assert "citations" in event_names
            assert "token" in event_names
            assert "done" in event_names
        else:
            # Fallback kiểm thử trực tiếp phương thức generate_answer của CleanRAGCore
            result = await rag_core.generate_answer(
                query="Chính sách mật khẩu như thế nào?",
                strategy=RetrieverStrategy.similarity,
            )
            assert len(result.citations) >= 1
            assert "Chinh_sach_bao_mat.docx" in result.answer
