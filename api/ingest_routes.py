"""
api/ingest_routes.py
FastAPI Router cho Document Ingestion & Vector Store Management.
Cung cấp các API:
1. POST /api/v1/documents/upload: Tải lên tài liệu (.pdf, .docx, .txt, .csv, .md) và nạp vào ChromaDB.
2. GET /api/v1/documents/stats: Xem trạng thái và số lượng vector trong ChromaDB.
3. POST /api/v1/documents/reset: Xóa và tái tạo lại Collection trong ChromaDB.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from core.config import get_settings
from core.exceptions import (
    DocumentIngestionError,
    DocumentLoadError,
    DuplicateDocumentError,
    UnsupportedFileTypeError,
)
from documents.hasher import DocumentHasher
from documents.ingestion import DocumentIngestionPipeline, IngestionResult
from documents.loader_factory import DocumentLoaderFactory
from rag.vector_store import ChromaVectorStoreManager, get_vector_store_manager

logger = logging.getLogger("api_ingest_routes")
router = APIRouter(prefix="/api/v1/documents", tags=["Document Ingestion & Vector Store"])

# Singleton instances
_ingestion_pipeline: Optional[DocumentIngestionPipeline] = None


def get_ingestion_pipeline() -> DocumentIngestionPipeline:
    global _ingestion_pipeline
    if _ingestion_pipeline is None:
        _ingestion_pipeline = DocumentIngestionPipeline()
    return _ingestion_pipeline


# --- Pydantic Schemas ---

class IngestResponse(BaseModel):
    """Phản hồi sau khi nạp tài liệu vào hệ thống."""
    success: bool
    file_name: str
    file_hash: str
    is_duplicate: bool
    total_raw_docs: int
    total_chunks: int
    new_chunks_added: int
    duplicate_chunks_ignored: int
    processing_time_ms: float
    message: str


class VectorStoreStatsResponse(BaseModel):
    """Thống kê trạng thái Vector Store."""
    collection_name: str
    persist_directory: str
    total_chunks_count: int
    embedding_provider: str
    supported_file_types: List[str]


# --- Endpoints ---

@router.post(
    "/upload",
    response_model=IngestResponse,
    summary="Upload và Ingest Tài liệu",
    description="Nhận file (.pdf, .docx, .txt, .csv, .md), chia tách thành chunks và lưu trữ vào ChromaDB persistent storage.",
)
async def upload_document(
    file: UploadFile = File(..., description="Tệp tài liệu cần nạp vào cơ sở tri thức"),
) -> IngestResponse:
    """
    Endpoint tải lên tài liệu mới vào hệ thống RAG:
    - Kiểm tra định dạng hợp lệ (.pdf, .docx, .txt, .csv, .md)
    - Tính mã băm SHA-256 chống nạp trùng
    - Chia tách văn bản RecursiveCharacterTextSplitter (1000/200)
    - Nạp vào ChromaDB Persistent Storage
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tên tệp không hợp lệ.",
        )

    file_ext = Path(file.filename).suffix.lower()
    if not DocumentLoaderFactory.is_supported(file.filename):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Định dạng tệp '{file_ext}' không được hỗ trợ. "
                f"Các định dạng hợp lệ: {', '.join(DocumentLoaderFactory.get_supported_extensions())}"
            ),
        )

    # Lưu tệp tạm thời để xử lý
    temp_dir = tempfile.mkdtemp(prefix="rag_upload_")
    temp_file_path = Path(temp_dir) / file.filename

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        pipeline = get_ingestion_pipeline()
        vstore = get_vector_store_manager()

        # Thực thi pipeline nạp tài liệu
        result: IngestionResult = pipeline.process_file(
            file_path=temp_file_path,
            skip_if_file_duplicate=False,
        )

        # Thêm các chunks mới vào ChromaDB
        added_ids: List[str] = []
        if result.chunks:
            added_ids = vstore.add_documents(result.chunks, skip_duplicates=True)

        msg = (
            f"Nạp thành công tài liệu '{file.filename}'. "
            f"Đã thêm {len(added_ids)} chunks mới vào ChromaDB."
        )
        if result.is_duplicate:
            msg = f"Tài liệu '{file.filename}' đã tồn tại, không có chunk nào cần nạp thêm."

        return IngestResponse(
            success=True,
            file_name=file.filename,
            file_hash=result.file_hash,
            is_duplicate=result.is_duplicate,
            total_raw_docs=result.total_raw_docs,
            total_chunks=result.total_chunks,
            new_chunks_added=len(added_ids),
            duplicate_chunks_ignored=result.duplicate_chunks,
            processing_time_ms=result.processing_time_ms,
            message=msg,
        )

    except Exception as exc:
        logger.exception("Lỗi khi nạp tệp %s: %s", file.filename, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quá trình nạp tài liệu thất bại: {str(exc)}",
        )
    finally:
        # Dọn dẹp tệp tạm
        shutil.rmtree(temp_dir, ignore_errors=True)


@router.get(
    "/stats",
    response_model=VectorStoreStatsResponse,
    summary="Thống kê Vector Database",
    description="Xem tổng số lượng vector chunks, tên collection và đường dẫn persistent storage.",
)
async def get_vector_store_stats() -> VectorStoreStatsResponse:
    """Trả về thông tin chi tiết về ChromaDB Vector Store."""
    settings = get_settings()
    vstore = get_vector_store_manager()
    total_count = vstore.count()

    return VectorStoreStatsResponse(
        collection_name=vstore.collection_name,
        persist_directory=vstore.persist_directory,
        total_chunks_count=total_count,
        embedding_provider=settings.EMBEDDING_PROVIDER,
        supported_file_types=DocumentLoaderFactory.get_supported_extensions(),
    )


@router.post(
    "/reset",
    summary="Reset Vector Database",
    description="Xóa toàn bộ các chunks trong collection hiện tại của ChromaDB.",
)
async def reset_vector_store() -> Dict[str, Any]:
    """Xóa trắng collection hiện tại phục vụ thử nghiệm lại."""
    vstore = get_vector_store_manager()
    vstore.reset()
    pipeline = get_ingestion_pipeline()
    pipeline.registry.clear()

    return {
        "success": True,
        "message": f"Đã reset thành công collection '{vstore.collection_name}'.",
        "remaining_chunks": vstore.count(),
    }
