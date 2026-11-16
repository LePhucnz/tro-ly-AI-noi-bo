"""
documents/ingestion.py
Quy trình nạp tài liệu toàn diện (Ingestion Pipeline) tích hợp kiểm tra chống trùng lặp.
Kết nối Loader Factory, SHA-256 Hasher và Recursive Text Splitter theo chuẩn Clean Architecture.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from core.exceptions import (
    DocumentIngestionError,
    DocumentLoadError,
    DuplicateDocumentError,
    UnsupportedFileTypeError,
)
from documents.hasher import DocumentHasher, DocumentHashRegistry
from documents.loader_factory import Document, DocumentLoaderFactory
from documents.splitter import DocumentSplitter

logger = logging.getLogger("document_ingestion_pipeline")


@dataclass
class IngestionResult:
    """Kết quả chi tiết của quá trình nạp tài liệu."""
    file_path: str
    file_name: str
    file_hash: str
    is_duplicate: bool
    total_raw_docs: int
    total_chunks: int
    new_chunks: int
    duplicate_chunks: int
    chunks: List[Document] = field(default_factory=list)
    chunk_ids: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi kết quả sang dạng Dictionary phục vụ API Response."""
        return {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "file_hash": self.file_hash,
            "is_duplicate": self.is_duplicate,
            "total_raw_docs": self.total_raw_docs,
            "total_chunks": self.total_chunks,
            "new_chunks": self.new_chunks,
            "duplicate_chunks": self.duplicate_chunks,
            "chunk_ids": self.chunk_ids,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


class DocumentIngestionPipeline:
    """
    Pipeline xử lý nạp tài liệu:
    1. Kiểm tra định dạng & tính mã băm SHA-256 của tệp.
    2. Kiểm tra chống trùng lặp tệp.
    3. Đọc dữ liệu qua DocumentLoaderFactory.
    4. Chia tách văn bản bằng DocumentSplitter (chunk_size=1000, overlap=200).
    5. Kiểm tra chống trùng lặp ở cấp độ từng chunk (chunk ID).
    """

    def __init__(
        self,
        splitter: Optional[DocumentSplitter] = None,
        hash_registry: Optional[DocumentHashRegistry] = None,
    ) -> None:
        self.splitter = splitter or DocumentSplitter()
        self.registry = hash_registry or DocumentHashRegistry()

    def process_file(
        self,
        file_path: Union[str, Path],
        existing_chunk_ids: Optional[Set[str]] = None,
        skip_if_file_duplicate: bool = True,
    ) -> IngestionResult:
        """
        Nạp và xử lý một tệp tài liệu cụ thể.

        :param file_path: Đường dẫn tệp cần nạp.
        :param existing_chunk_ids: Tập hợp ID các chunk đã có sẵn trong Vector Store để đối chiếu.
        :param skip_if_file_duplicate: Nếu True, bỏ qua nếu file hash đã tồn tại. Nếu False, vẫn xử lý.
        :return: IngestionResult chi tiết.
        """
        start_time = time.time()
        path = Path(file_path)

        if not path.exists():
            raise DocumentLoadError(str(path), "Tệp không tồn tại.")

        if not DocumentLoaderFactory.is_supported(path):
            raise UnsupportedFileTypeError(path.suffix.lower())

        file_hash = DocumentHasher.compute_file_sha256(path)
        existing_ids = existing_chunk_ids or set()

        # Kiểm tra trùng lặp tệp tin
        if skip_if_file_duplicate and self.registry.is_file_processed(file_hash):
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info("Phát hiện tệp trùng lặp (SHA-256: %s), bỏ qua.", file_hash[:12])
            return IngestionResult(
                file_path=str(path),
                file_name=path.name,
                file_hash=file_hash,
                is_duplicate=True,
                total_raw_docs=0,
                total_chunks=0,
                new_chunks=0,
                duplicate_chunks=0,
                processing_time_ms=elapsed_ms,
            )

        # 1. Đọc tài liệu qua Loader Factory
        raw_docs = DocumentLoaderFactory.load_document(path)

        # 2. Chia tách văn bản thành các chunks
        chunks = self.splitter.split_documents(raw_docs)

        # 3. Chống trùng lặp ở cấp độ chunk
        new_chunks: List[Document] = []
        new_chunk_ids: List[str] = []
        duplicate_chunk_count = 0

        for chunk in chunks:
            cid = chunk.metadata.get("chunk_id", "")
            if cid in existing_ids or self.registry.is_chunk_processed(cid):
                duplicate_chunk_count += 1
            else:
                new_chunks.append(chunk)
                new_chunk_ids.append(cid)
                self.registry.register_chunk(cid)

        # Ghi nhận tệp đã xử lý
        self.registry.register_file(file_hash)

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            "Nạp thành công '%s': tổng %d chunks, %d chunks mới, %d chunks trùng lặp (%.2f ms).",
            path.name,
            len(chunks),
            len(new_chunks),
            duplicate_chunk_count,
            elapsed_ms,
        )

        return IngestionResult(
            file_path=str(path),
            file_name=path.name,
            file_hash=file_hash,
            is_duplicate=False,
            total_raw_docs=len(raw_docs),
            total_chunks=len(chunks),
            new_chunks=len(new_chunks),
            duplicate_chunks=duplicate_chunk_count,
            chunks=new_chunks,
            chunk_ids=new_chunk_ids,
            processing_time_ms=elapsed_ms,
        )

    def process_directory(
        self,
        directory_path: Union[str, Path],
        existing_chunk_ids: Optional[Set[str]] = None,
        recursive: bool = True,
    ) -> List[IngestionResult]:
        """
        Quét và nạp toàn bộ các tệp được hỗ trợ trong thư mục.
        """
        dir_path = Path(directory_path)
        if not dir_path.is_dir():
            raise DocumentLoadError(str(dir_path), "Thư mục không tồn tại.")

        pattern = "**/*" if recursive else "*"
        supported_exts = set(DocumentLoaderFactory.get_supported_extensions())
        results: List[IngestionResult] = []

        for p in dir_path.glob(pattern):
            if p.is_file() and p.suffix.lower() in supported_exts:
                try:
                    res = self.process_file(p, existing_chunk_ids=existing_chunk_ids)
                    results.append(res)
                except Exception as e:
                    logger.error("Lỗi khi xử lý tệp %s: %s", p.name, e)

        return results
