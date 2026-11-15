"""
documents/splitter.py
Bộ chia tách văn bản tích hợp RecursiveCharacterTextSplitter theo thông số tối ưu:
chunk_size=1000, chunk_overlap=200.
Tự động làm giàu Metadata cho từng chunk: chunk_index, total_chunks, chunk_id, chunk_hash.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from core.config import get_settings
from core.exceptions import DocumentSplitError
from documents.hasher import DocumentHasher
from documents.loader_factory import Document

logger = logging.getLogger("document_splitter")

# Cố gắng import RecursiveCharacterTextSplitter từ LangChain
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
    except ImportError:
        # Fallback triển khai đệ quy chuẩn xác theo đúng thuật toán LangChain
        class RecursiveCharacterTextSplitter:  # type: ignore
            def __init__(
                self,
                chunk_size: int = 1000,
                chunk_overlap: int = 200,
                separators: Optional[List[str]] = None,
            ):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap
                self.separators = separators or ["\n\n", "\n", " ", ""]

            def split_text(self, text: str) -> List[str]:
                if not text:
                    return []
                if len(text) <= self.chunk_size:
                    return [text]

                separator = self.separators[-1]
                for s in self.separators:
                    if s == "":
                        separator = ""
                        break
                    if s in text:
                        separator = s
                        break

                splits = text.split(separator) if separator else list(text)
                good_splits: List[str] = []
                current_chunk: List[str] = []
                current_len = 0

                for piece in splits:
                    piece_len = len(piece) + (len(separator) if current_chunk else 0)
                    if current_len + piece_len > self.chunk_size and current_chunk:
                        chunk_str = separator.join(current_chunk)
                        good_splits.append(chunk_str)

                        # Giữ lại phần overlap
                        overlap_chunks: List[str] = []
                        overlap_len = 0
                        for p in reversed(current_chunk):
                            if overlap_len + len(p) <= self.chunk_overlap:
                                overlap_chunks.insert(0, p)
                                overlap_len += len(p) + len(separator)
                            else:
                                break
                        current_chunk = overlap_chunks
                        current_len = sum(len(p) for p in current_chunk) + (
                            len(separator) * (len(current_chunk) - 1) if current_chunk else 0
                        )

                    current_chunk.append(piece)
                    current_len += piece_len

                if current_chunk:
                    good_splits.append(separator.join(current_chunk))

                return [s for s in good_splits if s.strip()]


class DocumentSplitter:
    """
    Bộ chia tách văn bản chuẩn hóa.
    Được cấu hình mặc định: chunk_size=1000, chunk_overlap=200.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None,
    ) -> None:
        settings = get_settings()
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.separators = separators or settings.CHUNK_SEPARATORS

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap phải nhỏ hơn chunk_size.")

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chia tách danh sách Document thành các chunks nhỏ hơn.
        Gán mã ID duy nhất (Deterministic ID) và mã SHA-256 cho từng chunk.
        """
        if not documents:
            return []

        try:
            chunks: List[Document] = []
            for doc in documents:
                raw_text = doc.page_content.strip()
                if not raw_text:
                    continue

                split_texts = self._splitter.split_text(raw_text)
                total_chunks = len(split_texts)

                doc_hash = doc.metadata.get(
                    "file_hash",
                    DocumentHasher.compute_sha256(raw_text),
                )

                for chunk_idx, text_chunk in enumerate(split_texts):
                    # Sinh ID tiền định và hash cho từng chunk
                    chunk_hash = DocumentHasher.compute_sha256(text_chunk)
                    chunk_id = DocumentHasher.generate_chunk_id(
                        doc_hash=doc_hash,
                        chunk_index=chunk_idx,
                        chunk_content=text_chunk,
                    )

                    # Kế thừa toàn bộ metadata của tài liệu gốc và bổ sung thông tin chunk
                    chunk_metadata = dict(doc.metadata)
                    chunk_metadata.update(
                        {
                            "chunk_id": chunk_id,
                            "chunk_index": chunk_idx,
                            "total_chunks": total_chunks,
                            "chunk_char_count": len(text_chunk),
                            "chunk_hash": chunk_hash,
                            "parent_doc_hash": doc_hash,
                        }
                    )

                    chunks.append(
                        Document(
                            page_content=text_chunk,
                            metadata=chunk_metadata,
                        )
                    )

            logger.info(
                "Chia tách thành công %d tài liệu ban đầu thành %d chunks "
                "(chunk_size=%d, overlap=%d).",
                len(documents),
                len(chunks),
                self.chunk_size,
                self.chunk_overlap,
            )
            return chunks

        except Exception as e:
            logger.exception("Lỗi khi chia tách tài liệu: %s", e)
            raise DocumentSplitError(str(e))
