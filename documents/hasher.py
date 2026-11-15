"""
documents/hasher.py
Cơ chế băm mã SHA-256 chống trùng lặp tài liệu và các đoạn văn bản (chunks).
Đảm bảo tính toàn vẹn dữ liệu và ngăn chặn việc nạp lặp dữ liệu vào ChromaDB.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Set, Union


class DocumentHasher:
    """
    Tiện ích tính toán mã băm SHA-256 cho tệp và nội dung văn bản.
    Hỗ trợ sinh ID định danh duy nhất (Deterministic ID) cho từng chunk.
    """

    CHUNK_READ_SIZE = 65536  # Đọc từng khối 64KB để tối ưu bộ nhớ khi xử lý file lớn

    @classmethod
    def compute_sha256(cls, content: Union[str, bytes]) -> str:
        """
        Tính mã băm SHA-256 từ chuỗi ký tự hoặc mảng bytes.
        """
        hasher = hashlib.sha256()
        if isinstance(content, str):
            hasher.update(content.encode("utf-8"))
        elif isinstance(content, (bytes, bytearray)):
            hasher.update(content)
        else:
            hasher.update(str(content).encode("utf-8"))
        return hasher.hexdigest()

    @classmethod
    def compute_file_sha256(cls, file_path: Union[str, Path]) -> str:
        """
        Tính mã băm SHA-256 của toàn bộ tệp tin vật lý trên đĩa.
        Đọc theo dạng luồng nhị phân (binary stream) chống tràn RAM.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Tệp không tồn tại: {file_path}")

        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(cls.CHUNK_READ_SIZE):
                hasher.update(chunk)
        return hasher.hexdigest()

    @classmethod
    def generate_chunk_id(
        cls,
        doc_hash: str,
        chunk_index: int,
        chunk_content: Optional[str] = None,
    ) -> str:
        """
        Sinh ID định danh duy nhất và có tính tiền định (deterministic) cho từng chunk.
        Công thức: sha256(doc_hash + chunk_index + chunk_content_hash).
        Giúp ChromaDB nhận diện ngay chunk đã tồn tại khi nạp lại tài liệu.
        """
        content_hash = cls.compute_sha256(chunk_content) if chunk_content else ""
        raw_seed = f"{doc_hash}::{chunk_index}::{content_hash}"
        return f"chk_{cls.compute_sha256(raw_seed)[:32]}"


class DocumentHashRegistry:
    """
    Bộ quản lý và ghi nhớ danh sách mã băm SHA-256 trong bộ nhớ.
    Được sử dụng để kiểm tra nhanh chống trùng lặp trước khi gọi Vector Store.
    """

    def __init__(self) -> None:
        self._processed_file_hashes: Set[str] = set()
        self._processed_chunk_ids: Set[str] = set()

    def is_file_processed(self, file_hash: str) -> bool:
        """Kiểm tra tệp tin đã từng được nạp chưa."""
        return file_hash in self._processed_file_hashes

    def register_file(self, file_hash: str) -> None:
        """Đăng ký mã băm tệp vào bộ nhớ."""
        self._processed_file_hashes.add(file_hash)

    def is_chunk_processed(self, chunk_id: str) -> bool:
        """Kiểm tra chunk ID đã tồn tại chưa."""
        return chunk_id in self._processed_chunk_ids

    def register_chunk(self, chunk_id: str) -> None:
        """Đăng ký chunk ID vào bộ nhớ."""
        self._processed_chunk_ids.add(chunk_id)

    def clear(self) -> None:
        """Xóa sạch bộ ghi nhớ."""
        self._processed_file_hashes.clear()
        self._processed_chunk_ids.clear()
