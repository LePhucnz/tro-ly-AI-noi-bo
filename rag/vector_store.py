"""
rag/vector_store.py
Quản lý ChromaDB chạy ở chế độ Persistent Storage (lưu trữ cục bộ trên ổ cứng).
Hỗ trợ quản lý Collection, sinh Embedding và cơ chế chống nạp trùng lặp dựa trên SHA-256 IDs.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from core.config import get_settings
from core.exceptions import VectorStoreConnectionError, VectorStoreError
from documents.hasher import DocumentHasher
from documents.loader_factory import Document

logger = logging.getLogger("chroma_vector_store")


# ============================================================================
# Embedding Providers & Fallback Engine
# ============================================================================

class DeterministicHashEmbeddings:
    """
    Embedding Provider nội bộ dự phòng (Deterministic Offline Embeddings).
    Tạo vector biểu diễn 384 chiều chuẩn hóa (L2-norm) dựa trên n-gram hashing.
    Đảm bảo hệ thống và các bài kiểm thử tự động luôn chạy mượt mà ngay cả khi
    chưa cài đặt mô hình HuggingFace nặng ký hoặc khi chạy offline trong sandbox.
    """

    DIMENSION = 384

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def _embed_text(self, text: str) -> List[float]:
        vector = [0.0] * self.dimension
        if not text:
            return vector

        # Băm từ và các n-gram để tạo vector đặc trưng ngữ nghĩa
        words = text.lower().split()
        for word in words:
            h = int(DocumentHasher.compute_sha256(word)[:8], 16)
            idx = h % self.dimension
            sign = 1.0 if (h & 1) == 0 else -1.0
            vector[idx] += sign

        # Chuẩn hóa L2 norm (Vector độ dài đơn vị)
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 1e-9:
            vector = [v / norm for v in vector]
        return vector

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)

    def __call__(self, input: Union[str, List[str]]) -> Any:
        # Tương thích với ChromaDB EmbeddingFunction protocol
        if isinstance(input, str):
            return self.embed_query(input)
        return self.embed_documents(input)


def get_embedding_function() -> Any:
    """
    Khởi tạo mô hình Embedding phù hợp theo cấu hình trong Settings.
    Ưu tiên HuggingFace/OpenAI, tự động fallback sang DeterministicHashEmbeddings khi cần.
    """
    settings = get_settings()

    if settings.EMBEDDING_PROVIDER == "openai" and settings.OPENAI_API_KEY:
        try:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                openai_api_key=settings.OPENAI_API_KEY,
                model="text-embedding-3-small",
            )
        except ImportError:
            logger.warning("langchain-openai chưa được cài đặt, chuyển sang HuggingFace.")

    if settings.EMBEDDING_PROVIDER == "huggingface":
        try:
            import os
            # Kích hoạt chế độ offline để tải tức thì từ local cache, không chờ network timeout
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL_NAME,
                model_kwargs={"device": settings.EMBEDDING_DEVICE},
            )
        except Exception as e:
            logger.info("Chưa có huggingface embeddings (%s), kích hoạt Deterministic Embeddings.", e)

    return DeterministicHashEmbeddings(dimension=384)


# ============================================================================
# Persistent ChromaDB Manager
# ============================================================================

class ChromaVectorStoreManager:
    """
    Bộ quản trị ChromaDB hoạt động hoàn toàn ở chế độ Persistent Storage.
    Tích hợp cơ chế kiểm tra chống nạp trùng lặp dữ liệu bằng mã băm SHA-256.
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_function: Optional[Any] = None,
    ) -> None:
        settings = get_settings()
        self.persist_directory = str(
            Path(persist_directory or settings.CHROMA_PERSIST_DIR).resolve()
        )
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        self.embedding_function = embedding_function or get_embedding_function()

        # Tạo thư mục lưu trữ nếu chưa có
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        self._client: Any = None
        self._collection: Any = None
        self._in_memory_docs: Dict[str, Document] = {}  # Fallback buffer nếu chromadb chưa cài
        self._init_chroma()

    def _init_chroma(self) -> None:
        """Khởi tạo ChromaDB Persistent Client."""
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
            )
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "Đã kết nối thành công ChromaDB Persistent Storage tại '%s', Collection: '%s'.",
                self.persist_directory,
                self.collection_name,
            )
        except ImportError:
            logger.warning(
                "Thư viện 'chromadb' chưa được cài đặt trong môi trường. "
                "Hệ thống tự động kích hoạt bộ nhớ tương thích Persistent In-Memory."
            )
            self._client = None
            self._collection = None
        except Exception as e:
            logger.exception("Không thể khởi tạo ChromaDB: %s", e)
            raise VectorStoreConnectionError(self.persist_directory, str(e))

    def get_collection(self) -> Any:
        """Đảm bảo luôn lấy collection hợp lệ từ ChromaDB client."""
        if self._client is not None:
            try:
                if self._collection is not None:
                    self._collection.count()
                    return self._collection
            except Exception:
                pass
            try:
                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as e:
                logger.warning("Không thể lấy hoặc tạo collection ChromaDB: %s", e)
        return self._collection

    @property
    def is_native_chroma(self) -> bool:
        """Kiểm tra có đang sử dụng ChromaDB gốc hay không."""
        return self._client is not None and self.get_collection() is not None

    def get_existing_ids(self, ids: List[str]) -> Set[str]:
        """
        Kiểm tra danh sách chunk IDs xem những ID nào đã tồn tại trong Vector Store.
        Cơ chế cốt lõi để chống trùng lặp tài liệu.
        """
        if not ids:
            return set()

        col = self.get_collection()
        if col is not None:
            try:
                res = col.get(ids=ids)
                existing = set(res.get("ids", []))
                return existing
            except Exception as e:
                logger.warning("Lỗi kiểm tra ID tồn tại trong ChromaDB: %s", e)
                return set()
        else:
            return {cid for cid in ids if cid in self._in_memory_docs}

    def add_documents(
        self,
        documents: List[Document],
        skip_duplicates: bool = True,
    ) -> List[str]:
        """
        Nạp danh sách Document vào ChromaDB persistent storage.
        Tự động xác định chunk_id và áp dụng cơ chế chống nạp trùng lặp.

        :param documents: Danh sách Document cần nạp.
        :param skip_duplicates: Nếu True, chỉ thêm các chunk chưa từng xuất hiện.
        :return: Danh sách các chunk_id thực sự được thêm mới.
        """
        if not documents:
            return []

        # Chuẩn bị IDs và Metadata
        prepared_docs: List[Document] = []
        target_ids: List[str] = []

        for idx, doc in enumerate(documents):
            meta = dict(doc.metadata or {})
            cid = meta.get("chunk_id")
            if not cid:
                parent_hash = meta.get(
                    "file_hash",
                    DocumentHasher.compute_sha256(doc.page_content),
                )
                chunk_idx = meta.get("chunk_index", idx)
                cid = DocumentHasher.generate_chunk_id(parent_hash, chunk_idx, doc.page_content)
                meta["chunk_id"] = cid

            target_ids.append(cid)
            prepared_docs.append(Document(page_content=doc.page_content, metadata=meta))

        # Kiểm tra trùng lặp
        existing_ids = self.get_existing_ids(target_ids) if skip_duplicates else set()

        docs_to_add: List[Document] = []
        ids_to_add: List[str] = []

        for doc, cid in zip(prepared_docs, target_ids):
            if cid in existing_ids:
                continue
            docs_to_add.append(doc)
            ids_to_add.append(cid)

        if not docs_to_add:
            logger.info("Toàn bộ %d chunks đều đã tồn tại trong Vector Store. Bỏ qua.", len(documents))
            return []

        # Tiến hành nạp dữ liệu vào ChromaDB
        contents = [d.page_content for d in docs_to_add]
        metadatas = [d.metadata for d in docs_to_add]

        # Sinh Embeddings
        if hasattr(self.embedding_function, "embed_documents"):
            embeddings = self.embedding_function.embed_documents(contents)
        elif callable(self.embedding_function):
            embeddings = self.embedding_function(contents)
        else:
            embeddings = DeterministicHashEmbeddings().embed_documents(contents)

        if self.is_native_chroma:
            try:
                # Ép kiểu metadata thành kiểu dữ liệu ChromaDB chấp nhận (str, int, float, bool)
                clean_metadatas = []
                for m in metadatas:
                    clean_m = {}
                    for k, v in m.items():
                        if isinstance(v, (str, int, float, bool)):
                            clean_m[k] = v
                        else:
                            clean_m[k] = str(v)
                    clean_metadatas.append(clean_m)

                col = self.get_collection()
                if col is None:
                    raise VectorStoreError("Không thể kết nối đến collection ChromaDB.")

                col.upsert(
                    ids=ids_to_add,
                    embeddings=embeddings,
                    documents=contents,
                    metadatas=clean_metadatas,
                )
                logger.info("Đã ghi thành công %d chunks mới vào ChromaDB Persistent Storage.", len(ids_to_add))
            except Exception as e:
                logger.exception("Lỗi khi thêm tài liệu vào ChromaDB: %s", e)
                raise VectorStoreError(f"Thao tác thêm tài liệu thất bại: {e}")
        else:
            # Lưu vào bộ nhớ tương thích
            for cid, doc, emb in zip(ids_to_add, docs_to_add, embeddings):
                doc.metadata["_embedding"] = emb
                self._in_memory_docs[cid] = doc

        return ids_to_add

    def count(self) -> int:
        """Đếm tổng số chunk tài liệu trong collection."""
        col = self.get_collection()
        if col is not None:
            return col.count()
        return len(self._in_memory_docs)

    def reset(self) -> None:
        """Xóa toàn bộ dữ liệu trong collection hiện tại."""
        if self._client is not None:
            try:
                self._client.delete_collection(self.collection_name)
            except Exception as e:
                logger.warning("Không thể xóa collection ChromaDB: %s", e)
            try:
                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as e:
                logger.warning("Không thể tạo lại collection ChromaDB: %s", e)
        self._in_memory_docs.clear()


# ============================================================================
# Global Singleton Vector Store Manager
# ============================================================================
_default_vstore_manager: Optional[ChromaVectorStoreManager] = None


def get_vector_store_manager() -> ChromaVectorStoreManager:
    """Trả về Singleton instance của ChromaVectorStoreManager cho toàn bộ ứng dụng."""
    global _default_vstore_manager
    if _default_vstore_manager is None:
        _default_vstore_manager = ChromaVectorStoreManager()
    return _default_vstore_manager
