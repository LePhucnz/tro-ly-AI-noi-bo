"""
core/config.py
Cấu hình tập trung cho toàn bộ hệ thống RAG & Ingestion sử dụng Pydantic Settings.
Hỗ trợ đọc biến môi trường từ tệp .env và cung cấp giá trị mặc định tối ưu theo Clean Architecture.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    _SETTINGS_BASE = BaseSettings
except ImportError:
    try:
        from pydantic import BaseSettings  # Pydantic v1 fallback
        SettingsConfigDict = None  # type: ignore
        _SETTINGS_BASE = BaseSettings
    except ImportError:
        # Minimal standalone fallback if pydantic is not installed
        class _StandaloneSettings:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        _SETTINGS_BASE = _StandaloneSettings  # type: ignore
        SettingsConfigDict = None  # type: ignore

try:
    from pydantic import Field
except ImportError:
    def Field(default=None, **kwargs):  # type: ignore
        return default


class Settings(_SETTINGS_BASE):
    """
    Cấu hình toàn hệ thống Hỏi đáp nội bộ (RAG Backend).
    Tự động nạp giá trị từ các biến môi trường hoặc tệp .env.
    """

    # --- Thông tin dịch vụ & Môi trường ---
    APP_NAME: str = Field(
        default="Trợ lý AI Nội bộ - RAG & Document Ingestion Core",
        description="Tên ứng dụng hệ thống",
    )
    APP_ENV: str = Field(
        default="development",
        description="Môi trường chạy: development, staging, production",
    )
    DEBUG: bool = Field(default=True, description="Chế độ debug")
    LOG_LEVEL: str = Field(default="INFO", description="Mức độ logging")

    # --- Cấu hình ChromaDB (Persistent Storage) ---
    CHROMA_PERSIST_DIR: str = Field(
        default="./data/chroma_db",
        description="Đường dẫn thư mục lưu trữ ChromaDB ở chế độ Persistent Storage",
    )
    CHROMA_COLLECTION_NAME: str = Field(
        default="internal_knowledge_base",
        description="Tên collection mặc định trong ChromaDB",
    )

    # --- Cấu hình Mô hình Embedding ---
    EMBEDDING_PROVIDER: str = Field(
        default="huggingface",
        description="Nhà cung cấp embedding: 'huggingface', 'openai', hoặc 'local'",
    )
    EMBEDDING_MODEL_NAME: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Tên mô hình embedding tạo vector biểu diễn tài liệu",
    )
    EMBEDDING_DEVICE: str = Field(
        default="cpu",
        description="Thiết bị thực thi embedding (cpu hoặc cuda)",
    )

    # --- Cấu hình Chia tách văn bản (Text Splitter) ---
    CHUNK_SIZE: int = Field(
        default=1000,
        description="Kích thước ký tự tối đa của một đoạn văn bản (chunk)",
    )
    CHUNK_OVERLAP: int = Field(
        default=200,
        description="Số ký tự gối đầu giữa các đoạn liền kề để bảo toàn ngữ cảnh",
    )
    CHUNK_SEPARATORS: List[str] = Field(
        default=["\n\n", "\n", " ", ""],
        description="Danh sách ký tự phân tách ưu tiên khi cắt đoạn",
    )

    # --- Cấu hình Advanced Retriever ---
    DEFAULT_RETRIEVER_STRATEGY: str = Field(
        default="similarity",
        description="Chiến lược tìm kiếm mặc định: 'similarity', 'mmr', 'similarity_score_threshold'",
    )
    DEFAULT_TOP_K: int = Field(
        default=4,
        description="Số lượng tài liệu trích xuất tối đa trả về cho mỗi truy vấn",
    )
    DEFAULT_SCORE_THRESHOLD: float = Field(
        default=0.5,
        description="Ngưỡng điểm tương đồng tối thiểu áp dụng cho chiến lược similarity_score_threshold",
    )
    MMR_LAMBDA_MULT: float = Field(
        default=0.7,
        description="Hệ số lambda trong thuật toán MMR (càng gần 1 càng ưu tiên độ liên quan, gần 0 ưu tiên độ đa dạng)",
    )
    MMR_FETCH_K: int = Field(
        default=20,
        description="Số lượng tài liệu ban đầu được lấy ra trước khi lọc đa dạng hóa MMR",
    )

    # --- Cấu hình LLM & Anti-Hallucination ---
    LLM_PROVIDER: str = Field(
        default="mock",
        description="Nhà cung cấp LLM: 'openai', 'gemini', 'anthropic', hoặc 'mock'",
    )
    LLM_MODEL_NAME: str = Field(
        default="gpt-4o-mini",
        description="Tên mô hình ngôn ngữ lớn dùng để sinh câu trả lời",
    )
    LLM_TEMPERATURE: float = Field(
        default=0.0,
        description="Độ sáng tạo của LLM (0.0 giúp câu trả lời chuẩn xác, hạn chế ảo giác)",
    )
    OPENAI_API_KEY: Optional[str] = Field(
        default=None,
        description="Khóa API OpenAI (nếu sử dụng OpenAI LLM/Embeddings)",
    )
    GOOGLE_API_KEY: Optional[str] = Field(
        default=None,
        description="Khóa API Google Gemini (nếu sử dụng Gemini LLM)",
    )

    # Cấu hình Pydantic v2
    if SettingsConfigDict is not None:
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            case_sensitive=False,
            extra="ignore",
        )
    else:
        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
            case_sensitive = False
            extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """
    Hàm tạo và cache đối tượng cấu hình Singleton.
    Đảm bảo việc đọc cấu hình chỉ thực hiện một lần trong suốt vòng đời ứng dụng.
    """
    # Khởi tạo thư mục persistent nếu chưa tồn tại
    settings = Settings()
    try:
        persist_path = Path(settings.CHROMA_PERSIST_DIR)
        persist_path.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return settings
