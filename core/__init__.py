"""
core module
Cấu hình hệ thống (Settings) và Ngoại lệ phân cấp (Domain Exceptions).
"""

from core.config import Settings, get_settings
from core.exceptions import (
    BaseRAGException,
    CollectionNotFoundError,
    DocumentIngestionError,
    DocumentLoadError,
    DocumentNotFoundError,
    DocumentSplitError,
    DuplicateDocumentError,
    InvalidRetrieverStrategyError,
    LLMGenerationError,
    RetrieverError,
    UnsupportedFileTypeError,
    VectorStoreConnectionError,
    VectorStoreError,
    setup_exception_handlers,
)

__all__ = [
    "Settings",
    "get_settings",
    "BaseRAGException",
    "DocumentIngestionError",
    "UnsupportedFileTypeError",
    "DocumentLoadError",
    "DuplicateDocumentError",
    "DocumentSplitError",
    "VectorStoreError",
    "VectorStoreConnectionError",
    "CollectionNotFoundError",
    "RetrieverError",
    "InvalidRetrieverStrategyError",
    "DocumentNotFoundError",
    "LLMGenerationError",
    "setup_exception_handlers",
]
