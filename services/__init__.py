"""
services module: Quản lý bộ nhớ, Output Parser, RAG Core Interface, Coordinator
và Phân hệ Sàng lọc CV (CV Screening Core).
"""

from services.cv_screener import (
    CVExtractor,
    CVScorer,
    CVScreeningService,
    CVSummarizer,
    get_cv_screening_service,
)
from services.memory import ChatMessage, MemoryManager, get_memory_manager
from services.output_parser import RAGOutputParser, RAGParsedOutput, SourceCitation
from services.rag_coordinator import RAGCoordinator
from services.rag_core_interface import (
    BaseRAGCore,
    DefaultInternalRAGCore,
    RetrieverStrategy,
    get_rag_core,
    register_rag_core,
)

__all__ = [
    "ChatMessage",
    "MemoryManager",
    "get_memory_manager",
    "SourceCitation",
    "RAGParsedOutput",
    "RAGOutputParser",
    "BaseRAGCore",
    "DefaultInternalRAGCore",
    "RetrieverStrategy",
    "get_rag_core",
    "register_rag_core",
    "RAGCoordinator",
    "CVExtractor",
    "CVScorer",
    "CVSummarizer",
    "CVScreeningService",
    "get_cv_screening_service",
]
