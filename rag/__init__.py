"""
rag module
Module RAG Core & Vector Store:
- ChromaDB Client & Persistent Storage Manager
- Advanced Retriever (similarity, mmr, similarity_score_threshold)
- Anti-Hallucination Prompt Template
- CleanRAGCore Engine (BaseRAGCore implementation)
"""

from rag.engine import CleanRAGCore
from rag.prompts import ANTI_HALLUCINATION_SYSTEM_PROMPT, AntiHallucinationPrompt
from rag.retriever import AdvancedRetriever
from rag.vector_store import (
    ChromaVectorStoreManager,
    DeterministicHashEmbeddings,
    get_embedding_function,
)

__all__ = [
    "CleanRAGCore",
    "ChromaVectorStoreManager",
    "AdvancedRetriever",
    "AntiHallucinationPrompt",
    "ANTI_HALLUCINATION_SYSTEM_PROMPT",
    "DeterministicHashEmbeddings",
    "get_embedding_function",
]
