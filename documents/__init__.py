"""
documents module
Module quản lý Document Ingestion Pipeline & CV Preprocessing:
- Loader Factory (.pdf, .docx, .txt, .csv, .md)
- Recursive Text Splitter (chunk_size=1000, chunk_overlap=200)
- SHA-256 Hasher & Deduplication Engine
- CV Preprocessor & Normalizer
"""

from documents.cv_preprocessor import CVPreprocessor
from documents.hasher import DocumentHasher, DocumentHashRegistry
from documents.ingestion import DocumentIngestionPipeline, IngestionResult
from documents.loader_factory import Document, DocumentLoaderFactory
from documents.splitter import DocumentSplitter
from documents.text_cleaner import TextCleaner

__all__ = [
    "Document",
    "DocumentHasher",
    "DocumentHashRegistry",
    "DocumentLoaderFactory",
    "DocumentSplitter",
    "DocumentIngestionPipeline",
    "IngestionResult",
    "TextCleaner",
    "CVPreprocessor",
]
