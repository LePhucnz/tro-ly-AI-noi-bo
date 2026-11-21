"""
core/exceptions.py
Hệ thống xử lý ngoại lệ phân cấp (Hierarchical Domain Exceptions) cho Core RAG & Document Ingestion.
Được thiết kế theo Clean Architecture để tách biệt lỗi Domain với tầng Transport (FastAPI API Layer).
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class BaseRAGException(Exception):
    """
    Lớp ngoại lệ cơ sở cho toàn bộ hệ thống RAG & Ingestion.
    Chứa mã lỗi chuẩn (error_code), thông báo thân thiện và HTTP status code tương ứng.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "INTERNAL_RAG_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi ngoại lệ thành Dictionary phục vụ phản hồi API."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "status_code": self.status_code,
            "details": self.details,
        }


# ============================================================================
# Nhóm 1: Ngoại lệ xử lý Tài liệu & Ingestion (Document Ingestion Domain)
# ============================================================================

class DocumentIngestionError(BaseRAGException):
    """Lỗi chung phát sinh trong quá trình nạp và xử lý tài liệu."""

    def __init__(
        self,
        message: str,
        error_code: str = "DOCUMENT_INGESTION_ERROR",
        status_code: int = 400,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, error_code=error_code, status_code=status_code, details=details)


class UnsupportedFileTypeError(DocumentIngestionError):
    """Lỗi khi định dạng tệp tải lên không nằm trong danh sách hỗ trợ (.pdf, .docx, .txt, .csv, .md)."""

    def __init__(
        self,
        file_extension: str,
        supported_types: Optional[list[str]] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        supported = supported_types or [".pdf", ".docx", ".txt", ".csv", ".md"]
        message = (
            f"Định dạng tệp '{file_extension}' không được hỗ trợ. "
            f"Hệ thống chỉ chấp nhận các định dạng: {', '.join(supported)}."
        )
        det = details or {}
        det.update({"file_extension": file_extension, "supported_types": supported})
        super().__init__(
            message=message,
            error_code="UNSUPPORTED_FILE_TYPE",
            status_code=415,  # Unsupported Media Type
            details=det,
        )


class DocumentLoadError(DocumentIngestionError):
    """Lỗi khi đọc hoặc phân tích nội dung tệp tin."""

    def __init__(
        self,
        file_path: str,
        reason: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        message = f"Không thể đọc nội dung tài liệu '{file_path}': {reason}"
        det = details or {}
        det.update({"file_path": file_path, "reason": reason})
        super().__init__(
            message=message,
            error_code="DOCUMENT_LOAD_FAILED",
            status_code=422,  # Unprocessable Entity
            details=det,
        )


class DuplicateDocumentError(DocumentIngestionError):
    """Lỗi khi phát hiện tài liệu hoặc đoạn văn bản đã tồn tại trong Vector Store (trùng mã SHA-256)."""

    def __init__(
        self,
        file_hash: str,
        file_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        name_str = f"'{file_name}' " if file_name else ""
        message = (
            f"Tài liệu {name_str}với mã băm SHA-256 '{file_hash[:12]}...' "
            "đã được nạp trước đó. Hệ thống tự động bỏ qua để tránh trùng lặp dữ liệu."
        )
        det = details or {}
        det.update({"file_hash": file_hash, "file_name": file_name})
        super().__init__(
            message=message,
            error_code="DUPLICATE_DOCUMENT_IGNORED",
            status_code=409,  # Conflict
            details=det,
        )


class DocumentSplitError(DocumentIngestionError):
    """Lỗi trong quá trình cắt chia văn bản thành các đoạn (chunks)."""

    def __init__(
        self,
        reason: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        message = f"Lỗi trong quá trình chia tách văn bản: {reason}"
        super().__init__(
            message=message,
            error_code="DOCUMENT_SPLIT_FAILED",
            status_code=422,
            details=details,
        )


# ============================================================================
# Nhóm 2: Ngoại lệ Vector Store & ChromaDB (Vector Store Domain)
# ============================================================================

class VectorStoreError(BaseRAGException):
    """Lỗi chung phát sinh trong tầng Vector Database."""

    def __init__(
        self,
        message: str,
        error_code: str = "VECTOR_STORE_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, error_code=error_code, status_code=status_code, details=details)


class VectorStoreConnectionError(VectorStoreError):
    """Lỗi không thể kết nối hoặc khởi tạo ChromaDB persistent storage."""

    def __init__(
        self,
        persist_dir: str,
        reason: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        message = f"Không thể khởi tạo hoặc kết nối ChromaDB tại '{persist_dir}': {reason}"
        det = details or {}
        det.update({"persist_dir": persist_dir, "reason": reason})
        super().__init__(
            message=message,
            error_code="VECTOR_STORE_CONNECTION_FAILED",
            status_code=503,  # Service Unavailable
            details=det,
        )


class CollectionNotFoundError(VectorStoreError):
    """Lỗi khi không tìm thấy Collection được chỉ định trong ChromaDB."""

    def __init__(
        self,
        collection_name: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        message = f"Collection '{collection_name}' không tồn tại trong ChromaDB."
        det = details or {}
        det.update({"collection_name": collection_name})
        super().__init__(
            message=message,
            error_code="COLLECTION_NOT_FOUND",
            status_code=404,
            details=det,
        )


# ============================================================================
# Nhóm 3: Ngoại lệ Truy vấn & Tìm kiếm (Retriever Domain)
# ============================================================================

class RetrieverError(BaseRAGException):
    """Lỗi trong quá trình tìm kiếm tài liệu từ Vector Store."""

    def __init__(
        self,
        message: str,
        error_code: str = "RETRIEVER_ERROR",
        status_code: int = 400,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, error_code=error_code, status_code=status_code, details=details)


class InvalidRetrieverStrategyError(RetrieverError):
    """Lỗi khi người dùng truyền chiến lược tìm kiếm không hợp lệ."""

    def __init__(
        self,
        strategy: str,
        valid_strategies: Optional[list[str]] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        valid = valid_strategies or ["similarity", "mmr", "similarity_score_threshold"]
        message = (
            f"Chiến lược tìm kiếm '{strategy}' không hợp lệ. "
            f"Các chiến lược hỗ trợ: {', '.join(valid)}."
        )
        det = details or {}
        det.update({"invalid_strategy": strategy, "valid_strategies": valid})
        super().__init__(
            message=message,
            error_code="INVALID_RETRIEVER_STRATEGY",
            status_code=400,
            details=det,
        )


class DocumentNotFoundError(RetrieverError):
    """Lỗi khi không tìm thấy bất kỳ tài liệu nào phù hợp với câu hỏi."""

    def __init__(
        self,
        query: str,
        threshold: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        thresh_info = f" vượt qua ngưỡng lọc {threshold}" if threshold is not None else ""
        message = f"Không tìm thấy tài liệu liên quan{thresh_info} cho truy vấn: '{query}'."
        det = details or {}
        det.update({"query": query, "threshold": threshold})
        super().__init__(
            message=message,
            error_code="DOCUMENT_NOT_FOUND",
            status_code=404,
            details=det,
        )


# ============================================================================
# Nhóm 4: Ngoại lệ Tầng LLM & Sinh câu trả lời (LLM Domain)
# ============================================================================

class LLMGenerationError(BaseRAGException):
    """Lỗi trong quá trình gọi LLM để sinh câu trả lời RAG."""

    def __init__(
        self,
        reason: str,
        model_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        model_info = f" (Model: {model_name})" if model_name else ""
        message = f"Lỗi sinh câu trả lời từ LLM{model_info}: {reason}"
        det = details or {}
        det.update({"reason": reason, "model_name": model_name})
        super().__init__(
            message=message,
            error_code="LLM_GENERATION_FAILED",
            status_code=502,  # Bad Gateway
            details=det,
        )


# ============================================================================
# Nhóm 5: Ngoại lệ Sàng lọc CV (CV Screening Domain)
# ============================================================================

class CVScreeningError(BaseRAGException):
    """Lỗi chung phát sinh trong quá trình xử lý và sàng lọc CV ứng viên."""

    def __init__(
        self,
        message: str,
        error_code: str = "CV_SCREENING_ERROR",
        status_code: int = 400,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, error_code=error_code, status_code=status_code, details=details)


class CVLoadError(CVScreeningError):
    """Lỗi khi tải hoặc đọc nội dung hồ sơ CV."""

    def __init__(
        self,
        file_path_or_info: str,
        reason: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        message = f"Không thể tải hoặc đọc nội dung CV '{file_path_or_info}': {reason}"
        det = details or {}
        det.update({"file_info": file_path_or_info, "reason": reason})
        super().__init__(
            message=message,
            error_code="CV_LOAD_FAILED",
            status_code=422,
            details=det,
        )


class InvalidCriteriaError(CVScreeningError):
    """Lỗi khi bộ tiêu chí tuyển dụng không hợp lệ."""

    def __init__(
        self,
        reason: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        message = f"Bộ tiêu chí tuyển dụng không hợp lệ: {reason}"
        det = details or {}
        det.update({"reason": reason})
        super().__init__(
            message=message,
            error_code="INVALID_JOB_CRITERIA",
            status_code=422,
            details=det,
        )


class CVExtractionError(CVScreeningError):
    """Lỗi khi bóc tách thông tin ứng viên từ văn bản CV."""

    def __init__(
        self,
        reason: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        message = f"Lỗi bóc tách thông tin hồ sơ CV: {reason}"
        det = details or {}
        det.update({"reason": reason})
        super().__init__(
            message=message,
            error_code="CV_EXTRACTION_FAILED",
            status_code=500,
            details=det,
        )


class CVScoringError(CVScreeningError):
    """Lỗi khi tính điểm hoặc đối chiếu tiêu chí ứng viên."""

    def __init__(
        self,
        reason: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        message = f"Lỗi đánh giá và chấm điểm CV: {reason}"
        det = details or {}
        det.update({"reason": reason})
        super().__init__(
            message=message,
            error_code="CV_SCORING_FAILED",
            status_code=500,
            details=det,
        )


# ============================================================================
# FastAPI Exception Handler Registration Helper
# ============================================================================

def setup_exception_handlers(app: Any) -> None:
    """
    Đăng ký Exception Handler cho FastAPI để tự động bắt các Domain Exceptions
    và trả về JSONResponse chuẩn hóa theo chuẩn RESTful API.
    """
    try:
        from fastapi import Request
        from fastapi.responses import JSONResponse

        @app.exception_handler(BaseRAGException)
        async def rag_domain_exception_handler(request: Request, exc: BaseRAGException):
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "success": False,
                    "error": exc.error_code,
                    "message": exc.message,
                    "path": request.url.path,
                    "details": exc.details,
                },
            )
    except ImportError:
        pass
