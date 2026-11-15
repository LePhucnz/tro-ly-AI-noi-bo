"""
server.py
Khởi tạo và cấu hình ứng dụng FastAPI cho hệ thống Trợ lý AI Nội bộ (RAG Backend).
Bao gồm Middleware CORS, Router API v1, Exception Handlers và tài liệu OpenAPI.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import router as api_router

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rag_server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Quản lý vòng đời khởi động và kết thúc của ứng dụng."""
    logger.info("Khởi động FastAPI RAG Backend Service...")
    yield
    logger.info("Dừng FastAPI RAG Backend Service. Giải phóng tài nguyên.")


app = FastAPI(
    title="Trợ lý AI Nội bộ - RAG API Service",
    description=(
        "Hệ thống Backend cung cấp API Hỏi - Đáp nội bộ tích hợp RAG, "
        "quản lý phiên làm việc đa người dùng và truyền phát câu trả lời thời gian thực qua Server-Sent Events (SSE)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Cấu hình CORS để giao tiếp thông suốt với Streamlit hoặc Web Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đăng ký Router
app.include_router(api_router)


@app.get("/", tags=["General"])
async def root_endpoint():
    """Endpoint gốc điều hướng và kiểm tra service."""
    return {
        "service": "Trợ lý AI Nội bộ - RAG API Backend",
        "status": "online",
        "documentation": "/docs",
        "api_prefix": "/api/v1",
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Bắt các ngoại lệ không lường trước để trả về phản hồi an toàn và chi tiết."""
    logger.exception("Ngoại lệ không bắt được tại %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "Hệ thống gặp sự cố trong quá trình xử lý yêu cầu.",
            "detail": str(exc),
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

