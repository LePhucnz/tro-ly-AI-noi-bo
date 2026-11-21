"""
api/cv_routes.py
FastAPI Router cho phân hệ Sàng lọc CV (CV Screening API).
Cung cấp các API:
1. POST /api/v1/cv/screen: Tải lên một hoặc nhiều tệp CV (.pdf, .docx, .txt, .md), chấm điểm theo lô và sắp xếp ứng viên.
2. POST /api/v1/cv/screen-text: Sàng lọc trực tiếp từ chuỗi văn bản CV (JSON Request).
3. GET /api/v1/cv/sample-criteria: Lấy danh sách các bộ tiêu chí tuyển dụng mẫu có sẵn.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from api.schemas import BatchCVScreeningResponse, ScreenTextRequest
from core.cv_schemas import (
    CriterionType,
    CVScreeningResult,
    JobCriteria,
    JobCriterion,
)
from core.exceptions import (
    CVExtractionError,
    CVLoadError,
    CVScoringError,
    CVScreeningError,
    InvalidCriteriaError,
    UnsupportedFileTypeError,
)
from documents.loader_factory import DocumentLoaderFactory
from services.cv_screener import get_cv_screening_service

logger = logging.getLogger("api_cv_routes")
router = APIRouter(prefix="/api/v1/cv", tags=["CV Screening & HR"])


# ==============================================================================
# Danh sách Tiêu chí Tuyển dụng Mẫu cho Doanh nghiệp
# ==============================================================================
SAMPLE_JOB_CRITERIA: List[Dict[str, Any]] = [
    {
        "id": "crit-senior-backend-ai",
        "job_title": "Senior Backend & AI Engineer",
        "department": "Khối Công nghệ & AI",
        "min_total_experience_years": 3.0,
        "required_skills": ["Python", "FastAPI", "PostgreSQL", "Docker"],
        "preferred_skills": ["LangChain", "RAG", "Redis", "ChromaDB", "Kubernetes"],
        "passing_score": 70.0,
        "criteria_list": [
            {
                "name": "Kinh nghiệm thiết kế REST API & Microservices",
                "criterion_type": "required",
                "weight": 1.5,
                "description": "Thành thạo xây dựng API hiệu năng cao, tối ưu hóa truy vấn cơ sở dữ liệu.",
            },
            {
                "name": "Kinh nghiệm tích hợp LLM & RAG Vector Database",
                "criterion_type": "preferred",
                "weight": 1.2,
                "description": "Từng triển khai thực tế hệ thống RAG hoặc tương tác với OpenAI/Gemini API.",
            },
        ],
    },
    {
        "id": "crit-senior-frontend",
        "job_title": "Senior Frontend Developer",
        "department": "Khối Sản phẩm Số",
        "min_total_experience_years": 3.0,
        "required_skills": ["JavaScript", "TypeScript", "React", "HTML/CSS"],
        "preferred_skills": ["Next.js", "TailwindCSS", "Redux", "GraphQL"],
        "passing_score": 70.0,
        "criteria_list": [
            {
                "name": "Tối ưu hóa UI/UX & Responsive Web Design",
                "criterion_type": "required",
                "weight": 1.5,
                "description": "Xây dựng giao diện mượt mà, tối ưu tốc độ tải trang và trải nghiệm người dùng.",
            },
        ],
    },
    {
        "id": "crit-devops-cloud",
        "job_title": "DevOps / Cloud Engineer",
        "department": "Khối Hạ tầng & Bảo mật",
        "min_total_experience_years": 3.0,
        "required_skills": ["Linux", "Docker", "Kubernetes", "CI/CD"],
        "preferred_skills": ["AWS", "Terraform", "Ansible", "Prometheus", "Grafana"],
        "passing_score": 70.0,
        "criteria_list": [
            {
                "name": "Xây dựng và bảo trì Pipeline CI/CD tự động",
                "criterion_type": "required",
                "weight": 1.5,
                "description": "Tự động hóa quy trình kiểm thử, build image và deploy lên cụm máy chủ.",
            },
        ],
    },
    {
        "id": "crit-data-engineer",
        "job_title": "Data Engineer",
        "department": "Khối Dữ liệu",
        "min_total_experience_years": 2.0,
        "required_skills": ["Python", "SQL", "PostgreSQL", "ETL"],
        "preferred_skills": ["Apache Spark", "Airflow", "Kafka", "Data Warehouse"],
        "passing_score": 65.0,
        "criteria_list": [],
    },
]


@router.get(
    "/sample-criteria",
    summary="Lấy danh sách bộ tiêu chí tuyển dụng mẫu",
    description="Cung cấp các mẫu tiêu chí tuyển dụng chuẩn cho các vị trí công nghệ phổ biến.",
)
async def get_sample_criteria() -> List[Dict[str, Any]]:
    """Trả về danh sách tiêu chí mẫu giúp HR chọn nhanh trên giao diện."""
    return SAMPLE_JOB_CRITERIA


@router.post(
    "/screen-text",
    response_model=CVScreeningResult,
    summary="Sàng lọc CV từ nội dung văn bản (JSON)",
    description="Nhận nội dung văn bản CV trực tiếp cùng bộ tiêu chí tuyển dụng và trả về kết quả đánh giá chi tiết.",
)
async def screen_cv_text_endpoint(payload: ScreenTextRequest) -> CVScreeningResult:
    """Sàng lọc CV qua chuỗi văn bản JSON."""
    service = get_cv_screening_service()
    try:
        result = await service.ascreen_cv(
            cv_input=payload.cv_text,
            criteria=payload.criteria,
        )
        if payload.file_name:
            result.metadata["file_name"] = payload.file_name
        return result
    except (CVLoadError, InvalidCriteriaError, CVExtractionError, CVScoringError) as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except Exception as exc:
        logger.exception("Lỗi khi xử lý screen-text: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sự cố hệ thống trong quá trình sàng lọc: {str(exc)}",
        )


@router.post(
    "/screen",
    response_model=BatchCVScreeningResponse,
    summary="Sàng lọc một hoặc nhiều hồ sơ CV (Tải lên tệp)",
    description=(
        "Nhận một hoặc nhiều tệp CV (.pdf, .docx, .txt, .md) kèm cấu hình tiêu chí tuyển dụng (dạng JSON Form). "
        "Tự động chấm điểm, chống ảo giác và trả về bảng xếp hạng ứng viên có điểm số cao nhất."
    ),
)
async def screen_cv_files_endpoint(
    files: List[UploadFile] = File(..., description="Danh sách tệp CV (.pdf, .docx, .txt, .md)"),
    criteria: str = Form(..., description="Chuỗi JSON định nghĩa JobCriteria"),
) -> BatchCVScreeningResponse:
    """
    Endpoint chính xử lý sàng lọc CV theo lô (Batch Processing):
    - Nhận nhiều file cùng lúc cho cùng một vị trí tuyển dụng.
    - Chấm điểm và phân loại ứng viên độc lập.
    - Sắp xếp thứ hạng (ranking) ứng viên theo điểm số từ cao xuống thấp.
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Vui lòng tải lên ít nhất một tệp CV.",
        )

    # 1. Parse JSON tiêu chí
    try:
        criteria_dict = json.loads(criteria)
        job_criteria = JobCriteria.model_validate(criteria_dict)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Cấu hình tiêu chí tuyển dụng (criteria) không hợp lệ: {str(exc)}",
        )

    # 2. Xử lý từng tệp CV
    service = get_cv_screening_service()
    supported_exts = set(DocumentLoaderFactory.get_supported_extensions())
    results: List[CVScreeningResult] = []

    for upload_file in files:
        suffix = Path(upload_file.filename or "cv.txt").suffix.lower()
        if suffix not in supported_exts:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Tệp '{upload_file.filename}' có định dạng '{suffix}' không được hỗ trợ. "
                       f"Hệ thống chỉ chấp nhận: {', '.join(supported_exts)}.",
            )

        # Lưu tạm ra đĩa để DocumentLoaderFactory bóc tách
        with tempfile.NamedTemporaryFile("wb", suffix=suffix, delete=False) as tmp_file:
            content = await upload_file.read()
            if not content or len(content.strip()) == 0:
                Path(tmp_file.name).unlink(missing_ok=True)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Tệp CV '{upload_file.filename}' rỗng (0 bytes).",
                )
            tmp_file.write(content)
            tmp_path = Path(tmp_file.name)

        try:
            res = await service.ascreen_cv(cv_input=tmp_path, criteria=job_criteria)
            res.metadata["file_name"] = upload_file.filename
            results.append(res)
        finally:
            tmp_path.unlink(missing_ok=True)

    # 3. Sắp xếp thứ hạng ứng viên theo điểm overall_score giảm dần
    results.sort(key=lambda r: r.overall_score, reverse=True)

    # 4. Tính toán thống kê tổng quan
    total_cvs = len(results)
    pass_count = sum(1 for r in results if r.is_passed)
    fail_count = total_cvs - pass_count
    avg_score = round(sum(r.overall_score for r in results) / total_cvs, 1) if total_cvs > 0 else 0.0
    top_candidate = results[0].candidate_info.full_name if results else None

    return BatchCVScreeningResponse(
        job_title=job_criteria.job_title,
        total_cvs=total_cvs,
        pass_count=pass_count,
        fail_count=fail_count,
        average_score=avg_score,
        top_candidate=top_candidate,
        results=results,
    )
