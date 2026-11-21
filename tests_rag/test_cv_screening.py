"""
tests_rag/test_cv_screening.py
Kiểm thử toàn diện phân hệ Sàng lọc CV (CV Screening Core):
- Tiền xử lý và nạp CV (CVPreprocessor)
- Bóc tách hồ sơ ứng viên (CVExtractor)
- Đối chiếu tiêu chí và chống ảo giác (CVScorer & Anti-Hallucination Guard)
- Tổng hợp báo cáo, điểm mạnh, điểm thiếu & câu hỏi phỏng vấn (CVSummarizer)
- Dịch vụ sàng lọc đồng bộ & bất đồng bộ (CVScreeningService)
- Xử lý các ngoại lệ Domain (CVLoadError, InvalidCriteriaError)
"""

import tempfile
from pathlib import Path
import pytest

from core.cv_schemas import (
    CriterionType,
    CVScreeningResult,
    JobCriteria,
    JobCriterion,
    MatchRating,
    StructuredCandidateProfile,
)
from core.exceptions import CVLoadError, InvalidCriteriaError
from documents.cv_preprocessor import CVPreprocessor
from services.cv_screener import (
    CVExtractor,
    CVScorer,
    CVScreeningService,
    CVSummarizer,
    get_cv_screening_service,
)

SAMPLE_CV_SENIOR_PYTHON = """
NGUYỄN VĂN AN
Email: an.nguyen@example.com | Điện thoại: 0912345678
Địa chỉ: Cầu Giấy, Hà Nội
GitHub: https://github.com/annguyen-dev | LinkedIn: https://linkedin.com/in/annguyen-dev

TÓM TẮT NGHỀ NGHIỆP
Kỹ sư phần mềm cao cấp với hơn 5 năm kinh nghiệm chuyên sâu về kiến trúc Backend và Trí tuệ Nhân tạo.
Có thế mạnh về thiết kế hệ thống phân tán, xử lý dữ liệu lớn và tích hợp mô hình LLM vào sản phẩm thực tế.

KỸ NĂNG CHUYÊN MÔN
- Ngôn ngữ lập trình: Python, TypeScript, SQL
- Frameworks & Thư viện: FastAPI, Django, LangChain, PyTorch
- Cơ sở dữ liệu: PostgreSQL, MongoDB, Redis, ChromaDB, Qdrant
- Hạ tầng & Công cụ: Docker, Git, Linux, REST API, Microservices

KINH NGHIỆM LÀM VIỆC
Công ty Công nghệ AI Tech (2021 - 2024)
Vị trí: Senior Backend & AI Engineer
- Thiết kế và phát triển hệ thống RAG nội bộ tra cứu tri thức doanh nghiệp sử dụng FastAPI, LangChain và ChromaDB.
- Tối ưu hóa hiệu năng API giảm độ trễ truy vấn từ 800ms xuống 250ms với Redis Caching.
- Triển khai dịch vụ microservices đóng gói bằng Docker trên môi trường Linux.

Công ty Cổ phần Giải pháp Phần mềm ABC (2019 - 2021)
Vị trí: Backend Developer
- Phát triển các dịch vụ web RESTful API với Python và Django phục vụ hơn 500,000 người dùng hoạt động hàng tháng.
- Thiết kế cơ sở dữ liệu quan hệ PostgreSQL và viết Unit Tests đảm bảo độ bao phủ mã trên 85%.

HỌC VẤN
Trường Đại học Bách Khoa Hà Nội (2014 - 2019)
Bằng cấp: Kỹ sư Công nghệ Thông tin
"""

SAMPLE_CV_JUNIOR_FRONTEND = """
TRẦN THỊ BÍCH
Email: bich.tran@gmail.com | Phone: 0987654321
Location: TP. Hồ Chí Minh

MỤC TIÊU
Lập trình viên Frontend muốn phát triển sự nghiệp trong mảng Web Development.

KỸ NĂNG
JavaScript, React, HTML, CSS, TailwindCSS, Git

KINH NGHIỆM
Công ty Startup XYZ (2023 - 2024)
Junior Frontend Developer
- Xây dựng giao diện Landing Page và Dashboard quản trị bằng React và TailwindCSS.

HỌC VẤN
Đại học Khoa học Tự nhiên TP.HCM (2019 - 2023)
Cử nhân Khoa học Máy tính
"""


# ============================================================================
# 1. Tests cho CVPreprocessor
# ============================================================================

def test_cv_preprocessor_raw_text():
    text, meta = CVPreprocessor.load_and_preprocess(SAMPLE_CV_SENIOR_PYTHON)
    assert "NGUYỄN VĂN AN" in text
    assert meta["source"] == "raw_text_input"
    assert meta["file_type"] == "text"
    assert meta["file_size_bytes"] > 0
    assert len(meta["file_hash"]) == 64


def test_cv_preprocessor_from_file():
    with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False) as f:
        f.write(SAMPLE_CV_SENIOR_PYTHON)
        tmp_path = Path(f.name)

    try:
        text, meta = CVPreprocessor.load_and_preprocess(tmp_path)
        assert "an.nguyen@example.com" in text
        assert meta["file_name"] == tmp_path.name
        assert meta["file_type"] == "txt"
    finally:
        tmp_path.unlink()


def test_cv_preprocessor_empty_or_invalid_raises():
    with pytest.raises(CVLoadError):
        CVPreprocessor.load_and_preprocess("   \n\n  ")

    with pytest.raises(CVLoadError):
        CVPreprocessor.load_and_preprocess(Path("/duong/dan/khong/ton/tai.pdf"))


# ============================================================================
# 2. Tests cho CVExtractor
# ============================================================================

def test_cv_extractor_candidate_info():
    profile = CVExtractor.extract_profile(SAMPLE_CV_SENIOR_PYTHON)

    assert profile.candidate_info.full_name == "NGUYỄN VĂN AN"
    assert profile.candidate_info.email == "an.nguyen@example.com"
    assert profile.candidate_info.phone == "0912345678"
    assert "Hà Nội" in (profile.candidate_info.location or "")
    assert len(profile.candidate_info.links) >= 1
    assert any("github.com/annguyen-dev" in l for l in profile.candidate_info.links)


def test_cv_extractor_skills_and_experience():
    profile = CVExtractor.extract_profile(SAMPLE_CV_SENIOR_PYTHON)

    skills_lower = [s.lower() for s in profile.skills]
    assert "python" in skills_lower
    assert "fastapi" in skills_lower
    assert "docker" in skills_lower
    assert "postgresql" in skills_lower

    # Kiểm tra ước tính kinh nghiệm (2021-2024 = 3 năm + 2019-2021 = 2 năm = 5 năm)
    assert profile.total_experience_years_estimated is not None
    assert profile.total_experience_years_estimated >= 4.0

    # Học vấn
    assert len(profile.education) > 0
    assert any("Bách Khoa" in edu.institution for edu in profile.education)


# ============================================================================
# 3. Tests cho CVScorer & Anti-Hallucination Guard
# ============================================================================

def test_cv_scorer_strong_fit():
    profile = CVExtractor.extract_profile(SAMPLE_CV_SENIOR_PYTHON)

    criteria = JobCriteria(
        job_title="Senior Python & Backend Engineer",
        min_total_experience_years=3.0,
        required_skills=["Python", "FastAPI", "PostgreSQL"],
        preferred_skills=["Docker", "LangChain"],
        passing_score=70.0,
    )

    overall_score, match_rating, is_passed, evaluations = CVScorer.evaluate(profile, criteria)

    assert overall_score >= 85.0
    assert match_rating == MatchRating.STRONG_FIT
    assert is_passed is True
    assert len(evaluations) >= 5

    # Đảm bảo các tiêu chí bắt buộc đều có bằng chứng trích xuất từ CV
    for ev in evaluations:
        if ev.criterion_type == CriterionType.REQUIRED:
            assert ev.matched is True
            assert ev.score >= 80.0
            assert ev.evidence != "Không tìm thấy thông tin trong CV"


def test_cv_scorer_anti_hallucination_guard():
    """
    KIỂM SOÁT CHỐNG ẢO GIÁC:
    Ứng viên không hề có kỹ năng 'Kubernetes' và 'Golang'.
    Hệ thống KHÔNG ĐƯỢC PHÉP tự suy đoán là ứng viên biết,
    phải chấm điểm 0 và ghi rõ bằng chứng: 'Không tìm thấy thông tin trong CV'.
    """
    profile = CVExtractor.extract_profile(SAMPLE_CV_SENIOR_PYTHON)

    criteria = JobCriteria(
        job_title="DevOps / Cloud Engineer",
        required_skills=["Kubernetes", "Golang"],
        criteria_list=[
            JobCriterion(
                name="Chứng chỉ AWS Solutions Architect Professional",
                criterion_type=CriterionType.REQUIRED,
                weight=2.0,
            )
        ],
    )

    overall_score, match_rating, is_passed, evaluations = CVScorer.evaluate(profile, criteria)

    k8s_eval = next(e for e in evaluations if "Kubernetes" in e.criterion_name)
    assert k8s_eval.matched is False
    assert k8s_eval.score == 0.0
    assert k8s_eval.evidence == "Không tìm thấy thông tin trong CV"

    go_eval = next(e for e in evaluations if "Golang" in e.criterion_name)
    assert go_eval.matched is False
    assert go_eval.score == 0.0
    assert go_eval.evidence == "Không tìm thấy thông tin trong CV"

    assert is_passed is False
    assert match_rating == MatchRating.NOT_FIT


def test_cv_scorer_mismatched_role():
    """Ứng viên Frontend nộp vào vị trí Backend/AI Engineer."""
    profile = CVExtractor.extract_profile(SAMPLE_CV_JUNIOR_FRONTEND)

    criteria = JobCriteria(
        job_title="Senior AI / RAG Engineer",
        min_total_experience_years=4.0,
        required_skills=["Python", "FastAPI", "Vector DB"],
        passing_score=70.0,
    )

    overall_score, match_rating, is_passed, evaluations = CVScorer.evaluate(profile, criteria)

    assert overall_score < 50.0
    assert match_rating == MatchRating.NOT_FIT
    assert is_passed is False


# ============================================================================
# 4. Tests cho CVSummarizer & Output Report
# ============================================================================

def test_cv_summarizer_outputs():
    profile = CVExtractor.extract_profile(SAMPLE_CV_SENIOR_PYTHON)
    criteria = JobCriteria(
        job_title="Senior Backend Engineer",
        required_skills=["Python", "FastAPI"],
        preferred_skills=["Golang"],  # Ứng viên thiếu Golang
    )

    overall_score, match_rating, is_passed, evaluations = CVScorer.evaluate(profile, criteria)
    exec_summary, strengths, missing, questions = CVSummarizer.generate_summary(
        profile=profile,
        criteria=criteria,
        overall_score=overall_score,
        match_rating=match_rating,
        is_passed=is_passed,
        evaluations=evaluations,
    )

    # 1. Điểm mạnh phải có trích dẫn
    assert len(strengths) > 0
    assert any("Khớp tiêu chí" in s for s in strengths)

    # 2. Điểm thiếu phải chỉ rõ Golang
    assert len(missing) > 0
    assert any("Golang" in m for m in missing)

    # 3. Câu hỏi phỏng vấn
    assert len(questions) >= 3
    # Phải có câu hỏi liên quan đến điểm còn thiếu hoặc đào sâu kỹ năng
    assert any("Golang" in q or "FastAPI" in q for q in questions)

    # 4. Executive summary cho HR
    assert "NGUYỄN VĂN AN" in exec_summary
    assert "Khuyến nghị tuyển dụng" in exec_summary


# ============================================================================
# 5. Tests cho CVScreeningService (Facade)
# ============================================================================

def test_cv_screening_service_sync():
    service = get_cv_screening_service()
    criteria = JobCriteria(
        job_title="Python Developer",
        required_skills=["Python", "FastAPI"],
    )

    result = service.screen_cv(SAMPLE_CV_SENIOR_PYTHON, criteria)

    assert isinstance(result, CVScreeningResult)
    assert result.candidate_info.full_name == "NGUYỄN VĂN AN"
    assert result.job_title == "Python Developer"
    assert result.overall_score >= 80.0
    assert result.is_passed is True
    assert len(result.criteria_evaluations) >= 2
    assert "processing_time_sec" in result.metadata

    # Kiểm tra serialization sang dict
    dict_data = result.model_dump()
    assert dict_data["job_title"] == "Python Developer"
    assert "executive_summary" in dict_data


@pytest.mark.asyncio
async def test_cv_screening_service_async():
    service = get_cv_screening_service()
    criteria = JobCriteria(
        job_title="Backend Developer",
        required_skills=["Python"],
    )

    result = await service.ascreen_cv(SAMPLE_CV_SENIOR_PYTHON, criteria)
    assert result.overall_score > 0
    assert result.match_rating in [MatchRating.STRONG_FIT, MatchRating.POTENTIAL_FIT]


def test_invalid_criteria_validation():
    service = get_cv_screening_service()
    # Bộ tiêu chí không có bất kỳ kỹ năng hay yêu cầu nào
    empty_criteria = JobCriteria(
        job_title="Empty Role",
        required_skills=[],
        preferred_skills=[],
        criteria_list=[],
    )

    with pytest.raises(InvalidCriteriaError):
        service.screen_cv(SAMPLE_CV_SENIOR_PYTHON, empty_criteria)
