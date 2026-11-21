"""
core/cv_schemas.py
Pydantic Schemas định nghĩa cấu trúc dữ liệu cho phân hệ Sàng lọc CV (CV Screening Core).
Bao gồm: Bộ tiêu chí tuyển dụng, Hồ sơ ứng viên có cấu trúc, Đánh giá từng tiêu chí,
và Báo cáo tổng hợp dành cho phòng Nhân sự (HR).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class CriterionType(str, Enum):
    """Phân loại tiêu chí tuyển dụng."""
    REQUIRED = "required"          # Bắt buộc phải có (Must-have)
    PREFERRED = "preferred"        # Ưu tiên / Điểm cộng (Nice-to-have)
    EXPERIENCE = "experience"      # Số năm hoặc lĩnh vực kinh nghiệm
    EDUCATION = "education"        # Trình độ học vấn / Bằng cấp
    SOFT_SKILL = "soft_skill"      # Kỹ năng mềm / Quản lý
    OTHER = "other"                # Tiêu chí khác


class MatchRating(str, Enum):
    """Mức độ phù hợp tổng thể của ứng viên với vị trí tuyển dụng."""
    STRONG_FIT = "STRONG_FIT"        # Rất phù hợp (Điểm >= 85)
    POTENTIAL_FIT = "POTENTIAL_FIT"  # Phù hợp / Đạt yêu cầu (70 <= Điểm < 85)
    BORDERLINE = "BORDERLINE"        # Cân nhắc / Cần phỏng vấn kỹ (50 <= Điểm < 70)
    NOT_FIT = "NOT_FIT"              # Không phù hợp (Điểm < 50)


class JobCriterion(BaseModel):
    """Một tiêu chí đánh giá cụ thể trong bộ tiêu chí tuyển dụng."""
    name: str = Field(..., min_length=1, description="Tên hoặc nội dung tiêu chí (VD: 'Thành thạo Python & FastAPI')")
    criterion_type: CriterionType = Field(default=CriterionType.REQUIRED, description="Loại tiêu chí")
    weight: float = Field(default=1.0, ge=0.1, le=10.0, description="Trọng số đánh giá (từ 0.1 đến 10.0)")
    min_years: Optional[float] = Field(default=None, ge=0, description="Số năm kinh nghiệm tối thiểu nếu có")
    description: Optional[str] = Field(default=None, description="Mô tả chi tiết yêu cầu để chấm điểm chính xác")


class JobCriteria(BaseModel):
    """
    Bộ tiêu chí tuyển dụng đầy đủ cho một vị trí công việc.
    Hỗ trợ cả danh sách tiêu chí chi tiết và shortcut tiện dụng (required_skills, preferred_skills).
    """
    job_title: str = Field(..., min_length=1, description="Tên vị trí tuyển dụng (VD: 'Senior Backend Engineer')")
    department: Optional[str] = Field(default=None, description="Phòng ban tuyển dụng")
    min_total_experience_years: Optional[float] = Field(default=None, ge=0, description="Tổng số năm kinh nghiệm tối thiểu")
    required_skills: List[str] = Field(default_factory=list, description="Danh sách kỹ năng bắt buộc")
    preferred_skills: List[str] = Field(default_factory=list, description="Danh sách kỹ năng ưu tiên / điểm cộng")
    criteria_list: List[JobCriterion] = Field(default_factory=list, description="Danh sách tiêu chí đánh giá mở rộng")
    passing_score: float = Field(default=70.0, ge=0.0, le=100.0, description="Ngưỡng điểm đỗ (thang điểm 100)")

    @field_validator("job_title")
    @classmethod
    def validate_job_title(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("Tên vị trí tuyển dụng (job_title) không được để trống.")
        return clean

    def build_effective_criteria(self) -> List[JobCriterion]:
        """
        Gộp tất cả các tiêu chí từ required_skills, preferred_skills, kinh nghiệm và criteria_list
        thành một danh sách tiêu chí chuẩn hóa duy nhất phục vụ việc chấm điểm.
        """
        effective: List[JobCriterion] = []
        seen_names = set()

        # 1. Tiêu chí kinh nghiệm tổng thể
        if self.min_total_experience_years is not None and self.min_total_experience_years > 0:
            c_name = f"Tổng số năm kinh nghiệm tối thiểu: {self.min_total_experience_years} năm"
            effective.append(
                JobCriterion(
                    name=c_name,
                    criterion_type=CriterionType.EXPERIENCE,
                    weight=1.5,
                    min_years=self.min_total_experience_years,
                    description=f"Ứng viên cần tích lũy tối thiểu {self.min_total_experience_years} năm kinh nghiệm làm việc thực tế.",
                )
            )
            seen_names.add(c_name.lower())

        # 2. Kỹ năng bắt buộc (Required skills)
        for skill in self.required_skills:
            clean_skill = skill.strip()
            if clean_skill and clean_skill.lower() not in seen_names:
                effective.append(
                    JobCriterion(
                        name=clean_skill,
                        criterion_type=CriterionType.REQUIRED,
                        weight=1.5,
                        description=f"Kỹ năng bắt buộc: {clean_skill}",
                    )
                )
                seen_names.add(clean_skill.lower())

        # 3. Kỹ năng ưu tiên (Preferred skills)
        for skill in self.preferred_skills:
            clean_skill = skill.strip()
            if clean_skill and clean_skill.lower() not in seen_names:
                effective.append(
                    JobCriterion(
                        name=clean_skill,
                        criterion_type=CriterionType.PREFERRED,
                        weight=1.0,
                        description=f"Kỹ năng ưu tiên / điểm cộng: {clean_skill}",
                    )
                )
                seen_names.add(clean_skill.lower())

        # 4. Tiêu chí bổ sung trong criteria_list
        for crit in self.criteria_list:
            if crit.name.lower() not in seen_names:
                effective.append(crit)
                seen_names.add(crit.name.lower())

        return effective


class CandidateInfo(BaseModel):
    """Thông tin liên hệ cơ bản của ứng viên."""
    full_name: Optional[str] = Field(default=None, description="Họ và tên ứng viên")
    email: Optional[str] = Field(default=None, description="Địa chỉ email liên hệ")
    phone: Optional[str] = Field(default=None, description="Số điện thoại liên hệ")
    location: Optional[str] = Field(default=None, description="Địa điểm / Nơi cư trú")
    links: List[str] = Field(default_factory=list, description="Liên kết LinkedIn, GitHub, Portfolio...")


class CandidateWorkExperience(BaseModel):
    """Chi tiết kinh nghiệm làm việc của ứng viên."""
    company: Optional[str] = Field(default=None, description="Tên công ty / tổ chức")
    role: Optional[str] = Field(default=None, description="Chức danh / vị trí công việc")
    duration_years: Optional[float] = Field(default=None, description="Thời gian làm việc tính bằng năm")
    description: Optional[str] = Field(default=None, description="Mô tả công việc và thành tựu")
    technologies: List[str] = Field(default_factory=list, description="Công nghệ và công cụ sử dụng")


class CandidateEducation(BaseModel):
    """Trình độ học vấn của ứng viên."""
    institution: Optional[str] = Field(default=None, description="Trường đại học / cao đẳng / viện đào tạo")
    degree: Optional[str] = Field(default=None, description="Bằng cấp (Cử nhân, Kỹ sư, Thạc sĩ...)")
    major: Optional[str] = Field(default=None, description="Chuyên ngành đào tạo")
    graduation_year: Optional[int] = Field(default=None, description="Năm tốt nghiệp")


class StructuredCandidateProfile(BaseModel):
    """Hồ sơ ứng viên được bóc tách và chuẩn hóa từ văn bản CV."""
    candidate_info: CandidateInfo = Field(default_factory=CandidateInfo)
    summary: Optional[str] = Field(default=None, description="Tóm tắt hồ sơ / Mục tiêu nghề nghiệp")
    skills: List[str] = Field(default_factory=list, description="Danh sách kỹ năng chuyên môn trích xuất từ CV")
    work_experience: List[CandidateWorkExperience] = Field(default_factory=list, description="Lịch sử kinh nghiệm làm việc")
    education: List[CandidateEducation] = Field(default_factory=list, description="Trình độ học vấn")
    certifications: List[str] = Field(default_factory=list, description="Chứng chỉ chuyên môn")
    total_experience_years_estimated: Optional[float] = Field(default=None, description="Ước tính tổng số năm kinh nghiệm")
    raw_text: str = Field(default="", description="Nội dung văn bản CV đã được chuẩn hóa")


class CriterionEvaluation(BaseModel):
    """
    Kết quả đánh giá đối chiếu chi tiết cho từng tiêu chí.
    Tuân thủ nguyên tắc chống ảo giác: bằng chứng phải trích trực tiếp từ CV.
    """
    criterion_name: str = Field(..., description="Tên tiêu chí đánh giá")
    criterion_type: CriterionType = Field(default=CriterionType.REQUIRED, description="Loại tiêu chí")
    weight: float = Field(default=1.0, description="Trọng số áp dụng")
    score: float = Field(..., ge=0.0, le=100.0, description="Điểm đánh giá tiêu chí trên thang điểm 100")
    matched: bool = Field(..., description="Ứng viên có đạt tiêu chí này không")
    evidence: str = Field(
        ...,
        description="Bằng chứng trích xuất trực tiếp từ CV. Nếu không có thông tin, ghi rõ 'Không tìm thấy thông tin trong CV'.",
    )
    reasoning: str = Field(..., description="Lý giải ngắn gọn căn cứ theo bằng chứng trích xuất")


class CVScreeningResult(BaseModel):
    """
    Báo cáo kết quả sàng lọc CV hoàn chỉnh gửi cho phòng Nhân sự (HR).
    Bao gồm đánh giá chi tiết, phân loại phù hợp, điểm mạnh, điểm thiếu và câu hỏi phỏng vấn.
    """
    candidate_info: CandidateInfo = Field(..., description="Thông tin cơ bản ứng viên")
    job_title: str = Field(..., description="Vị trí tuyển dụng đang ứng tuyển")
    overall_score: float = Field(..., ge=0.0, le=100.0, description="Điểm tổng thể theo trọng số (thang 100)")
    match_rating: MatchRating = Field(..., description="Xếp loại mức độ phù hợp")
    is_passed: bool = Field(..., description="Kết luận đạt hay không đạt ngưỡng xét tuyển")
    criteria_evaluations: List[CriterionEvaluation] = Field(default_factory=list, description="Bảng điểm chi tiết từng tiêu chí")
    key_strengths: List[str] = Field(default_factory=list, description="Danh sách các điểm mạnh nổi bật của ứng viên kèm bằng chứng")
    missing_or_unclear_points: List[str] = Field(default_factory=list, description="Các điểm còn thiếu hoặc chưa rõ ràng trong CV")
    executive_summary: str = Field(..., description="Bản tóm tắt ngắn gọn và khuyến nghị cho HR")
    interview_questions: List[str] = Field(default_factory=list, description="Danh sách câu hỏi phỏng vấn gợi ý")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata phụ (thời gian xử lý, model, file hash...)")
