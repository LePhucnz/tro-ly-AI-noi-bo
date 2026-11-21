"""
services/cv_screener.py
Phân hệ Sàng lọc CV (CV Screening Core):
1. CVExtractor: Bóc tách hồ sơ ứng viên thành StructuredCandidateProfile.
2. CVScorer: Đối chiếu bộ tiêu chí, chấm điểm có trọng số, chống ảo giác (Anti-Hallucination Guard).
3. CVSummarizer: Sinh tóm tắt cho HR, điểm mạnh, điểm thiếu và câu hỏi phỏng vấn gợi ý.
4. CVScreeningService: Facade điều phối toàn bộ chu trình sàng lọc.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from core.cv_schemas import (
    CandidateEducation,
    CandidateInfo,
    CandidateWorkExperience,
    CriterionEvaluation,
    CriterionType,
    CVScreeningResult,
    JobCriteria,
    JobCriterion,
    MatchRating,
    StructuredCandidateProfile,
)
from core.exceptions import (
    CVExtractionError,
    CVLoadError,
    CVScoringError,
    CVScreeningError,
    InvalidCriteriaError,
)
from documents.cv_preprocessor import CVPreprocessor

logger = logging.getLogger("cv_screener")


class CVExtractor:
    """
    Trích xuất và chuẩn hóa thông tin ứng viên từ nội dung văn bản CV:
    - Thông tin cá nhân: Họ tên, Email, Số điện thoại, Địa điểm, Liên kết
    - Kỹ năng chuyên môn
    - Kinh nghiệm làm việc & Dự án
    - Học vấn & Chứng chỉ
    """

    # Danh mục các kỹ năng công nghệ phổ biến để đối chiếu từ khóa
    TECH_SKILL_KEYWORDS = [
        "python", "fastapi", "django", "flask", "langchain", "rag", "vector db", "chromadb", "qdrant",
        "openai", "llm", "pytorch", "tensorflow", "scikit-learn", "nlp", "transformers", "huggingface",
        "javascript", "typescript", "react", "next.js", "vue", "angular", "node.js", "express",
        "java", "spring boot", "c++", "c#", ".net", "golang", "go", "rust", "php", "laravel",
        "sql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "kafka", "rabbitmq",
        "docker", "kubernetes", "k8s", "aws", "gcp", "azure", "ci/cd", "git", "linux", "rest api",
        "graphql", "microservices", "unit test", "system design", "data structures", "algorithms",
    ]

    @classmethod
    def extract_profile(cls, raw_text: str) -> StructuredCandidateProfile:
        """
        Bóc tách toàn diện văn bản CV thành StructuredCandidateProfile.
        """
        if not raw_text or not raw_text.strip():
            raise CVExtractionError("Văn bản CV rỗng, không thể trích xuất hồ sơ.")

        clean_text = CVPreprocessor.normalize_cv_text(raw_text)

        # 1. Trích xuất thông tin liên hệ
        candidate_info = cls._extract_candidate_info(clean_text)

        # 2. Trích xuất kỹ năng
        skills = cls._extract_skills(clean_text)

        # 3. Trích xuất kinh nghiệm làm việc
        work_experience, est_years = cls._extract_work_experience(clean_text)

        # 4. Trích xuất học vấn
        education = cls._extract_education(clean_text)

        # 5. Trích xuất chứng chỉ
        certifications = cls._extract_certifications(clean_text)

        # 6. Trích xuất tóm tắt mục tiêu nghề nghiệp
        summary = cls._extract_summary(clean_text)

        return StructuredCandidateProfile(
            candidate_info=candidate_info,
            summary=summary,
            skills=skills,
            work_experience=work_experience,
            education=education,
            certifications=certifications,
            total_experience_years_estimated=est_years,
            raw_text=clean_text,
        )

    @classmethod
    def _extract_candidate_info(cls, text: str) -> CandidateInfo:
        """Trích xuất email, phone, links, name, location."""
        # Email
        email_match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
        email = email_match.group(0) if email_match else None

        # Phone (hỗ trợ toàn diện định dạng SĐT Việt Nam: cố định Hà Nội/HCM 024/028, di động 03/05/07/08/09 & quốc tế +84)
        phone = None
        # Ưu tiên 1: Tìm theo nhãn tiền tố (Độ chính xác cao nhất)
        prefix_phone_match = re.search(
            r"(?:Điện\s*thoại|Đt|SĐT|SDT|Telephone|Phone|Tel|Mobile|Hotline)\s*[:：]?\s*([+0-9(][+0-9()\s.-]{7,22}[0-9])",
            text,
            re.IGNORECASE,
        )
        if prefix_phone_match:
            candidate_phone = prefix_phone_match.group(1).strip()
            digits = re.sub(r"\D", "", candidate_phone)
            if 8 <= len(digits) <= 15:
                phone = candidate_phone

        # Ưu tiên 2: Tìm theo mẫu số chuẩn nếu không có tiền tố
        if not phone:
            general_phone_patterns = [
                # Cố định VN có ngoặc: (024) 6680 5588 hoặc (+84 24) 6680 5588
                r"\((?:02\d{1,2}|\+84\s*2\d{1,2})\)[-.\s]?\d{3,4}[-.\s]?\d{4}\b",
                # Có mã quốc gia: +84 hoặc (+84)
                r"(?:\+84|\(\+84\))[-.\s]?(?:2\d{1,2}|3\d|5\d|7\d|8\d|9\d)[-.\s0-9]{6,12}\b",
                # Di động / Cố định VN bắt đầu bằng 0: 02x, 03x, 05x, 07x, 08x, 09x
                r"\b0(?:2\d{1,2}|3\d|5\d|7\d|8\d|9\d)[-.\s0-9]{7,11}\b",
                # Chuẩn quốc tế: (xxx) xxx-xxxx
                r"\(?\d{3,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}\b",
            ]
            for pat in general_phone_patterns:
                m = re.search(pat, text)
                if m:
                    cand = m.group(0).strip()
                    digits = re.sub(r"\D", "", cand)
                    if 8 <= len(digits) <= 15:
                        phone = cand
                        break

        # Links (LinkedIn, GitHub, Portfolio)
        link_matches = re.findall(
            r"(?:https?://)?(?:www\.)?(?:github\.com|linkedin\.com/in|gitlab\.com)/[a-zA-Z0-9_.-]+",
            text,
            re.IGNORECASE,
        )
        # Chuẩn hóa: đảm bảo mọi link đều có https:// prefix
        links = []
        for lnk in dict.fromkeys(link_matches):
            if not lnk.startswith(("http://", "https://")):
                lnk = "https://" + lnk
            links.append(lnk)

        # Họ tên ứng viên
        full_name = cls._extract_candidate_name(text, email)

        # Địa điểm
        location = None
        loc_patterns = [
            r"(?:Địa chỉ|Nơi ở|Khu vực|Location|Address)\s*[:\-]\s*([^\n,]+(?:,\s*[^\n,]+)?)",
            r"\b(Hà Nội|TP\.?\s*Hồ Chí Minh|Hồ Chí Minh|Đà Nẵng|Cần Thơ|Hải Phòng|Bình Dương)\b",
        ]
        for pat in loc_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                location = m.group(1).strip()
                break

        return CandidateInfo(
            full_name=full_name,
            email=email,
            phone=phone,
            location=location,
            links=links,
        )

    @classmethod
    def _extract_candidate_name(cls, text: str, email: Optional[str]) -> Optional[str]:
        """Bóc tách tên ứng viên dựa trên vị trí đầu tài liệu, tiền tố hoặc so khớp email."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return None

        # Danh sách đen các đề mục thường gặp để tránh nhận diện nhầm tiêu đề thành tên
        HEADER_BLACKLIST = {
            "WORK EXPERIENCE", "EXPERIENCE", "EMPLOYMENT HISTORY", "CAREER HISTORY",
            "EDUCATION", "ACADEMIC BACKGROUND", "HỌC VẤN", "TRÌNH ĐỘ HỌC VẤN",
            "SKILLS", "TECHNICAL SKILLS", "ADDITIONAL SKILLS", "EXPERTISE", "KỸ NĂNG",
            "PROJECTS", "KEY PROJECTS", "DỰ ÁN", "DỰ ÁN TIÊU BIỂU",
            "SUMMARY", "PROFESSIONAL SUMMARY", "OBJECTIVE", "CAREER OBJECTIVE",
            "TÓM TẮT", "MỤC TIÊU NGHỀ NGHIỆP", "PROFILE", "THÔNG TIN", "CONTACT", "LIÊN HỆ",
            "CURRICULUM VITAE", "RESUME", "CV", "CERTIFICATIONS", "AWARDS",
        }

        # Cách 1: Tiền tố rõ ràng: "Họ và tên: Nguyễn Văn A", "Full Name: John Doe"
        prefix_pattern = r"(?:Họ\s*(?:và|&)?\s*tên|Full\s*name|Name)\s*[:\-]\s*([A-ZÀ-Ỹa-zà-ỹ\s]+)"
        for line in lines[:10]:
            m = re.search(prefix_pattern, line, re.IGNORECASE)
            if m:
                candidate_name = m.group(1).strip()
                if 2 <= len(candidate_name.split()) <= 6:
                    return candidate_name

        # Cách 2: Tìm kiếm tên dựa vào email trong các dòng đầu tài liệu
        if email:
            user_part = email.split("@")[0]
            tokens = [p.lower() for p in re.split(r"[._-]", user_part) if len(p) >= 2 and p.isalpha()]
            if len(tokens) >= 2:
                for line in lines[:6]:
                    clean_l = re.sub(r"https?://\S+|www\.\S+|\S+@\S+|\+?[0-9()\s.-]{7,22}|\S+\.(?:com|vn|org|io|dev|net)", " ", line)
                    clean_l = re.sub(r"[^a-zA-ZÀ-Ỹà-ỹ\s]", " ", clean_l).strip()
                    words = clean_l.split()
                    matched_words = [w for w in words if w.lower() in tokens]
                    if len(matched_words) >= 2:
                        return " ".join(w.capitalize() for w in matched_words)

        # Cách 3: Dòng đầu tiên hoặc thứ hai viết hoa toàn bộ hoặc dạng Tiêu đề (không chứa trong blacklist)
        for line in lines[:4]:
            if line.startswith(("#", "-", "*", "http", "www", "@")) or "@" in line:
                continue
            # Làm sạch URL, email, SĐT trước khi kiểm tra
            clean_l = re.sub(r"https?://\S+|www\.\S+|\S+@\S+|\+?[0-9()\s.-]{7,22}|\S+\.(?:com|vn|org|io|dev|net)", " ", line)
            clean_l = re.sub(r"[^a-zA-ZÀ-Ỹà-ỹ\s]", " ", clean_l).strip()
            if clean_l.upper() in HEADER_BLACKLIST:
                continue
            words = clean_l.split()
            if 2 <= len(words) <= 5 and all(w[0].isupper() for w in words if w):
                return clean_l

        # Cách 4: Fallback suy luận từ địa chỉ email
        if email:
            user_part = email.split("@")[0]
            name_parts = re.split(r"[._-]", user_part)
            if len(name_parts) >= 2 and all(p.isalpha() for p in name_parts):
                return " ".join(p.capitalize() for p in name_parts)

        return None

    @classmethod
    def _extract_skills(cls, text: str) -> List[str]:
        """Bóc tách danh sách kỹ năng chuyên môn từ CV."""
        skills_found: List[str] = []
        text_lower = text.lower()

        # 1. Quét theo danh mục từ khóa kỹ năng chuẩn
        for kw in cls.TECH_SKILL_KEYWORDS:
            # Dùng regex word boundary để tránh khớp từ con (ví dụ "c" trong "react")
            pattern = r"(?<![a-zA-Z0-9#+])" + re.escape(kw) + r"(?![a-zA-Z0-9#+])"
            if re.search(pattern, text_lower):
                skills_found.append(kw.upper() if len(kw) <= 4 and kw not in ["git", "java", "rust"] else kw.title())

        # 2. Bóc tách từ phần "KỸ NĂNG / SKILLS" nếu có định dạng danh sách
        skills_section_match = re.search(
            r"(?:KỸ NĂNG|KỸ NĂNG CHUYÊN MÔN|SKILLS|TECHNICAL SKILLS)\s*[:\-]?\n([\s\S]*?)(?=\n[A-ZÀ-Ỹ\s]{4,}|\Z)",
            text,
            re.IGNORECASE,
        )
        if skills_section_match:
            section_content = skills_section_match.group(1)
            raw_items = re.split(r"[,;•\n\-\*\|]", section_content)
            for item in raw_items:
                item_clean = item.strip()
                if 2 <= len(item_clean) <= 40 and not item_clean.startswith("http"):
                    # Chuẩn hóa nếu chưa có trong danh sách
                    if not any(item_clean.lower() == s.lower() for s in skills_found):
                        skills_found.append(item_clean)

        return list(dict.fromkeys(skills_found))

    @classmethod
    def _extract_work_experience(cls, text: str) -> Tuple[List[CandidateWorkExperience], Optional[float]]:
        """
        Bóc tách các vị trí công tác và ước tính tổng số năm kinh nghiệm:
        - Giới hạn quét trong phạm vi mục Kinh nghiệm làm việc để tránh nhầm với số năm Đại học/Học vấn.
        - Hỗ trợ toàn diện tên tháng tiếng Anh (May, November...) và tiếng Việt (Tháng 5...).
        - Khử trùng lặp (Interval Merging) giữa các dự án/công việc làm song song.
        """
        experiences: List[CandidateWorkExperience] = []

        # 1. Tách riêng phân vùng 'Kinh nghiệm làm việc' (nếu có đề mục rõ ràng)
        work_headers = [
            "Kinh nghiệm làm việc", "Kinh nghiệm", "Quá trình làm việc", "Lịch sử làm việc",
            "Work Experience", "Experience", "Employment History", "Professional Experience",
        ]
        stop_headers = [
            "Học vấn", "Trình độ học vấn", "Bằng cấp", "Education", "Academic Background",
            "Kỹ năng", "Kỹ năng chuyên môn", "Skills", "Technical Skills", "Additional Skills",
            "Dự án", "Dự án tiêu biểu", "Projects", "Key Projects",
            "Chứng chỉ", "Chứng nhận", "Certifications", "Certificates",
            "Sở thích", "Hoạt động", "Giải thưởng", "Awards", "Activities", "Interests",
        ]
        t_join = "|".join(re.escape(h) for h in work_headers)
        s_join = "|".join(re.escape(h) for h in stop_headers)
        start_pattern = r"(?:^|\n)\s*(?:" + t_join + r")\s*[:\-]?(?:\n|\Z)"
        stop_pattern = r"(?:^|\n)\s*(?:" + s_join + r")\s*[:\-]?(?:\n|\Z)"

        start_m = re.search(start_pattern, text, re.IGNORECASE)
        if start_m:
            work_text = text[start_m.end():]
            stop_m = re.search(stop_pattern, work_text, re.IGNORECASE)
            if stop_m:
                work_text = work_text[:stop_m.start()]
        else:
            work_text = text

        MONTH_MAP = {
            "jan": 1, "january": 1, "tháng 1": 1, "thg 1": 1, "t1": 1,
            "feb": 2, "february": 2, "tháng 2": 2, "thg 2": 2, "t2": 2,
            "mar": 3, "march": 3, "tháng 3": 3, "thg 3": 3, "t3": 3,
            "apr": 4, "april": 4, "tháng 4": 4, "thg 4": 4, "t4": 4,
            "may": 5, "tháng 5": 5, "thg 5": 5, "t5": 5,
            "jun": 6, "june": 6, "tháng 6": 6, "thg 6": 6, "t6": 6,
            "jul": 7, "july": 7, "tháng 7": 7, "thg 7": 7, "t7": 7,
            "aug": 8, "august": 8, "tháng 8": 8, "thg 8": 8, "t8": 8,
            "sep": 9, "sept": 9, "september": 9, "tháng 9": 9, "thg 9": 9, "t9": 9,
            "oct": 10, "october": 10, "tháng 10": 10, "thg 10": 10, "t10": 10,
            "nov": 11, "november": 11, "tháng 11": 11, "thg 11": 11, "t11": 11,
            "dec": 12, "december": 12, "tháng 12": 12, "thg 12": 12, "t12": 12,
        }
        month_regex = (
            r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
            r"aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|"
            r"tháng\s*\d{1,2}|thg\s*\d{1,2}|0?[1-9]|1[0-2])"
        )

        exp_pattern = re.compile(
            rf"(?:({month_regex})[\s\/\.-]+)?((?:19|20)\d{{2}})\s*(?:[-–—tođến]|hết)\s*"
            rf"(?:(?:({month_regex})[\s\/\.-]+)?((?:19|20)\d{{2}})|(nay|hiện tại|present|current))",
            re.IGNORECASE,
        )

        current_year = 2026
        current_month = 5
        intervals: List[Tuple[int, int]] = []

        matches = list(exp_pattern.finditer(work_text))
        for m in matches:
            start_m_str, start_y_str, end_m_str, end_y_str, present_str = m.groups()
            start_y = int(start_y_str)
            start_m_val = MONTH_MAP.get(start_m_str.lower(), 1) if start_m_str else 1

            if present_str:
                end_y = current_year
                end_m_val = current_month
            else:
                end_y = int(end_y_str) if end_y_str else current_year
                end_m_val = MONTH_MAP.get(end_m_str.lower(), 12) if end_m_str else 12

            start_month_index = start_y * 12 + start_m_val
            end_month_index = max(start_month_index + 1, end_y * 12 + end_m_val)
            intervals.append((start_month_index, end_month_index))

            duration_years = round(max(0.5, (end_month_index - start_month_index) / 12), 1)

            # Lấy ngữ cảnh xung quanh mốc thời gian để làm snippet
            start_pos = max(0, m.start() - 60)
            end_pos = min(len(work_text), m.end() + 150)
            context = work_text[start_pos:end_pos].strip()

            experiences.append(
                CandidateWorkExperience(
                    company="Doanh nghiệp / Dự án",
                    role="Chuyên viên / Kỹ sư",
                    duration_years=duration_years,
                    description=context,
                )
            )

        # 2. Hợp nhất các khoảng thời gian bị overlap để tính tổng số năm chuẩn xác
        total_years = 0.0
        if intervals:
            intervals.sort(key=lambda x: (x[0], x[1]))
            merged: List[List[int]] = []
            for s, e in intervals:
                if not merged or merged[-1][1] < s:
                    merged.append([s, e])
                else:
                    merged[-1][1] = max(merged[-1][1], e)
            total_months = sum(e - s for s, e in merged)
            total_years = round(total_months / 12, 1)

        # 3. Quét các câu khẳng định số năm kinh nghiệm ở phần tóm tắt/mục tiêu (cả tiếng Anh & Việt)
        exp_text_match = re.search(
            r"(\d+(?:\.\d+)?)\s*\+?\s*(?:năm|years?|yrs?)(?:\s*(?:kinh\s*nghiệm|trong\s*nghề|(?:of\s*)?experience))?",
            text,
            re.IGNORECASE,
        )
        if exp_text_match:
            stated_years = float(exp_text_match.group(1))
            # Nếu mốc tính toán chưa có hoặc mốc tự khai lớn hơn một cách hợp lý
            if total_years == 0.0 or (stated_years > total_years and stated_years <= total_years + 2.0):
                total_years = max(total_years, stated_years)

        # Ước tính trần kinh nghiệm hợp lý
        est_years = round(min(30.0, total_years), 1) if total_years > 0 else None
        return experiences, est_years

    @classmethod
    def _extract_education(cls, text: str) -> List[CandidateEducation]:
        """Trích xuất thông tin trường lớp, học vấn."""
        educations: List[CandidateEducation] = []
        edu_keywords = [
            "đại học", "học viện", "cao đẳng", "university", "college", "institute",
            "cử nhân", "kỹ sư", "thạc sĩ", "tiến sĩ", "bachelor", "master", "engineer",
        ]
        lines = text.splitlines()
        for idx, line in enumerate(lines):
            line_lower = line.lower()
            if any(kw in line_lower for kw in edu_keywords):
                # Gom 2 dòng xung quanh làm ngữ cảnh
                snippet = " - ".join(lines[max(0, idx - 1) : min(len(lines), idx + 2)]).strip()
                degree = "Cử nhân / Kỹ sư" if any(d in line_lower for d in ["cử nhân", "kỹ sư", "bachelor", "engineer"]) else "Đại học"
                educations.append(
                    CandidateEducation(
                        institution=line.strip()[:80],
                        degree=degree,
                        description=snippet[:200],
                    )
                )
                if len(educations) >= 3:
                    break

        return educations

    @classmethod
    def _extract_certifications(cls, text: str) -> List[str]:
        """Trích xuất chứng chỉ chuyên môn."""
        certs = []
        cert_keywords = ["aws", "azure", "gcp", "pmp", "ielts", "toeic", "scrum", "ckad", "cka", "ccna"]
        for kw in cert_keywords:
            m = re.search(r"(?i)\b(" + re.escape(kw) + r"[^\n,;]{0,30})\b", text)
            if m:
                certs.append(m.group(0).strip())
        return list(dict.fromkeys(certs))

    @classmethod
    def _extract_summary(cls, text: str) -> Optional[str]:
        """Trích xuất phần tóm tắt mục tiêu / profile."""
        summary_match = re.search(
            r"(?:TÓM TẮT|MỤC TIÊU NGHỀ NGHIỆP|SUMMARY|OBJECTIVE|ABOUT ME)\s*[:\-]?\n([\s\S]*?)(?=\n[A-ZÀ-Ỹ\s]{4,}|\Z)",
            text,
            re.IGNORECASE,
        )
        if summary_match:
            return summary_match.group(1).strip()[:500]
        return None


class CVScorer:
    """
    Bộ đánh giá và chấm điểm CV theo bộ tiêu chí (JobCriteria).
    Tuân thủ nguyên tắc Anti-Hallucination:
    - Bắt buộc phải có bằng chứng (evidence) trích trực tiếp từ CV.
    - Không có bằng chứng -> Điểm 0, ghi rõ "Không tìm thấy thông tin trong CV".
    - Tính điểm tổng hợp theo trọng số (Weighted Score).
    """

    @classmethod
    def evaluate(
        cls,
        profile: StructuredCandidateProfile,
        criteria: JobCriteria,
    ) -> Tuple[float, MatchRating, bool, List[CriterionEvaluation]]:
        """
        Đối chiếu hồ sơ ứng viên với bộ tiêu chí và trả về:
        (Điểm tổng thể, Xếp loại phù hợp, Kết luận đạt/không, Danh sách đánh giá từng tiêu chí).
        """
        # Validate tiêu chí đầu vào
        effective_criteria = criteria.build_effective_criteria()
        if not effective_criteria:
            raise InvalidCriteriaError(
                "Bộ tiêu chí tuyển dụng phải có ít nhất 1 tiêu chí đánh giá "
                "(từ required_skills, preferred_skills, kinh nghiệm hoặc criteria_list)."
            )

        evaluations: List[CriterionEvaluation] = []
        weighted_score_sum = 0.0
        total_weight = 0.0
        required_criteria_failed = 0

        raw_text_lower = profile.raw_text.lower()
        skills_set = {s.lower() for s in profile.skills}

        for crit in effective_criteria:
            weight = max(0.1, float(crit.weight))
            total_weight += weight

            # Đánh giá từng loại tiêu chí
            score, matched, evidence, reasoning = cls._evaluate_single_criterion(
                crit=crit,
                profile=profile,
                raw_text_lower=raw_text_lower,
                skills_set=skills_set,
            )

            evaluations.append(
                CriterionEvaluation(
                    criterion_name=crit.name,
                    criterion_type=crit.criterion_type,
                    weight=weight,
                    score=score,
                    matched=matched,
                    evidence=evidence,
                    reasoning=reasoning,
                )
            )

            weighted_score_sum += score * weight

            if crit.criterion_type == CriterionType.REQUIRED and not matched:
                required_criteria_failed += 1

        # Tính điểm tổng thể (Weighted Average, thang điểm 100)
        overall_score = round(weighted_score_sum / total_weight, 1) if total_weight > 0 else 0.0

        # Phân loại mức độ phù hợp (MatchRating)
        if overall_score >= 85.0 and required_criteria_failed == 0:
            match_rating = MatchRating.STRONG_FIT
        elif overall_score >= 70.0 and required_criteria_failed <= 1:
            match_rating = MatchRating.POTENTIAL_FIT
        elif overall_score >= 50.0:
            match_rating = MatchRating.BORDERLINE
        else:
            match_rating = MatchRating.NOT_FIT

        # Kết luận đỗ/trượt
        is_passed = (overall_score >= criteria.passing_score) and (required_criteria_failed == 0)

        return overall_score, match_rating, is_passed, evaluations

    @classmethod
    def _evaluate_single_criterion(
        cls,
        crit: JobCriterion,
        profile: StructuredCandidateProfile,
        raw_text_lower: str,
        skills_set: set[str],
    ) -> Tuple[float, bool, str, str]:
        """
        Đánh giá một tiêu chí đơn lẻ. Bắt buộc kiểm soát chống ảo giác.
        """
        crit_name_lower = crit.name.lower()

        # 1. Tiêu chí kinh nghiệm (Kinh nghiệm tổng hoặc số năm chuyên biệt)
        if crit.criterion_type == CriterionType.EXPERIENCE or crit.min_years is not None:
            min_years = crit.min_years or 1.0
            candidate_years = profile.total_experience_years_estimated or 0.0

            # Cho phép dung sai làm tròn tháng nhỏ (<= 0.2 năm ~ 2 tháng)
            if candidate_years >= (min_years - 0.2):
                evidence = f"Hồ sơ ghi nhận ước tính {candidate_years} năm kinh nghiệm làm việc thực tế."
                return 100.0, True, evidence, f"Đạt yêu cầu kinh nghiệm tối thiểu ({candidate_years}/{min_years} năm)."
            elif candidate_years > 0:
                score = round((candidate_years / min_years) * 80.0, 1)
                evidence = f"Hồ sơ ghi nhận {candidate_years} năm kinh nghiệm, thấp hơn mức yêu cầu {min_years} năm."
                return score, False, evidence, f"Chưa đạt đủ số năm kinh nghiệm yêu cầu ({candidate_years}/{min_years} năm)."
            else:
                return 0.0, False, "Không tìm thấy thông tin trong CV", "Không tìm thấy dữ liệu về số năm kinh nghiệm làm việc trong CV."

        # 2. Tiêu chí kỹ năng hoặc bằng cấp
        # Kiểm tra xem từ khóa tiêu chí có xuất hiện trong danh sách kỹ năng hoặc nội dung văn bản không
        # Tìm cụm từ hoặc từ khóa chính
        keywords = [k.strip() for k in re.split(r"[,/&]|hoặc|or", crit_name_lower) if len(k.strip()) >= 2]
        if not keywords:
            keywords = [crit_name_lower]

        found_evidence: Optional[str] = None

        # a. Tìm trong skills set
        for kw in keywords:
            if kw in skills_set or any(kw in s for s in skills_set):
                # Tìm câu thực tế chứa từ khóa trong raw_text để làm bằng chứng
                found_evidence = cls._find_sentence_containing(profile.raw_text, kw)
                break

        # b. Tìm trong raw text nếu chưa có bằng chứng
        if not found_evidence:
            for kw in keywords:
                found_evidence = cls._find_sentence_containing(profile.raw_text, kw)
                if found_evidence:
                    break

        # Đánh giá dựa trên bằng chứng
        if found_evidence:
            score = 100.0 if crit.criterion_type != CriterionType.PREFERRED else 90.0
            reasoning = f"Ứng viên có kỹ năng / kinh nghiệm phù hợp với tiêu chí '{crit.name}'."
            return score, True, found_evidence, reasoning
        else:
            # ANTI-HALLUCINATION: Không có trong CV -> Tuyệt đối không suy diễn!
            evidence = "Không tìm thấy thông tin trong CV"
            score = 0.0
            reasoning = f"Không tìm thấy đề cập nào liên quan đến '{crit.name}' trong hồ sơ ứng viên."
            return score, False, evidence, reasoning

    @classmethod
    def _find_sentence_containing(cls, text: str, keyword: str) -> Optional[str]:
        """Tìm và trích xuất câu văn bản trực tiếp chứa từ khóa làm bằng chứng."""
        pattern = r"(?i)(?:[^\n.!?]*\b" + re.escape(keyword) + r"\b[^\n.!?]*)"
        m = re.search(pattern, text)
        if m:
            snippet = m.group(0).strip()
            # Giới hạn độ dài đoạn trích dẫn
            return snippet[:250]
        return None


class CVSummarizer:
    """
    Bộ tổng hợp thông tin và sinh báo cáo trực quan cho HR:
    - Điểm mạnh nổi bật (kèm bằng chứng)
    - Điểm còn thiếu hoặc chưa rõ ràng
    - Bản tóm tắt nhận định tổng quan (Executive Summary)
    - Danh sách câu hỏi phỏng vấn gợi ý (Interview Questions)
    """

    @classmethod
    def generate_summary(
        cls,
        profile: StructuredCandidateProfile,
        criteria: JobCriteria,
        overall_score: float,
        match_rating: MatchRating,
        is_passed: bool,
        evaluations: List[CriterionEvaluation],
    ) -> Tuple[str, List[str], List[str], List[str]]:
        """
        Sinh ra:
        - executive_summary (str)
        - key_strengths (List[str])
        - missing_or_unclear_points (List[str])
        - interview_questions (List[str])
        """
        # 1. Danh sách điểm mạnh
        key_strengths = []
        for ev in evaluations:
            if ev.matched and ev.score >= 80.0:
                key_strengths.append(f"Khớp tiêu chí '{ev.criterion_name}': {ev.evidence}")
        if not key_strengths:
            key_strengths.append("Hồ sơ ứng viên có các thông tin cơ bản nhưng chưa có điểm mạnh vượt trội nổi bật.")

        # 2. Danh sách điểm còn thiếu hoặc chưa rõ
        missing_points = []
        for ev in evaluations:
            if not ev.matched:
                crit_tag = "[Bắt buộc]" if ev.criterion_type == CriterionType.REQUIRED else "[Ưu tiên]"
                missing_points.append(f"{crit_tag} Thiếu tiêu chí '{ev.criterion_name}': {ev.reasoning}")

        if not profile.total_experience_years_estimated:
            missing_points.append("Thời gian kinh nghiệm làm việc chưa được phân kỳ rõ ràng theo từng giai đoạn.")

        # 3. Gợi ý câu hỏi phỏng vấn
        interview_questions = cls._generate_interview_questions(profile, criteria, evaluations)

        # 4. Bản tóm tắt cho HR (Executive Summary)
        exec_summary = cls._generate_executive_summary(
            profile=profile,
            criteria=criteria,
            overall_score=overall_score,
            match_rating=match_rating,
            is_passed=is_passed,
            evaluations=evaluations,
            key_strengths=key_strengths,
            missing_points=missing_points,
        )

        return exec_summary, key_strengths, missing_points, interview_questions

    @classmethod
    def _generate_interview_questions(
        cls,
        profile: StructuredCandidateProfile,
        criteria: JobCriteria,
        evaluations: List[CriterionEvaluation],
    ) -> List[str]:
        """Tạo câu hỏi phỏng vấn có tính chọn lọc và đào sâu vào bằng chứng thực tế."""
        questions: List[str] = []
        c_name = profile.candidate_info.full_name or "Ứng viên"

        # a. Câu hỏi kiểm chứng các điểm còn thiếu / chưa rõ
        for ev in evaluations:
            if not ev.matched:
                questions.append(
                    f"Trong hồ sơ bạn chưa đề cập nhiều về '{ev.criterion_name}'. "
                    f"Bạn đã từng có trải nghiệm thực chiến hoặc học tập về phần này chưa? Hãy chia sẻ ví dụ cụ thể."
                )
            if len(questions) >= 2:
                break

        # b. Câu hỏi đào sâu vào kỹ năng thế mạnh
        matched_crits = [ev for ev in evaluations if ev.matched]
        for ev in matched_crits[:2]:
            questions.append(
                f"Bạn có nêu kinh nghiệm về '{ev.criterion_name}'. "
                f"Hãy chia sẻ về thử thách kỹ thuật lớn nhất bạn từng giải quyết khi làm việc với công nghệ này."
            )

        # c. Câu hỏi về văn hóa và phương pháp làm việc cho vị trí
        questions.append(
            f"Đối với vị trí {criteria.job_title}, bạn thường tiếp cận và tìm hiểu một hệ thống/quy trình mới "
            f"trong 30 ngày đầu tiên như thế nào?"
        )

        return questions

    @classmethod
    def _generate_executive_summary(
        cls,
        profile: StructuredCandidateProfile,
        criteria: JobCriteria,
        overall_score: float,
        match_rating: MatchRating,
        is_passed: bool,
        evaluations: List[CriterionEvaluation],
        key_strengths: List[str],
        missing_points: List[str],
    ) -> str:
        """Tạo bản tóm tắt dành cho HR với cấu trúc rõ ràng."""
        c_name = profile.candidate_info.full_name or "Ứng viên"
        rating_labels = {
            MatchRating.STRONG_FIT: "Rất phù hợp (Strong Fit)",
            MatchRating.POTENTIAL_FIT: "Phù hợp / Tiềm năng (Potential Fit)",
            MatchRating.BORDERLINE: "Cần cân nhắc / Phỏng vấn kỹ (Borderline)",
            MatchRating.NOT_FIT: "Chưa phù hợp (Not Fit)",
        }
        rating_str = rating_labels.get(match_rating, match_rating.value)
        status_str = "ĐẠT NGƯỠNG XÉT TUYỂN" if is_passed else "CHƯA ĐẠT NGƯỠNG XÉT TUYỂN"

        exp_str = (
            f"khoảng {profile.total_experience_years_estimated} năm kinh nghiệm"
            if profile.total_experience_years_estimated
            else "chưa rõ số năm kinh nghiệm cụ thể"
        )

        summary_lines = [
            f"**Ứng viên**: {c_name} | **Vị trí**: {criteria.job_title}",
            f"**Đánh giá tổng quan**: {overall_score}/100 điểm - Xếp loại: **{rating_str}** ({status_str}).",
            f"**Hồ sơ kinh nghiệm**: Ứng viên có {exp_str}. Số tiêu chí đạt: {sum(1 for e in evaluations if e.matched)}/{len(evaluations)} tiêu chí.",
        ]

        if is_passed:
            summary_lines.append(
                f"👉 **Khuyến nghị tuyển dụng**: Ứng viên đáp ứng tốt các yêu cầu chính. "
                "Đề xuất phòng Nhân sự liên hệ sắp xếp lịch phỏng vấn chuyên môn vòng 1."
            )
        elif match_rating == MatchRating.BORDERLINE:
            summary_lines.append(
                f"👉 **Khuyến nghị tuyển dụng**: Ứng viên có tiềm năng nhưng còn thiếu một số kỹ năng quan trọng. "
                "Có thể tiến hành phỏng vấn sàng lọc ngắn (Phone screening 15 phút) trước khi quyết định."
            )
        else:
            summary_lines.append(
                f"👉 **Khuyến nghị tuyển dụng**: Hồ sơ chưa đáp ứng đủ các tiêu chí quan trọng của vị trí. "
                "Đề xuất lưu hồ sơ vào Talent Pool để cân nhắc cho các vị trí khác phù hợp hơn."
            )

        return "\n\n".join(summary_lines)


class CVScreeningService:
    """
    Facade chính cung cấp dịch vụ Sàng lọc CV hoàn chỉnh:
    - Nhận vào file CV (.pdf, .docx, .txt, .md) hoặc chuỗi text + JobCriteria
    - Thực hiện Pipeline: Preprocessing -> Extract -> Score -> Summarize
    - Trả về CVScreeningResult chuẩn hóa
    """

    def __init__(self):
        self.preprocessor = CVPreprocessor()
        self.extractor = CVExtractor()
        self.scorer = CVScorer()
        self.summarizer = CVSummarizer()

    def screen_cv(
        self,
        cv_input: Union[str, Path],
        criteria: JobCriteria,
    ) -> CVScreeningResult:
        """
        Xử lý sàng lọc CV đồng bộ (Synchronous).
        """
        start_time = time.time()

        try:
            # 1. Đọc và tiền xử lý CV
            normalized_text, metadata = self.preprocessor.load_and_preprocess(cv_input)

            # 2. Bóc tách hồ sơ ứng viên
            profile = self.extractor.extract_profile(normalized_text)

            # 3. Chấm điểm và đối chiếu tiêu chí (Anti-Hallucination)
            overall_score, match_rating, is_passed, evaluations = self.scorer.evaluate(
                profile=profile,
                criteria=criteria,
            )

            # 4. Sinh bản tóm tắt, điểm mạnh, điểm thiếu và câu hỏi phỏng vấn
            exec_summary, strengths, missing, questions = self.summarizer.generate_summary(
                profile=profile,
                criteria=criteria,
                overall_score=overall_score,
                match_rating=match_rating,
                is_passed=is_passed,
                evaluations=evaluations,
            )

            processing_time = round(time.time() - start_time, 3)
            metadata["processing_time_sec"] = processing_time

            return CVScreeningResult(
                candidate_info=profile.candidate_info,
                job_title=criteria.job_title,
                overall_score=overall_score,
                match_rating=match_rating,
                is_passed=is_passed,
                criteria_evaluations=evaluations,
                key_strengths=strengths,
                missing_or_unclear_points=missing,
                executive_summary=exec_summary,
                interview_questions=questions,
                metadata=metadata,
            )

        except (CVLoadError, InvalidCriteriaError, CVExtractionError, CVScoringError):
            raise
        except Exception as e:
            logger.exception("Lỗi không mong muốn trong quá trình sàng lọc CV: %s", e)
            raise CVScoringError(f"Quá trình sàng lọc CV gặp sự cố: {str(e)}")

    async def ascreen_cv(
        self,
        cv_input: Union[str, Path],
        criteria: JobCriteria,
    ) -> CVScreeningResult:
        """
        Xử lý sàng lọc CV bất đồng bộ (Asynchronous).
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.screen_cv(cv_input, criteria))


# Singleton instance
_cv_screening_service_instance: Optional[CVScreeningService] = None


def get_cv_screening_service() -> CVScreeningService:
    """Lấy hoặc khởi tạo instance singleton của CVScreeningService."""
    global _cv_screening_service_instance
    if _cv_screening_service_instance is None:
        _cv_screening_service_instance = CVScreeningService()
    return _cv_screening_service_instance
