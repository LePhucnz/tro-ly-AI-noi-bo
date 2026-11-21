"""
documents/cv_preprocessor.py
Tiền xử lý và chuẩn hóa văn bản CV ứng viên.
Tái sử dụng DocumentLoaderFactory để hỗ trợ các định dạng .pdf, .docx, .txt, .md.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from core.exceptions import CVLoadError, DocumentLoadError, UnsupportedFileTypeError
from documents.hasher import DocumentHasher
from documents.loader_factory import Document, DocumentLoaderFactory

logger = logging.getLogger("cv_preprocessor")


class CVPreprocessor:
    """
    Bộ nạp và tiền xử lý hồ sơ CV:
    - Đọc từ tệp (.pdf, .docx, .txt, .md) hoặc chuỗi văn bản trực tiếp.
    - Chuẩn hóa ký tự, khoảng trắng, định dạng các đề mục chuẩn (Kinh nghiệm, Kỹ năng, Học vấn...).
    """

    # Các đề mục thường gặp trong CV tiếng Việt và tiếng Anh
    SECTION_HEADERS = [
        "THÔNG TIN CÁ NHÂN", "THÔNG TIN LIÊN HỆ", "PERSONAL INFORMATION", "CONTACT",
        "MỤC TIÊU NGHỀ NGHIỆP", "TÓM TẮT", "SUMMARY", "OBJECTIVE", "ABOUT ME",
        "KỸ NĂNG", "KỸ NĂNG CHUYÊN MÔN", "SKILLS", "TECHNICAL SKILLS", "EXPERTISE",
        "KINH NGHIỆM LÀM VIỆC", "KINH NGHIỆM", "WORK EXPERIENCE", "EXPERIENCE", "EMPLOYMENT HISTORY",
        "HỌC VẤN", "TRÌNH ĐỘ HỌC VẤN", "EDUCATION", "ACADEMIC BACKGROUND",
        "DỰ ÁN", "DỰ ÁN TIÊU BIỂU", "PROJECTS", "KEY PROJECTS",
        "CHỨNG CHỈ", "CHỨNG NHẬN", "CERTIFICATES", "CERTIFICATIONS",
        "HOẠT ĐỘNG", "GIẢI THƯỞNG", "ACTIVITIES", "AWARDS", "HONORS",
    ]

    @classmethod
    def load_and_preprocess(
        cls,
        cv_input: Union[str, Path],
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Nạp CV từ đường dẫn file hoặc chuỗi văn bản, sau đó làm sạch và chuẩn hóa.

        Returns:
            Tuple[str, Dict[str, Any]]: (Nội dung văn bản đã chuẩn hóa, Metadata tệp)
        """
        if isinstance(cv_input, Path) or (isinstance(cv_input, str) and (cv_input.endswith(tuple(DocumentLoaderFactory.get_supported_extensions())) or Path(cv_input).exists())):
            return cls._load_from_file(Path(cv_input))
        elif isinstance(cv_input, str):
            return cls._load_from_raw_text(cv_input)
        else:
            raise CVLoadError(
                str(cv_input),
                "Đầu vào CV không hợp lệ. Vui lòng cung cấp đường dẫn tệp hợp lệ hoặc chuỗi văn bản.",
            )

    @classmethod
    def _load_from_file(cls, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Đọc và bóc tách nội dung từ tệp tin qua DocumentLoaderFactory."""
        if not file_path.exists():
            raise CVLoadError(str(file_path), f"Tệp tin CV không tồn tại tại đường dẫn: {file_path}")

        if not file_path.is_file():
            raise CVLoadError(str(file_path), f"Đường dẫn không phải là một tệp hợp lệ: {file_path}")

        try:
            docs: List[Document] = DocumentLoaderFactory.load_document(file_path)
            if not docs:
                raise CVLoadError(str(file_path), "Tệp tin CV rỗng, không chứa nội dung văn bản.")

            raw_text = "\n\n".join(doc.page_content for doc in docs if doc.page_content.strip())
            if not raw_text.strip():
                raise CVLoadError(str(file_path), "Tệp tin CV rỗng sau khi bóc tách văn bản.")

            cleaned_text = cls.normalize_cv_text(raw_text)

            metadata: Dict[str, Any] = {
                "source": str(file_path),
                "file_name": file_path.name,
                "file_type": file_path.suffix.lstrip(".").lower(),
                "file_size_bytes": file_path.stat().st_size,
                "file_hash": DocumentHasher.compute_file_sha256(file_path),
                "total_pages": len(docs),
            }
            return cleaned_text, metadata

        except UnsupportedFileTypeError as e:
            raise CVLoadError(str(file_path), str(e))
        except DocumentLoadError as e:
            raise CVLoadError(str(file_path), e.message)
        except Exception as e:
            logger.exception("Lỗi bất ngờ khi đọc file CV %s: %s", file_path, e)
            raise CVLoadError(str(file_path), str(e))

    @classmethod
    def _load_from_raw_text(cls, raw_text: str) -> Tuple[str, Dict[str, Any]]:
        """Xử lý trực tiếp từ văn bản CV dạng chuỗi."""
        clean = cls.normalize_cv_text(raw_text)
        if not clean.strip():
            raise CVLoadError("raw_text", "Nội dung văn bản CV rỗng hoặc chỉ chứa khoảng trắng.")

        metadata: Dict[str, Any] = {
            "source": "raw_text_input",
            "file_name": "candidate_cv.txt",
            "file_type": "text",
            "file_size_bytes": len(raw_text.encode("utf-8")),
            "file_hash": DocumentHasher.compute_sha256(raw_text),
            "total_pages": 1,
        }
        return clean, metadata

    @classmethod
    def normalize_cv_text(cls, text: str) -> str:
        """
        Chuẩn hóa văn bản CV:
        - Chuẩn hóa ngắt dòng, khoảng trắng thừa
        - Loại bỏ ký tự điều khiển lạ (non-printable)
        - Làm nổi bật đề mục để dễ bóc tách
        """
        if not text:
            return ""

        # 1. Chuẩn hóa xuống dòng (\r\n -> \n)
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")

        # 2. Xóa các ký tự điều khiển (control characters) trừ \n và \t
        normalized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", normalized)

        # 3. Gộp các khoảng trắng liên tiếp trong cùng một dòng
        normalized = re.sub(r"[ \t]+", " ", normalized)

        # 4. Gộp nhiều dòng trống liên tiếp thành tối đa 2 dòng trống
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)

        # 5. Cắt tỉa đầu cuối
        return normalized.strip()
