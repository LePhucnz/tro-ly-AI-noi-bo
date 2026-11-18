"""
documents/text_cleaner.py
Tiền xử lý văn bản trước khi chunking: loại bỏ header/footer trang,
ký tự thừa, và chuẩn hóa khoảng trắng.

Mục tiêu: Giảm thiểu hiện tượng chunk bị cắt lửng do header/footer
trang PDF xen vào giữa câu.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from documents.loader_factory import Document

logger = logging.getLogger("text_cleaner")


# ==============================================================================
# Các pattern phổ biến cần loại bỏ từ tài liệu doanh nghiệp Việt Nam
# ==============================================================================

# Pattern 1: Header/footer dạng "Tên công ty: Tên tài liệu ... Trang X/Y"
# Ví dụ: "Bảo Minh: Quy chế nội bộ về quản trị công ty Trang 18/42"
_PAGE_HEADER_FOOTER_PATTERNS = [
    # "Tên: ... Trang X/Y" hoặc "Trang X / Y"
    re.compile(
        r"^.*?Trang\s+\d+\s*/\s*\d+\s*$",
        re.MULTILINE | re.IGNORECASE,
    ),
    # Dòng chỉ chứa số trang đơn: "- 18 -" hoặc "—18—"
    re.compile(
        r"^\s*[-—–]\s*\d+\s*[-—–]\s*$",
        re.MULTILINE,
    ),
    # Footer dạng "Page X of Y"
    re.compile(
        r"^\s*Page\s+\d+\s+of\s+\d+\s*$",
        re.MULTILINE | re.IGNORECASE,
    ),
    # Dòng chỉ chứa số trang đứng một mình
    re.compile(
        r"^\s*\d{1,3}\s*$",
        re.MULTILINE,
    ),
]

# Pattern 2: Nhiều dòng trống liên tiếp → gộp thành tối đa 2
_MULTIPLE_BLANK_LINES = re.compile(r"\n{3,}")

# Pattern 3: Khoảng trắng thừa trong dòng (nhiều space liên tiếp)
_EXCESSIVE_SPACES = re.compile(r"[ \t]{3,}")


class TextCleaner:
    """
    Bộ tiền xử lý văn bản trước khi đưa vào Text Splitter.
    Có thể bật/tắt từng bước xử lý.
    """

    def __init__(
        self,
        remove_headers_footers: bool = True,
        normalize_whitespace: bool = True,
        custom_patterns: Optional[List[re.Pattern]] = None,
    ) -> None:
        self.remove_headers_footers = remove_headers_footers
        self.normalize_whitespace = normalize_whitespace
        self.custom_patterns = custom_patterns or []

    def clean_text(self, text: str) -> str:
        """Áp dụng toàn bộ pipeline làm sạch trên chuỗi văn bản."""
        if not text or not text.strip():
            return text

        cleaned = text

        # Bước 1: Loại bỏ header/footer trang
        if self.remove_headers_footers:
            for pattern in _PAGE_HEADER_FOOTER_PATTERNS:
                cleaned = pattern.sub("", cleaned)

        # Bước 2: Áp dụng custom patterns (nếu có)
        for pattern in self.custom_patterns:
            cleaned = pattern.sub("", cleaned)

        # Bước 3: Chuẩn hóa khoảng trắng
        if self.normalize_whitespace:
            # Gộp nhiều dòng trống thành 2 dòng
            cleaned = _MULTIPLE_BLANK_LINES.sub("\n\n", cleaned)
            # Gộp khoảng trắng thừa trong dòng
            cleaned = _EXCESSIVE_SPACES.sub("  ", cleaned)

        # Bước 4: Trim đầu cuối mỗi dòng
        lines = cleaned.split("\n")
        lines = [line.rstrip() for line in lines]
        cleaned = "\n".join(lines).strip()

        return cleaned

    def clean_documents(self, documents: List[Document]) -> List[Document]:
        """
        Làm sạch page_content của từng Document.
        Trả về danh sách Document mới (không mutate bản gốc).
        """
        if not documents:
            return documents

        cleaned_docs = []
        total_removed = 0

        for doc in documents:
            original_len = len(doc.page_content)
            cleaned_text = self.clean_text(doc.page_content)
            removed = original_len - len(cleaned_text)
            total_removed += removed

            # Bỏ qua document rỗng sau khi clean
            if not cleaned_text.strip():
                logger.debug(
                    "Document từ '%s' trở thành rỗng sau khi clean, bỏ qua.",
                    doc.metadata.get("source", "?"),
                )
                continue

            cleaned_doc = Document(
                page_content=cleaned_text,
                metadata=dict(doc.metadata),
            )
            cleaned_docs.append(cleaned_doc)

        if total_removed > 0:
            logger.info(
                "TextCleaner: Đã làm sạch %d documents, loại bỏ %d ký tự header/footer/noise.",
                len(documents),
                total_removed,
            )

        return cleaned_docs


# Singleton
_cleaner_instance: Optional[TextCleaner] = None


def get_text_cleaner() -> TextCleaner:
    """Singleton TextCleaner cho toàn ứng dụng."""
    global _cleaner_instance
    if _cleaner_instance is None:
        _cleaner_instance = TextCleaner()
    return _cleaner_instance
