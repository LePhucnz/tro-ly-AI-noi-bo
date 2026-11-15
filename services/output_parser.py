"""
services/output_parser.py
Output Parser bóc tách rõ ràng phần Câu trả lời của AI và danh sách Nguồn tài liệu tham khảo
(Source Citations) từ RAG chain, đồng thời chuẩn hóa dữ liệu phát ra qua SSE streaming.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field


class SourceCitation(BaseModel):
    """Mô hình dữ liệu chi tiết của từng nguồn trích dẫn tài liệu tham khảo."""
    source: str = Field(..., description="Tên tệp, mã tài liệu hoặc nguồn dữ liệu")
    page: Optional[Union[int, str]] = Field(default=None, description="Số trang hoặc mục/điều khoản")
    score: Optional[float] = Field(default=None, description="Điểm tương đồng (relevance score)")
    snippet: str = Field(default="", description="Đoạn trích dẫn nội dung liên quan")
    url: Optional[str] = Field(default=None, description="Đường dẫn xem tài liệu gốc nếu có")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "page": self.page,
            "score": round(self.score, 4) if self.score is not None else None,
            "snippet": self.snippet.strip(),
            "url": self.url,
        }


class RAGParsedOutput(BaseModel):
    """Kết quả hoàn chỉnh sau khi bóc tách từ RAG Chain."""
    answer: str = Field(..., description="Nội dung câu trả lời chính của AI")
    citations: List[SourceCitation] = Field(default_factory=list, description="Danh sách các nguồn trích dẫn")


class RAGOutputParser:
    """
    Parser linh hoạt hỗ trợ nhiều cấu trúc trả về từ RAG Core:
    1. LangChain Document objects (page_content + metadata)
    2. Dict payloads (ví dụ: {"answer": ..., "source_documents": [...]})
    3. Trích xuất citations từ format văn bản của LLM (nếu LLM xuất kèm tag trích dẫn)
    """

    @classmethod
    def parse_documents(cls, raw_docs: List[Any]) -> List[SourceCitation]:
        """
        Chuyển đổi danh sách Document từ Retriever thành danh sách SourceCitation chuẩn.
        Hỗ trợ cả LangChain Document object và Python dict.
        """
        citations: List[SourceCitation] = []
        for doc in raw_docs:
            if hasattr(doc, "page_content") and hasattr(doc, "metadata"):
                # LangChain Document
                meta = doc.metadata or {}
                source = meta.get("source") or meta.get("file_name") or meta.get("title") or "Tài liệu nội bộ"
                page = meta.get("page") or meta.get("section")
                score = meta.get("score") or meta.get("similarity")
                snippet = doc.page_content[:300] if doc.page_content else ""
                url = meta.get("url")
            elif isinstance(doc, dict):
                # Dict format
                source = doc.get("source") or doc.get("file_name") or doc.get("title") or "Tài liệu nội bộ"
                page = doc.get("page") or doc.get("section")
                score = doc.get("score") or doc.get("similarity")
                snippet = doc.get("snippet") or doc.get("content") or doc.get("page_content") or ""
                snippet = snippet[:300] if isinstance(snippet, str) else str(snippet)
                url = doc.get("url")
            elif isinstance(doc, SourceCitation):
                citations.append(doc)
                continue
            else:
                source = "Tài liệu nội bộ"
                page = None
                score = None
                snippet = str(doc)[:300]
                url = None

            citations.append(
                SourceCitation(
                    source=str(source),
                    page=page,
                    score=float(score) if score is not None else None,
                    snippet=snippet,
                    url=url,
                )
            )

        return citations

    @classmethod
    def parse_raw_response(
        cls,
        raw_response: Union[Dict[str, Any], str],
        retrieved_docs: Optional[List[Any]] = None,
    ) -> RAGParsedOutput:
        """
        Bóc tách câu trả lời và citations từ kết quả trả về của RAG Core.
        """
        citations: List[SourceCitation] = []

        if retrieved_docs:
            citations = cls.parse_documents(retrieved_docs)

        if isinstance(raw_response, dict):
            # Cấu trúc dạng Dict từ RAG Chain
            answer = raw_response.get("answer") or raw_response.get("result") or raw_response.get("output") or ""
            doc_candidates = (
                raw_response.get("source_documents")
                or raw_response.get("citations")
                or raw_response.get("sources")
                or []
            )
            if doc_candidates and not citations:
                citations = cls.parse_documents(doc_candidates)
        elif isinstance(raw_response, str):
            answer = raw_response
        else:
            answer = str(raw_response)

        # Nếu chưa có citations từ docs, thử bóc tách từ text nếu có pattern
        if not citations and answer:
            clean_answer, extracted_citations = cls._extract_citations_from_text(answer)
            if extracted_citations:
                answer = clean_answer
                citations = extracted_citations

        return RAGParsedOutput(answer=answer.strip(), citations=citations)

    @classmethod
    def _extract_citations_from_text(cls, text: str) -> Tuple[str, List[SourceCitation]]:
        """
        Tách phần 'Nguồn tham khảo:' hoặc 'Sources:' ở cuối văn bản nếu LLM tự chèn vào text.
        """
        pattern = r"(?i)(?:\n+|---\n+)?(?:\*\*|#+\s*)?(?:Nguồn tài liệu tham khảo|Nguồn tham khảo|Tài liệu tham khảo|Sources|Citations):?\s*(?:\*\*)?\n([\s\S]+)$"
        match = re.search(pattern, text)
        if not match:
            return text, []

        clean_text = text[: match.start()].strip()
        citation_block = match.group(1).strip()
        citations: List[SourceCitation] = []

        # Tách từng dòng nguồn: ví dụ "- Sổ tay nhân viên (Trang 12)"
        for line in citation_block.splitlines():
            line = line.strip().lstrip("-*• ")
            if not line:
                continue
            # Regex bóc tách tên file và trang
            page_match = re.search(r"\(trang\s*(\d+)\)|\[trang\s*(\d+)\]|p\.?\s*(\d+)", line, re.IGNORECASE)
            page = None
            if page_match:
                page_str = next((g for g in page_match.groups() if g is not None), None)
                if page_str:
                    page = int(page_str)
            citations.append(
                SourceCitation(
                    source=line,
                    page=page,
                    snippet="Trích dẫn theo tài liệu văn bản",
                )
            )

        return clean_text, citations

    # ==========================
    # SSE Formatting Helpers
    # ==========================
    @classmethod
    def format_token_event(cls, token: str) -> Dict[str, Any]:
        """Tạo payload SSE cho từng token phát sinh."""
        return {
            "type": "token",
            "content": token,
        }

    @classmethod
    def format_citations_event(cls, citations: List[SourceCitation]) -> Dict[str, Any]:
        """Tạo payload SSE chứa toàn bộ danh sách nguồn trích dẫn."""
        return {
            "type": "citations",
            "citations": [c.to_dict() for c in citations],
        }

    @classmethod
    def format_done_event(
        cls,
        session_id: str,
        total_tokens: Optional[int] = None,
        strategy_used: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Tạo payload SSE báo hiệu hoàn thành lượt stream."""
        return {
            "type": "done",
            "session_id": session_id,
            "total_tokens": total_tokens,
            "strategy": strategy_used,
        }

    @classmethod
    def format_error_event(cls, error_msg: str, error_code: str = "INTERNAL_ERROR") -> Dict[str, Any]:
        """Tạo payload SSE báo lỗi an toàn tới client."""
        return {
            "type": "error",
            "message": error_msg,
            "code": error_code,
        }

