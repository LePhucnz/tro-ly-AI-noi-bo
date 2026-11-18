"""
services/rag_core_interface.py
Định nghĩa Interface chuẩn kết nối với RAG Core do Dev 1 xây dựng,
đồng thời cung cấp DefaultInternalRAGCore với tri thức doanh nghiệp mẫu
giúp toàn bộ hệ thống chạy và kiểm thử độc lập ngay lập tức.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional
from services.memory import ChatMessage
from services.output_parser import SourceCitation


class RetrieverStrategy(str, Enum):
    """Các chiến lược tìm kiếm tài liệu từ Vector Database."""
    similarity = "similarity"  # Độ tương đồng Cosine chuẩn
    mmr = "mmr"                # Maximal Marginal Relevance (đa dạng hóa nguồn tin)
    threshold = "threshold"    # Lọc điểm tương đồng theo ngưỡng tối thiểu
    similarity_score_threshold = "similarity_score_threshold"  # Alias tương thích đầy đủ
    hybrid = "hybrid"          # Kết hợp Dense Vector + BM25 Từ khóa (khử dấu)


class BaseRAGCore(ABC):
    """
    Interface trừu tượng cho RAG Core.
    Dev 1 chỉ cần kế thừa lớp này hoặc triển khai các phương thức tương đương:
    1. retrieve_documents: Tìm kiếm tài liệu theo chiến lược
    2. stream_answer: Sinh câu trả lời dạng streaming token
    """

    @abstractmethod
    async def retrieve_documents(
        self,
        query: str,
        strategy: RetrieverStrategy = RetrieverStrategy.similarity,
        k: int = 4,
        threshold: float = 0.5,
    ) -> List[SourceCitation]:
        """Truy vấn tài liệu từ Vector Store theo chiến lược đã chọn."""
        pass

    @abstractmethod
    async def stream_answer(
        self,
        query: str,
        history: List[ChatMessage],
        citations: List[SourceCitation],
    ) -> AsyncIterator[str]:
        """Sinh từng token của câu trả lời dựa trên query, lịch sử và tài liệu trích xuất."""
        pass


class DefaultInternalRAGCore(BaseRAGCore):
    """
    RAG Core mặc định (Mock/Fallback) tích hợp sẵn cơ sở tri thức nội bộ công ty.
    Hỗ trợ đầy đủ các thuật toán retriever (similarity, mmr, threshold)
    và cơ chế phát token chữ cái như ChatGPT.
    Khi Dev 1 tích hợp model thực tế, lớp này có thể dễ dàng thay thế hoặc kế thừa.
    """

    def __init__(self):
        # Cơ sở tri thức nội bộ mẫu cho hệ sinh thái công ty
        self._knowledge_base = [
            {
                "source": "Quy_che_lam_viec_tu_xa_2024.pdf",
                "page": 4,
                "score": 0.94,
                "keywords": ["làm việc từ xa", "remote", "wfh", "online", "nhà", "work from home"],
                "content": (
                    "Quy định làm việc từ xa (WFH): Nhân viên chính thức được phép đăng ký WFH tối đa 2 ngày/tuần "
                    "sau khi được Quản lý trực tiếp phê duyệt trước ít nhất 24 giờ trên hệ thống HRM nội bộ. "
                    "Trong thời gian WFH, nhân viên phải duy trì kết nối Slack, kiểm tra email và đảm bảo tiến độ công việc."
                ),
            },
            {
                "source": "Chinh_sach_an_toan_thong_tin_v3.docx",
                "page": 11,
                "score": 0.88,
                "keywords": ["bảo mật", "mật khẩu", "an toàn thông tin", "vpn", "dữ liệu", "password"],
                "content": (
                    "Chính sách bảo mật: Toàn bộ thiết bị làm việc phải kích hoạt mã hóa ổ đĩa BitLocker/FileVault. "
                    "Mật khẩu tài khoản công ty tối thiểu 12 ký tự, bao gồm chữ hoa, chữ thường, số và ký tự đặc biệt, "
                    "thay đổi định kỳ 90 ngày. Bắt buộc kích hoạt xác thực hai yếu tố (2FA/MFA) trên tất cả hệ thống."
                ),
            },
            {
                "source": "So_tay_nhan_su_va_phuc_loi.pdf",
                "page": 18,
                "score": 0.85,
                "keywords": ["nghỉ phép", "phép năm", "ốm đau", "chế độ", "phúc lợi", "bảo hiểm"],
                "content": (
                    "Chế độ nghỉ phép năm: Mỗi nhân viên có 12 ngày phép năm có lương. Sau mỗi 2 năm làm việc liên tục "
                    "được cộng thêm 1 ngày phép. Đơn xin nghỉ phép từ 3 ngày trở lên cần báo trước tối thiểu 3 ngày làm việc. "
                    "Công ty chi trả gói bảo hiểm sức khỏe nâng cao cho nhân viên sau khi hoàn thành thời gian thử việc."
                ),
            },
            {
                "source": "Huong_dan_onboarding_ky_thuat.md",
                "page": 2,
                "score": 0.81,
                "keywords": ["onboarding", "git", "ci/cd", "quy trình", "deploy", "server", "kỹ thuật"],
                "content": (
                    "Quy trình triển khai kỹ thuật: Mọi Pull Request (PR) phải có ít nhất 2 reviews từ Senior Engineer. "
                    "Các commit phải tuân thủ chuẩn Conventional Commits. Pipeline CI/CD tự động chạy unit tests "
                    "và security scan trước khi cho phép merge vào nhánh staging/main."
                ),
            },
            {
                "source": "Quy_trinh_thanh_toan_cong_tac_phi.pdf",
                "page": 7,
                "score": 0.77,
                "keywords": ["công tác phí", "hoàn ứng", "hóa đơn", "chi phí", "tài chính", "tiền"],
                "content": (
                    "Quy định hoàn ứng công tác phí: Hóa đơn đỏ (e-invoice) hợp lệ phải gửi về phòng Kế toán "
                    "trong vòng 7 ngày làm việc kể từ ngày kết thúc chuyến công tác. Định mức lưu trú khách sạn tối đa "
                    "1.200.000 VNĐ/đêm đối với các thành phố trực thuộc trung ương."
                ),
            },
        ]

    async def retrieve_documents(
        self,
        query: str,
        strategy: RetrieverStrategy = RetrieverStrategy.similarity,
        k: int = 4,
        threshold: float = 0.5,
    ) -> List[SourceCitation]:
        """Truy vấn tài liệu theo các chiến lược: similarity, mmr, threshold."""
        query_lower = query.lower()
        scored_docs = []

        for item in self._knowledge_base:
            # Tính điểm tương đồng dựa trên từ khóa khớp và base score
            matches = sum(1 for kw in item["keywords"] if kw in query_lower)
            if matches > 0:
                calculated_score = min(0.98, item["score"] + (matches * 0.04))
            else:
                calculated_score = max(0.2, item["score"] * 0.5)

            scored_docs.append({
                "source": item["source"],
                "page": item["page"],
                "score": calculated_score,
                "snippet": item["content"],
                "content": item["content"],
            })

        # Sắp xếp theo điểm tương đồng giảm dần
        scored_docs.sort(key=lambda x: x["score"], reverse=True)

        selected_docs: List[Dict[str, Any]] = []

        if strategy == RetrieverStrategy.similarity:
            # Lấy top k điểm cao nhất
            selected_docs = scored_docs[:k]

        elif strategy == RetrieverStrategy.threshold:
            # Lọc theo ngưỡng tối thiểu
            selected_docs = [doc for doc in scored_docs if doc["score"] >= threshold][:k]
            # Nếu không tìm thấy gì trên ngưỡng, không lấy gì để chống ảo giác
            if not selected_docs and scored_docs and scored_docs[0]["score"] >= 0.4:
                # Nếu câu hỏi quá chung chung, có thể trả về tài liệu có điểm tốt nhất
                pass

        elif strategy == RetrieverStrategy.mmr:
            # Giả lập MMR: vừa ưu tiên điểm cao, vừa loại bỏ các tài liệu có nguồn trùng lặp
            seen_sources = set()
            for doc in scored_docs:
                if doc["source"] not in seen_sources:
                    selected_docs.append(doc)
                    seen_sources.add(doc["source"])
                if len(selected_docs) >= k:
                    break
            # Nếu chưa đủ k thì bổ sung thêm
            if len(selected_docs) < k:
                for doc in scored_docs:
                    if doc not in selected_docs:
                        selected_docs.append(doc)
                    if len(selected_docs) >= k:
                        break

        # Chuyển đổi sang SourceCitation
        return [
            SourceCitation(
                source=d["source"],
                page=d["page"],
                score=d["score"],
                snippet=d["snippet"],
            )
            for d in selected_docs
        ]

    async def stream_answer(
        self,
        query: str,
        history: List[ChatMessage],
        citations: List[SourceCitation],
    ) -> AsyncIterator[str]:
        """
        Sinh câu trả lời từng token (từ/chữ) theo thời gian thực (giả lập LLM generation).
        Tích hợp tri thức từ citations và ngữ cảnh lịch sử hội thoại.
        """
        if not citations:
            answer_text = (
                f"Xin lỗi, tôi đã tra cứu trong toàn bộ cơ sở dữ liệu nội bộ nhưng không tìm thấy "
                f"tài liệu nào phù hợp với yêu cầu: '{query}'. "
                f"Vui lòng thử điều chỉnh lại từ khóa hoặc hạ thấp ngưỡng điểm lọc (threshold)."
            )
        else:
            # Tổng hợp câu trả lời dựa trên tài liệu tìm được
            best_snippet = citations[0].snippet
            source_names = ", ".join({f"'{c.source}'" for c in citations[:2]})
            
            # Kiểm tra ngữ cảnh lịch sử
            has_history = len([m for m in history if m.role == "user"]) > 1
            history_context = "Tiếp nối trao đổi trước đó, " if has_history else ""

            answer_text = (
                f"{history_context}Dựa trên tài liệu nội bộ ({source_names}), câu trả lời cho câu hỏi "
                f"'{query}' của bạn như sau:\n\n"
                f"{best_snippet}\n\n"
                f"Nếu cần thêm thông tin chi tiết hoặc hỗ trợ về thủ tục liên quan, "
                f"bạn có thể tham khảo mục tài liệu trích dẫn bên dưới hoặc liên hệ bộ phận phụ trách."
            )

        # Tách thành từng từ/cụm từ để stream mượt mà như ChatGPT
        words = answer_text.split(" ")
        for i, word in enumerate(words):
            # Thêm dấu cách sau mỗi từ trừ từ cuối cùng
            token = word + (" " if i < len(words) - 1 else "")
            yield token
            # Giả lập độ trễ token stream mượt mà (khoảng 20ms - 40ms)
            await asyncio.sleep(0.025)


# Global RAG Core registry
_current_rag_core: BaseRAGCore = DefaultInternalRAGCore()


def get_rag_core() -> BaseRAGCore:
    """Lấy instance RAG Core hiện đang được đăng ký."""
    global _current_rag_core
    return _current_rag_core


def register_rag_core(rag_core: BaseRAGCore) -> None:
    """Cho phép Dev 1 đăng ký RAG Core thực tế của họ vào hệ thống."""
    global _current_rag_core
    _current_rag_core = rag_core

