
"""
rag/engine.py
RAG Engine hoàn chỉnh thực thi BaseRAGCore theo mô hình Clean Architecture.
Kết nối ChromaVectorStoreManager, AdvancedRetriever, AntiHallucinationPrompt và OutputParser.
Hỗ trợ cả Streaming Response qua Server-Sent Events (SSE) và Synchronous Generation.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from core.config import get_settings
from core.exceptions import LLMGenerationError
from documents.loader_factory import Document
from rag.prompts import AntiHallucinationPrompt
from rag.retriever import AdvancedRetriever
from rag.vector_store import ChromaVectorStoreManager, get_vector_store_manager
try:
    from services.memory import ChatMessage
    from services.output_parser import RAGOutputParser, RAGParsedOutput, SourceCitation
    from services.rag_core_interface import BaseRAGCore, RetrieverStrategy
except ImportError:
    class ChatMessage:  # type: ignore
        def __init__(self, role: str, content: str, timestamp: float = 0.0, citations: Optional[List[Any]] = None):
            self.role = role
            self.content = content
            self.timestamp = timestamp
            self.citations = citations or []

    class SourceCitation:  # type: ignore
        def __init__(self, source: str, page: Optional[Union[int, str]] = None, score: Optional[float] = None, snippet: str = "", url: Optional[str] = None):
            self.source = source
            self.page = page
            self.score = score
            self.snippet = snippet
            self.url = url

        def to_dict(self) -> Dict[str, Any]:
            return {
                "source": self.source,
                "page": self.page,
                "score": round(self.score, 4) if self.score is not None else None,
                "snippet": self.snippet.strip(),
                "url": self.url,
            }

    class RAGParsedOutput:  # type: ignore
        def __init__(self, answer: str, citations: Optional[List[Any]] = None):
            self.answer = answer
            self.citations = citations or []

    class RAGOutputParser:  # type: ignore
        @classmethod
        def parse_documents(cls, raw_docs: List[Any]) -> List[SourceCitation]:
            citations: List[SourceCitation] = []
            for doc in raw_docs:
                meta = getattr(doc, "metadata", {}) or {}
                source = meta.get("file_name") or meta.get("source") or "Tài liệu nội bộ"
                page = meta.get("page")
                score = meta.get("score")
                snippet = getattr(doc, "page_content", str(doc)).strip()
                citations.append(SourceCitation(source=source, page=page, score=score, snippet=snippet))
            return citations

    from abc import ABC, abstractmethod
    class BaseRAGCore(ABC):  # type: ignore
        @abstractmethod
        async def retrieve_documents(self, query: str, strategy: Any = "similarity", k: int = 4, threshold: float = 0.5) -> List[SourceCitation]:
            pass

        @abstractmethod
        async def stream_answer(self, query: str, history: List[Any], citations: List[SourceCitation]) -> AsyncIterator[str]:
            pass

    from enum import Enum
    class RetrieverStrategy(str, Enum):  # type: ignore
        similarity = "similarity"
        mmr = "mmr"
        threshold = "threshold"
        similarity_score_threshold = "similarity_score_threshold"
        hybrid = "hybrid"

logger = logging.getLogger("rag_engine")


class CleanRAGCore(BaseRAGCore):
    """
    RAG Core hoàn chỉnh kế thừa BaseRAGCore của hệ thống.
    Thay thế DefaultInternalRAGCore bằng hệ thống truy vấn Vector Database ChromaDB thực tế,
    Advanced Retriever với 3 chiến lược và Prompt Chống ảo giác (Anti-Hallucination).
    """

    def __init__(
        self,
        retriever: Optional[AdvancedRetriever] = None,
        vector_store_manager: Optional[ChromaVectorStoreManager] = None,
    ) -> None:
        self.vector_store_manager = vector_store_manager or get_vector_store_manager()
        self.retriever = retriever or AdvancedRetriever(self.vector_store_manager)
        self.settings = get_settings()

    async def retrieve_documents(
        self,
        query: str,
        strategy: Union[RetrieverStrategy, str] = RetrieverStrategy.similarity,
        k: int = 4,
        threshold: float = 0.5,
    ) -> List[SourceCitation]:
        """
        Truy vấn tài liệu từ Vector Store theo chiến lược đã chọn và chuẩn hóa sang SourceCitation.

        :param query: Câu hỏi người dùng.
        :param strategy: 'similarity' | 'mmr' | 'similarity_score_threshold'.
        :param k: Số lượng tài liệu cần lấy.
        :param threshold: Ngưỡng lọc độ tương đồng.
        :return: Danh sách SourceCitation minh bạch nguồn gốc.
        """
        # Chuyển đổi RetrieverStrategy enum sang chuỗi nếu cần
        strat_str = strategy.value if isinstance(strategy, RetrieverStrategy) else str(strategy)

        # Chạy truy vấn bất đồng bộ qua threadpool để tránh chặn event loop
        loop = asyncio.get_running_loop()
        docs: List[Document] = await loop.run_in_executor(
            None,
            lambda: self.retriever.retrieve(
                query=query,
                strategy=strat_str,
                k=k,
                threshold=threshold,
            ),
        )

        # Chuyển đổi sang chuẩn SourceCitation qua RAGOutputParser
        citations = RAGOutputParser.parse_documents(docs)
        logger.info(
            "Truy vấn RAG ('%s', chiến lược=%s): tìm thấy %d tài liệu trích dẫn.",
            query[:30],
            strat_str,
            len(citations),
        )
        return citations

    async def stream_answer(
        self,
        query: str,
        history: List[ChatMessage],
        citations: List[SourceCitation],
    ) -> AsyncIterator[str]:
        """
        Sinh câu trả lời từng token (streaming) qua LLM thực tế.
        - Nếu không có trích dẫn: Từ chối lịch sự và dứt khoát.
        - Nếu có trích dẫn: Gửi context + câu hỏi đến LLM, stream phản hồi.
        - Fallback về mock nếu LLM không khả dụng.
        """
        # 1. Kiểm tra Anti-Hallucination: Từ chối lịch sự nếu không có tài liệu nào
        if not citations:
            refusal_message = (
                "Xin lỗi, tôi đã kiểm tra kỹ toàn bộ cơ sở dữ liệu tài liệu nội bộ "
                f"nhưng không tìm thấy thông tin phù hợp với câu hỏi: '{query}'. "
                "Vui lòng thử điều chỉnh lại từ khóa tìm kiếm hoặc liên hệ bộ phận liên quan để được hỗ trợ chính xác nhất."
            )
            words = refusal_message.split(" ")
            for idx, word in enumerate(words):
                yield word + (" " if idx < len(words) - 1 else "")
                await asyncio.sleep(0.02)
            return

        # 2. Xây dựng messages cho LLM từ Anti-Hallucination Prompt
        # Chuyển SourceCitation sang dạng Document-like để format_context xử lý
        citation_dicts = [
            {
                "source": c.source,
                "page": c.page,
                "snippet": c.snippet,
            }
            for c in citations
        ]

        # Chuyển chat_history sang dạng dict
        history_dicts = []
        for msg in history:
            if msg.content:
                history_dicts.append({"role": msg.role, "content": msg.content})

        messages = AntiHallucinationPrompt.build_prompt_messages(
            question=query,
            documents=citation_dicts,
            chat_history=history_dicts[-6:],  # Giới hạn 3 cặp hội thoại gần nhất
        )

        # 3. Gọi LLM thực tế (OpenAI-compatible API)
        settings = self.settings
        api_key = settings.OPENAI_API_KEY
        base_url = getattr(settings, "OPENAI_BASE_URL", None)
        llm_provider = settings.LLM_PROVIDER

        if api_key and llm_provider != "mock":
            try:
                from openai import AsyncOpenAI

                client_kwargs = {"api_key": api_key}
                if base_url:
                    client_kwargs["base_url"] = base_url

                client = AsyncOpenAI(**client_kwargs)

                stream = await client.chat.completions.create(
                    model=settings.LLM_MODEL_NAME,
                    messages=messages,
                    temperature=settings.LLM_TEMPERATURE,
                    max_tokens=1024,
                    stream=True,
                )

                async for chunk in stream:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta and delta.content:
                        yield delta.content

                return  # Thành công, không cần fallback

            except Exception as e:
                logger.warning(
                    "LLM streaming thất bại (%s), fallback về mock response: %s",
                    llm_provider,
                    str(e)[:200],
                )

        # 4. Fallback: Mock response (khi không có API key hoặc LLM lỗi)
        from pathlib import Path
        top_citations = citations[:3]
        clean_sources = list({Path(c.source).name for c in top_citations})
        sources_str = ", ".join(f"'{name}'" for name in clean_sources)

        main_content = top_citations[0].snippet.strip()

        full_answer = (
            f"Dựa trên tài liệu nội bộ ({sources_str}), "
            f"câu trả lời cho câu hỏi '{query}' của bạn như sau:\n\n"
            f"{main_content}\n\n"
            f"*(Chi tiết căn cứ và số trang được trích xuất ở mục tài liệu tham khảo bên dưới)*"
        )

        words = full_answer.split(" ")
        for idx, word in enumerate(words):
            yield word + (" " if idx < len(words) - 1 else "")
            await asyncio.sleep(0.02)

    async def generate_answer(
        self,
        query: str,
        history: Optional[List[ChatMessage]] = None,
        strategy: Union[RetrieverStrategy, str] = RetrieverStrategy.similarity,
        k: int = 4,
        threshold: float = 0.5,
    ) -> RAGParsedOutput:
        """
        Phương thức đồng bộ hóa (Non-streaming) trả về toàn bộ câu trả lời và trích dẫn cùng lúc.
        """
        chat_hist = history or []
        citations = await self.retrieve_documents(query=query, strategy=strategy, k=k, threshold=threshold)

        tokens = []
        async for token in self.stream_answer(query=query, history=chat_hist, citations=citations):
            tokens.append(token)

        complete_answer = "".join(tokens)
        return RAGParsedOutput(answer=complete_answer, citations=citations)
