"""
tests_rag/test_output_parser.py
Kiểm thử cho Output Parser và các hàm định dạng SSE.
"""

from services.output_parser import RAGOutputParser, SourceCitation


class MockDoc:
    def __init__(self, content, metadata):
        self.page_content = content
        self.metadata = metadata


def test_parse_documents_from_objects_and_dicts():
    mock_langchain_doc = MockDoc(
        content="Quy định làm việc từ xa tối đa 2 ngày/tuần.",
        metadata={"source": "Quy_che_wfh.pdf", "page": 3, "score": 0.95},
    )
    dict_doc = {
        "source": "Bao_mat.docx",
        "page": 10,
        "score": 0.88,
        "content": "Bảo mật mật khẩu 12 ký tự.",
    }

    citations = RAGOutputParser.parse_documents([mock_langchain_doc, dict_doc])

    assert len(citations) == 2
    assert citations[0].source == "Quy_che_wfh.pdf"
    assert citations[0].page == 3
    assert citations[0].score == 0.95
    assert "tối đa 2 ngày/tuần" in citations[0].snippet

    assert citations[1].source == "Bao_mat.docx"
    assert citations[1].page == 10
    assert citations[1].score == 0.88


def test_extract_citations_from_text():
    text_with_sources = (
        "Nhân viên được phép làm việc từ xa 2 ngày mỗi tuần.\n\n"
        "Nguồn tham khảo:\n"
        "- Quy_che_wfh_2024.pdf (trang 5)\n"
        "- Huong_dan_nhan_su.docx [trang 12]"
    )

    clean_answer, citations = RAGOutputParser._extract_citations_from_text(text_with_sources)

    assert "Nguồn tham khảo:" not in clean_answer
    assert "Nhân viên được phép làm việc từ xa" in clean_answer
    assert len(citations) == 2
    assert citations[0].page == 5
    assert citations[1].page == 12


def test_sse_event_formatters():
    token_evt = RAGOutputParser.format_token_event("Xin ")
    assert token_evt["type"] == "token"
    assert token_evt["content"] == "Xin "

    citation = SourceCitation(source="test.pdf", page=1, score=0.9, snippet="test snippet")
    cit_evt = RAGOutputParser.format_citations_event([citation])
    assert cit_evt["type"] == "citations"
    assert len(cit_evt["citations"]) == 1
    assert cit_evt["citations"][0]["source"] == "test.pdf"

    done_evt = RAGOutputParser.format_done_event(session_id="s1", total_tokens=15, strategy_used="similarity")
    assert done_evt["type"] == "done"
    assert done_evt["session_id"] == "s1"
    assert done_evt["total_tokens"] == 15
    assert done_evt["strategy"] == "similarity"

    err_evt = RAGOutputParser.format_error_event("Timeout", error_code="ERR_TIMEOUT")
    assert err_evt["type"] == "error"
    assert err_evt["message"] == "Timeout"
    assert err_evt["code"] == "ERR_TIMEOUT"

