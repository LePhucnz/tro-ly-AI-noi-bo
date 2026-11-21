"""
app.py
Giao diện Streamlit Frontend cho hệ thống Trợ lý AI Nội bộ & Sàng lọc CV Tuyển dụng:
1. Chức năng 1: Hỏi - Đáp Tri thức Nội bộ phong cách Google Gemini (RAG, SSE Streaming, Citations).
2. Chức năng 2: Sàng lọc CV Tuyển dụng (HR Dashboard: Batch Upload, Ranking, Anti-Hallucination, Interview Questions).
"""

from __future__ import annotations

import datetime
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx
import streamlit as st

# ==============================================================================
# 1. Cấu hình Trang Streamlit & Theme
# ==============================================================================
st.set_page_config(
    page_title="Trợ lý AI Nội bộ & Sàng lọc CV",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS tạo hiệu ứng Google Gemini & HR Dashboard
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Google Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Container chính căn giữa gọn gàng */
.main .block-container {
    max-width: 950px;
    padding-top: 1.8rem;
    padding-bottom: 6rem;
}

/* Gemini Hero Greeting */
.gemini-hero-container {
    padding: 2.0rem 0 1.2rem 0;
    text-align: left;
}

.gemini-sparkle-icon {
    font-size: 2.2rem;
    background: linear-gradient(135deg, #1BA1E2 0%, #4B68FF 30%, #8A2BE2 60%, #FF5A8D 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-block;
    margin-bottom: 0.5rem;
    animation: pulse 3s infinite;
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.08); }
    100% { transform: scale(1); }
}

.gemini-greeting {
    font-size: 2.6rem;
    font-weight: 700;
    line-height: 1.2;
    background: linear-gradient(90deg, #4285F4 0%, #9B72CB 40%, #D96570 80%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}

.gemini-subgreeting {
    font-size: 1.3rem;
    font-weight: 500;
    color: #5f6368;
    margin-top: 0.4rem;
    margin-bottom: 1.8rem;
}

/* Suggestion Prompt Cards */
.gemini-card {
    background-color: #F8FAFD;
    border: 1px solid #E1E8F5;
    border-radius: 16px;
    padding: 16px 20px;
    cursor: pointer;
    transition: all 0.25s ease;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.gemini-card:hover {
    background-color: #F0F4F9;
    border-color: #D3E3FD;
    box-shadow: 0 4px 12px rgba(66, 133, 244, 0.08);
    transform: translateY(-2px);
}

.gemini-card-title {
    font-weight: 600;
    font-size: 1.05rem;
    color: #1f1f1f;
    margin-bottom: 6px;
}

.gemini-card-desc {
    font-size: 0.88rem;
    color: #5f6368;
    line-height: 1.4;
}

.gemini-card-icon {
    font-size: 1.3rem;
    align-self: flex-end;
    margin-top: 10px;
    opacity: 0.85;
}

/* Chips & Badges */
.grounding-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background-color: #F0F4F9;
    border: 1px solid #D3E3FD;
    color: #0B57D0;
    border-radius: 9999px;
    padding: 4px 12px;
    font-size: 0.78rem;
    font-weight: 500;
    margin-right: 6px;
    margin-top: 6px;
}

.citation-card {
    background-color: #F8FAFC;
    border-left: 3.5px solid #4285F4;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 8px 0;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 600;
}
.status-online {
    background-color: #DEF7EC;
    color: #03543F;
}
.status-offline {
    background-color: #FDE8E8;
    color: #9B1C1C;
}

/* CV Screening Specific Badges */
.badge-strong {
    background-color: #DEF7EC;
    color: #03543F;
    padding: 3px 10px;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.82rem;
}
.badge-potential {
    background-color: #E1EFFE;
    color: #1E429F;
    padding: 3px 10px;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.82rem;
}
.badge-borderline {
    background-color: #FEF08A;
    color: #713F12;
    padding: 3px 10px;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.82rem;
}
.badge-notfit {
    background-color: #FDE8E8;
    color: #9B1C1C;
    padding: 3px 10px;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.82rem;
}

.cv-summary-box {
    background-color: #F8FAFC;
    border-left: 4px solid #10B981;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 12px 0;
}

.evidence-box {
    background-color: #EFF6FF;
    border-left: 3px solid #3B82F6;
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 0.88rem;
    color: #1E3A8A;
    margin-top: 4px;
}

.missing-box {
    background-color: #FEF2F2;
    border-left: 3px solid #EF4444;
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 0.88rem;
    color: #991B1B;
    margin-top: 4px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# 2. Quản lý Trạng thái Phiên (Session State Management)
# ==============================================================================
if "session_id" not in st.session_state:
    st.session_state.session_id = f"gemini-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_url" not in st.session_state:
    st.session_state.api_url = "http://127.0.0.1:8000"

if "cv_batch_results" not in st.session_state:
    st.session_state.cv_batch_results = None

if "cv_active_tab" not in st.session_state:
    st.session_state.cv_active_tab = "batch"


def create_new_chat():
    """Tạo một phiên hội thoại mới tinh."""
    st.session_state.session_id = f"gemini-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
    st.session_state.messages = []


def check_api_health(base_url: str) -> tuple[bool, str]:
    """Kiểm tra tình trạng kết nối tới FastAPI Backend."""
    urls_to_try = [base_url]
    if "localhost" in base_url:
        urls_to_try.append(base_url.replace("localhost", "127.0.0.1"))
    elif "127.0.0.1" in base_url:
        urls_to_try.append(base_url.replace("127.0.0.1", "localhost"))

    for u in urls_to_try:
        try:
            with httpx.Client(timeout=2.5) as client:
                resp = client.get(f"{u}/")
                if resp.status_code == 200:
                    st.session_state.api_url = u
                    return True, "Hệ thống Backend Online"
        except Exception:
            continue
    return False, "Không kết nối được Backend"


def fetch_vector_stats(base_url: str) -> Optional[Dict[str, Any]]:
    """Lấy thông số thống kê từ ChromaDB Vector Store."""
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(f"{base_url}/api/v1/documents/stats")
            if resp.status_code == 200:
                return resp.json()
    except Exception:
        pass
    return None


def fetch_sample_criteria(base_url: str) -> List[Dict[str, Any]]:
    """Lấy danh sách các bộ tiêu chí tuyển dụng mẫu."""
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(f"{base_url}/api/v1/cv/sample-criteria")
            if resp.status_code == 200:
                return resp.json()
    except Exception:
        pass
    return []


# ==============================================================================
# 3. SIDEBAR: Điều hướng Chức năng & Cấu hình
# ==============================================================================
with st.sidebar:
    # Logo & Header
    st.markdown(
        """
        <div style='display: flex; align-items: center; gap: 8px; margin-bottom: 0.8rem;'>
            <span style='font-size: 1.6rem; color: #4285F4;'>✦</span>
            <span style='font-size: 1.35rem; font-weight: 700; color: #1f1f1f;'>Trợ lý AI Doanh nghiệp</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Lựa chọn Chức năng Chính
    app_mode = st.radio(
        "Chức năng hoạt động:",
        options=["💬 Hỏi - Đáp Tri thức Nội bộ", "📋 Sàng lọc CV Tuyển dụng (HR)"],
        index=0,
    )

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # Kiểm tra trạng thái Backend
    is_online, status_msg = check_api_health(st.session_state.api_url)
    if is_online:
        st.markdown(f"<span class='status-pill status-online'>🟢 {status_msg}</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"<span class='status-pill status-offline'>🔴 {status_msg}</span>", unsafe_allow_html=True)

    st.divider()

    # ----------------------------------------------------
    # Sidebar cho Chức năng 1: Hỏi đáp Tri thức Nội bộ
    # ----------------------------------------------------
    if app_mode == "💬 Hỏi - Đáp Tri thức Nội bộ":
        if st.button("➕ Cuộc trò chuyện mới", use_container_width=True, type="primary"):
            create_new_chat()
            st.rerun()

        stats = fetch_vector_stats(st.session_state.api_url) if is_online else None
        if stats:
            st.caption(f"📚 Vector DB: **{stats.get('total_chunks_count', 0)} chunks** | Model: `{stats.get('embedding_provider')}`")

        # Quản lý Session ID
        with st.expander("🔑 Quản lý Phiên (Session)", expanded=False):
            current_sess = st.text_input(
                "Session ID hiện tại:",
                value=st.session_state.session_id,
            )
            if current_sess != st.session_state.session_id:
                st.session_state.session_id = current_sess
                st.session_state.messages = []
                st.rerun()

            if st.button("🗑️ Xóa ngữ cảnh phiên này", use_container_width=True):
                try:
                    with httpx.Client(timeout=3.0) as client:
                        client.delete(f"{st.session_state.api_url}/api/v1/sessions/{st.session_state.session_id}")
                except Exception:
                    pass
                st.session_state.messages = []
                st.success("Đã xóa bộ nhớ phiên!")
                st.rerun()

        # Cấu hình RAG Retriever
        with st.expander("⚙️ Cấu hình RAG Retriever", expanded=True):
            strategy_labels = {
                "similarity": "🎯 Similarity (Tương đồng Cosine)",
                "hybrid": "⚡ Hybrid (Vector + BM25 Từ khóa)",
                "mmr": "⚖️ MMR (Đa dạng hóa tài liệu)",
                "similarity_score_threshold": "🛡️ Threshold (Lọc ngưỡng an toàn)",
            }
            selected_strategy = st.selectbox(
                "Chiến lược tìm kiếm:",
                options=list(strategy_labels.keys()),
                format_func=lambda x: strategy_labels[x],
                index=0,
            )

            top_k = st.slider(
                "Số tài liệu trích xuất (Top-K):",
                min_value=1,
                max_value=10,
                value=6,
            )

            threshold_val = 0.35
            if selected_strategy in ("similarity_score_threshold", "threshold"):
                threshold_val = st.slider(
                    "Ngưỡng tin cậy (Threshold):",
                    min_value=0.1,
                    max_value=0.8,
                    value=0.35,
                    step=0.05,
                )

        # Nạp thêm tài liệu vào ChromaDB
        with st.expander("📤 Nạp thêm Tài liệu vào Vector DB", expanded=False):
            uploaded_doc = st.file_uploader(
                "Chọn tệp (.pdf, .docx, .txt, .csv, .md):",
                type=["pdf", "docx", "txt", "csv", "md"],
            )
            if uploaded_doc is not None:
                if st.button("🚀 Nạp vào Cơ sở Tri thức", use_container_width=True):
                    with st.spinner("Đang nạp tài liệu qua Ingestion Pipeline..."):
                        try:
                            files = {"file": (uploaded_doc.name, uploaded_doc.getvalue(), uploaded_doc.type or "application/octet-stream")}
                            with httpx.Client(timeout=60.0) as client:
                                resp = client.post(f"{st.session_state.api_url}/api/v1/documents/upload", files=files)
                                if resp.status_code == 200:
                                    res_data = resp.json()
                                    if res_data.get("is_duplicate"):
                                        st.info(f"ℹ️ {res_data.get('message')}")
                                    else:
                                        st.success(f"✅ Đã nạp thành công! Thêm {res_data.get('new_chunks_added')} chunks mới.")
                                    st.rerun()
                                else:
                                    st.error(f"Lỗi: {resp.status_code} - {resp.text}")
                        except Exception as e:
                            st.error(f"Không thể tải lên: {e}")

    # ----------------------------------------------------
    # Sidebar cho Chức năng 2: Sàng lọc CV Tuyển dụng (HR)
    # ----------------------------------------------------
    else:
        st.markdown("### 🎯 Tiêu chí Tuyển dụng")
        sample_crits = fetch_sample_criteria(st.session_state.api_url) if is_online else []

        sample_options = ["Tùy chỉnh riêng..."] + [c["job_title"] for c in sample_crits]
        chosen_sample = st.selectbox("Chọn vị trí mẫu hoặc tự nhập:", sample_options, index=1 if sample_crits else 0)

        # Điền dữ liệu mặc định dựa theo vị trí mẫu
        matched_sample = next((c for c in sample_crits if c["job_title"] == chosen_sample), None)

        default_title = matched_sample["job_title"] if matched_sample else "Senior Backend & AI Engineer"
        default_exp = float(matched_sample.get("min_total_experience_years") or 3.0) if matched_sample else 3.0
        default_req = ", ".join(matched_sample.get("required_skills", [])) if matched_sample else "Python, FastAPI, PostgreSQL, Docker"
        default_pref = ", ".join(matched_sample.get("preferred_skills", [])) if matched_sample else "LangChain, RAG, Redis, Kubernetes"
        default_pass = float(matched_sample.get("passing_score", 70.0)) if matched_sample else 70.0

        job_title_input = st.text_input("Vị trí tuyển dụng:", value=default_title)
        min_exp_input = st.number_input("Số năm kinh nghiệm tối thiểu:", min_value=0.0, max_value=20.0, value=default_exp, step=0.5)
        req_skills_input = st.text_area("Kỹ năng bắt buộc (Required - cách nhau dấu phẩy):", value=default_req, height=80)
        pref_skills_input = st.text_area("Kỹ năng ưu tiên (Preferred - cách nhau dấu phẩy):", value=default_pref, height=80)
        pass_score_input = st.slider("Ngưỡng điểm đạt (Thang 100):", min_value=50.0, max_value=90.0, value=default_pass, step=5.0)

        # Lưu tiêu chí vào session state
        st.session_state.current_criteria = {
            "job_title": job_title_input.strip() or "Vị trí tuyển dụng",
            "min_total_experience_years": min_exp_input,
            "required_skills": [s.strip() for s in req_skills_input.split(",") if s.strip()],
            "preferred_skills": [s.strip() for s in pref_skills_input.split(",") if s.strip()],
            "passing_score": pass_score_input,
        }


# ==============================================================================
# 4. GIAO DIỆN CHÍNH (MAIN VIEW)
# ==============================================================================

# ==============================================================================
# TAB 1: GIAO DIỆN HỎI ĐÁP NỘI BỘ (RAG CHAT)
# ==============================================================================
if app_mode == "💬 Hỏi - Đáp Tri thức Nội bộ":
    # Màn hình chào mừng phong cách Gemini nếu chưa có tin nhắn
    if not st.session_state.messages:
        st.markdown(
            """
            <div class='gemini-hero-container'>
                <div class='gemini-sparkle-icon'>✦</div>
                <h1 class='gemini-greeting'>Xin chào!</h1>
                <div class='gemini-subgreeting'>Tôi có thể giúp bạn tra cứu quy định hoặc tài liệu nào hôm nay?</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        cards = [
            {
                "title": "Làm việc từ xa (WFH)",
                "desc": "Quy định số ngày WFH tối đa và quy trình đăng ký?",
                "icon": "🏠",
                "prompt": "Quy định làm việc từ xa (WFH) của công ty như thế nào? Được WFH tối đa mấy ngày?",
            },
            {
                "title": "Bảo mật & Thiết bị",
                "desc": "Chính sách mật khẩu, VPN và tiêu chuẩn an toàn?",
                "icon": "🔐",
                "prompt": "Chính sách bảo mật thông tin và tiêu chuẩn mật khẩu của công ty là gì?",
            },
            {
                "title": "Nghỉ phép & Chế độ",
                "desc": "Số ngày phép năm, bảo hiểm y tế nâng cao?",
                "icon": "🏖️",
                "prompt": "Chế độ nghỉ phép năm và phúc lợi bảo hiểm cho nhân viên như thế nào?",
            },
            {
                "title": "Quy trình Kỹ thuật",
                "desc": "Quy định commit, review Pull Request và CI/CD?",
                "icon": "💻",
                "prompt": "Quy trình triển khai kỹ thuật và review Pull Request quy định thế nào?",
            },
        ]

        col1, col2 = st.columns(2)
        for idx, card in enumerate(cards):
            target_col = col1 if idx % 2 == 0 else col2
            with target_col:
                st.markdown(
                    f"""
                    <div class='gemini-card'>
                        <div>
                            <div class='gemini-card-title'>{card['title']}</div>
                            <div class='gemini-card-desc'>{card['desc']}</div>
                        </div>
                        <div class='gemini-card-icon'>{card['icon']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(f"Hỏi về: {card['title']}", key=f"btn_card_{idx}", use_container_width=True):
                    st.session_state.pending_prompt = card["prompt"]
                    st.rerun()

        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)

    # Hiển thị lịch sử tin nhắn
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])
                citations = msg.get("citations", [])
                if citations:
                    chip_html = "<div style='margin-top: 10px; margin-bottom: 4px;'>"
                    for c in citations[:4]:
                        raw_source = c.get("source", "Tài liệu")
                        source_name = Path(raw_source).name if ("/" in raw_source or "\\" in raw_source) else raw_source
                        score = c.get("score")
                        score_str = f" • {int(score * 100)}%" if score is not None else ""
                        chip_html += f"<span class='grounding-chip'>📄 {source_name}{score_str}</span>"
                    chip_html += "</div>"
                    st.markdown(chip_html, unsafe_allow_html=True)

                    with st.expander(f"🔍 Xem {len(citations)} trích dẫn nguồn chi tiết", expanded=False):
                        for idx, cit in enumerate(citations, 1):
                            raw_src = cit.get("source", "Tài liệu nội bộ")
                            source = Path(raw_src).name if ("/" in raw_src or "\\" in raw_src) else raw_src
                            page = cit.get("page")
                            page_str = f" • Trang/Dòng: {page}" if page else ""
                            score = cit.get("score")
                            score_str = f" • Điểm khớp: {score:.2f}" if score is not None else ""
                            snippet = cit.get("snippet", "")
                            st.markdown(
                                f"""
                                <div class='citation-card'>
                                    <div style='font-weight: 600; color: #1E40AF; font-size: 0.9rem;'>[{idx}] {source} {page_str} {score_str}</div>
                                    <div style='font-size: 0.88rem; color: #374151; margin-top: 4px; line-height: 1.4;'>{snippet}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

    # Chat Input & Streaming SSE
    user_input = st.chat_input("Hỏi bất kỳ điều gì về quy định, chính sách, tài liệu nội bộ...")
    if hasattr(st.session_state, "pending_prompt") and st.session_state.pending_prompt:
        user_input = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_text = ""
            citations_list = []
            has_error = False

            response_placeholder.markdown("🔍 *Đang tra cứu cơ sở tri thức nội bộ...*")

            try:
                api_endpoint = f"{st.session_state.api_url}/api/v1/chat/stream"
                payload = {
                    "session_id": st.session_state.session_id,
                    "query": user_input,
                    "strategy": selected_strategy,
                    "k": top_k,
                    "threshold": threshold_val,
                }

                with httpx.Client(timeout=60.0) as client:
                    with client.stream(
                        "POST",
                        api_endpoint,
                        json=payload,
                        headers={"Accept": "text/event-stream"},
                    ) as response:
                        if response.status_code != 200:
                            has_error = True
                            err_content = response.read().decode("utf-8", errors="replace")
                            st.error(f"Lỗi phản hồi Backend: HTTP {response.status_code} - {err_content}")
                        else:
                            current_event = None
                            for line in response.iter_lines():
                                if not line:
                                    continue
                                line = line.strip()

                                if line.startswith("event:"):
                                    current_event = line[len("event:"):].strip()
                                    continue

                                if line.startswith("data:"):
                                    raw_json = line[len("data:"):].strip()
                                    try:
                                        data_obj = json.loads(raw_json)
                                    except json.JSONDecodeError:
                                        continue

                                    evt_type = data_obj.get("type") or current_event

                                    if evt_type == "token":
                                        token = data_obj.get("content", "")
                                        full_text += token
                                        response_placeholder.markdown(full_text + " ▌")

                                    elif evt_type == "citations":
                                        citations_list = data_obj.get("citations", [])

                                    elif evt_type == "done":
                                        response_placeholder.markdown(full_text)

                                    elif evt_type == "error":
                                        has_error = True
                                        err_msg = data_obj.get("message", "Đã có lỗi phát sinh.")
                                        st.error(f"⚠️ {err_msg}")
                                        break

                if not has_error:
                    response_placeholder.markdown(full_text)

                    if citations_list:
                        chip_html = "<div style='margin-top: 10px; margin-bottom: 4px;'>"
                        for c in citations_list[:4]:
                            raw_source = c.get("source", "Tài liệu")
                            source_name = Path(raw_source).name if ("/" in raw_source or "\\" in raw_source) else raw_source
                            score = c.get("score")
                            score_str = f" • {int(score * 100)}%" if score is not None else ""
                            chip_html += f"<span class='grounding-chip'>📄 {source_name}{score_str}</span>"
                        chip_html += "</div>"
                        st.markdown(chip_html, unsafe_allow_html=True)

                        with st.expander(f"🔍 Xem {len(citations_list)} trích dẫn nguồn chi tiết", expanded=False):
                            for idx, cit in enumerate(citations_list, 1):
                                raw_src = cit.get("source", "Tài liệu nội bộ")
                                source = Path(raw_src).name if ("/" in raw_src or "\\" in raw_src) else raw_src
                                page = cit.get("page")
                                page_str = f" • Trang/Dòng: {page}" if page else ""
                                score = cit.get("score")
                                score_str = f" • Điểm khớp: {score:.2f}" if score is not None else ""
                                snippet = cit.get("snippet", "")
                                st.markdown(
                                    f"""
                                    <div class='citation-card'>
                                        <div style='font-weight: 600; color: #1E40AF; font-size: 0.9rem;'>[{idx}] {source} {page_str} {score_str}</div>
                                        <div style='font-size: 0.88rem; color: #374151; margin-top: 4px; line-height: 1.4;'>{snippet}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": full_text,
                            "citations": citations_list,
                        }
                    )

            except httpx.ConnectError:
                response_placeholder.empty()
                st.error("❌ Không thể kết nối đến Backend tại " + st.session_state.api_url)
            except Exception as exc:
                response_placeholder.empty()
                st.error(f"❌ Lỗi: {str(exc)}")


# ==============================================================================
# TAB 2: GIAO DIỆN SÀNG LỌC CV TUYỂN DỤNG (HR DASHBOARD)
# ==============================================================================
else:
    st.markdown(
        """
        <div style='margin-bottom: 1.5rem;'>
            <h1 style='font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.3rem;'>
                📋 Sàng lọc Hồ sơ CV Tuyển dụng
            </h1>
            <div style='color: #4B5563; font-size: 1.05rem;'>
                Trích xuất hồ sơ, đối chiếu tiêu chí có trọng số, kiểm soát <b>chống ảo giác</b> và sinh câu hỏi phỏng vấn cho HR.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    criteria_data = st.session_state.get("current_criteria", {})
    st.info(
        f"🎯 Đang áp dụng tiêu chí cho vị trí: **{criteria_data.get('job_title')}** | "
        f"Kinh nghiệm: **{criteria_data.get('min_total_experience_years')} năm** | "
        f"Kỹ năng bắt buộc: `{', '.join(criteria_data.get('required_skills', []))}`"
    )

    tab_upload, tab_paste = st.tabs(["📁 Tải lên nhiều tệp CV (Batch)", "✍️ Dán nội dung văn bản CV"])

    # 1. Tab Tải tệp CV
    with tab_upload:
        uploaded_cvs = st.file_uploader(
            "Tải lên danh sách tệp CV ứng viên (.pdf, .docx, .txt, .md):",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            help="Hệ thống hỗ trợ nạp nhiều CV cùng lúc và tự động chấm điểm xếp hạng ứng viên.",
        )

        if uploaded_cvs:
            st.caption(f"Đã chọn **{len(uploaded_cvs)} tệp CV** sẵn sàng phân tích.")
            if st.button("🚀 Bắt đầu Sàng lọc Hàng loạt", type="primary", use_container_width=True):
                with st.spinner("Đang phân tích, trích xuất và đối chiếu tiêu chí từng hồ sơ..."):
                    try:
                        multipart_files = [
                            ("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
                            for f in uploaded_cvs
                        ]
                        form_data = {"criteria": json.dumps(criteria_data)}

                        with httpx.Client(timeout=120.0) as client:
                            resp = client.post(
                                f"{st.session_state.api_url}/api/v1/cv/screen",
                                data=form_data,
                                files=multipart_files,
                            )
                            if resp.status_code == 200:
                                st.session_state.cv_batch_results = resp.json()
                                st.success(f"✅ Đã sàng lọc hoàn tất {len(uploaded_cvs)} hồ sơ CV!")
                            else:
                                st.error(f"Lỗi từ máy chủ: HTTP {resp.status_code} - {resp.text}")
                    except Exception as e:
                        st.error(f"Lỗi kết nối khi gửi yêu cầu sàng lọc: {e}")

    # 2. Tab Nhập text trực tiếp
    with tab_paste:
        raw_cv_text = st.text_area(
            "Dán nội dung văn bản CV vào đây:",
            placeholder="Dán toàn bộ nội dung CV ứng viên bao gồm kỹ năng, kinh nghiệm, học vấn...",
            height=250,
        )
        candidate_label = st.text_input("Tên tệp/Nhãn ứng viên:", value="CV_Ung_Vien.txt")

        if st.button("🚀 Sàng lọc CV này", use_container_width=True):
            if not raw_cv_text.strip():
                st.warning("Vui lòng nhập nội dung văn bản CV trước khi sàng lọc.")
            else:
                with st.spinner("Đang bóc tách và đánh giá hồ sơ..."):
                    try:
                        payload = {
                            "cv_text": raw_cv_text,
                            "criteria": criteria_data,
                            "file_name": candidate_label,
                        }
                        with httpx.Client(timeout=60.0) as client:
                            resp = client.post(f"{st.session_state.api_url}/api/v1/cv/screen-text", json=payload)
                            if resp.status_code == 200:
                                single_res = resp.json()
                                # Định dạng tương thích batch response
                                st.session_state.cv_batch_results = {
                                    "job_title": criteria_data.get("job_title"),
                                    "total_cvs": 1,
                                    "pass_count": 1 if single_res.get("is_passed") else 0,
                                    "fail_count": 0 if single_res.get("is_passed") else 1,
                                    "average_score": single_res.get("overall_score"),
                                    "top_candidate": single_res.get("candidate_info", {}).get("full_name"),
                                    "results": [single_res],
                                }
                                st.success("✅ Đã sàng lọc hoàn tất hồ sơ!")
                            else:
                                st.error(f"Lỗi: {resp.status_code} - {resp.text}")
                    except Exception as e:
                        st.error(f"Lỗi kết nối: {e}")

    # ----------------------------------------------------
    # BẢNG KẾT QUẢ SÀNG LỌC & XẾP HẠNG ỨNG VIÊN
    # ----------------------------------------------------
    batch_res = st.session_state.get("cv_batch_results")
    if batch_res and batch_res.get("results"):
        st.markdown("---")
        st.markdown("### 📊 Tổng kết & Xếp hạng Ứng viên")

        # 4 Thẻ Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📄 Tổng số CV", batch_res.get("total_cvs", 0))
        m2.metric("✅ Đạt yêu cầu", batch_res.get("pass_count", 0))
        m3.metric("❌ Chưa đạt", batch_res.get("fail_count", 0))
        m4.metric("⭐ Điểm trung bình", f"{batch_res.get('average_score', 0)}/100")

        # Bảng Tổng quan Thứ hạng (Ranking Overview)
        results = batch_res.get("results", [])
        overview_data = []
        for rank, r in enumerate(results, start=1):
            c_info = r.get("candidate_info", {})
            overview_data.append({
                "Thứ hạng": f"#{rank}",
                "Ứng viên": c_info.get("full_name") or "Chưa rõ họ tên",
                "Điểm số": f"{r.get('overall_score')}/100",
                "Xếp loại": r.get("match_rating"),
                "Xét tuyển": "✅ ĐẠT" if r.get("is_passed") else "❌ CHƯA ĐẠT",
                "Email": c_info.get("email") or "—",
                "SĐT": c_info.get("phone") or "—",
                "Tên file": r.get("metadata", {}).get("file_name", "—"),
            })

        st.dataframe(overview_data, use_container_width=True)

        st.markdown("### 📑 Chi tiết Từng Ứng viên")

        for idx, r in enumerate(results, start=1):
            c_info = r.get("candidate_info", {})
            c_name = c_info.get("full_name") or f"Ứng viên #{idx}"
            score = r.get("overall_score", 0)
            rating = r.get("match_rating", "NOT_FIT")
            is_passed = r.get("is_passed", False)
            file_name = r.get("metadata", {}).get("file_name", "")

            # Phân màu badge
            badge_class = "badge-strong" if rating == "STRONG_FIT" else (
                "badge-potential" if rating == "POTENTIAL_FIT" else (
                    "badge-borderline" if rating == "BORDERLINE" else "badge-notfit"
                )
            )

            expander_title = (
                f"#{idx}. {c_name} — Điểm: {score}/100 [{rating}] — {'✅ ĐẠT' if is_passed else '❌ CHƯA ĐẠT'} ({file_name})"
            )

            with st.expander(expander_title, expanded=(idx == 1)):
                # Thông tin liên hệ
                col_c1, col_c2, col_c3 = st.columns(3)
                col_c1.markdown(f"**Email:** `{c_info.get('email') or 'Không có'}`")
                col_c2.markdown(f"**Điện thoại:** `{c_info.get('phone') or 'Không có'}`")
                col_c3.markdown(f"**Địa điểm:** `{c_info.get('location') or 'Không có'}`")

                if c_info.get("links"):
                    st.caption("🔗 Liên kết: " + " | ".join(f"[{l}]({l})" for l in c_info.get("links", [])))

                # Bản tóm tắt cho HR
                st.markdown(
                    f"""
                    <div class='cv-summary-box'>
                        <div style='font-weight: 700; color: #065F46; font-size: 1rem; margin-bottom: 6px;'>
                            📋 Bản tóm tắt dành cho HR Tuyển dụng:
                        </div>
                        <div style='font-size: 0.93rem; color: #1F2937; line-height: 1.5;'>
                            {r.get('executive_summary')}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # 2 Cột: Điểm mạnh vs Điểm thiếu
                col_str, col_mis = st.columns(2)
                with col_str:
                    st.markdown("#### 🌟 Điểm mạnh & Khớp tiêu chí")
                    strengths = r.get("key_strengths", [])
                    if strengths:
                        for s in strengths:
                            st.markdown(f"<div class='evidence-box'>✓ {s}</div>", unsafe_allow_html=True)
                    else:
                        st.caption("Chưa có điểm mạnh nổi bật.")

                with col_mis:
                    st.markdown("#### ⚠️ Điểm còn thiếu / Chưa rõ")
                    missing = r.get("missing_or_unclear_points", [])
                    if missing:
                        for m in missing:
                            st.markdown(f"<div class='missing-box'>✗ {m}</div>", unsafe_allow_html=True)
                    else:
                        st.caption("Không có thiếu sót lớn.")

                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

                # Bảng chấm điểm tiêu chí chi tiết
                st.markdown("#### ⚖️ Bảng điểm chi tiết từng tiêu chí")
                evals = r.get("criteria_evaluations", [])
                eval_table = []
                for ev in evals:
                    eval_table.append({
                        "Tiêu chí": ev.get("criterion_name"),
                        "Loại": ev.get("criterion_type"),
                        "Điểm": f"{ev.get('score')}/100",
                        "Kết quả": "✅ Đạt" if ev.get("matched") else "❌ Không",
                        "Bằng chứng từ CV": ev.get("evidence"),
                        "Lý giải": ev.get("reasoning"),
                    })
                st.dataframe(eval_table, use_container_width=True)

                # Câu hỏi phỏng vấn gợi ý
                st.markdown("#### 🎙️ Gợi ý Câu hỏi Phỏng vấn")
                questions = r.get("interview_questions", [])
                if questions:
                    for q_idx, q in enumerate(questions, start=1):
                        st.markdown(f"**{q_idx}.** {q}")
                else:
                    st.caption("Không có câu hỏi gợi ý bổ sung.")
