"""
app.py








Giao diện Streamlit Frontend mang phong cách Google Gemini cho hệ thống Trợ lý AI Nội bộ.
- Thiết kế thẩm mỹ chuẩn Gemini: Header Gradient, Sparkle Icons, Prompt Suggestion Cards, Grounding Citations.
- Streaming thời gian thực qua Server-Sent Events (SSE).
- Quản lý phiên hội thoại đa người dùng (Session History).
- Tích hợp Upload & Ingest tài liệu trực tiếp vào ChromaDB từ giao diện.
- Tùy chọn linh hoạt các chiến lược RAG (Similarity, MMR, Threshold).
"""

from __future__ import annotations

import datetime
import json
import uuid
from typing import Any, Dict, List, Optional
import httpx
import streamlit as st

# ==============================================================================
# 1. Cấu hình Trang Streamlit & Theme Google Gemini
# ==============================================================================
st.set_page_config(
    page_title="Trợ lý AI Nội bộ",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS tạo hiệu ứng Google Gemini cao cấp
GEMINI_CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Google Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Ẩn các thành phần mặc định không cần thiết của Streamlit */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Container chính căn giữa gọn gàng */
.main .block-container {
    max-width: 900px;
    padding-top: 2rem;
    padding-bottom: 6rem;
}

/* Gemini Hero Greeting */
.gemini-hero-container {
    padding: 2.5rem 0 1.5rem 0;
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
    font-size: 2.8rem;
    font-weight: 700;
    line-height: 1.2;
    background: linear-gradient(90deg, #4285F4 0%, #9B72CB 40%, #D96570 80%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}

.gemini-subgreeting {
    font-size: 1.4rem;
    font-weight: 500;
    color: #5f6368;
    margin-top: 0.4rem;
    margin-bottom: 2rem;
}

/* Suggestion Prompt Cards phong cách Gemini */
.gemini-cards-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
    margin-bottom: 2.5rem;
}

.gemini-card {
    background-color: rgba(240, 244, 249, 0.7);
    border: 1px solid rgba(220, 227, 237, 0.8);
    border-radius: 16px;
    padding: 16px;
    cursor: pointer;
    transition: all 0.2s ease-in-out;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.gemini-card:hover {
    background-color: #E9EEF6;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(66, 133, 244, 0.08);
}

.gemini-card-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #1f1f1f;
    margin-bottom: 6px;
}

.gemini-card-desc {
    font-size: 0.82rem;
    color: #444746;
    line-height: 1.35;
}

.gemini-card-icon {
    font-size: 1.3rem;
    align-self: flex-end;
    margin-top: 10px;
    opacity: 0.85;
}

/* Bong bóng chat người dùng & trợ lý */
.user-bubble {
    background: #F0F4F9;
    border-radius: 20px 20px 4px 20px;
    padding: 12px 18px;
    color: #1f1f1f;
    font-size: 1rem;
    line-height: 1.5;
    margin-bottom: 12px;
    display: inline-block;
    max-width: 85%;
    float: right;
    clear: both;
}

.assistant-bubble-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
    font-weight: 600;
    font-size: 0.95rem;
    color: #4285F4;
}

/* Gemini Grounding / Citation Chips */
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
    text-decoration: none;
    transition: background 0.2s;
}

.grounding-chip:hover {
    background-color: #D3E3FD;
}

.citation-card {
    background-color: #F8FAFC;
    border-left: 3.5px solid #4285F4;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 8px 0;
}

/* Nút tạo phiên mới phong cách Gemini */
.stButton > button[kind="primary"] {
    border-radius: 28px;
    font-weight: 600;
    background: linear-gradient(90deg, #1a73e8, #4285f4);
    border: none;
}

/* Sidebar tinh gọn */
section[data-testid="stSidebar"] {
    background-color: #F8FAFD;
    border-right: 1px solid #E5E7EB;
}

/* Status indicator badge */
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
</style>
"""
st.markdown(GEMINI_CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# 2. Quản lý Trạng thái Phiên (Session State Management)
# ==============================================================================
if "session_id" not in st.session_state:
    st.session_state.session_id = f"gemini-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_url" not in st.session_state:
    st.session_state.api_url = "http://127.0.0.1:8000"

if "upload_status" not in st.session_state:
    st.session_state.upload_status = None


def create_new_chat():
    """Tạo một phiên hội thoại mới tinh, làm sạch màn hình."""
    st.session_state.session_id = f"gemini-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
    st.session_state.messages = []
    st.session_state.upload_status = None


def check_api_health(base_url: str) -> tuple[bool, str]:
    """Kiểm tra tình trạng kết nối tới FastAPI Backend (tự động thử cả 127.0.0.1 và localhost)."""
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
                    return True, "Hệ thống RAG Online"
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


# ==============================================================================
# 3. SIDEBAR: Quản lý Phiên, Cấu hình RAG & Tải lên Tài liệu
# ==============================================================================
with st.sidebar:
    # Header phong cách Google Gemini
    st.markdown(
        """
        <div style='display: flex; align-items: center; gap: 8px; margin-bottom: 1.2rem;'>
            <span style='font-size: 1.6rem; color: #4285F4;'>✦</span>
            <span style='font-size: 1.35rem; font-weight: 700; color: #1f1f1f;'>Trợ lý Nội bộ</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Nút Cuộc trò chuyện mới (+ New Chat)
    if st.button("➕ Cuộc trò chuyện mới", use_container_width=True, type="primary"):
        create_new_chat()
        st.rerun()

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Hiển thị trạng thái Backend & Vector Store
    is_online, status_msg = check_api_health(st.session_state.api_url)
    if is_online:
        st.markdown(f"<span class='status-pill status-online'>🟢 {status_msg}</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"<span class='status-pill status-offline'>🔴 {status_msg}</span>", unsafe_allow_html=True)

    stats = fetch_vector_stats(st.session_state.api_url) if is_online else None
    if stats:
        st.caption(f"📚 Vector DB: **{stats.get('total_chunks_count', 0)} chunks** | Model: `{stats.get('embedding_provider')}`")

    st.divider()

    # Quản lý Session ID
    with st.expander("🔑 Quản lý Phiên (Session)", expanded=False):
        current_sess = st.text_input(
            "Session ID hiện tại:",
            value=st.session_state.session_id,
            help="Mỗi session_id lưu giữ lịch sử ngữ cảnh hội thoại riêng biệt trong bộ nhớ.",
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

    # Cấu hình Chiến lược RAG Retriever
    with st.expander("⚙️ Cấu hình Chiến lược RAG", expanded=True):
        strategy_labels = {
            "similarity": "🎯 Similarity (Tương đồng Cosine)",
            "hybrid": "⚡ Hybrid (Kết hợp Vector + BM25 Từ khóa)",
            "mmr": "⚖️ MMR (Đa dạng hóa nguồn tin)",
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
            help="Số lượng đoạn văn bản có điểm cao nhất được đưa vào prompt LLM.",
        )

        threshold_val = 0.35
        if selected_strategy in ("similarity_score_threshold", "threshold"):
            threshold_val = st.slider(
                "Ngưỡng tin cậy (Threshold):",
                min_value=0.1,
                max_value=0.8,
                value=0.35,
                step=0.05,
                help="Loại bỏ tài liệu có điểm nhỏ hơn ngưỡng để phòng chống ảo giác.",
            )
            st.caption("💡 Khuyến nghị: Chọn **Similarity** hoặc **MMR** để có câu trả lời tốt nhất. Với tài liệu tiếng Việt, ngưỡng Threshold nên đặt khoảng 0.25 - 0.40.")

    # Tích hợp Nạp tài liệu trực tiếp vào ChromaDB
    with st.expander("📤 Nạp thêm Tài liệu vào Vector DB", expanded=False):
        uploaded_file = st.file_uploader(
            "Chọn tệp (.pdf, .docx, .txt, .csv, .md):",
            type=["pdf", "docx", "txt", "csv", "md"],
            help="Hệ thống tự động băm SHA-256 chống nạp trùng lặp, cắt 1000/200 và lưu vào ChromaDB.",
        )

        if uploaded_file is not None:
            if st.button("🚀 Nạp vào Cơ sở Tri thức", use_container_width=True):
                with st.spinner("Đang xử lý nạp tài liệu qua Ingestion Pipeline..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "application/octet-stream")}
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

        if st.button("🔄 Reset Collection ChromaDB", use_container_width=True, help="Xóa trắng dữ liệu trong Vector DB"):
            try:
                with httpx.Client(timeout=5.0) as client:
                    resp = client.post(f"{st.session_state.api_url}/api/v1/documents/reset")
                    if resp.status_code == 200:
                        st.success("Đã làm sạch Vector Store!")
                        st.rerun()
            except Exception as e:
                st.error(f"Lỗi reset: {e}")


# ==============================================================================
# 4. KHUNG CHAT CHÍNH (Gemini Chat Interface)
# ==============================================================================

# Nếu chưa có tin nhắn nào trong phiên, hiển thị Màn hình Chào mừng phong cách Gemini
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

    # 4 Thẻ Gợi ý Nhanh (Gemini Suggestion Cards)
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

# Hiển thị lịch sử tin nhắn đã trao đổi
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
            citations = msg.get("citations", [])
            if citations:
                # Hiển thị các Grounding Chips phong cách Gemini
                from pathlib import Path
                chip_html = "<div style='margin-top: 10px; margin-bottom: 4px;'>"
                for c in citations[:4]:
                    raw_source = c.get("source", "Tài liệu")
                    source_name = Path(raw_source).name if ("/" in raw_source or "\\" in raw_source) else raw_source
                    score = c.get("score")
                    score_str = f" • {int(score * 100)}%" if score is not None else ""
                    chip_html += f"<span class='grounding-chip'>📄 {source_name}{score_str}</span>"
                chip_html += "</div>"
                st.markdown(chip_html, unsafe_allow_html=True)

                # Accordion chi tiết đoạn trích dẫn
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


# ==============================================================================
# 5. Thanh Nhập Câu hỏi & Xử lý Streaming SSE
# ==============================================================================
user_input = st.chat_input("Hỏi bất kỳ điều gì về quy định, chính sách, tài liệu nội bộ...")
if hasattr(st.session_state, "pending_prompt") and st.session_state.pending_prompt:
    user_input = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if user_input:
    # 1. Hiển thị tin nhắn người dùng
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Xử lý phản hồi Streaming thời gian thực từ Backend
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_text = ""
        citations_list = []
        has_error = False

        # Hiệu ứng nạp dữ liệu thanh nhã
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

                                # Token stream
                                if evt_type == "token":
                                    token = data_obj.get("content", "")
                                    full_text += token
                                    response_placeholder.markdown(full_text + " ▌")

                                # Citations
                                elif evt_type == "citations":
                                    citations_list = data_obj.get("citations", [])

                                # Hoàn tất
                                elif evt_type == "done":
                                    response_placeholder.markdown(full_text)

                                # Báo lỗi
                                elif evt_type == "error":
                                    has_error = True
                                    err_msg = data_obj.get("message", "Đã có lỗi phát sinh.")
                                    st.error(f"⚠️ {err_msg}")
                                    break

            if not has_error:
                response_placeholder.markdown(full_text)

                # Hiển thị Grounding chips & nguồn
                if citations_list:
                    from pathlib import Path
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

                # Lưu vào lịch sử phiên
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": full_text,
                        "citations": citations_list,
                    }
                )

        except httpx.ConnectError:
            response_placeholder.empty()
            st.error(
                "❌ **Không thể kết nối đến FastAPI Backend!**\n\n"
                "Vui lòng đảm bảo Backend đang chạy tại `http://localhost:8000`:\n"
                "```bash\n"
                "python3 main.py\n"
                "```"
            )
        except Exception as exc:
            response_placeholder.empty()
            st.error(f"❌ Lỗi: {str(exc)}")
