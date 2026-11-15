"""
app.py
Giao diện Streamlit Frontend cho hệ thống Trợ lý AI Nội bộ (RAG Assistant).
Tích hợp Streaming SSE (Server-Sent Events) qua httpx, quản lý phiên session_id,
tùy chọn chiến lược retriever (similarity/mmr/threshold), và hiển thị Source Citations trực quan.
"""

from __future__ import annotations

import datetime
import json
import uuid
import httpx
import streamlit as st

# ==========================================
# Cấu hình Trang Streamlit
# ==========================================
st.set_page_config(
    page_title="Trợ lý AI Nội bộ - RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS tinh chỉnh giao diện chuyên nghiệp
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .citation-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        margin-top: 8px;
        margin-bottom: 8px;
        border-radius: 4px;
    }
    .citation-title {
        font-weight: 600;
        color: #1E40AF;
        font-size: 0.95rem;
    }
    .citation-score {
        background-color: #DBEAFE;
        color: #1E40AF;
        font-size: 0.75rem;
        padding: 2px 6px;
        border-radius: 9999px;
        font-weight: 600;
    }
    .status-badge-online {
        color: #16A34A;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-badge-offline {
        color: #DC2626;
        font-weight: 600;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# Quản lý State trong Session
# ==========================================
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_url" not in st.session_state:
    st.session_state.api_url = "http://localhost:8000"


def generate_new_session():
    """Tạo session_id mới và xóa lịch sử hiển thị trên giao diện."""
    st.session_state.session_id = f"session-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
    st.session_state.messages = []


def check_backend_health(base_url: str) -> tuple[bool, str]:
    """Kiểm tra tình trạng kết nối tới FastAPI Backend."""
    try:
        with httpx.Client(timeout=2.0) as client:
            resp = client.get(f"{base_url}/api/v1/health")
            if resp.status_code == 200:
                data = resp.json()
                return True, f"Online ({data.get('rag_core', 'RAG')})"
            return False, f"HTTP {resp.status_code}"
    except Exception:
        return False, "Không thể kết nối"


def clear_backend_session(base_url: str, session_id: str):
    """Gửi yêu cầu xóa phiên lên backend."""
    try:
        with httpx.Client(timeout=3.0) as client:
            client.delete(f"{base_url}/api/v1/sessions/{session_id}")
    except Exception as e:
        st.sidebar.warning(f"Không thể xóa session trên backend: {e}")


# ==========================================
# SIDEBAR: Cấu hình và Quản lý Phiên
# ==========================================
with st.sidebar:
    st.markdown("### 🏢 Hệ thống RAG Nội bộ")

    # Kiểm tra trạng thái Backend
    is_online, status_text = check_backend_health(st.session_state.api_url)
    if is_online:
        st.markdown(f"Trạng thái Backend: <span class='status-badge-online'>🟢 {status_text}</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"Trạng thái Backend: <span class='status-badge-offline'>🔴 {status_text}</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🔑 Quản lý Phiên (`session_id`)")

    # Quản lý Session ID
    session_input = st.text_input(
        "Mã phiên làm việc hiện tại:",
        value=st.session_state.session_id,
        help="Mỗi session_id lưu trữ một ngữ cảnh lịch sử hội thoại riêng biệt trong MemoryManager.",
    )
    if session_input != st.session_state.session_id:
        st.session_state.session_id = session_input
        st.session_state.messages = []
        st.rerun()

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("➕ Phiên mới", use_container_width=True, help="Tạo một phiên hội thoại hoàn toàn mới"):
            generate_new_session()
            st.rerun()

    with col_btn2:
        if st.button("🗑️ Xóa phiên", use_container_width=True, help="Xóa bộ nhớ của phiên này"):
            clear_backend_session(st.session_state.api_url, st.session_state.session_id)
            st.session_state.messages = []
            st.success("Đã xóa lịch sử!")
            st.rerun()

    st.markdown("---")
    st.markdown("#### ⚙️ Chiến lược RAG Retriever")

    # Lựa chọn chiến lược RAG
    strategy_options = {
        "similarity": "🎯 Similarity (Độ tương đồng trực tiếp)",
        "mmr": "⚖️ MMR (Đa dạng hóa tài liệu)",
        "threshold": "🛡️ Threshold (Lọc theo ngưỡng tối thiểu)",
    }

    selected_strategy_key = st.radio(
        "Chiến lược tìm kiếm:",
        options=list(strategy_options.keys()),
        format_func=lambda x: strategy_options[x],
        index=0,
        help="Lựa chọn giải thuật trích xuất văn bản từ cơ sở tri thức công ty.",
    )

    k_value = st.slider(
        "Top-K (Số tài liệu tối đa):",
        min_value=1,
        max_value=10,
        value=4,
        help="Số đoạn văn bản có điểm cao nhất được đưa vào prompt LLM.",
    )

    threshold_value = 0.5
    if selected_strategy_key == "threshold":
        threshold_value = st.slider(
            "Ngưỡng tương đồng tối thiểu (Threshold):",
            min_value=0.1,
            max_value=1.0,
            value=0.6,
            step=0.05,
            help="Chỉ giữ lại những tài liệu có điểm tương đồng lớn hơn hoặc bằng ngưỡng này.",
        )

    with st.expander("ℹ️ Giải thích các chiến lược RAG"):
        st.markdown(
            """
            - **Similarity**: Tìm các đoạn văn bản có độ tương đồng Cosine cao nhất với câu hỏi. Thích hợp cho câu hỏi trực diện, cụ thể.
            - **MMR (Maximal Marginal Relevance)**: Cân bằng giữa độ tương đồng và tính đa dạng, tránh việc chọn nhiều đoạn tài liệu bị trùng lặp nội dung.
            - **Threshold**: Chỉ nhận tài liệu vượt ngưỡng tin cậy, giúp phòng tránh hiện tượng AI bịa đặt (hallucination) khi không có tài liệu liên quan.
            """
        )

    st.markdown("---")
    st.markdown("#### 💡 Câu hỏi mẫu")
    sample_queries = [
        "Quy định làm việc từ xa (WFH) là gì?",
        "Chính sách bảo mật mật khẩu và thiết bị?",
        "Chế độ nghỉ phép năm và bảo hiểm y tế?",
        "Quy trình review Pull Request kỹ thuật?",
        "Quy định thanh toán công tác phí?",
    ]
    for sq in sample_queries:
        if st.button(sq, key=f"sq_{sq}", use_container_width=True):
            st.session_state.pending_prompt = sq
            st.rerun()


# ==========================================
# KHUNG CHAT CHÍNH (Main Chat Interface)
# ==========================================
st.markdown("<div class='main-header'>🤖 Trợ lý AI Nội bộ Doanh nghiệp</div>", unsafe_allow_html=True)
st.markdown(
    f"<div class='sub-header'>Phiên: <code>{st.session_state.session_id}</code> | "
    f"Chiến lược: <b>{strategy_options[selected_strategy_key]}</b></div>",
    unsafe_allow_html=True,
)

# Hiển thị lịch sử hội thoại hiện có
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        citations = msg.get("citations", [])
        if citations:
            with st.expander(f"📚 {len(citations)} Nguồn tài liệu tham khảo (Citations)", expanded=False):
                for idx, cit in enumerate(citations, 1):
                    source = cit.get("source", "Tài liệu nội bộ")
                    page = cit.get("page")
                    page_str = f" • Trang {page}" if page else ""
                    score = cit.get("score")
                    score_str = f" • Điểm: {score:.2f}" if score is not None else ""
                    snippet = cit.get("snippet", "")

                    st.markdown(
                        f"""
                        <div class='citation-box'>
                            <div class='citation-title'>[{idx}] {source} {page_str} {score_str}</div>
                            <div style='font-size: 0.9rem; color: #374151; margin-top: 4px;'>{snippet}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# Nhận input từ người dùng (hoặc từ câu hỏi mẫu bấm nhanh)
user_prompt = st.chat_input("Nhập câu hỏi của bạn về quy định, tài liệu nội bộ...")
if hasattr(st.session_state, "pending_prompt") and st.session_state.pending_prompt:
    user_prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if user_prompt:
    # 1. Hiển thị tin nhắn người dùng ngay lập tức
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # 2. Xử lý phản hồi Streaming từ Backend qua Server-Sent Events (SSE)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        collected_citations = []
        error_occurred = False

        # Hiển thị hiệu ứng chờ phản hồi
        message_placeholder.markdown("🔍 *Đang tra cứu tài liệu và tổng hợp câu trả lời...*")

        try:
            # Gọi tới FastAPI Backend bằng httpx với cơ chế SSE
            api_endpoint = f"{st.session_state.api_url}/api/v1/chat/stream"
            request_payload = {
                "session_id": st.session_state.session_id,
                "query": user_prompt,
                "strategy": selected_strategy_key,
                "k": k_value,
                "threshold": threshold_value,
            }

            with httpx.Client(timeout=60.0) as client:
                with client.stream(
                    "POST",
                    api_endpoint,
                    json=request_payload,
                    headers={"Accept": "text/event-stream"},
                ) as response:

                    if response.status_code != 200:
                        error_occurred = True
                        st.error(f"Lỗi máy chủ Backend: HTTP {response.status_code} - {response.text}")
                    else:
                        # Đọc stream từng dòng SSE
                        current_event_type = None

                        for line in response.iter_lines():
                            if not line:
                                continue

                            line = line.strip()

                            # Bắt tên sự kiện: event: token / event: citations / event: done
                            if line.startswith("event:"):
                                current_event_type = line[len("event:"):].strip()
                                continue

                            # Bắt phần dữ liệu: data: {...}
                            if line.startswith("data:"):
                                raw_data = line[len("data:"):].strip()
                                try:
                                    payload = json.loads(raw_data)
                                except json.JSONDecodeError:
                                    continue

                                msg_type = payload.get("type") or current_event_type

                                # 1. Nhận token chữ cái
                                if msg_type == "token":
                                    token = payload.get("content", "")
                                    full_response += token
                                    message_placeholder.markdown(full_response + " ▌")

                                # 2. Nhận danh sách trích dẫn nguồn
                                elif msg_type == "citations":
                                    collected_citations = payload.get("citations", [])

                                # 3. Nhận sự kiện hoàn thành
                                elif msg_type == "done":
                                    message_placeholder.markdown(full_response)

                                # 4. Nhận sự kiện báo lỗi
                                elif msg_type == "error":
                                    error_occurred = True
                                    err_msg = payload.get("message", "Đã có lỗi xảy ra.")
                                    st.error(f"⚠️ {err_msg}")
                                    break

            # Hiển thị hoàn thiện câu trả lời
            if not error_occurred:
                message_placeholder.markdown(full_response)

                # Hiển thị box nguồn tài liệu tham khảo (Source Citations)
                if collected_citations:
                    with st.expander(f"📚 {len(collected_citations)} Nguồn tài liệu tham khảo (Citations)", expanded=True):
                        for idx, cit in enumerate(collected_citations, 1):
                            source = cit.get("source", "Tài liệu nội bộ")
                            page = cit.get("page")
                            page_str = f" • Trang {page}" if page else ""
                            score = cit.get("score")
                            score_str = f" • Điểm: {score:.2f}" if score is not None else ""
                            snippet = cit.get("snippet", "")

                            st.markdown(
                                f"""
                                <div class='citation-box'>
                                    <div class='citation-title'>[{idx}] {source} {page_str} {score_str}</div>
                                    <div style='font-size: 0.9rem; color: #374151; margin-top: 4px;'>{snippet}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                # Lưu vào lịch sử hội thoại của session_state
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": full_response,
                        "citations": collected_citations,
                    }
                )

        except httpx.ConnectError:
            message_placeholder.empty()
            st.error(
                "❌ **Không thể kết nối đến FastAPI Backend!**\n\n"
                "Vui lòng đảm bảo Backend đã được khởi chạy với lệnh:\n"
                "```bash\n"
                "uvicorn server:app --host 0.0.0.0 --port 8000 --reload\n"
                "```"
            )
        except Exception as exc:
            message_placeholder.empty()
            st.error(f"❌ Đã xảy ra lỗi không mong muốn: {str(exc)}")

