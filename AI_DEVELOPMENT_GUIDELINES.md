# 📋 KỊCH BẢN & NGUYÊN TẮC BẮT BUỘC DÀNH CHO AI KHI PHÁT TRIỂN DỰ ÁN
> **Tài liệu tham chiếu chuẩn (AI Engineering & Coding Standard)**
> Áp dụng cho mọi tác vụ phát triển tính năng, tái cấu trúc hoặc sửa lỗi trong hệ thống `tro-ly-AI-noi-bo`.

---

## 🛑 NGUYÊN TẮC VÀNG (THE GOLDEN RULES)

### 1. TUYỆT ĐỐI KHÔNG ĐƯỢC LÀM (NEVER DO)
1. ❌ **Không tự ý làm bừa (No Hallucination & No Wild Assumptions)**:
   - Tuyệt đối không tự suy diễn các API, thư viện hoặc cấu trúc không tồn tại trong dự án.
   - Không tự tiện cài đặt các thư viện lạ nếu các thư viện hiện tại (`FastAPI`, `Pydantic`, `httpx`, `sse-starlette`, `Streamlit`) đã đáp ứng đủ.
2. ❌ **Không tự ý sửa/xóa các thành phần đang hoạt động ổn định (No Breaking Existing Features)**:
   - Không được ghi đè hoặc thay đổi các interface cốt lõi (`BaseRAGCore`, `MemoryManager`, `RAGCoordinator`, schema API) nếu không có yêu cầu cụ thể từ người dùng.
   - Khi chỉnh sửa file cũ, phải bảo toàn tính tương thích ngược (backward compatibility) và không làm hỏng các bài kiểm thử (`tests_rag/`).
3. ❌ **Không viết code khi chưa hiểu rõ bối cảnh (No Coding Without Reading)**:
   - Cấm nhảy vào viết code ngay lập tức khi chưa đọc kỹ mã nguồn hiện có của các file liên quan.
   - Cấm bỏ qua bất kỳ gạch đầu dòng nào trong yêu cầu của người dùng.

---

## 🔄 QUY TRÌNH 5 BƯỚC PHÁT TRIỂN BẮT BUỘC (5-STEP AI WORKFLOW)

Mỗi khi nhận một yêu cầu phát triển mới từ người dùng, AI **bắt buộc** phải tuân thủ nghiêm ngặt chu trình sau:

```
[Bước 1: Đọc & Khảo sát] ➡️ [Bước 2: Lập Kế hoạch] ➡️ [Bước 3: Viết Code Chuẩn] ➡️ [Bước 4: Tự Test] ➡️ [Bước 5: Bàn giao]
```

### Bước 1: Đọc kỹ yêu cầu & Khảo sát hiện trạng (Deep Discovery)
- Đọc kỹ 100% prompt của người dùng, phân tích từng điều kiện ràng buộc.
- Kiểm tra các file hiện có liên quan trong thư mục (`api/`, `services/`, `app.py`, `tests_rag/`).
- Xác định rõ:
  - Tính năng này tác động đến những module nào?
  - Dữ liệu đầu vào (Input) và dữ liệu đầu ra (Output) là gì?
  - Có nguy cơ làm gãy module nào của Dev khác (ví dụ: Dev 1 RAG Core) không?

### Bước 2: Đối chiếu & Lập kế hoạch (Alignment & Design)
- Tóm tắt lại cho người dùng hiểu: *Mục tiêu cần làm* và *Các file sẽ tạo mới hoặc cập nhật*.
- Nếu yêu cầu của người dùng bị mâu thuẫn hoặc thiếu dữ liệu, **phải dừng lại đặt câu hỏi làm rõ**, không tự đoán mò rồi làm sai.

### Bước 3: Triển khai mã nguồn có kiểm soát (Strict Coding)
- **Kiến trúc phân tầng (Clean Architecture)**:
  - `api/`: Chỉ xử lý tầng giao tiếp HTTP, validate dữ liệu với Pydantic, không nhồi nhét logic nghiệp vụ nặng.
  - `services/`: Xử lý logic hội thoại, trích xuất dữ liệu, điều phối RAG.
  - `app.py`: Chỉ xử lý hiển thị giao diện, gọi API bất đồng bộ, không truy cập trực tiếp database/vectorstore.
- **Tiêu chuẩn code**:
  - Viết Type Hints đầy đủ (`typing`, `pydantic`).
  - Viết Docstrings và comment tiếng Việt rõ ràng, giải thích "Tại sao làm thế này".
  - Luôn có cơ chế bắt ngoại lệ (`try...except`) an toàn, không để chương trình crash trắng trang.

### Bước 4: Tự kiểm thử & Thẩm định (Self-Testing & Verification)
- Sau khi viết mã xong, **luôn luôn chạy lại bộ kiểm thử tự động**:
  ```bash
  pytest -c pytest.ini tests_rag/ -v
  ```
- Nếu thêm tính năng mới, **bắt buộc viết thêm unit test tương ứng** trong `tests_rag/`.
- Nếu có lỗi phát sinh (failed tests), phải phân tích nguyên nhân gốc rễ và sửa dứt điểm, không được xóa test để qua mặt.

### Bước 5: Bàn giao & Giải trình minh bạch (Clear Delivery)
- Báo cáo rõ ràng:
  - Những file nào đã thay đổi/tạo mới (kèm link `file:///...`).
  - Kết quả kiểm thử trước và sau khi chỉnh sửa.
  - Hướng dẫn cụ thể cách người dùng kiểm tra lại bằng tay.

---

## 📝 MẪU PROMPT CHUẨN ĐỂ NGƯỜI DÙNG GIAO VIỆC CHO AI (REUSABLE PROMPT TEMPLATE)

> 💡 **Mẹo dành cho Bạn**: Mỗi khi muốn AI làm một tính năng mới trong tương lai, hãy copy đoạn khung prompt dưới đây và điền thông tin vào:

```markdown
### YÊU CẦU PHÁT TRIỂN TÍNH NĂNG MỚI: [Tên tính năng]

#### 1. Nguyên tắc tuân thủ:
- Tuân thủ nghiêm ngặt các điều khoản trong file `AI_DEVELOPMENT_GUIDELINES.md`.
- ĐỌC KỸ mã nguồn của các file liên quan trước khi chỉnh sửa.
- TUYỆT ĐỐI KHÔNG làm hỏng các tính năng cũ và phải giữ nguyên 14 test cases hiện tại trong `tests_rag/`.

#### 2. Mô tả tính năng cần làm:
- [Mô tả chi tiết mục tiêu bạn muốn làm]

#### 3. Đầu vào & Đầu ra mong muốn:
- Input: [Ví dụ: dữ liệu client gửi lên...]
- Output: [Ví dụ: định dạng trả về, giao diện Streamlit hiển thị thế nào...]

#### 4. Các file dự kiến can thiệp:
- [Ví dụ: services/memory.py, api/routes.py, hoặc app.py...]

#### 5. Yêu cầu kiểm thử:
- Chạy lại toàn bộ pytest và viết thêm test case cho tính năng mới này.
```

---

## 📌 BẢNG CHECKLIST KIỂM SOÁT TRƯỚC KHI BÀN GIAO (AI PRE-FLIGHT CHECKLIST)

Trước khi gửi câu trả lời hoàn tất cho người dùng, AI phải tự tick đủ các tiêu chí sau:

- [ ] Đã đọc và hiểu trọn vẹn yêu cầu người dùng chưa?
- [ ] Có tự chế ra thư viện hay cấu trúc nào ngoài lề không?
- [ ] Code có type hints và docstrings đầy đủ không?
- [ ] Có đảm bảo không phá vỡ interface kết nối của Dev 1 (`BaseRAGCore`) không?
- [ ] Đã chạy lệnh `pytest` và đảm bảo toàn bộ tests đều PASSED chưa?
- [ ] Đã tạo liên kết file trực quan (`file:///...`) đúng quy định chưa?

