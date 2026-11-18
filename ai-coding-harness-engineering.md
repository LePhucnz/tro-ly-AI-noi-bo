Mô tả: Áp dụng các nguyên lý Harness Engineering cho các AI coding agent (như Claude Code) để xây dựng hệ thống code tự động vững chắc thông qua lập kế hoạch, hooks, phân quyền, subagent, MCP tools và kiểm tra mục tiêu.

Nội dung cốt lõi của Skill:

Skill này tổng hợp 6 lớp bảo vệ (6 Pillars of Harness) giúp kiểm soát AI coding agent hiệu quả thay vì chỉ phụ thuộc vào việc đổi model:
Lập kế hoạch (/plan):

        Chạy ở chế độ chỉ đọc (read-only) trước tiên để phân tích và tạo file trạng thái (ví dụ: features.json) với các tính năng mặc định là passes: false, giải quyết từng việc một.

    Hệ thống Hooks:

        Thiết lập PostToolUse để tự động chạy kiểm thử (test) ngay sau mỗi lần sửa file.

        Stop hook đóng vai trò là cửa ải cuối cùng: nếu test chưa xanh (chưa qua), tuyệt đối không cho phép agent dừng lại.

    Phân quyền cứng (Permissions):

        Đặt luật chặn trực tiếp (ví dụ: chặn quyền sửa file test deny Edit(tests/**)) thay vì chỉ nhắc nhở qua câu lệnh văn bản trong file hướng dẫn.

    Sử dụng Subagent:

        Chia nhỏ mỗi tính năng cho một subagent riêng biệt với ngữ cảnh (context) độc lập. Agent con chỉ gửi về bản tóm tắt kết quả, giúp cửa sổ ngữ cảnh chính luôn gọn gàng.

    Công cụ MCP (ví dụ: Playwright):

        Trao quyền cho agent tự động mở trình duyệt, bấm thử từng nút giống như người dùng thực tế trước khi xác nhận hoàn thành công việc.

    Kiểm tra mục tiêu (/goal):

        Thiết lập điều kiện hoàn thành rõ ràng. Một cơ chế đánh giá (hoặc model phụ trợ nhỏ hơn) sẽ chấm điểm sau mỗi lượt thực thi trước khi chuyển sang bước tiếp theo.