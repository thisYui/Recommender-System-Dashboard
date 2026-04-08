# Giải thích: Hàm dung hợp Đa phương thức (Multimodal Fusion)

Dựa trên bài báo **"Một khảo sát về Hệ thống Gợi ý Đa phương thức: Những tiến bộ gần đây và Hướng đi tương lai"** (Paper 01 - IEEE 2025), tài liệu này giải thích chi tiết về các cơ chế dung hợp trong Hệ thống Gợi ý Đa phương thức (MRS).

---

## 1. Khái niệm cốt lõi
Trong MRS, **Dung hợp (Fusion)** là quá trình kết hợp các loại dữ liệu khác nhau (Hình ảnh, Văn bản, Âm thanh, Hành vi) thành một biểu diễn thống nhất. Mục tiêu là tận dụng tối đa thông tin bổ trợ từ các phương thức để nâng cao độ chính xác của gợi ý.

Bài khảo sát phân loại các chiến lược dung hợp theo 2 góc độ: **Thời điểm (Timing)** và **Chiến lược (Strategy)**.

---

## 2. Góc độ Thời điểm (Timing Perspective)

Góc độ này xác định giai đoạn mà các phương thức được kết hợp với nhau trong mô hình.

### A. Dung hợp Sớm (Early Fusion)
*   **Cơ chế:** Kết hợp các đặc trưng ($I_{visual}, I_{text},...$) **trước khi** đưa vào bộ mã hóa (Encoder). 
*   **Ưu điểm:** Cho phép mô hình học được các mối quan hệ tương tác ẩn (hidden correlations) giữa các phương thức ngay từ đầu.
*   **Công thức:** $I_{unified} = \text{Aggr}(I_{visual}, I_{text}, ...)$ sau đó $\hat{R} = \text{Model}(U, I_{unified})$.

### B. Dung hợp Muộn (Late Fusion)
*   **Cơ chế:** Mỗi phương thức được xử lý bởi một bộ mã hóa riêng biệt để tạo ra các dự đoán (scores) độc lập. Các kết quả này sau đó mới được gộp lại ở bước cuối cùng.
*   **Ưu điểm:** Tận dụng tối đa thế mạnh riêng biệt của từng loại dữ liệu. Phù hợp khi một số phương thức có độ tin cậy cao hơn hẳn các phương thức khác.
*   **Công thức:** $\hat{R} = \text{Aggr}(\text{Score}_{visual}, \text{Score}_{text}, ...)$.

---

## 3. Góc độ Chiến lược (Strategy Perspective)

Góc độ này xác định phép toán toán học được sử dụng để gộp các vector.

| Chiến lược | Phép toán / Cơ chế | Đặc điểm |
| :--- | :--- | :--- |
| **Cộng/Trung bình (Element-wise)** | $v = v_1 + v_2$ hoặc $v = \text{avg}(v_1, v_2)$ | Đơn giản, giả định các phương thức có vai trò ngang hàng. |
| **Nối vector (Concatenation)** | $v = [v_1; v_2]$ | Giữ nguyên toàn bộ thông tin nhưng làm tăng số chiều vector. |
| **Chú ý (Attentive)** | $v = \sum \alpha_m v_m$ | **SOTA.** Sử dụng trọng số $\alpha$ có thể học được để nhấn mạnh phương thức quan trọng nhất. |
| **Quy tắc (Heuristic)** | Trọng số cố định | Dựa trên kinh nghiệm (ví dụ: gán 0.7 cho Hình ảnh và 0.3 cho Văn bản). |

---

## 4. Hàm Aggr(·) trong các bộ mã hóa

Hàm $\text{Aggr}(\cdot)$ (Aggregation) đóng vai trò là hàm dung hợp cụ thể trong mã nguồn:

### Trong MF-based Encoder:
*   **Hợp nhất (Unified):** $I = \text{Aggr}(I_m)$. Gom đặc trưng vào một vector mục duy nhất trước khi nhân với vector người dùng $U$.
*   **Nhiều bộ (Multiple):** $\hat{R} = \text{Aggr}(U I_m^T)$. Tính điểm cho từng phương thức rồi mới gộp điểm lại.

### Trong Graph-based Encoder (GCN):
*   Dung hợp không chỉ ở vector nhúng mà còn ở **Cấu trúc Đồ thị**. 
*   Hàm $\text{Aggr}$ có thể dùng để tạo ra một **Đồ thị hợp nhất** từ nhiều đồ thị đa phương thức (ví dụ: đồ thị tương quan hình ảnh + đồ thị tương quan văn bản).

---

## 5. Tóm tắt Pipeline Multimodal
1.  **Feature Extraction:** Dùng ResNet (Visual) và BERT/Sentence-Transformer (Textual) để lấy đặc trưng thô.
2.  **Multimodal Fusion:** Dùng hàm $\text{Aggr}(\cdot)$ (thường là cơ chế **Attention**) để kết hợp chúng lại.
3.  **Encoder:** Đưa qua GCN hoặc MF để học các mối quan hệ cộng tác (collaborative filtering).
4.  **Loss Function:** Tối ưu hóa mô hình bằng hàm mất mát (như BPR Loss hoặc Contrastive Loss).

---
*Tài liệu tóm tắt dựa trên Hình 2 và Phần V của bài khảo sát IEEE 2025.*
