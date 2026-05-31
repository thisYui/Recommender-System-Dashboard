# ChatCRS - Incorporating External Knowledge and Goal Guidance for LLM-based Conversational Recommender Systems

**Tên mô hình:** ChatCRS
**Tác giả:** Xinyun Li, Yang Deng, Meng Hu, Min-Yen Kan, Lizi Liao (Singapore Management University, National University of Singapore, Zhejiang University)
**Nơi công bố/Năm:** Đầu năm 2025 (Từ kho lưu trữ Awesome-LLM-KG-RecSys)
**Lĩnh vực:** Hệ thống gợi ý đàm thoại (Conversational Recommender Systems - CRSs), Dẫn dắt Mục tiêu (Goal Guidance), LLM Agents.

---

## 1. Bối cảnh & Điểm nghẽn (The Problem)

Đưa Large Language Models (LLMs) như ChatGPT vào làm Hệ thống gợi ý đàm thoại (CRS) đang là xu hướng vì chúng nói chuyện rất tự nhiên. Tuy nhiên, nếu bạn đã từng dùng ChatGPT để nhờ tư vấn mua hàng, bạn sẽ nhận ra **2 điểm yếu chết người**:

1. **Thiếu tính Chủ động (Lack of Proactivity / Goal Guidance):** ChatGPT thường hành động như một "kẻ thụ động". Bạn hỏi gì nó trả lời nấy. Một nhân viên sales giỏi phải biết cách *dẫn dắt câu chuyện*, chủ động đặt câu hỏi khơi gợi nhu cầu (Elicit preference) và hướng khách hàng đến việc chốt đơn (Recommendation). LLM hiện tại không biết cách lên kế hoạch cho cuộc trò chuyện.
2. **Ảo giác Tri thức (Knowledge Hallucination):** LLM không được kết nối với cơ sở dữ liệu kho hàng (Item database) hay Đồ thị tri thức (KG) theo thời gian thực. Nó có thể gợi ý cho bạn một bộ phim không có trên Netflix hoặc một món đồ đã hết hàng.

---

## 2. Ý tưởng Đột phá: Mô hình ChatCRS

Để biến LLM thành một "nhân viên sales" thực thụ, nhóm tác giả đề xuất **ChatCRS** – một hệ thống trang bị cho LLM khả năng **Lập kế hoạch mục tiêu (Goal Planning)** và **Sử dụng Công cụ (Tool Use)** để truy xuất tri thức ngoài.

Mô hình hoạt động theo quy trình 3 bước khép kín:

### Bước 1: Lập Kế hoạch Mục tiêu với sự hỗ trợ của Công cụ (Tool-Assisted Goal Planning)

- Khi nhận được tin nhắn của khách hàng, hệ thống không vội trả lời ngay. Nó yêu cầu LLM phải "Suy nghĩ" xem bước tiếp theo nên làm gì.
- Mục tiêu (Goal) có thể là: `[Hỏi thêm về sở thích]`, `[Chit-chat làm thân]`, hoặc `[Đưa ra gợi ý]`.
- LLM được cấp quyền sử dụng các Công cụ (Tools) để gọi API tra cứu Đồ thị tri thức hoặc kho hàng, xem xét tình hình thực tế trước khi chốt Mục tiêu.

### Bước 2: Gợi ý Nhận thức Tri thức (Knowledge-Aware Recommendation)

- Nếu Mục tiêu được xác định là `[Đưa ra gợi ý]`, hệ thống sẽ kích hoạt mô-đun Truy xuất (Retrieval).
- Dựa vào lịch sử chat và các công cụ vừa dùng, hệ thống lục lọi trong Đồ thị tri thức (KG) để lấy ra các sản phẩm phù hợp nhất, kèm theo các "Sự thật" (Facts - ví dụ: Phim này đạt giải Oscar) để làm bối cảnh (Context).

### Bước 3: Sinh Phản hồi theo Dẫn dắt Mục tiêu (Goal-Guided Response Generation)

- LLM giờ đây nhận được một gói thông tin hoàn chỉnh bao gồm: *[Lịch sử chat] + [Mục tiêu bắt buộc phải làm] + [Sản phẩm từ Đồ thị tri thức]*.
- LLM buộc phải sinh ra một câu trả lời bám sát Mục tiêu đã đề ra và sử dụng đúng dữ liệu thực tế từ Đồ thị. Không được phép bịa chuyện, không được phép trả lời lan man.

### Tinh chỉnh Mô hình (Fine-Tuning)

Để LLM mã nguồn mở (như LLaMA-2 hoặc Mistral) biết cách làm 3 bước này, nhóm tác giả đã xây dựng một bộ dữ liệu Huấn luyện Chỉ thị (Instruction Tuning dataset) chất lượng cao. Các câu lệnh được định dạng theo cấu trúc: `[Context] -> [Thought/Tools] -> [Goal] -> [Action/Rec] -> [Response]`.

---

## 3. Thiết lập Thí nghiệm & Kết quả (Results)

- **Datasets:** Sử dụng 2 bộ dữ liệu hội thoại lớn: **ReDial** (Phim ảnh) và **TG-ReDial** (Topic-Guided ReDial).
- **Baselines:** So sánh với GPT-3.5, GPT-4 (dạng Zero-shot/Few-shot), và các mô hình CRS truyền thống mạnh nhất (KBRD, KGSF, RevCore).
- **Kết quả cực kỳ ấn tượng:**
  - **Vượt trội GPT-4:** Mặc dù chỉ dùng LLaMA-2 7B (rất nhỏ), ChatCRS sau khi được tinh chỉnh đã đánh bại cả GPT-3.5 và GPT-4 (không có công cụ) về độ chính xác gợi ý (Recall, NDCG) lẫn chất lượng hội thoại (Fluency, Informativeness).
  - **Tính Chủ động (Proactivity):** Đo lường cho thấy ChatCRS có khả năng bám sát chủ đề (Topic hit rate) và dẫn dắt người dùng đến đích (Recommendation success rate) nhanh hơn hẳn các mô hình khác. Nó biết cách đặt câu hỏi khéo léo thay vì im lặng chờ đợi.
  - **Khử Ảo giác:** Nhờ bắt buộc phải tra cứu qua Tool và KG trước khi chốt đơn, tỉ lệ gợi ý sai thông tin thực tế giảm xuống mức tối thiểu.

---

## 4. Ví dụ Trực quan (Dễ hình dung)

Bạn nhắn cho Chatbot: *"Cuối tuần này tôi muốn xem một bộ phim giải trí nhẹ nhàng cùng gia đình."*

- **ChatGPT (Không có Goal Guidance):** *"Tuyệt quá! Xem phim cùng gia đình rất vui. Bạn có thích phim hoạt hình không?"* $\rightarrow$ Rất thụ động, hỏi một câu vô thưởng vô phạt.
- **ChatCRS:**
  1. **[Nghĩ thầm & Dùng Tool]:** *Khách muốn xem phim gia đình. Mình dùng Tool tra cứu KG xem phim Gia đình nào đang hot.* $\rightarrow$ KG trả về: Phim *Coco* (Pixar).
  2. **[Lập kế hoạch Mục tiêu]:** *Chủ đề hiện tại là phim gia đình. Kế hoạch: [Chuyển hướng sang Gợi ý Phim] (Goal: Recommendation).*
  3. **[Sinh câu trả lời]:** *"Trải qua một tuần mệt mỏi thì xem phim gia đình là lựa chọn tuyệt vời! Tôi gợi ý cho bạn phim hoạt hình **Coco**. Nó không chỉ hài hước, hình ảnh rực rỡ mà còn mang thông điệp rất ý nghĩa về tình cảm gia đình. Bạn có muốn xem thử trailer không?"*

$\rightarrow$ Câu trả lời không chỉ tự nhiên mà còn **Dẫn dắt trực tiếp** khách hàng đến quyết định, thể hiện phong thái của một nhân viên tư vấn xuất sắc!
