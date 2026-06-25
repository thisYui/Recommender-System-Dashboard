# Tổng Quan Tài Liệu & Nhận Diện Khoảng Trống Nghiên Cứu (Literature Review & Gap)

## 1. Bối cảnh: LLM trong Hệ thống Gợi ý (RecSys)
Sự bùng nổ của các Mô hình Ngôn ngữ Lớn (LLMs) đã mở ra một kỷ nguyên mới cho Generative Recommendation (Gợi ý Tạo sinh). Trong kiến trúc này, lịch sử tương tác của người dùng được mô hình hóa dưới dạng một chuỗi các token (ví dụ: `[Item_A, Item_B, Item_C]`). Để LLM hiểu được thứ tự của các token này, cơ chế **Rotary Position Embedding (RoPE)** được sử dụng rộng rãi, đánh số các token bằng một mảng chỉ số nguyên rời rạc ($m = 1, 2, 3...$).

Tuy nhiên, đặc thù của dữ liệu hành vi người dùng trong RecSys là **tính không đồng bộ (asynchronous)**. Các tương tác hiếm khi xảy ra ở các khoảng cách đều đặn. Việc RoPE ép các khoảng trống thời gian khác biệt (ví dụ: 5 phút và 5 tháng) thành cùng một khoảng cách chỉ số $\Delta m = 1$ được gọi là **Uniform Spacing Fallacy** (Sai lầm khoảng cách đồng đều). Điều này làm mất đi khả năng của mô hình trong việc nhận thức sự suy giảm sở thích (interest decay) hoặc tính chu kỳ (seasonality).

## 2. Các Hướng Giải Quyết Gần Đây & Hạn Chế
Giới học thuật đã nhận ra vấn đề này và bắt đầu đưa thời gian vật lý vào cơ chế Attention trong thời gian rất gần đây (2024 - 2025). 

### 2.1. Nhóm 1: Non-RoPE Time-aware Attention (Attention Nhận thức Thời gian Không dùng RoPE)
Các nghiên cứu như **TiSASRec** và **MEANTIME** đưa thời gian vào mô hình bằng cách chia nhỏ (binning) khoảng cách thời gian thành các bucket rời rạc, sau đó cộng thêm một embedding tương đối (relative position bias) vào Attention Matrix. 
*   **Hạn chế:** Các mô hình này không tận dụng được sức mạnh ngoại suy của RoPE, làm mất khả năng tương thích với các kiến trúc LLM hiện đại được tối ưu bằng FlashAttention.

### 2.2. Nhóm 2: Continuous Time-aware RoPE (RoPE Thời gian Liên tục - SOTA Hiện Tại)
Đây là các nghiên cứu ra mắt gần nhất, giải quyết trực tiếp Uniform Spacing Fallacy bằng cách điều chỉnh RoPE:
*   **TO-RoPE (Time-and-Order RoPE - 2024):** Coi cả "vị trí thứ tự" (Order) và "thời gian thực" (Time) là nguồn để xoay vector. TO-RoPE chia không gian chiều ẩn (split-by-dim) để một nửa số chiều xử lý thứ tự, nửa còn lại xử lý thời gian trôi qua.
*   **RoTE (Multi-Level Rotary Time Embedding):** Phân rã thời gian thành cấu trúc phân cấp (Năm, Tháng, Ngày) và áp dụng các phép xoay độc lập để bắt được sự chuyển dịch sở thích đa mức độ.
*   **Time-Aware RoPE Extensions:** Nhiều biến thể nhận ra rằng dùng thời gian trôi (elapsed time) tuyến tính dễ làm mô hình nổ (explode), nên họ áp dụng phép biến đổi logarit $\log(1 + \Delta t)$ vào góc xoay để chuẩn hóa sự suy giảm.

## 3. Khoảng Trống Nghiên Cứu (The Research Gap)
Mặc dù TO-RoPE và RoTE đã rất thành công trong việc biến RoPE thành "nhận thức thời gian" (Time-aware), chúng cùng chia sẻ một **điểm mù chí mạng**: **Tính Cứng Nhắc Trong Cảm Nhận Thời Gian (Static Time Perception).**

Trong các mô hình SOTA hiện tại, 5 ngày đối với *Người dùng A* (truy cập mỗi giờ) và 5 ngày đối với *Người dùng B* (truy cập mỗi tháng) được ánh xạ vào cùng một góc xoay tuyệt đối. Điều này trái ngược hoàn toàn với lý thuyết hành vi: tốc độ quên lãng và vòng đời sở thích phụ thuộc mạnh mẽ vào **mật độ tương tác (interaction density)** của từng cá nhân.

## 4. Đóng góp của Nghiên cứu: Context-Aware Dynamic Time Warping RoPE
Để lấp đầy khoảng trống này, chúng tôi đề xuất **Personalized ChronoRoPE** (hoặc *Dynamic Time-Warped RoPE*). 

Chúng tôi là nghiên cứu đầu tiên cá nhân hóa (Personalize) cơ chế quay của RoPE. Thay vì sử dụng một hàm biến đổi thời gian tĩnh (như $\log(t)$ với các tham số cố định), mô hình của chúng tôi tự động sinh ra các tần số xoay (rotation frequencies) dựa trên sự kết hợp giữa **Khoảng cách thời gian vật lý** và **Mật độ hoạt động ngữ cảnh của User**. 

Bằng cách này, thời gian trong LLM không trôi đi theo nhịp độ của đồng hồ vật lý, mà trôi đi theo **nhịp độ sinh học (behavioral clock)** của chính người dùng đó.
