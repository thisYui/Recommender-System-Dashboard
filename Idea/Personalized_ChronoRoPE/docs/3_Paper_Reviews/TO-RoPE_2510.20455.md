# Phân Tích Bài Báo: TO-RoPE (Rotate Both Ways: Time-and-Order RoPE for Generative Recommendation)

**Tác giả:** Xiaokai Wei, Jiajun Wu, Daiyao Yi, Reza Shirkavand, Michelle Gong (Roblox & University of Maryland)  
**Năm xuất bản:** Tháng 10, 2025 (arXiv)  
**Lĩnh vực:** Generative Recommendation, Transformers, Positional Embedding  

---

## 1. Vấn đề cốt lõi (Problem Statement)
Các mô hình Generative Recommendation (GR) hiện đại (như SASRec, HSTU) thường coi lịch sử người dùng là một chuỗi tuần tự (Sequence) và áp dụng cơ chế tự chú ý (Self-Attention). Tuy nhiên, hành vi người dùng trong thực tế bị chi phối bởi 2 trục thông tin:
1. **Discrete Order (Vị trí/Thứ tự):** Món hàng A được mua trước món hàng B.
2. **Occurrence Time (Thời gian thực):** Khoảng thời gian thực tế giữa 2 lần mua sắm (Burstiness - mua dồn dập, hoặc Calendar effects - tính chu kỳ theo ngày/tuần).

**Nhược điểm của các phương pháp cũ:**
- **Index-only RoPE (RoPE cơ bản):** Chỉ biết thứ tự (1, 2, 3), hoàn toàn mù tịt về việc món thứ 2 cách món thứ 3 một ngày hay một năm.
- **Time-only RoPE:** Chỉ dùng thời gian thực. Nếu 2 món hàng được mua rất sát nhau (khoảng cách thời gian gần như bằng 0), mô hình sẽ không phân biệt được thứ tự trước sau của chúng.
- **Relative Bias (HSTU, T5):** Cộng thêm điểm bias vào Attention Score. Rất khó thiết kế, cồng kềnh và không trực tiếp can thiệp vào bộ não (vector feature) của món hàng.

## 2. Giải pháp của TO-RoPE (Time-and-Order RoPE)
Để giải quyết bài toán này, nhóm tác giả từ Roblox đề xuất **TO-RoPE** - một khung lý thuyết (framework) kết hợp song song cả Thứ tự ($i$) và Thời gian ($\tau$) vào bên trong góc xoay của RoPE.

### Công thức Tổng quát
Góc xoay của một mặt phẳng (rotary plane) thứ $k$ tại vị trí thứ $i$ với mốc thời gian $\tau$ được định nghĩa bằng sự kết hợp tuyến tính:
$$ \theta_k(i) = (1 - \lambda_k)\alpha_k i \omega_k + \lambda_k \alpha_k \tau \omega_k $$

Trong đó:
- $i$: Thứ tự (Index).
- $\tau$: Thời gian thực (đã chuẩn hóa).
- $\omega_k$: Tần số xoay (Frequency bank).
- $\lambda_k \in [0, 1]$: Cổng kiểm soát (Gate) quyết định tỷ lệ đóng góp của Thời gian và Vị trí.

### 3 Biến thể (Instantiations) của TO-RoPE
Nếu kết hợp bừa bãi Vị trí và Thời gian vào cùng một góc xoay, mô hình sẽ bị **Giao thoa triệt tiêu (Destructive Interference)**: Góc thời gian và góc vị trí triệt tiêu lẫn nhau khiến hàm Cosine/Sine bị nhiễu loạn. Để khắc phục, tác giả đề xuất 3 cách cài đặt:

1. **Early Fusion (Trộn chung):** 
   Cộng trực tiếp góc thời gian và vị trí trong cùng một mặt phẳng. (Hiệu năng không ổn định do bị giao thoa triệt tiêu).
   
2. **Split-by-Dimension (Tách theo Chiều):** 
   Chia Vector 64 chiều thành 2 nhóm. 
   - Nhóm 1 (ví dụ 32 chiều đầu) chuyên xoay theo Vị trí ($i$).
   - Nhóm 2 (32 chiều sau) chuyên xoay theo Thời gian ($\tau$).
   👉 *Loại bỏ hoàn toàn sự giao thoa.*

3. **Split-by-Head (Tách theo Đầu Attention):**
   Trong kiến trúc Multi-Head Attention, dành ra một số Head chỉ để đo Vị trí (Index Heads) và một số Head chỉ để đo Thời gian (Time Heads). Các Head này sẽ tự dung hợp thông tin với nhau ở lớp Feed Forward.

## 3. Kết quả Thực nghiệm
- **Datasets:** MovieLens-20M và Dataset Nội bộ của Roblox (hàng chục triệu user).
- **Kết quả:** TO-RoPE (đặc biệt là biến thể Split-by-Head và Split-by-Dim) đánh bại hoàn toàn các phương pháp Positional Embedding trước đây (Absolute PE, Relative Bias của HSTU, Index-only RoPE).
- **Tỷ lệ Split vàng (Capacity Allocation):** Phân bổ khoảng 30% đến 50% tài nguyên (Head hoặc Dimension) cho Thời gian, phần còn lại cho Vị trí là mang lại kết quả HR@10 và NDCG tốt nhất.

## 4. Tầm nhìn cho ChronoRoPE của chúng ta
- **Sự kế thừa:** Chúng ta có thể thấy TO-RoPE đã chỉ ra rằng việc sử dụng chung (Fusion) cả Thời gian và Vị trí là chìa khóa chiến thắng.
- **Hạn chế của TO-RoPE:** TO-RoPE coi Thời gian $\tau$ là một biến số tuyến tính chung cho tất cả mọi người. Nó dùng một góc xoay "một kích cỡ vừa cho tất cả" (one-size-fits-all).
- **Cơ hội của ChronoRoPE:** Đây chính là tử huyệt của TO-RoPE mà chúng ta sẽ đánh vào! Bằng cách thay thế mốc $\tau$ tuyến tính bằng một hàm phi tuyến được cá nhân hóa qua Meta-MLP (đọc mật độ tương tác của user), ChronoRoPE sẽ bóp méo thời gian sao cho phù hợp với nhịp sinh học mua sắm của từng cá nhân.
