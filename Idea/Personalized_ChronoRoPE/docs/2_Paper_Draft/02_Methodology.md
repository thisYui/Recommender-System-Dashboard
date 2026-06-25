# Phương pháp luận (Methodology)

Nghiên cứu này giới thiệu **Personalized ChronoRoPE**, một cơ chế nhúng vị trí (Positional Embedding) động, kết hợp trực tiếp thời gian vật lý và đặc điểm hành vi của người dùng vào các góc xoay (rotation angles) của Transformer.

## 1. Cơ sở: Rotary Position Embedding (RoPE)
Cho một query vector $\mathbf{q}$ tại vị trí $m$, RoPE áp dụng phép xoay hình học để mã hóa vị trí tuyệt đối $m$:
$$ \mathbf{q}_m = \mathbf{q} \odot \cos(m \Theta) + \mathbf{q}_{\text{rotate}} \odot \sin(m \Theta) $$
Trong đó, $\Theta = \{\theta_l\}$ là tập hợp các tần số giảm dần. Tích vô hướng (Attention score) giữa Query tại $m$ và Key tại $n$ tự động trở thành hàm của khoảng cách tương đối $(m - n)$:
$$ \langle \mathbf{q}_m, \mathbf{k}_n \rangle \propto \cos((m - n) \Theta) $$

## 2. Bóp méo Thời gian Động (Dynamic Time Warping)
Thay vì sử dụng chỉ số chuỗi đồng đều $m$, chúng tôi thay thế nó bằng một hàm bóp méo thời gian liên tục $\tau(t)$, trong đó $t$ là khoảng thời gian vật lý trôi qua kể từ lần tương tác đầu tiên của người dùng.

Để giải quyết vấn đề ngoại suy (Extrapolation Failure) của các mô hình tuyến tính, chúng tôi đề xuất 2 dạng hàm phi tuyến (Xem chi tiết chứng minh tại **[Khung Lý thuyết](04_Theoretical_Framework_and_Proofs.md)**):

1. **Logarithmic Decay (Dành cho E-commerce):**
   $$ \tau(t) = \alpha \cdot \log(1 + \beta \cdot t) $$
   Mô phỏng khả năng giữ trí nhớ dài hạn (Long-term memory) về sở thích cũ.
2. **Exponential Decay (Dành cho Short-Video/News):**
   $$ \tau(t) = \gamma \cdot (1 - e^{-\lambda \cdot t}) $$
   Quên triệt để quá khứ xa xôi, thiết lập ngưỡng giới hạn tuyệt đối.

## 3. Tốc Độ Thời Gian Cá Nhân Hóa thông qua Meta-MLP
Sự đột phá của thuật toán nằm ở việc các tham số ($\alpha, \beta$ hoặc $\gamma, \lambda$) **không phải là các hằng số cố định** mà được sinh ra động cho từng người dùng thông qua Mạng Điều phối Siêu nhỏ (Meta-MLP).

### 3.1 Trích xuất Mật độ Tương tác (Interaction Density)
Tốc độ Hoạt động (Activity Rate) $\rho_u$ của user $u$:
$$ \rho_u = \frac{N_u}{T_{\text{last}} - T_{\text{first}} + \epsilon} $$
*Xử lý Cold-start Users:* Đối với người dùng có dưới 3 tương tác, hệ thống tự động gán $\rho_u = \rho_{avg}$ (mật độ trung bình) hoặc fallback về cơ chế RoPE truyền thống để giữ tính ổn định cho mạng lưới.

### 3.2 Sinh Tham Số Bằng Mạng Nơ-ron
Đặc trưng mật độ $\rho_u$ được đưa qua một Meta-MLP sử dụng hàm kích hoạt Softplus để đảm bảo các tham số thời gian luôn dương:
$$ \begin{bmatrix} \alpha_u \\ \beta_u \end{bmatrix} = \text{Softplus}(\mathbf{W}_2 \cdot \text{GELU}(\mathbf{W}_1 \cdot \rho_u + \mathbf{b}_1) + \mathbf{b}_2) $$

### Hiệu ứng đạt được:
- **Người dùng dày đặc (Dense user - $\rho_u$ cao):** Tham số sinh ra lớn, đường cong thời gian dốc đứng. Mô hình "quên" các item cũ rất nhanh để tập trung vào xu hướng mua sắm dồn dập ở hiện tại.
- **Người dùng thưa thớt (Sparse user - $\rho_u$ thấp):** Tham số sinh ra tiệm cận 0. Tần số xoay diễn ra vô cùng chậm. Mô hình bảo toàn "trí nhớ" về các item từ rất lâu trong quá khứ của họ.

## 4. Tương thích Cấu trúc (Architectural Compatibility)
Bởi vì sự can thiệp của chúng tôi chỉ xảy ra ở bước tiền xử lý **Position Index Generation**, cấu trúc nhân vô hướng của ma trận $Q$ và $K$ không hề thay đổi.
Điều này giúp Personalized ChronoRoPE **tương thích 100% với FlashAttention-2/3**, giữ nguyên độ phức tạp không gian và thời gian ở mức $\mathcal{O}(N)$ thay vì $\mathcal{O}(N^2)$ như phương pháp TiSASRec. Meta-MLP vô cùng nhẹ (<100 tham số), lý tưởng để áp dụng làm module LoRA cho các LLMs cỡ lớn.
