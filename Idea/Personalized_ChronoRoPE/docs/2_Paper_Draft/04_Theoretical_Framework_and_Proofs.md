# Khung Lý Thuyết và Chứng Minh Toán Học (Theoretical Framework & Proofs)

Trong phần này, chúng tôi cung cấp các chứng minh toán học nghiêm ngặt để giải thích tại sao **Personalized ChronoRoPE** không chỉ mang lại hiệu suất thực nghiệm cao mà còn vượt trội về mặt nền tảng lý thuyết so với các phương pháp nhúng vị trí (Positional Embedding) hiện hành.

---

## 1. Định lý 1: Tính Bất Biến Tịnh Tiến trong Không gian Thời gian Bóp méo (Translation Invariance in Warped Time Space)

Cơ sở của RoPE (Rotary Position Embedding) truyền thống nằm ở việc biến đổi góc xoay sao cho phép nhân vô hướng (dot product) giữa Query và Key phụ thuộc hoàn toàn vào khoảng cách tương đối $m - n$.

**Định lý 1:** *Khi thay thế chỉ số chuỗi rời rạc $m \in \mathbb{N}$ bằng một hàm thời gian liên tục $\tau(t) \in \mathbb{R}$, phép nhân vô hướng của RoPE vẫn bảo toàn tính bất biến tịnh tiến (translation invariance), nhưng ở trong không gian thời gian đã bị bóp méo.*

**Chứng minh:**
Giả sử vector query $\mathbf{q}$ tại thời điểm $t_q$ và vector key $\mathbf{k}$ tại thời điểm $t_k$. Phép xoay RoPE được định nghĩa thông qua ma trận xoay khối (block-diagonal rotation matrix) $\mathbf{R}_{\Theta,\tau(t)}$, với $\Theta = \{\theta_l = b^{-2l/d}\}_{l=1}^{d/2}$ là tập các tần số gốc.

Ta có:
$$ \tilde{\mathbf{q}} = \mathbf{R}_{\Theta, \tau(t_q)} \mathbf{q} $$
$$ \tilde{\mathbf{k}} = \mathbf{R}_{\Theta, \tau(t_k)} \mathbf{k} $$

Tích vô hướng (Attention Score) trước khi qua Softmax được tính bằng:
$$ \langle \tilde{\mathbf{q}}, \tilde{\mathbf{k}} \rangle = (\mathbf{R}_{\Theta, \tau(t_q)} \mathbf{q})^T (\mathbf{R}_{\Theta, \tau(t_k)} \mathbf{k}) $$
$$ = \mathbf{q}^T \mathbf{R}_{\Theta, \tau(t_q)}^T \mathbf{R}_{\Theta, \tau(t_k)} \mathbf{k} $$

Theo tính chất ma trận trực giao (orthogonal) của phép xoay: $\mathbf{R}_{\alpha}^T \mathbf{R}_{\beta} = \mathbf{R}_{\beta - \alpha}$. Do đó:
$$ \langle \tilde{\mathbf{q}}, \tilde{\mathbf{k}} \rangle = \mathbf{q}^T \mathbf{R}_{\Theta, \tau(t_q) - \tau(t_k)} \mathbf{k} $$

**Hệ quả:** Điểm Attention không phụ thuộc vào mốc thời gian tuyệt đối của hệ thống (như năm 2023 hay 2024), mà chỉ phụ thuộc duy nhất vào **khoảng cách thời gian tương đối đã bị bóp méo** $\Delta \tau = \tau(t_q) - \tau(t_k)$. Điều này hoàn toàn thỏa mãn lý thuyết quy nạp của mô hình Sequence Modeling.

---

## 2. Phân tích và So sánh Các Hàm Bóp méo Thời gian (Time-Warping Functions)

Để định hình $\tau(t)$, chúng ta cần một hàm số mô phỏng chính xác "Đường cong quên lãng" (Ebbinghaus Forgetting Curve) trong tâm lý học con người. Chúng tôi đề xuất và phân tích hai hàm phi tuyến: **Logarithmic Decay** và **Exponential Decay**.

### 2.1 Hàm Logarit (Logarithmic Time Warping)
Định nghĩa: 
$$ \tau_{log}(t) = \alpha \log(1 + \beta t) $$
Với $\alpha, \beta > 0$. Tham số $\beta$ kiểm soát tốc độ phân rã, $\alpha$ kiểm soát biên độ tần số.

**Đạo hàm (Tốc độ thay đổi cảm nhận thời gian):**
$$ \tau'_{log}(t) = \frac{\alpha \beta}{1 + \beta t} $$
*Nhận xét:* Tốc độ thay đổi cao ở những $t$ nhỏ (các sự kiện gần đây), và tiến dần về $0$ một cách cực kỳ chậm (Long-tail) khi $t$ lớn. Hàm này rất phù hợp cho lĩnh vực Thương mại điện tử (E-commerce), nơi mà một món đồ mua cách đây 2 năm vẫn mang giá trị phản ánh sở thích dài hạn (Long-term preference) của người dùng.

### 2.2 Hàm Mũ Suy Giảm (Exponential Decay Time Warping)
Định nghĩa:
$$ \tau_{exp}(t) = \gamma (1 - e^{-\lambda t}) $$
Với $\gamma, \lambda > 0$.

**Đạo hàm:**
$$ \tau'_{exp}(t) = \gamma \lambda e^{-\lambda t} $$
*Nhận xét:* Hàm số này hội tụ cực kỳ nhanh về ngưỡng cận trên $\gamma$. Nếu khoảng thời gian $\Delta t$ đủ lớn, $\tau_{exp}(t)$ gần như bằng $\gamma$, làm cho khoảng cách tương đối giữa các item cũ trở thành $0$. Hàm này "xóa sổ" hoàn toàn quá khứ xa xôi, cực kỳ phù hợp cho các nền tảng biến đổi thị hiếu chóng mặt như Short-video (TikTok/Reels).

---

## 3. Định lý 2: Tính An toàn Ngoại suy (Extrapolation Safety)

**Vấn đề của các mô hình tiền nhiệm (như TO-RoPE):** 
TO-RoPE sử dụng biến đổi tuyến tính $\tau_{to}(t) = c \cdot t$. Nếu một người dùng "ngủ đông" (inactive) trong 3 năm ($t = 1000$ ngày), hàm tuyến tính sẽ tạo ra một góc xoay khổng lồ $\theta \cdot 1000$. Mức xoay này chưa từng xuất hiện trong tập Training, đẩy mô hình ra khỏi vùng phân phối (Out-of-Distribution - OOD), dẫn đến kết quả nhiễu loạn (Extrapolation Failure).

**Định lý 2:** *Các hàm $\tau_{log}(t)$ và $\tau_{exp}(t)$ giới hạn sự bùng nổ của góc xoay, đảm bảo tính an toàn ngoại suy (Extrapolation Safety) trong mọi trường hợp $t \to \infty$.*

**Chứng minh:**
- Đối với hàm Exponential: 
  $$ \lim_{t \to \infty} \tau_{exp}(t) = \lim_{t \to \infty} \gamma (1 - e^{-\lambda t}) = \gamma $$
  Góc xoay đạt cực đại tuyệt đối tại $\gamma \Theta$. Mô hình hoàn toàn an toàn bất chấp thời gian $t$ dài đến đâu.
  
- Đối với hàm Logarithmic:
  Đánh giá tốc độ tăng trưởng (Growth rate) bằng giới hạn:
  $$ \lim_{t \to \infty} \frac{\tau_{log}(t)}{t} = \lim_{t \to \infty} \frac{\alpha \log(1 + \beta t)}{t} = 0 $$
  Mặc dù $\tau_{log}(t)$ tiến ra vô cùng, nhưng sự tăng trưởng của nó là **dưới tuyến tính (sub-linear)**. Một khoảng cách 3 năm ($1000$ ngày) sẽ bị nén lại thành kích thước tương đương $\sim \log(1000) \approx 6.9$. Do đó, ma trận Key/Query hiếm khi vượt ra khỏi vùng giá trị mà mô hình đã học, loại bỏ hiện tượng phát nổ gradient.

---

## 4. Phân tích Cực biên: Xử lý Cold-start và Heavy Users (Corner Cases)

Tham số $\alpha, \beta$ không cố định mà được sinh ra từ Mạng Nơ-ron Meta-MLP nhận đầu vào là mật độ tương tác $\rho_u$.

- **Heavy Users (Người dùng mua sắm cực độ):** $\rho_u \gg 1$. Meta-MLP sẽ sinh ra $\alpha, \beta$ rất lớn. Đường cong thời gian dốc đứng. Khoảng cách 1 ngày đối với họ bị khuếch đại thành góc xoay lớn, ép mô hình "quên" các item của ngày hôm trước vì sở thích của họ thay đổi từng giờ.
- **Cold-start / Sparse Users (Người dùng mới):** $\rho_u \approx 0$.
  - Nếu số lượng item $< 3$: Mật độ không thể tính toán chính xác. Hệ thống sẽ **tự động fallback** (lùi về) một hằng số $\rho_{avg}$ (mật độ trung bình của toàn bộ dataset) hoặc gán $\beta = 0$, biến RoPE trở về dạng Positional Embedding tiêu chuẩn $m$ để giữ an toàn tuyệt đối.
  - Khi $\rho_u$ cực nhỏ, $\alpha$ và $\beta$ sinh ra tiến sát về 0. Hàm Logarit tiệm cận với $0$, đồng nghĩa với việc góc xoay gần như không đổi. Mô hình sẽ giữ lại toàn bộ trí nhớ về người dùng này suốt nhiều năm.

---

## 5. Bảng Phân Tích Độ Phức Tạp (Complexity Analysis) so với SOTA

Một trong những đóng góp lớn nhất của **Personalized ChronoRoPE** là phá vỡ nút thắt cổ chai về bộ nhớ của các mô hình Time-Aware (như TiSASRec).

Giả sử $B$ là Batch Size, $S$ là Sequence Length, $D$ là Hidden Dimension.

| Phương pháp | Loại | Không gian tính toán Time Matrix | Độ Phức Tạp Không Gian | Hỗ trợ FlashAttention-2/3 |
| :--- | :--- | :--- | :--- | :--- |
| **SASRec** | Không dùng thời gian | $\text{N/A}$ | $\mathcal{O}(B \cdot S \cdot D)$ | **Có** |
| **TiSASRec** | Relative Time Matrix | Xây dựng ma trận khoảng cách thời gian giữa mọi cặp $m, n$ | $\mathcal{O}(B \cdot S^2)$ | Không |
| **TO-RoPE** | Tuyến tính, Tĩnh | Modulate thẳng vào $Q, K$ | $\mathcal{O}(B \cdot S \cdot D)$ | **Có** |
| **ChronoRoPE (Đề xuất)** | Phi tuyến, Cá nhân hóa | Modulate thẳng vào $Q, K$ với $\tau(t)$ | $\mathcal{O}(B \cdot S \cdot D)$ | **Có** |

**Kết luận:** Phương pháp của chúng tôi bổ sung khả năng "Cảm nhận thời gian cá nhân hóa" (Personalized Time-Awareness) với sức mạnh cao hơn TiSASRec, nhưng lại duy trì độ phức tạp bộ nhớ tuyến tính $\mathcal{O}(B \cdot S)$ của SASRec, cho phép mô hình dễ dàng "scale-up" lên các LLMs hàng tỷ tham số thông qua công nghệ FlashAttention.
