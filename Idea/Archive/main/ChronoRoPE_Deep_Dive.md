# ChronoRoPE: Đi Sâu Vào Nhúng Vị Trí Thời Gian Liên Tục

Tài liệu này giải thích phần cốt lõi về toán học và kiến trúc của **ChronoRoPE**, loại bỏ mọi thuật ngữ sáo rỗng. Nó trả lời ba câu hỏi: Điểm yếu chính xác của các LLM hiện tại trong RecSys là gì? Làm thế nào để sửa chữa nó về mặt toán học? Và nó được triển khai như thế nào?

---

## 1. Lỗ hổng Căn bản: "Sai lầm Khoảng cách Đồng đều" (Uniform Spacing Fallacy)

Trong các Mô hình Ngôn ngữ Lớn (LLMs) hiện đại như LLaMA hay Mistral, thứ tự của chuỗi được mã hóa bằng cách sử dụng **Rotary Position Embedding (RoPE)** (Nhúng Vị trí Xoay).

Khi một chuỗi các tương tác của người dùng được đưa vào LLM: `[Item A, Item B, Item C]`, RoPE gán một **chỉ số nguyên rời rạc ($m$)** cho mỗi món hàng:
*   Item A: $m = 1$
*   Item B: $m = 2$
*   Item C: $m = 3$

### Toán học của RoPE Tiêu chuẩn
RoPE hoạt động bằng cách xoay các vector Query ($q$) và Key ($k$) trong cơ chế Attention. Góc xoay tỷ lệ thuận với chỉ số vị trí $m$.
Đối với một attention head và chiều (dimension) cụ thể, tích vô hướng (điểm attention) giữa $q$ tại vị trí $m$ và $k$ tại vị trí $n$ trở thành một hàm của khoảng cách tương đối giữa chúng:

$$\langle q_m, k_n \rangle \propto \cos((m - n) \theta)$$

*(Trong đó $\theta$ là một tần số cơ sở cố định).*

### Vấn đề trong Thương mại Điện tử (Thời gian Không Đồng bộ)
Giả sử:
*   Người dùng mua Item A vào **Thứ Hai, 8:00 AM**.
*   Người dùng mua Item B vào **Thứ Hai, 8:05 AM** (5 phút sau).
*   Người dùng mua Item C vào **Thứ Sáu, 8:00 AM** (4 ngày sau).

Trong RoPE tiêu chuẩn, khoảng cách giữa A và B là $|2 - 1| = 1$. Khoảng cách giữa B và C là $|3 - 2| = 1$.
**LLM tính toán $\cos(1 \times \theta)$ cho cả hai cặp.**
Cơ chế Attention hoàn toàn mù mờ trước thực tế rằng Item C xảy ra sau đó tận 4 ngày. Nó coi khoảng trống 5 phút và khoảng trống 4 ngày là y hệt nhau. Nó ép thời gian vật lý vào một mạng lưới số nguyên được phân bổ đồng đều.

---

## 2. Giải pháp: ChronoRoPE (RoPE Thời gian Liên tục)

Thay vì sử dụng chuỗi chỉ số tùy ý ($m = 1, 2, 3$), chúng ta sử dụng **mốc thời gian vật lý tuyệt đối** ($t$) của lần tương tác.

Tuy nhiên, chúng ta không thể đơn giản lấy raw timestamp $t$ (ví dụ: unix timestamp `1718000000`) nhét trực tiếp vào góc xoay, bởi vì thời gian thô là tuyến tính và không được chuẩn hóa (unscaled), sẽ khiến hàm cosine dao động hỗn loạn.

### Hàm Bóp méo Thời gian $\tau(t)$
Chúng tôi giới thiệu một Mạng Nơ-ron siêu nhẹ, có thể học được $\tau(t)$ để "bóp méo" (warp) thời gian vật lý đưa vào không gian vị trí của LLM.

Thay vì $m$, vị trí bây giờ được định nghĩa là:
$$\text{Position} = \tau(t_i)$$

Trong đó $\tau(t_i)$ là một Multi-Layer Perceptron (MLP) 2 lớp đơn giản:
$$\tau(t_i) = W_2 \cdot \text{GELU}(W_1 \cdot t_i + b_1) + b_2$$

### Toán học Attention Mới
Tích vô hướng giữa query tại thời điểm $t_q$ và key tại thời điểm $t_k$ trở thành:

$$\langle q_{t_q}, k_{t_k} \rangle \propto \cos\Big( \big(\tau(t_q) - \tau(t_k)\big) \theta \Big)$$

### Tại sao đây lại là một Đột phá:
1.  **Sự Suy giảm Tự nhiên (Natural Decay):** Nếu $\tau(t_q) - \tau(t_k)$ trở nên lớn (ví dụ: khoảng cách 3 năm), hàm cosine tự nhiên dịch chuyển, làm thay đổi trọng số attention. LLM sẽ "quên" đi các món hàng ở quá khứ xa xôi một cách tự nhiên mà không cần các công thức suy giảm (decay formulas) thủ công.
2.  **Tính Chu kỳ có thể Học được (Learnable Periodicity):** Bởi vì MLP $\tau$ ánh xạ thời gian theo cách phi tuyến tính trước khi truyền nó vào hàm $\cos$ có tính chu kỳ, mô hình có thể tự thân học được các hành vi mang tính chu kỳ (ví dụ: mua áo khoác mùa đông cứ mỗi 12 tháng) trực tiếp bên trong ma trận attention.
3.  **Không Phình to Tham số (Zero Parameter Bloat):** Chúng ta không làm phình to kích thước từ vựng $|V|$, cũng không phải tính các tích phân khổng lồ như trong Temporal Point Processes (TPPs). Mạng MLP $\tau$ yêu cầu chưa tới 100 tham số.

---

## 3. Cách Triển khai (PyTorch / HuggingFace)

Việc triển khai ChronoRoPE dễ đến mức kinh ngạc vì nó móc nối (hook) trực tiếp vào các kiến trúc LLM hiện có. Chúng ta không cần phải huấn luyện một LLM từ đầu.

### HuggingFace RoPE Tiêu chuẩn:
```python
# Trong LlamaRotaryEmbedding tiêu chuẩn
# position_ids là một tensor có kích thước [batch_size, seq_len] chứa [1, 2, 3...]
inv_freq = self.inv_freq # [dim / 2]
# Tích ngoài (Outer product) để lấy góc: [1, 2, 3]^T x [freqs]
freqs = torch.einsum("i,j->ij", position_ids.float(), inv_freq) 
```

### Chỉnh sửa ChronoRoPE:
```python
class ChronoRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        
        # Đóng góp cốt lõi: Một mạng MLP nhỏ xíu để bóp méo thời gian
        self.time_warper = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )

    def forward(self, timestamps, seq_len=None):
        # timestamps có kích thước: [batch_size, seq_len] chứa thời gian vật lý thực tế (ví dụ: số ngày kể từ tương tác đầu tiên)
        
        # 1. Bóp méo thời gian vật lý vào không gian vị trí
        # Đầu vào của MLP phải là [batch_size * seq_len, 1]
        warped_time = self.time_warper(timestamps.unsqueeze(-1)).squeeze(-1) 
        
        # 2. Tính toán các tần số xoay dựa trên thời gian đã bóp méo
        freqs = torch.einsum("bi,j->bij", warped_time, self.inv_freq.to(timestamps.device))
        
        # 3. Tạo các ma trận cos và sin cho việc xoay (logic RoPE tiêu chuẩn)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()
```

## 4. Chiến lược Huấn luyện (LoRA)
Bởi vì chúng ta đang sử dụng một LLM đã được huấn luyện trước (ví dụ: LLaMA-3-8B), chúng ta sẽ **đóng băng (freeze) toàn bộ LLM**.
Chúng ta chỉ set `requires_grad=True` cho:
1.  Các Adapter LoRA (để tinh chỉnh LLM thích nghi với miền RecSys).
2.  Mạng MLP `self.time_warper`.

Hàm mất mát (loss function) vẫn là Dự đoán Token Tiếp theo (Next-Token Prediction - Cross-Entropy) tiêu chuẩn. Gradient sẽ chảy một cách mượt mà từ hàm mất mát dự đoán, xuyên qua ma trận attention, xuống thẳng mạng MLP `time_warper`, dạy cho nó biết chính xác cách kéo giãn hoặc nén thời gian vật lý sao cho độ chính xác của khuyến nghị đạt mức tối đa.