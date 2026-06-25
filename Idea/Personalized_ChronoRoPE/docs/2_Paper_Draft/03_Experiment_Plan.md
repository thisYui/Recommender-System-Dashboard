# Kế Hoạch Thực Nghiệm (Experiment Plan)

Tài liệu này vạch ra chiến lược đánh giá để chứng minh sức mạnh của **Personalized ChronoRoPE** trước các hội đồng phản biện. Chúng tôi tập trung vào 2 mục tiêu: (1) Tính hiệu quả trên dữ liệu thưa thớt (Sparse data) và (2) Khả năng ngoại suy thời gian (Zero-shot Temporal Extrapolation).

## 1. Dữ liệu Đánh giá (Datasets)

Chúng tôi sẽ sử dụng các tập dữ liệu công khai chuẩn mực, có chứa thông tin `Timestamp` chi tiết và mức độ hoạt động của user đa dạng (từ rất active đến rất sparse).

1. **Amazon Product Reviews (Beauty / Toys / Sports):** Dữ liệu siêu thưa thớt (ultra-sparse), các lần mua hàng cách nhau hàng tháng hoặc hàng năm. Lý tưởng để test sức mạnh của Log-Time Warping.
2. **Taobao User Behavior:** Dữ liệu hành vi dày đặc (clicks/carts/purchases) trong 9 ngày. Lý tưởng để test tốc độ cá nhân hóa (Personalization) trên short-term intent.

### 1.1 Kỹ thuật Tiền xử lý (Pre-processing)

Đối với mỗi user $u$, chúng tôi trích xuất:

- Chuỗi $N$ item tương tác: $S_u = [i_1, i_2, ..., i_n]$
- Chuỗi thời gian tuyệt đối (Unix epoch): $T_u = [t_1, t_2, ..., t_n]$
- Chuẩn hóa thời gian (Đặt mốc 0): $\tilde{t}_k = t_k - t_1$ (Thời gian trôi qua kể từ tương tác đầu tiên).
- Tính mật độ (Density): $\rho_u = \frac{n}{\tilde{t}_n + 1}$

## 2. Các Mô hình So sánh (Baselines)

Để chứng minh sự vượt trội, chúng tôi sẽ so sánh ChronoRoPE với 3 nhóm Baseline:

- **Base (Không có thời gian):** SASRec, LLaMA-Rec (với RoPE tiêu chuẩn).
- **Time-aware (Rời rạc/Cộng Bias):** TiSASRec (sử dụng ma trận thời gian tương đối).
- **Time-aware RoPE (SOTA - Tĩnh):** TO-RoPE (Time-and-Order RoPE) hoặc kiến trúc thay thế RoPE bằng elapsed time $\log(t)$ nhưng không có tham số cá nhân hóa.

## 3. Thiết kế Nhiệm vụ Đánh giá (Evaluation Tasks)

### Task 1: Tiên đoán Chuỗi Tiêu chuẩn (Standard Sequential Recommendation)

- **Mô tả:** Nhiệm vụ Next-item prediction cổ điển. Cho biết lịch sử $n-1$ item, dự đoán item thứ $n$.
- **Metrics:** Hit Rate (HR@10, HR@20), NDCG@10, NDCG@20.
- **Kỳ vọng:** ChronoRoPE đánh bại TO-RoPE ít nhất 5-10% trên dataset Amazon nhờ khả năng điều chỉnh nhịp thời gian cá nhân hóa.

### Task 2: Thử nghiệm Lọc User theo Mật độ (Density-based Ablation)

- **Mô tả:** Chia test-set thành 3 nhóm: Sparse Users (ít mua), Normal Users, và Dense Users (mua liên tục). Đo lường hiệu suất trên từng nhóm.
- **Kỳ vọng:** Mô hình Time-aware tĩnh (như TO-RoPE) sẽ hoạt động tốt trên nhóm Normal, nhưng thất bại ở nhóm Sparse và Dense. ChronoRoPE sẽ duy trì phong độ cao trên cả 3 nhóm nhờ Mạng $\text{MLP}_{\text{meta}}$ tự động cân bằng $\alpha, \beta$.

### Task 3: Đột phá Ngoại suy Thời gian (Zero-Shot Temporal Extrapolation)

- **Mô tả:** Đây là bài test sát thủ để chứng minh độ bền vững của mô hình (Robustness).
- **Cách làm:**
  - Trong tập Train, chúng tôi chủ ý **chỉ giữ lại** các chuỗi tương tác có tổng vòng đời (lifespan) $\le 30$ ngày.
  - Trong tập Test, chúng tôi chạy mô hình trên các user có khoảng thời gian ngắt quãng (time gap) lên tới **1 năm**.
- **Kỳ vọng:** Các mô hình dùng MLP bóp méo thời gian trực tiếp sẽ nổ (collapse) do out-of-distribution. Trong khi đó, Log-Time Warping của ChronoRoPE sẽ vẫn tự động suy giảm attention mượt mà, giúp dự đoán đúng dù khoảng cách thời gian chưa từng xuất hiện trong tập Train.

## 4. Kiến trúc Thực thi (Implementation)

- **Khung kỹ thuật (Framework):** PyTorch + HuggingFace `transformers`.
- **Mô hình lõi:** Áp dụng trên 2 quy mô (Scale):
  1. *Small-Scale:* Xây dựng kiến trúc Transformer từ đầu (giống SASRec) với 2-4 layer, tích hợp Context-aware ChronoRoPE để train nhanh trên 1 GPU.
  2. *Large-Scale:* Tích hợp vào LLaMA-3-8B thông qua việc override hàm `LlamaRotaryEmbedding`, sau đó Fine-tune bằng **LoRA** (chỉ mở gradient cho LoRA adapters và mạng $\text{MLP}_{\text{meta}}$ siêu nhỏ).
- **Tối ưu hóa:** Sử dụng FlashAttention-2 để đảm bảo tốc độ training.
