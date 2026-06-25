# 3. Mổ Xẻ RoPE & Khai Sinh ChronoRoPE

Như đã chỉ ra ở Bài 2, Self-Attention gốc không có khái niệm về "khoảng cách". Để ép góc giữa Query và Key phải tự động nới rộng ra khi khoảng cách lịch sử tăng lên, chúng ta áp dụng một phép xoay hình học gọi là **RoPE (Rotary Position Embedding)**.

Dưới đây là sơ đồ mô tả cách RoPE cắt lớp không gian và xoay Vector (Trích xuất từ bài báo RoPE gốc).

![Cơ chế cắt lớp và xoay của RoPE](/home/hieumagic/Repositories/Recommender-System-Dashboard/Idea/Personalized_ChronoRoPE/docs/images/2104.09864_roformer_RoPE_v2.png)

## 1. Cơ Chế Hoạt Động Của RoPE (Geometric Rotation)

Thay vì cộng (+) một hằng số vào Vector như Absolute Position, RoPE sử dụng phép **nhân ma trận xoay (Rotation Matrix)**.

### Không gian cắt lớp (Feature Slicing)

Như hình ảnh bên trên minh họa, RoPE không xoay toàn bộ vector 64 chiều (bởi vì phép xoay đa chiều cực kỳ tốn tài nguyên toán học).

- Nó chia vector 64 chiều thành **32 cặp 2D độc lập**.
- Mỗi cặp được biểu diễn thành tọa độ $(x, y)$ trên một mặt phẳng hình tròn.
- Có tổng cộng 32 mặt phẳng, mỗi mặt phẳng có một **tần số xoay ($\theta_i$)** riêng biệt. Mặt phẳng số 1 xoay rất nhanh, mặt phẳng số 32 xoay siêu chậm.

### Phép Xoay

Góc xoay của mũi tên trên mỗi mặt phẳng được tính bằng: $\text{Vị trí} \times \text{Tần số mặt phẳng}$.

Giả sử Query $Q$ nằm ở vị trí $m$, và Key $K$ nằm ở vị trí $n$. Chúng sẽ bị xoay lần lượt đi một góc $m\theta$ và $n\theta$.

Khi tính Attention (Tích vô hướng - Dot Product), toán học hình học khẳng định rằng: **Góc lệch giữa $Q$ và $K$ lúc này chính xác bằng $(m - n)\theta$.**
Khoảng cách $(m - n)$ càng lớn, góc lệch càng rộng, $\cos(\text{góc})$ càng nhỏ, khiến Attention Score tự động rơi tự do. Hiệu ứng này được gọi là **Long-Term Decay** (Suy giảm dài hạn).

![Đường cong suy giảm dài hạn của RoPE](/home/hieumagic/Repositories/Recommender-System-Dashboard/Idea/Personalized_ChronoRoPE/docs/images/2104.09864_long-term-decay.png)
*(Biểu đồ cho thấy khi khoảng cách tương đối càng xa, điểm Attention tự động dao động và triệt tiêu dần về 0)*

## 2. Sự Giới Hạn Của RoPE Trong RecSys

Toán học của RoPE rất hoàn hảo cho Ngôn ngữ tự nhiên (NLP). Tuy nhiên, có một lỗ hổng chí mạng khi mang nó sang RecSys:
Trong NLP, các từ đứng cạnh nhau luôn cách đều nhau 1 đơn vị. Khoảng cách giữa từ số 1 và số 2 là $\Delta m = 1$.
Nhưng trong RecSys, **thời gian mua sắm không hề đều đặn**.

- Lần mua 1: [iPhone] (Ngày 1)
- Lần mua 2: [Ốp lưng] (Ngày 2) $\rightarrow$ Cách 1 ngày
- Lần mua 3: [Tai nghe] (Ngày 100) $\rightarrow$ Cách 98 ngày

Nếu dùng RoPE gốc (như SASRec), nó vẫn coi Lần 1, 2, 3 cách đều nhau 1 đơn vị $(\Delta m = 1)$. Điều này làm biến dạng hoàn toàn không gian thời gian của người dùng.

## 3. Lời Giải: Context-Aware Dynamic Time Warping (ChronoRoPE)

Để vá lỗ hổng này, **Personalized ChronoRoPE** của chúng ta đã ra đời! Chúng ta can thiệp thẳng vào hệ số xoay của RoPE.

Thay vì dùng vị trí tĩnh $m, n$, chúng ta thay bằng hàm thời gian vật lý liên tục đã bị bóp méo: $\tau_u(t)$.
Góc xoay lúc này trở thành: $\tau_u(t) \cdot \theta$.

Và $\tau_u(t)$ không cố định, nó được sinh ra từ một mạng Meta-MLP dựa trên **mật độ tương tác (Interaction Density)** của chính người dùng đó:

$$
\tau_u(t) = \alpha_u \cdot \log(1 + \beta_u \cdot t)
$$

**Sự kỳ diệu xảy ra:**
Nếu người dùng đó mua sắm điên cuồng (Dense User), Meta-MLP đẩy hệ số $\alpha_u$ lên cao. Mũi tên bị xoay cực gắt chỉ sau vài giờ trôi qua. Mô hình ép giỏ hàng phải quên ngay lập tức món đồ cũ để đuổi theo trend hiện tại.
Ngược lại, nếu người dùng ít khi mua (Sparse User), $\alpha_u$ tiến về 0. Mũi tên hầu như không xoay. Điểm Attention được bảo toàn vĩnh viễn suốt hàng năm trời.

> [!IMPORTANT]
> Toàn bộ kiến trúc và các công thức chứng minh toán học chuyên sâu cho cơ chế bóp méo thời gian này đã được trình bày chặt chẽ tại tài liệu **`04_Theoretical_Framework_and_Proofs.md`**.
