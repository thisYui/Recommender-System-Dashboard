# 2. Cơ Chế Self-Attention: Trái Tim Của Transformer

Nếu Transformer là một bộ não, thì **Self-Attention** chính là mảng nơ-ron chịu trách nhiệm liên kết các ký ức lại với nhau.

Mục đích của Self-Attention trong RecSys là trả lời câu hỏi: *"Trong giỏ hàng của user, món đồ hiện tại có liên quan mật thiết đến món đồ nào trong quá khứ?"*. Dưới đây là hình ảnh minh họa cơ chế Attention trích xuất từ bài báo MEANTIME:

![Cơ chế Self-Attention (Nguồn: MEANTIME)](/home/hieumagic/Repositories/Recommender-System-Dashboard/Idea/Personalized_ChronoRoPE/docs/images/meantime_attention-1.png)

## 1. Q, K, V (Query, Key, Value)

Để mô phỏng quá trình "tìm kiếm sự liên quan", Transformer biến mỗi món hàng thành 3 nhân cách khác nhau:

- **Query (Q - Người đi hỏi):** Đại diện cho món hàng ở hiện tại đang muốn tìm hiểu về quá khứ. Nó mang câu hỏi: *"Tôi là Ốp lưng, có ai liên quan đến tôi không?"*
- **Key (K - Người trả lời):** Đại diện cho các món hàng trong quá khứ. Nó mang câu trả lời: *"Tôi là iPhone, tôi là đồ điện tử..."*
- **Value (V - Giá trị thực):** Thông tin cốt lõi của món hàng sẽ được truyền đi nếu Query và Key khớp nhau.

## 2. Đo Lường Sự Tương Đồng Bằng Tích Vô Hướng (Dot Product)

Làm sao máy tính biết được "Câu hỏi" (Query) và "Câu trả lời" (Key) có khớp nhau không? Câu trả lời nằm ở môn Hình học cấp 3: **Tích Vô Hướng (Dot Product)**.

Query và Key bản chất là 2 mũi tên (Vector) trong không gian.

- Nếu 2 mũi tên **chĩa cùng một hướng** (Góc kẹp giữa chúng rất nhỏ $\rightarrow \cos(0^\circ) = 1$), Tích vô hướng sẽ cực kỳ lớn. Máy tính hiểu rằng 2 món hàng này rất liên quan.
- Nếu 2 mũi tên **vuông góc với nhau** ($\cos(90^\circ) = 0$), Tích vô hướng bằng 0. Máy tính hiểu rằng 2 món hàng này chả liên quan gì nhau.

Công thức cốt lõi của Attention:

$$
\text{Attention Score} = \text{Softmax} \left( \frac{Q \times K^T}{\sqrt{d}} \right) \times V
$$

1. Nó lấy Query của món hàng hiện tại nhân vô hướng ($Q \times K^T$) với toàn bộ các Key trong quá khứ.
2. Nó dùng hàm **Softmax** để biến các điểm số này thành xác suất (tổng bằng 100%).
3. Nó nhân xác suất này với **Value** để rút ra thông tin cuối cùng.

> [!TIP]
> **Nhìn vào hình minh họa ở trên:** Bạn sẽ thấy một khối lập phương (Attention Matrix) có các ô màu đậm nhạt khác nhau. Ô màu càng đậm nghĩa là Dot Product giữa Query của món hàng $i$ và Key của món hàng $j$ càng cao $\rightarrow$ Chúng đang "chú ý" (Attend) tới nhau rất mạnh!

## 3. Lỗ Hổng Chết Người Của Self-Attention Gốc

Hãy nhớ lại: Tích vô hướng đo góc giữa 2 Vector. Nhưng các Vector Query và Key sinh ra chỉ dựa vào **tính chất của món hàng** (Ví dụ: Ốp lưng và iPhone luôn có góc nhỏ vì chúng hợp nhau).

**Vấn đề:** Nếu một User mua [iPhone] từ tận năm 2010, và hôm nay (2026) User mua [Ốp lưng]. Mặc dù tính chất của 2 món hàng này rất khớp nhau, nhưng vì khoảng cách thời gian quá xa, lẽ ra sự liên quan phải bị giảm đi đáng kể!
Nhưng Self-Attention gốc không hề biết được khoảng cách này. Góc giữa Query và Key của chúng vẫn nhỏ, và Attention Score vẫn cao một cách sai lệch!

👉 **Đây chính là lúc chúng ta phải triệu hồi RoPE (Rotary Position Embedding). Kỹ thuật này sẽ "ép" góc của Query và Key phải tự động tách xa nhau ra nếu khoảng cách thời gian của chúng càng lớn.** (Xem tiếp Bài 3)
