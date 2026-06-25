# Hồi 1: Bức Tranh Toàn Cảnh - Transformer-based Sequential Recommendation

Để đi sâu vào bất kỳ cải tiến cốt lõi nào, chúng ta cần hiểu rõ "sân chơi" mà chúng ta đang đứng. Sân chơi của chúng ta chính là bài toán **Sequential Recommendation (Gợi ý dựa trên chuỗi)**, và cụ thể hơn là nhánh: **Transformer-based SR** (Các hệ thống gợi ý sử dụng kiến trúc Transformer).

## 1. Sequential Recommendation (SR) là gì?

Khác với các hệ thống gợi ý truyền thống (như Collaborative Filtering) chỉ quan tâm đến việc *User A từng mua Item B*, SR quan tâm đến **Thứ tự của các hành vi theo thời gian**.

**Ví dụ thực tế:**
Giả sử một người dùng có lịch sử mua hàng như sau:
`[Điện thoại iPhone] ➔ [Ốp lưng] ➔ [Tai nghe AirPods] ➔ [?]`

Một hệ thống SR giỏi phải nhìn vào **trình tự** này để suy luận được ý định (Intent) hiện tại của người dùng: "Họ vừa mua một hệ sinh thái Apple, món tiếp theo rất có thể là Củ sạc nhanh hoặc Apple Watch", thay vì gợi ý một chiếc điện thoại Samsung (dù điện thoại Samsung rất phổ biến).

## 2. Pipeline chung của Transformer-based SR

Để biến lịch sử mua hàng của user thành một lời dự đoán chính xác, các mô hình Transformer-based SR (tiêu biểu như SASRec, BERT4Rec) đều tuân theo một quy trình (Pipeline) chuẩn mực gồm 4 bước.

### Bước 1: Tiền xử lý chuỗi hành vi (Sequence Construction)

Đầu tiên, hệ thống gom toàn bộ lịch sử tương tác của người dùng thành một mảng các Item ID, được sắp xếp theo thời gian từ cũ đến mới.
Nếu chuỗi quá dài, hệ thống sẽ cắt bớt (Truncate). Nếu chuỗi quá ngắn, hệ thống sẽ đệm thêm (Pad) để độ dài chuỗi luôn cố định (thường là $L = 50$ hoặc $L = 200$).

*Ví dụ:* Chuỗi độ dài $L=5$: `[Pad, Pad, Item_A, Item_B, Item_C]`

### Bước 2: Tầng Nhúng (Embedding Layer)

Máy tính không hiểu được ID `Item_A` là gì. Nó cần chuyển ID này thành các vector số học (ví dụ: vector 64 chiều). Ở bước này, có **hai thành phần độc lập** được sinh ra và cộng lại với nhau:

1. **Item Embedding:** Một cuốn từ điển chuyển `Item_A` thành một vector $E_A$. Vector này đại diện cho tính chất của món hàng (VD: màu sắc, giá cả, thể loại).
2. **Positional Encoding (Mã hóa vị trí):** Vì chuỗi có thứ tự, hệ thống phải báo cho mô hình biết `Item_A` đứng thứ 3, `Item_B` đứng thứ 4. Do đó, hệ thống tạo ra các vector vị trí tuyệt đối (Absolute Position) $P_3, P_4$... và **cộng thẳng** vào Item Embedding.

$$
Vector\_Đầu\_Vào = E_{Item} + P_{Position}
$$

### Bước 3: Trái tim hệ thống - Transformer Encoder

Đây là cỗ máy tiêu hóa dữ liệu. Các vector đầu vào (đã mang thông tin hàng hóa + vị trí) sẽ được ném vào các khối **Self-Attention**.
Mục tiêu của Self-Attention là tính toán sự liên quan giữa mọi món hàng trong quá khứ với nhau.

- Nó sẽ học được rằng: `Item_B (Ốp lưng)` có liên quan mật thiết đến `Item_A (iPhone)`.
- Sau nhiều lớp xử lý, Transformer nhả ra một **Vector Đại diện (User Intent Vector)**. Vector này mang ý nghĩa: *"Tại thời điểm hiện tại, đây là thứ người dùng đang khao khát"*.

### Bước 4: Dự đoán và Khớp (Prediction & Matching)

Cuối cùng, hệ thống lấy **User Intent Vector** đem đi so sánh (tính tích vô hướng - Dot Product) với toàn bộ Vector của hàng triệu Item trong kho hàng.
Món hàng nào cho ra điểm số cao nhất sẽ được chọn làm Top-1 để gợi ý cho người dùng.

---

## 3. Khâu cốt lõi và Sự khởi nguồn của vấn đề

Tại sao kiến trúc này lại thành công? Đó là nhờ sự xuất sắc của cơ chế **Self-Attention** ở Bước 3. Nó có khả năng nhìn thấu mối liên kết phức tạp giữa các món hàng.

Tuy nhiên, bản thân cơ chế Self-Attention tự nhiên lại **hoàn toàn mù quáng về mặt vị trí/thứ tự** (Permutation Equivariant). Nếu không có Bước 2 "nhét" thêm vector Positional Encoding vào, Transformer sẽ coi chuỗi `[iPhone ➔ Ốp lưng]` y hệt như `[Ốp lưng ➔ iPhone]`.

> **Vấn đề xuất hiện:**
> Cách người ta "cộng" Positional Encoding ở Bước 2 là cách học lỏm từ Xử lý ngôn ngữ tự nhiên (NLP) - nơi các từ cách đều nhau. Nhưng trong mua sắm, các lần tương tác không hề cách đều nhau!
>
> Vậy Self-Attention bên trong hoạt động cụ thể như thế nào mà lại bị mù vị trí? Và người ta đã dùng kỹ thuật gì để bắt nó nhìn thấy được vị trí một cách thông minh hơn?
