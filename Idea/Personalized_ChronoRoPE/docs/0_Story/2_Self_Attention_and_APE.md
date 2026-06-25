# Hồi 2: Trái Tim Self-Attention và Lỗ Hổng Của Việc "Cộng" Vị Trí

Như đã đề cập ở cuối Hồi 1, Transformer có một khả năng kỳ diệu trong việc tìm ra sự liên hệ giữa các món hàng. Tuy nhiên, nó lại mắc một "căn bệnh bẩm sinh": **Mù vị trí** (Permutation Equivariant).

Để hiểu tại sao, chúng ta phải phẫu thuật trái tim của Transformer: cơ chế **Self-Attention**.

## 1. Bản chất của Self-Attention: Cơ chế Q, K, V

Khi một chuỗi các món hàng (đã biến thành vector) đi vào Transformer, mỗi vector món hàng sẽ tự nhân bản để tạo ra 3 vector hoàn toàn mới thông qua các phép nhân ma trận. Đó là:

1. **Q (Query - Truy vấn):** Đại diện cho câu hỏi *"Hiện tại tôi đang tìm kiếm thông tin/đặc điểm gì?"*
2. **K (Key - Chìa khóa):** Đại diện cho câu trả lời *"Tôi đang nắm giữ đặc điểm gì?"*
3. **V (Value - Giá trị):** Chứa *"Nội dung thực sự"* của món hàng đó.

**Làm sao để biết món hàng A (hiện tại) có liên quan đến món hàng B (trong quá khứ) hay không?**
Hệ thống sẽ lấy **Truy vấn của A ($Q_A$)** nhân vô hướng (Dot Product) với **Chìa khóa của B ($K_B$)**.
Nếu kết quả nhân ra điểm số cao, nghĩa là "Câu hỏi" và "Chìa khóa" khớp nhau ➔ Món B rất quan trọng để suy luận ra Món A.

> **Điểm mù toán học:** Phép nhân vô hướng (Dot Product) giữa hai vector hoàn toàn không bị ảnh hưởng bởi thứ tự. Cho dù `Item_B` nằm ngay sát `Item_A` hay nằm cách xa 100 vị trí, phép nhân $Q_A \cdot K_B$ vẫn cho ra cùng một kết quả.
> 👉 Đó là lý do bản thân cấu trúc của Transformer không hề biết được thứ tự trước sau! Nó nhìn mọi món hàng như một mớ bòng bong không có trình tự.

## 2. Giải pháp sơ khai: Absolute Positional Encoding (APE)

Để "chữa mù" cho Transformer, các nhà nghiên cứu ban đầu nghĩ ra một cách rất thô sơ học lỏm từ Xử lý ngôn ngữ tự nhiên: **Absolute Positional Encoding (Mã hóa vị trí tuyệt đối)**.

Họ tạo ra các vector vị trí tĩnh mang định danh tuyệt đối (ví dụ: $P_1, P_2, P_3...$) và **Cộng thẳng (Add)** vào vector món hàng ban đầu trước khi tạo ra Q, K, V.

- `Item_A` (ở vị trí 1) trở thành `Item_A + P_1`
- `Item_B` (ở vị trí 2) trở thành `Item_B + P_2`

**Nhược điểm chí mạng của phép cộng:**

1. **Gây nhiễu dữ liệu:** Phép cộng làm "nhòe" đi thông tin biểu diễn gốc của món hàng (Item Embedding). Món hàng mang thêm thông tin vị trí, nhưng lại mất đi một phần sự trong sáng của đặc trưng gốc.
2. **Khó học khoảng cách tương đối:** Mô hình bị nhồi nhét vị trí tuyệt đối, nên rất khó để nó tự suy luận ra được **khoảng cách tương đối** (VD: khoảng cách giữa vị trí 5 và 6 cũng giống hệt như vị trí 100 và 101).
3. **Giới hạn mở rộng:** Khi áp dụng cho Recommender Systems (Hệ thống gợi ý), phép cộng này không thể biểu diễn được **Khoảng cách thời gian thực (Timestamp)** vì các vị trí chỉ là index tĩnh đếm từ 1 đến N.

## 3. Câu hỏi đặt ra (Dẫn vào Hồi 3)

Nếu phép **CỘNG** vector mang lại quá nhiều nhược điểm như vậy, liệu có phép toán nào khác có thể giúp Transformer vừa nhận biết được khoảng cách vị trí tương đối một cách hoàn hảo, vừa giữ nguyên được đặc trưng của món hàng?

Và đó chính là lúc giới Toán học và AI chứng kiến một cú "Big Bang" mang tên **Rotary Position Embedding (RoPE)**.

👉 Chúng ta sẽ đi sâu vào cơ chế lượng giác tuyệt mỹ của RoPE ở **Hồi 3**.
