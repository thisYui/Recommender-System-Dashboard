# RoTE, TO-RoPE và Lỗ hổng

Nhờ kiến trúc lượng giác xoay góc của RoPE (như đã nói ở Hồi 3), chúng ta đã giải quyết được việc đo lường **khoảng cách tương đối** một cách vô cùng thanh lịch mà không làm nhiễu dữ liệu gốc.

Nhưng hãy nhớ lại công thức khởi thủy của RoPE bên Xử lý ngôn ngữ tự nhiên (NLP): Góc xoay được tính bằng $m \times \theta$, với $m$ là vị trí Index (1, 2, 3...).

## 1. Sự thức tỉnh của RecSys: Thay Index bằng Thời gian (Timestamp)

Trong thương mại điện tử, khoảng cách giữa lần mua thứ 1 và thứ 2 có thể là 1 ngày, nhưng khoảng cách giữa lần thứ 2 và thứ 3 có thể là 1 năm. Nếu vẫn dùng Index $m$ (1, 2, 3), mô hình sẽ cào bằng hai khoảng cách này là "cách nhau 1 bước".

Sự vô lý này dẫn đến sự bùng nổ của các nghiên cứu tiên phong (năm 2025-2026) nhằm đưa **Thời gian thực** vào RoPE. Hai đại diện xuất sắc nhất là **RoTE** và **TO-RoPE**.

### RoTE (2026): Phân rã thời gian

Các tác giả của RoTE nhận thấy Timestamp (dấu thời gian Unix, ví dụ: `171638201`) rất khó để mô hình tiêu hóa trực tiếp. Do đó, họ băm nhỏ thời gian thành cấu trúc phân cấp: **Năm, Tháng, Ngày**. Họ áp dụng phép xoay RoPE riêng biệt cho từng cấp độ này rồi tổng hợp lại. Cách này giúp mô hình nhận biết được tính chu kỳ của thời gian.

### TO-RoPE (2025): Trộn lẫn Thời gian và Vị trí

TO-RoPE (thiết kế riêng cho Generative RS) lại đi theo một hướng kết hợp. Nhận ra rằng đôi khi thời gian bị nhiễu hoặc có nhiều món hàng mua cùng một lúc (cùng Timestamp), họ tính góc xoay dựa trên một sự kết hợp tuyến tính giữa **Vị trí (Index)** và **Thời gian (Timestamp)** để chúng bổ trợ lẫn nhau.

## 2. Lỗ hổng của RoTE và TO-RoPE

Phải công nhận, RoTE và TO-RoPE đã đẩy giới hạn của Sequential Recommendation lên một tầm cao mới. Tuy nhiên, dưới góc độ thấu hiểu hành vi con người, chúng vướng phải một sai lầm về mặt logic thiết kế: **Chúng giả định thời gian trôi qua với một tốc độ giống hệt nhau đối với mọi User!**

Hãy cùng xét một ví dụ thực tế về 2 người dùng có hành vi mua sắm hoàn toàn trái ngược:

- **User A (Tín đồ nghiện mua sắm):** Mua hàng trên Shopee mỗi ngày. Với người này, nếu họ **dừng mua sắm trong 1 tuần**, đó là một tín hiệu bất thường rất lớn (có thể họ vừa chuyển nhà, hoặc đổi nền tảng). Khoảng trống 1 tuần đối với User A mang trọng số thông tin cực kỳ cao.
- **User B (Người mua thời vụ):** Chỉ mua hàng 6 tháng 1 lần (vào dịp Tết và ngày sinh nhật). Với người này, **khoảng trống 1 tuần** không có ý nghĩa gì cả, nó trôi qua như một cái chớp mắt.

> **Sự "bằng" của Toán học:**
> Dù dùng RoTE hay TO-RoPE, khi tính góc xoay cho khoảng thời gian 1 tuần, công thức của họ sẽ tính ra **một góc xoay có biên độ y hệt nhau** cho cả User A và User B.
>
> Chúng đã thất bại hoàn toàn trong việc thấu hiểu **"Nhịp độ sinh học" (Temporal Density)** của từng cá nhân. Một khoảng thời gian không bao giờ mang cùng một ý nghĩa đối với hai người khác nhau!

## 3. Lời giải của chúng ta: Sự xuất hiện của Personalized ChronoRoPE

Vậy làm thế nào để giải quyết sự "bằng" vô lý này? Làm sao để góc xoay của mô hình tự động "nở ra" khi gặp tín đồ mua sắm, và "co lại" khi gặp người mua thời vụ?

Đó chính là lúc chúng ta tung ra đòn quyết định: **Personalized ChronoRoPE (Cá nhân hóa cơ chế RoPE theo nhịp độ thời gian)**.

Thay vì đưa Timestamp trực tiếp vào công thức góc xoay một cách khô khan và cứng nhắc, chúng ta sẽ biến Thời gian thành một đại lượng có tính chất **Co giãn (Time-Warping)** dựa trên Mật độ tương tác (Density) của từng cá nhân.

👉 Mạng nơ-ron nào sẽ đảm nhận nhiệm vụ "bóp méo thời gian" này, và kiến trúc của nó được thiết kế ra sao? Chúng ta sẽ mổ xẻ nó trong **Hồi 5**.
