# 1. Tổng Quan Kiến Trúc Transformer trong RecSys

Trong lĩnh vực Hệ thống Gợi ý (Recommender System), **Sequential Recommendation** (Gợi ý dựa trên chuỗi) là bài toán cốt lõi. Nhiệm vụ của mô hình là nhìn vào chuỗi lịch sử mua hàng của một người dùng và dự đoán món hàng tiếp theo họ sẽ mua.

Kiến trúc **Transformer** (được giới thiệu lần đầu trong bài báo *Attention is All You Need*) đã trở thành tiêu chuẩn vàng để giải quyết bài toán này, vượt qua các mô hình cũ như RNN hay CNN. Hình ảnh dưới đây (trích xuất từ bài báo **MEANTIME**) mô tả kiến trúc tiêu chuẩn của một mô hình Transformer áp dụng cho RecSys.

![Kiến trúc Transformer trong RecSys (Nguồn: MEANTIME)](/home/hieumagic/Repositories/Recommender-System-Dashboard/Idea/Personalized_ChronoRoPE/docs/images/meantime_model_figure-1.png)

## Hành trình của Dữ liệu (Data Flow)

Để hiểu hình ảnh trên, chúng ta hãy theo dõi hành trình của một lịch sử mua hàng:
`Lịch sử User A: [iPhone] -> [Ốp lưng] -> [Airpods]`

### Bước 1: Input Embedding (Nhúng Dữ liệu Đầu vào)
Máy tính không hiểu từ "iPhone", nó chỉ hiểu các con số.
- **Item Embedding (Biểu diễn món hàng):** Mỗi món hàng được cấp một mã số ID. Hệ thống sẽ tra cứu bảng (Lookup Table) để biến ID này thành một Vector chứa nhiều con số (ví dụ Vector 64 chiều).
- **Position Embedding (Nhúng vị trí):** Transformer có một nhược điểm chí mạng là nó xử lý dữ liệu song song (parallel) thay vì xử lý từng từ một như RNN. Do đó, nó bị **mù thứ tự**. Nếu không có Position Embedding, mô hình sẽ coi `[iPhone -> Ốp lưng]` giống hệt `[Ốp lưng -> iPhone]`. 
  - Trong hình trên, bạn có thể thấy các ô vuông màu xanh lá cây `[P_1, P_2, P_3]` được cộng/nhân vào các món hàng tương ứng để mô hình biết món nào mua trước, món nào mua sau. **Đây chính là nơi RoPE và ChronoRoPE sẽ can thiệp!**

### Bước 2: Transformer Blocks (Lõi Tính Toán)
Sau khi dữ liệu đã được nhúng thành Vector, chúng đi vào các khối Transformer (khung nét đứt ở giữa hình). Một khối Transformer gồm 2 thành phần chính:
1. **Multi-Head Self-Attention (Lõi chú ý):** Giúp các món hàng trong lịch sử "nhìn" lẫn nhau. Ví dụ: [Airpods] sẽ nhận ra rằng nó liên quan mật thiết tới [iPhone] hơn là tới [Ốp lưng]. (Chúng ta sẽ đi sâu vào phần này ở Bài 2).
2. **Feed-Forward Network (FFN):** Một mạng nơ-ron truyền thống giúp xử lý và chắt lọc thông tin phi tuyến tính từ kết quả của Attention.

Các khối Transformer này có thể được xếp chồng lên nhau (như Lớp 1, Lớp 2, Lớp L) để mô hình có khả năng suy luận sâu hơn.

### Bước 3: Prediction (Dự đoán)
Sau khi đi qua L lớp Transformer, vector cuối cùng của chuỗi (mang toàn bộ ngữ cảnh lịch sử) sẽ được nhân với ma trận của toàn bộ các món hàng trong kho (Item Catalog). Món hàng nào có điểm số (Dot Product) cao nhất sẽ được chọn làm dự đoán tiếp theo.

---

> [!NOTE]
> **Điểm nghẽn học thuật:** Hầu hết các mô hình Transformer hiện nay (như SASRec, BERT4Rec) đều sử dụng **Absolute Position Embedding (Vị trí tuyệt đối)** ở Bước 1. Nghĩa là chúng chỉ gán thứ tự $1, 2, 3$ một cách cứng nhắc mà không quan tâm đến việc [iPhone] mua cách [Ốp lưng] 1 ngày, nhưng [Ốp lưng] cách [Airpods] tận 3 tháng. Đây là lý do chúng ta cần đến sự tiến hóa mang tên **RoPE**!
