# Recommender Systems with Generative Retrieval

**Tên mô hình:** TIGER (Transformer Index for GEnerative Recommenders)
**Tác giả:** Shashank Rajput, Raghunandan Keshavan, Yi Tay, et al. (Google & Google DeepMind)
**Nơi công bố/Năm:** NeurIPS 2023
**Lĩnh vực:** Generative Recommendation (Gợi ý Tạo sinh), Semantic IDs (Định danh Ngữ nghĩa)

---

## 1. Bối cảnh & Vấn đề (The Problem)

Truyền thống, các hệ thống gợi ý (Recommender Systems - RS) hoạt động theo cơ chế **Retrieve-and-Rank** (Truy xuất và Xếp hạng). Trong đó, mỗi sản phẩm được gán một **Atomic ID** (một mã định danh nguyên tử, ví dụ: `item_1024`, `item_64`).

**Hạn chế của phương pháp cũ (Atomic IDs):**

1. **Thiếu ngữ nghĩa:** Mã `1024` và `1025` không hề có mối liên hệ nào về mặt nội dung (dù chúng có thể cùng là áo thun).
2. **Cold-start (Khởi động lạnh):** Khi một sản phẩm mới tinh được thêm vào kho, nó mang một ID mới chưa từng xuất hiện trong lịch sử mua hàng của ai. Mô hình không thể học được nhúng (embedding) của nó, dẫn đến việc không bao giờ gợi ý được sản phẩm mới này.
3. **Phình to bộ nhớ (Memory Bloat):** Nếu kho hàng có hàng tỷ sản phẩm, hệ thống phải lưu hàng tỷ vector nhúng khổng lồ cho từng ID, gây tốn kém bộ nhớ cực kỳ lớn.

---

## 2. Giải pháp Đột phá: Mô hình TIGER

Để giải quyết, nhóm nghiên cứu tại Google đề xuất một hệ tư tưởng hoàn toàn mới: Biến bài toán Gợi ý thành bài toán **Truy xuất Tạo sinh (Generative Retrieval)** bằng cách sử dụng **Semantic IDs** (Định danh Ngữ nghĩa).

Mô hình hoạt động qua 2 giai đoạn chính:

### Giai đoạn 1: Sinh Định danh Ngữ nghĩa bằng RQ-VAE (Semantic ID Generation)

Thay vì dùng số ngẫu nhiên, TIGER đọc nội dung (text, title, category) của sản phẩm, biến chúng thành một tuple (chuỗi) các "từ vựng" ngắn gọn (ví dụ: `(5, 23, 55)`).

- **Thuật toán RQ-VAE (Residual-Quantized VAE):**
  - **Mã hóa (Encode):** Đưa text của sản phẩm qua mô hình `Sentence-T5` để lấy 1 vector ngữ nghĩa.
  - **Lượng tử hóa phân cấp (Hierarchical Quantization):**
    - *Mức 1:* So khớp vector với Bảng mã số 1 (Codebook 1) để tìm ra token gần nhất $\rightarrow$ Ra được token đầu tiên (đại diện cho Danh mục lớn, vd: `5` là Giày).
    - *Mức 2:* Lấy phần chênh lệch (Residual) giữa vector gốc và token 1, tiếp tục so khớp với Bảng mã số 2 $\rightarrow$ Ra được token thứ hai (Danh mục nhỏ, vd: `23` là Giày chạy bộ).
    - *Mức 3:* Tiếp tục lấy phần dư để tính ra token thứ 3.
  - **Xử lý đụng độ (Handling Collisions):** Nếu 2 đôi giày quá giống nhau và cùng ra mã `(5, 23, 55)`, mô hình sẽ tự động gắn thêm token số 4 ở cuối (ví dụ `0` và `1`) để đảm bảo mỗi ID là duy nhất.

=> Kết quả: Kho hàng tỷ sản phẩm được tóm gọn lại bằng sự kết hợp của vài nghìn mã token. Sản phẩm càng giống nhau, chung tiền tố (prefix) càng dài.

### Giai đoạn 2: Mô hình Gợi ý Tạo sinh (Seq2Seq Transformer)

- Đầu vào: Chuỗi các Semantic ID mà người dùng đã mua trong quá khứ (Vd: mua mã `(5, 25, 78)`, sau đó mua `(5, 23, 55)`).
- Đầu ra: Mô hình **Transformer (Encoder-Decoder)** sẽ hoạt động y hệt như ChatGPT. Nó tự động **sinh ra (generate) từng token một** cho món đồ tiếp theo.
- Quá trình suy luận dùng thuật toán **Beam Search** để sinh ra luôn Top-K sản phẩm phù hợp nhất, hoàn toàn bỏ qua bước dùng thuật toán ANN (Approximate Nearest Neighbor) truyền thống.

---

## 3. Kết quả Thí nghiệm (Results)

Mô hình TIGER đã được test trên bộ dữ liệu Amazon (Beauty, Sports, Toys) và đem lại những kết quả chấn động:

1. **Hiệu suất SOTA:** Vượt qua hoàn toàn các mô hình Sequential RecSys hàng đầu trước đó (như SASRec, S3-Rec, BERT4Rec, P5) ở các chỉ số Recall@K và NDCG@K.
2. **Khả năng Cold-Start xuất sắc:** TIGER có thể gợi ý những sản phẩm *chưa từng xuất hiện trong tập huấn luyện (Unseen items)*! Bởi vì dựa trên tiền tố của Semantic ID, mô hình biết món đồ đó thuộc nhóm nào và tự động sinh ra mã của nó.
3. **Điều chỉnh tính Đa dạng (Diversity):** Tương tự LLM, bằng cách chỉnh thông số "Nhiệt độ" (Temperature) khi dùng thuật toán Beam Search, hệ thống có thể tạo ra các gợi ý đa dạng hơn (vượt ra khỏi nhóm ngành hàng quá quen thuộc), tránh việc người dùng bị nhốt trong "buồng vang" (filter bubble).
4. **Tiết kiệm Bộ nhớ (Memory Efficient):** Thay vì phải lưu 20.000 vector embeddings cho 20.000 sản phẩm, TIGER chỉ cần lưu vector cho các Token trong bảng mã Codebook (256 token x 4 mức = 1024 vector nhúng). Thu nhỏ kích thước mô hình theo cấp số nhân!

---

## 4. Ví dụ Trực quan (Dễ hình dung)

Hãy tưởng tượng bạn đang quản lý một cửa hàng thời trang:

- **Hệ thống cũ:**

  - Khách vừa mua `Áo_thun_A` (ID: 101) và `Áo_thun_B` (ID: 550).
  - Bạn mới nhập về một cái `Áo_thun_C` mới tinh (ID: 999). Vì chưa có ai mua cái áo 999 này, hệ thống cũ "mù tịt" và không dám gợi ý cho vị khách kia.
- **Hệ thống TIGER (Semantic ID):**

  - Áo_thun_A được RQ-VAE mã hóa thành `(10, 5, 2)`. (Ý nghĩa ngầm: 10 = Quần áo, 5 = Áo thun, 2 = Ngắn tay).
  - Áo_thun_B được mã hóa thành `(10, 5, 8)`.
  - Bạn nhập cái `Áo_thun_C` mới về. Chỉ cần quét qua dòng mô tả văn bản, RQ-VAE tự động gán cho nó mã `(10, 5, 9)` dù chưa ai mua nó cả.
  - Khi vị khách lướt web, mô hình AI (Transformer) nhìn thấy lịch sử `(10, 5, 2)` $\rightarrow$ `(10, 5, 8)`. Nó tự động đoán (generate) rằng token tiếp theo khách muốn sẽ bắt đầu bằng tiền tố `(10, 5, ...)`.
  - Hệ thống "sinh ra" mã `(10, 5, 9)` $\rightarrow$ Áo_thun_C (mới nhập) lập tức xuất hiện ngay trên trang chủ của khách hàng!
