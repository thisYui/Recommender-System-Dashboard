# Enhanced multimodal recommendation systems through reviews integration

**Tên mô hình:** PMPR (Personalized Multi-Preference Recommender)
**Tác giả:** Hong Fang, Jindong Liang, Leiyuxin Sha (Shanghai Polytechnic University, Trung Quốc)
**Tạp chí/Năm công bố:** Knowledge and Information Systems - Tháng 1/2025
**Lĩnh vực:** Hệ thống Gợi ý Đa phương thức (Multimodal Recommender Systems)

---

## 1. Bối cảnh

Các hệ thống gợi ý đa phương thức (như LATTICE hay FREEDOM) thường kết hợp văn bản (Text) và hình ảnh (Visual) với đồ thị tương tác để cải thiện hiệu suất. Tuy nhiên, bài báo này chỉ ra một điểm yếu rất thực tế của các tập dữ liệu hiện hành (ví dụ: Amazon): **Vấn đề Khuyết thiếu Phương thức (Modality Missing).**

- **Vấn đề 1 (Khuyết thiếu dữ liệu):** Rất nhiều sản phẩm trên e-commerce bị thiếu phần mô tả văn bản, hoặc mô tả rất chung chung ("áo thun"), hoặc hình ảnh bị lỗi/mờ. Khi dữ liệu của một phương thức bị khuyết, mô hình không thể tìm ra điểm chung giữa các sản phẩm.
- **Vấn đề 2 (Độ nhiễu của Bình luận):** Dữ liệu Bình luận (Reviews) chứa rất nhiều thông tin phản ánh trực tiếp sở thích của người dùng (ví dụ: "áo mặc rất mát và hợp đi biển"). Tuy nhiên, bình luận cũng chứa đầy "rác" và "nhiễu" (ví dụ: "giao hàng chậm", "đóng gói xấu"). Việc nhét trực tiếp bình luận vào mô hình sẽ làm giảm độ chính xác.

---

## 2. Giải pháp: Mô hình PMPR (Tích hợp Bình luận làm Phương thức thứ 3)

Để giải quyết bài toán trên, tác giả đề xuất mô hình **PMPR** nhằm trích xuất Sở thích Đa chiều (Multi-Preference) bằng cách tích hợp trực tiếp **Bình luận (Reviews)** như một phương thức độc lập, đồng thời xử lý triệt để yếu tố nhiễu.

Mô hình gồm 4 bước hoạt động cốt lõi:

### 2.1. Trích xuất và Khử nhiễu Bình luận (Review Feature Extraction & Denoising)

- Thay vì gộp chung tất cả bình luận thành một đoạn văn dài, PMPR dùng mô hình ngôn ngữ `Sentence-Transformers` để xử lý từng bình luận một (đưa về cùng không gian vector với văn bản mô tả sản phẩm).
- **Cơ chế Chú ý Cá nhân hóa (User-specific Attention Mechanism):** Hệ thống dùng ID của người dùng để tạo ra một "Bộ lọc Chú ý" (Attention Vector). Nó tự động đánh giá xem bình luận nào là *quan trọng và mang tính sở thích* (gán trọng số cao), và bình luận nào là *nhảm nhí/rác* (gán trọng số thấp). Sau đó, nó tổng hợp lại thành một Biểu diễn Sở thích tinh khiết.

### 2.2. Học Đồ thị Đồng nhất (Homogeneous Graph Learning - Item-Item)

- Kế thừa tư duy của mô hình FREEDOM, PMPR xây dựng đồ thị nối các sản phẩm giống nhau (Item-Item Graph).
- Tuy nhiên, thay vì chỉ dùng Hình ảnh và Mô tả, nó **thêm Đặc trưng Bình luận vào làm phương thức thứ 3**. Sử dụng k-NN để tính độ tương đồng Cosine, lọc lấy Top-K láng giềng.
- Đồ thị này sau đó được chuyển thành đồ thị không trọng số và **được đóng băng (Frozen)** trong quá trình Training để tăng tốc độ và giữ tính ổn định cho ngữ nghĩa của sản phẩm.

### 2.3. Học Đồ thị Không đồng nhất (Heterogeneous Graph Learning - User-Item)

- Xây dựng một đồ thị tương tác User-Item đặc thù dựa trên Đặc trưng Bình luận.
- Chạy thuật toán **LightGCN** trên đồ thị này để lan truyền và tổng hợp thông tin, giúp nhào nặn ra các Vector biểu diễn tiềm ẩn (Latent embeddings) chứa đầy đủ sở thích cá nhân của người dùng được trích xuất từ các bình luận.

### 2.4. Tích hợp và Dự đoán (Integration & Prediction)

- Vector của User và Item thu được từ các đồ thị sẽ được dung hợp lại. Hệ thống tính điểm số tương tác (Dot product) và dùng hàm **BPR Loss** để tối ưu hóa, cuối cùng đưa ra danh sách Top-K sản phẩm phù hợp nhất.

---

## 3. Thiết lập Thí nghiệm & Kết quả (Experiments & Results)

- **Datasets:** 5 tập dữ liệu công khai (bao gồm các tập phổ biến của Amazon như Baby, Sports, Clothing).
- **Baselines so sánh:** BPR, LightGCN, VBPR, MMGCN, GRCN, DualGNN, SLMRec, **LATTICE**, và **FREEDOM** (SOTA bài số 4).
- **Kết quả:**
  - **Vượt qua FREEDOM:** PMPR chứng minh sự vượt trội hoàn toàn so với mô hình mạnh nhất là FREEDOM trên cả 5 tập dữ liệu.
  - **Mức độ cải thiện:** Đạt mức cải thiện trung bình từ **2.76% đến 22.52%** so với các mô hình SOTA trên những thang đo Recall, NDCG, Precision, và F1.
  - Điều này khẳng định rằng: Nếu Hình ảnh và Mô tả sản phẩm bị thiếu hụt, việc dùng Bình luận đã được "lọc nhiễu" để làm cầu nối là một chiến lược bù đắp cực kỳ hoàn hảo.

---

## 4. Ví dụ Trực quan (Dễ hình dung)

Hãy tưởng tượng bạn đang lướt Shopee tìm mua một chiếc **"Váy đi biển"**:

- **Vấn đề (Khuyết thiếu dữ liệu):** Chủ shop đăng chiếc váy lên nhưng lười viết mô tả (chỉ ghi chữ "Váy"), và bức ảnh chụp bị thiếu sáng, nhìn không ra họa tiết hoa. Các hệ thống cũ (như LATTICE hay FREEDOM) sẽ "bó tay" vì không có cơ sở nào để xếp chiếc váy này vào nhóm "Thời trang đi biển", dẫn đến việc bạn sẽ không bao giờ được gợi ý chiếc váy đó.
- **Cách PMPR xử lý:** PMPR bắt đầu đọc **Bình luận** của khách hàng cũ.
  - Khách hàng A viết: *"Váy hoa này mặc đi dạo biển ngắm hoàng hôn rất mát và bay bổng"*.
  - Khách hàng B viết: *"Giao hàng cho shipper thái độ lồi lõm"*.
- **Cơ chế lọc nhiễu:** PMPR tự động dập tắt bình luận của B (vì là nhiễu, không mang tính sở thích) và đẩy mạnh trọng số bình luận của A.
- **Kết nối Đồ thị:** Mặc dù không có "Mô tả sản phẩm" hay "Hình ảnh đẹp", nhưng dựa vào bình luận của khách A (xem như phương thức thứ 3), PMPR ngay lập tức nối chiếc váy đó với các món đồ đi biển khác trong hệ thống.
  $\rightarrow$ Kết quả: Khi bạn tìm kiếm đồ đi biển, chiếc váy bị "khuyết thiếu thông tin" đó vẫn xuất hiện chính xác trên trang chủ của bạn!
