# Tóm tắt Chi tiết (Phân tích Chuyên sâu): A Tale of Two Graphs - Freezing and Denoising Graph Structures for Multimodal Recommendation

**Tên mô hình:** FREEDOM (FREEzing and DenOising Graph Structures for Multimodal Recommendation)  
**Tác giả:** Xin Zhou và Zhiqi Shen (Nanyang Technological University, Alibaba-NTU)  
**Hội nghị/Năm:** ACM Multimedia (MM '23) - 2023  
**Mã nguồn:** [https://github.com/enoche/FREEDOM](https://github.com/enoche/FREEDOM)  
**Lĩnh vực:** Hệ thống Gợi ý Đa phương thức (Multimodal Recommender Systems)

---

## 1. Lời mở đầu & Phản biện SOTA (The Problem)

Bài báo này đưa ra một lời phản biện cực kỳ đanh thép nhắm thẳng vào mô hình **LATTICE** - mô hình từng giữ vị trí State-of-the-Art (SOTA) trong giới Gợi ý Đa phương thức bằng việc học cấu trúc đồ thị ẩn giữa các sản phẩm (latent item-item graph).

Tác giả của bài báo khẳng định rằng: **Cơ chế học đồ thị ẩn của LATTICE là "cồng kềnh, kém hiệu quả và hoàn toàn thừa thãi"!**

**2 Vấn đề cốt lõi được chỉ ra:**
1. **Sự rườm rà của việc học Đồ thị Item-Item:** LATTICE dùng Mạng Nơ-ron (MLP) để cập nhật động (dynamic) và liên tục biến đổi mạng lưới kết nối giữa các sản phẩm (Item-Item Graph) trong suốt quá trình Training. Việc này tiêu tốn dung lượng RAM khổng lồ $O(N^2 d_m)$ và thời gian tính toán rất chậm, đặc biệt trên các tập dữ liệu lớn.
2. **Nhiễu trong Đồ thị User-Item:** Các mô hình trước đây mặc định tin tưởng tuyệt đối vào mọi tương tác của User (cứ có click là tốt). Nhưng thực tế, người dùng có xu hướng "bấm nhầm" hoặc "bấm theo phong trào" (unintentional interactions), gây ra các đường nối giả/nhiễu (false-positive) làm sai lệch mô hình.

---

## 2. Ý tưởng Đột phá: Mô hình FREEDOM

Tác giả đề xuất mô hình **FREEDOM**, viết tắt của hai hành động: **FREEzes** (Đóng băng đồ thị Item-Item) và **DenOises** (Khử nhiễu đồ thị User-Item).

### 2.1. FREEZING: Đóng băng Đồ thị Item-Item (I-I)
Thay vì để đồ thị I-I thay đổi liên tục, FREEDOM làm một việc cực kì đơn giản nhưng táo bạo:
- Dùng đặc trưng thô (Raw Features) của Hình ảnh và Văn bản để tính độ tương đồng Cosine giữa các Item.
- Lọc lấy **Top-K (K=10)** láng giềng giống nhất cho mỗi sản phẩm.
- Biến đồ thị này thành đồ thị không trọng số (Unweighted Graph) gồm toàn số 0 và 1.
- **ĐÓNG BĂNG (Freeze)** đồ thị này vĩnh viễn trước khi Training. Trong lúc mạng Nơ-ron huấn luyện, đồ thị I-I này đứng im không suy suyển.
- **Chứng minh Toán học (Phân tích Phổ - Spectral Analysis):** Bằng toán học, tác giả chứng minh rằng một đồ thị đóng băng hoạt động như một **Bộ lọc Tần số Thấp (Low-pass Filter)** chặt chẽ hơn, chặn đứng được các tín hiệu nhiễu ở tần số cao tốt hơn rất nhiều so với đồ thị động của LATTICE.

### 2.2. DENOISING: Khử nhiễu Đồ thị User-Item (U-I)
Để xử lý các tương tác rác, FREEDOM giới thiệu kỹ thuật **Tỉa Cạnh nhạy cảm với Bậc (Degree-sensitive Edge Pruning)**.
- Thay vì cắt bỏ cạnh ngẫu nhiên (như kĩ thuật DropEdge nổi tiếng), FREEDOM tính toán xác suất giữ lại cạnh.
- **Quy tắc:** Sản phẩm nào có "Bậc" (Degree) càng cao (tức là càng nổi tiếng, được mua/click cực nhiều) thì cạnh nối với nó **càng có nguy cơ bị cắt đứt cao hơn**.
- **Nguyên lý:** Các sản phẩm siêu Hot thường chịu trách nhiệm lớn nhất gây ra hiện tượng *Over-smoothing* (trơn hóa quá mức) vì chúng kéo tuột mọi vector User về phía chúng. Việc chặt bớt đường nối với các món đồ Hot giúp giữ lại được bản sắc riêng của từng User. Kỹ thuật này được áp dụng liên tục ở mỗi vòng lặp (Epoch).

---

## 3. Quá trình Tích hợp và Dự đoán

Giống như cấu trúc đa đồ thị thông thường, FREEDOM chạy song song:
1. **Truyền thông tin trên đồ thị Item-Item đã ĐÓNG BĂNG** bằng LightGCN.
2. **Truyền thông tin trên đồ thị User-Item đã TỈA BỚT CẠNH** bằng LightGCN.

Sau đó, tổng hợp kết quả của cả hai bên lại (bằng phép cộng). Việc dự đoán mua hàng (Interaction Score) chỉ dùng Vector biểu diễn Lịch sử (ID Embeddings) nhân vô hướng với nhau. Đặc trưng hình ảnh/văn bản chỉ làm "chất xúc tác" mồi ở phía sau, không được đem ra để dự đoán trực tiếp. Tối ưu bằng hàm BPR Loss quen thuộc.

---

## 4. Thiết lập Thí nghiệm & Kết quả (Experiments & Results)

- **Datasets:** Ba bộ dữ liệu siêu thưa thớt của Amazon: Baby, Sports, Clothing (chuẩn 5-core). 
- **Đặc trưng:** Vision = 4096 dimensions, Text = 384 dimensions (Sentence-BERT).
- **Baselines:** So sánh với BPR, LightGCN, VBPR, MMGCN, GRCN, DualGNN, LATTICE, và SLMRec.
- **Cấu hình mô hình:** Tỷ lệ tỉa cạnh $\rho \in \{0.8, 0.9\}$ (Cắt bỏ tới 80-90% đồ thị gốc!).

### Kết quả chấn động:
1. **Đè bẹp LATTICE:** Dù đơn giản hóa và "đóng băng" thuật toán, FREEDOM đánh bại LATTICE với biên độ trung bình lên tới **19.07%** trên mọi chỉ số (Recall/NDCG).
2. **Tiết kiệm RAM khủng khiếp:** Bằng việc bỏ đi phép tính toán động đồ thị I-I, FREEDOM tiết kiệm bộ nhớ GPU gấp **6 LẦN** (Giảm tới 6x memory cost) trên các tập dữ liệu lớn.
3. **Bài học rút ra:** Đôi khi, "tĩnh" lại tốt hơn "động". Việc cho thuật toán quá nhiều tự do (học đồ thị động) lại vô tình làm thuật toán học sai và tốn tài nguyên.

---

## 5. Ví dụ Trực quan (Dễ hình dung)

Hãy tưởng tượng bạn đang kinh doanh một trung tâm thương mại điện tử (Shopee):

- **Cách làm của LATTICE:** Giống như một **đội ngũ kỹ sư sắp xếp kệ hàng**. Mỗi ngày (mỗi Epoch), đội kỹ sư này đều xem xét lại *toàn bộ* hình ảnh, mô tả của các sản phẩm và hì hục di chuyển, đổi chỗ các kệ hàng sao cho những món đồ giống nhau đứng cạnh nhau. Việc này cực kỳ mệt mỏi, tốn kém nhân lực (Tốn RAM, tốn thời gian) và nhiều khi vì di chuyển quá nhiều nên đồ đạc bị sai vị trí (Over-fitting).
- **Cách làm của FREEDOM (Đóng băng đồ thị I-I):** Giống như một **bản vẽ quy hoạch chốt cứng từ ngày đầu**. Người quản lý nhìn hình ảnh, mô tả sản phẩm đúng 1 lần duy nhất trước khi mở cửa siêu thị. Chốt vị trí kệ hàng vĩnh viễn (Đóng băng). Kết quả là: Không tốn chi phí dọn dẹp hàng ngày, và khách hàng thấy đồ đạc ổn định, dễ tìm hơn!
- **Khử nhiễu đồ thị (Denoising U-I Graph):** Chiếc "Áo thun cơ bản" được 1 triệu người mua (Sản phẩm Hot). Nếu giữ nguyên sổ sách, ai vào Shopee cũng sẽ bị gợi ý cái áo đó. FREEDOM lén **xóa bớt 80-90% hóa đơn** mua chiếc áo thun đó. Nhờ vậy, hồ sơ của người dùng nổi bật lên được các món đồ *độc đáo* (như Vợt cầu lông, Đồ câu cá) thay vì bị cái áo thun che lấp hết!