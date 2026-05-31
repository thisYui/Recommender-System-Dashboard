# Tóm tắt Chi tiết (Phân tích Chuyên sâu): Multi-View Graph Convolutional Network for Multimedia Recommendation

**Tên mô hình:** MGCN (Multi-View Graph Convolutional Network)  
**Tác giả:** Penghang Yu, Zhiyi Tan, Guanming Lu, và Bing-Kun Bao (Đại học Bưu chính Viễn thông Nam Kinh, Trung Quốc)  
**Hội nghị/Năm:** ACM Multimedia (MM '23) - 2023  
**Mã nguồn:** [https://github.com/demonph10/MGCN](https://github.com/demonph10/MGCN)  
**Lĩnh vực:** Hệ thống Gợi ý Đa phương thức (Multimedia/Multimodal Recommender Systems)

---

## 1. Bối cảnh & Điểm nghẽn của các mô hình SOTA (The Problem)

Các mạng nơ-ron đồ thị (GCN) đang thống trị mảng hệ thống gợi ý đa phương thức, tuy nhiên các mô hình trước đó (như MMGCN) đang vướng phải **hai điểm yếu chí mạng**:

1. **Sự ô nhiễm nhiễu đa phương thức (Modality Noise Contamination):** 
   Dữ liệu hình ảnh và văn bản chứa rất nhiều "nhiễu" không liên quan đến sở thích thực sự (ví dụ: nền ảnh lộn xộn, độ sáng tối của ảnh, hay những từ mô tả sáo rỗng). Các mô hình GCN trước đây đem trộn chung đặc trưng đa phương thức này với đặc trưng hành vi (collaborative/behavior features) rồi truyền thẳng lên đồ thị User-Item. Hậu quả là: *Nhiễu lây lan từ node này sang node khác, làm vẩn đục toàn bộ biểu diễn của người dùng và sản phẩm.*

2. **Mô hình hóa sở thích không hoàn chỉnh (Incomplete user preference modeling):**
   Các mô hình cũ thường dung hợp (fuse) Hình ảnh và Văn bản bằng cách nối (concatenate) hoặc cộng tuyến tính, coi trọng cả hai như nhau. Nhưng thực tế, người dùng có sở thích khác nhau tùy ngữ cảnh: *Họ có thể chỉ nhìn hình ảnh khi xem TikTok (Visual quan trọng), nhưng lại đọc kỹ mô tả khi mua Sách (Textual quan trọng).* Việc đối xử bình đẳng các phương thức dẫn đến mô hình hóa sở thích thiếu chiều sâu.

---

## 2. Kiến trúc Mô hình MGCN (Methodology)

Để giải quyết 2 vấn đề trên, MGCN đưa ra một quy trình 3 bước chặt chẽ: Lọc nhiễu -> Làm giàu đặc trưng Đa góc nhìn (Multi-view) -> Dung hợp thông minh.

### 2.1. Bộ lọc dẫn dắt bởi hành vi (Behavior-Guided Purifier)
- **Mục tiêu:** Xử lý triệt để "Modality Noise Contamination" ngay từ vòng gửi xe.
- **Cách hoạt động:** 
  Dùng biểu diễn Lịch sử hành vi (ID Embeddings - thứ chứa thông tin "thuần khiết" nhất về thói quen mua sắm) để làm màng lọc (Cổng Sigmoid) đối với đặc trưng Hình ảnh và Văn bản. Nhờ sự dẫn dắt của ID, mô hình tự động giữ lại những nét đặc trưng liên quan đến sở thích và loại bỏ các chi tiết thừa thãi của bức ảnh/văn bản.

### 2.2. Bộ mã hóa thông tin Đa góc nhìn (Multi-View Information Encoder)
Thay vì truyền chung tất cả lên một đồ thị, MGCN tách ra làm **2 Góc nhìn (Views)** riêng biệt để làm giàu đặc trưng:
- **Góc nhìn User-Item (Hành vi):** Dùng thuật toán LightGCN truyền thông tin trên đồ thị tương tác để bắt các tín hiệu cộng tác bậc cao (High-order collaborative signals).
- **Góc nhìn Item-Item (Đa phương thức):** 
  - Xây dựng đồ thị K-láng giềng gần nhất (KNN) dựa trên độ tương đồng Cosine của hình ảnh/văn bản.
  - Sử dụng mạng GCN nông (chỉ 1 lớp) để truyền thông tin. Điều này giúp Item học hỏi thêm các đặc điểm thẩm mỹ/ngữ nghĩa từ các Item giống nó, đồng thời tránh hiện tượng "Over-smoothing" (trơn hóa quá mức) nếu dùng mạng quá sâu.

### 2.3. Bộ dung hợp nhận thức hành vi (Behavior-Aware Fuser)
- **Mục tiêu:** Giải quyết vấn đề "Đối xử bình đẳng các phương thức" bằng cách học trọng số động.
- **Cách hoạt động:**
  - Dùng đặc trưng hành vi (ID) của người dùng để "vắt" ra (distill) **sở thích phương thức (Modality Preferences)**. 
  - Từ đó, mô hình tự động điều chỉnh xem với user này, sản phẩm này thì Hình ảnh hay Văn bản quan trọng hơn.
  - **Self-supervised Auxiliary Task (Học tự giám sát):** Để tránh mô hình học "lệch" hoặc đi tìm "đường tắt" (shortcut), tác giả thêm một hàm mất mát đối chiếu (Contrastive Loss - $L_C$). Hàm này ép mô hình tối đa hóa "thông tin tương hỗ" (Mutual Information) giữa biểu diễn Đa phương thức tổng hợp và biểu diễn Hành vi, buộc chúng phải hỗ trợ qua lại chặt chẽ.

---

## 3. Thiết lập Thí nghiệm (Experimental Setup)

- **Datasets:** Ba bộ dữ liệu siêu thưa thớt của Amazon: Baby, Sports, Clothing (Tỉ lệ thưa thớt tương tự các bài trước, $\approx 0.03\% - 0.11\%$).
- **Đặc trưng đa phương thức:** Dùng chuẩn chung là 4096-dim cho Hình ảnh và 384-dim cho Văn bản.
- **Baselines:** So sánh với MF, LightGCN, VBPR, MMGCN, GRCN, SLMRec, BM3, và MICRO.
- **Thông số chính:** Kích thước embedding = 64, Temperature $\tau = 0.2$ (cho Contrastive Learning). 

---

## 4. Kết quả & Đóng góp cốt lõi (Results & Findings)

### 4.1. Hiệu suất vượt trội
- MGCN đánh bại toàn bộ các mô hình khác (kể cả các mô hình SOTA chuyên về siêu đồ thị hay tự giám sát như SLMRec, BM3, MICRO) trên cả 3 tập dữ liệu.
- Đỉnh điểm, trên tập **Clothing**, MGCN đạt mức **cải thiện lên tới 23.3%** so với baseline tốt nhất (theo chỉ số Recall/NDCG). 

### 4.2. Khám phá thú vị về GCN và Nhiễu
Bài báo phát hiện ra một sự thật: **GCN rất nhạy cảm với nhiễu đa phương thức.** 
- Mô hình truyền thống như VBPR (chỉ nối vector, không dùng đồ thị) thì tốt hơn MF.
- Nhưng khi dùng GCN (MMGCN), hiệu suất lại... tệ hơn cả LightGCN (chỉ dùng ID). 
- *Lý do:* Cơ chế Message Passing (Truyền thông báo) của GCN đã vô tình phát tán "nhiễu" từ một Item sang toàn bộ mạng lưới, làm bẩn toàn bộ biểu diễn. Nhờ có **Behavior-Guided Purifier**, MGCN đã diệt trừ hoàn toàn vấn đề này, minh chứng cho sức mạnh của bước Lọc Nhiễu.

---

## 5. Ví dụ Trực quan (Dễ hình dung)

Hãy tưởng tượng bạn đang dùng TikTok và bấm vào một **Video bán Váy họa tiết hoa (Video A)**:

- **Lỗi của mô hình cũ (Ô nhiễm nhiễu):** Video A có cô người mẫu mặc váy hoa đứng trước một *Bức tường màu đỏ chói*. Mô hình cũ bê nguyên đặc điểm "Màu đỏ" (Nhiễu) đưa vào mạng lưới. Sau đó, nó đi gợi ý cho bạn một đống áo sơ mi nam, chảo chống dính... chỉ vì những video đó cũng có dính bức tường màu đỏ!
- **MGCN giải quyết bằng Bộ lọc Hành vi (Purifier):** Nó nhìn vào *lịch sử hành vi* của bạn (ID của bạn chỉ toàn xem đồ nữ). Nó sẽ nói với màng lọc hình ảnh rằng: "Khoan, anh chàng này chỉ quan tâm quần áo nữ thôi, hãy bỏ qua cái màu đỏ của bức tường đi!". Vậy là nhiễu "Bức tường đỏ" bị xóa sổ.
- **Góc nhìn Đa phương thức (Item-Item View):** Chiếc váy hoa này tự động liên kết với một chiếc Váy hoa khác (Video B) trong kho dữ liệu vì chúng giống nhau (Cosine Similarity cao). Nó vay mượn thêm đặc tính từ Video B để làm phong phú thêm đặc trưng cho Video A.
- **Bộ dung hợp nhận thức hành vi (Behavior-Aware Fuser):** Shopee/TikTok nhận ra bạn là người **rất lười đọc chữ** (phát hiện từ Lịch sử ID của bạn). Vậy nên, khi trộn đặc trưng của Video A, hệ thống sẽ đẩy trọng số của "Hình ảnh chiếc váy" lên 90% và hạ trọng số "Đoạn text mô tả" xuống 10%. Điều này giúp gợi ý tiếp theo khớp hoàn hảo với thói quen "thích nhìn hình" của bạn!

---
*Tóm tắt này cung cấp chi tiết (Pass 2+) về cấu trúc Mạng nơ-ron đa góc nhìn (Multi-view), cách xử lý nhiễu tinh tế ngay từ đầu, sự dung hợp linh hoạt (thay vì tĩnh) các phương thức, đi kèm ví dụ liên hệ thực tế.*