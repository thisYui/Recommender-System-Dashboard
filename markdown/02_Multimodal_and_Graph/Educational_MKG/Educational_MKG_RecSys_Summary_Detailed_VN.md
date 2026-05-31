# Research on the methodology of personalized recommender systems based on multimodal knowledge graphs

**Tên mô hình:** Educational MMKG RecSys (Hệ thống gợi ý giáo dục dựa trên Đồ thị Tri thức Đa phương thức)
**Tác giả:** Shaowu Bao và Jiajia Wang (Anhui Agricultural University, Trung Quốc)
**Tạp chí/Năm công bố:** Natural Language Processing Journal - 2025
**Lĩnh vực:** Đồ thị Tri thức Đa phương thức (MMKG), Hệ thống Gợi ý Giáo dục (Educational Recommender Systems)

---

## 1. Bối cảnh & Vấn đề (The Problem & Motivation)

Trong thời đại bùng nổ thông tin, các nền tảng giáo dục trực tuyến sở hữu kho tài nguyên khổng lồ (văn bản, video, hình ảnh). Tuy nhiên, **tỷ lệ học tập hiệu quả (effective learning rate) chỉ đạt 35%**. Lý do là các hệ thống gợi ý giáo dục hiện tại mắc phải 3 hạn chế cốt lõi:

1. **Hạn chế của Hệ thống truyền thống (CF, CB):** Dễ bị ảnh hưởng bởi "Cold-start" (người dùng mới) và dữ liệu thưa thớt. Hơn nữa, chúng chỉ nhìn bề nổi của tài liệu mà không hiểu được mối liên hệ sâu xa giữa các "Điểm kiến thức" (Knowledge Points).
2. **Hạn chế của Đồ thị tri thức Đơn phương thức (Single-modal KG):** Các KG giáo dục cũ thường chỉ dùng "Văn bản" (Text). Chúng không thể hiểu được các video bài giảng hay hình ảnh thí nghiệm sinh học. Thêm vào đó, chúng là các cấu trúc tĩnh, **không theo kịp sự thay đổi linh hoạt trong quá trình học tập (temporal behavior)** của học sinh.
3. **Thiếu tính Giải thích (Lack of Interpretability):** Đa số là các mô hình "hộp đen". Học sinh và giáo viên không hiểu *tại sao* hệ thống lại gợi ý tài liệu này, dẫn đến độ tin cậy thấp.

---

## 2. Giải pháp Đột phá: MMKG RecSys với Động lực học Thời gian

Để giải quyết triệt để vấn đề trên, tác giả đề xuất một hệ thống Gợi ý cá nhân hóa xây dựng trên mạng lưới ngữ nghĩa **"Điểm Kiến Thức – Học Sinh – Tài Nguyên"**.

Mô hình có 3 thành phần mang tính đột phá:

### 2.1. Căn chỉnh Thực thể Phân cấp (Hierarchical Entity Alignment)

Làm sao để hệ thống biết được từ "Quang hợp" trong sách giáo khoa (Text), bức ảnh chiếc lá (Image), và video thí nghiệm (Video) là cùng nói về một thứ?

- Tác giả chia đồ thị thành các **Tiểu đồ thị (Subgraphs)**: Tiểu đồ thị Văn bản, Tiểu đồ thị Hình ảnh, Tiểu đồ thị Cấu trúc.
- Dùng các mô hình SOTA chuyên biệt để nhận diện: BiLSTM+CRF cho văn bản, YOLOv5 cho hình ảnh, Wav2Vec2.0 + BERT cho video/audio.
- Dùng **Mạng nơ-ron chú ý đồ thị (GAT)** để trích xuất đặc trưng và thuật toán LS-SVM để học trọng số, từ đó "căn chỉnh" (align) chính xác các thực thể này vào cùng một không gian. Nhờ vậy, độ chính xác căn chỉnh đạt tới **87.6%**.

### 2.2. Nhúng Cặp Đường dẫn (Dual-Path Embedding: Node2vec + LSTM)

Hệ thống kết hợp sự "tĩnh" của kiến thức và sự "động" của học sinh:

- **Node2vec (Bắt ngữ nghĩa Cấu trúc - Static):** Dùng bước nhảy ngẫu nhiên (random walk) trên MMKG để tạo ra Vector 256 chiều đại diện cho **cấu trúc bất biến của điểm kiến thức**.
- **LSTM (Bắt hành vi Thời gian - Dynamic):** Phân tích lịch sử học tập đa phương thức theo tuần của học sinh. *Cổng quên (Forget gate)* lọc bỏ các lịch sử lướt web vô dụng từ tháng trước, *Cổng đầu vào (Input gate)* tập trung vào các video vừa xem tuần này. Output là một Vector 128 chiều đại diện cho **sự thay đổi Sở thích/Trình độ hiện tại**.
- **Dung hợp:** Hai vector này được nối lại và đưa qua MLP 3 lớp để tạo ra biểu diễn hoàn hảo nhất cho việc dự đoán.

### 2.3. Mô hình Gợi ý có tính Giải thích (Interpretable Recommendation)

Thay vì chỉ ném ra kết quả, hệ thống dùng thuật toán Tìm kiếm chiều sâu (DFS) trên đồ thị (tối đa 3 bước nhảy - 3 hops) để dò tìm con đường từ Học sinh đến Tài nguyên.

- Hệ thống chấm điểm các đường dẫn (Path Confidence) dựa trên tần suất và điểm tương tác.
- Lọc ra các đường dẫn Top-K và dịch nó sang **Ngôn ngữ tự nhiên** để giải thích cho người học lý do họ nhận được gợi ý này.

---

## 3. Thiết lập Thí nghiệm & Kết quả (Experiments & Results)

- **Datasets:** Dữ liệu thực tế từ nền tảng giáo dục với 100.000 học sinh, 50.000 tài nguyên, và 2.000.000 bản ghi tương tác.
- **Baselines:** So sánh với CF, GAT+CB, Node2vec (truyền thống), KGAT, RippleNet (KG-aware), và M3KGR (SOTA Multimodal).

### Kết quả Siêu việt:

1. **Độ chính xác (Accuracy):** Precision@5 đạt **43.5%**, Hits@10 đạt **62.7%**, đè bẹp hoàn toàn mô hình đa phương thức SOTA M3KGR (vượt lần lượt 8.5% và 4.2%).
2. **Hiệu quả Giáo dục (Educational Efficacy):**
   - Độ bao phủ điểm kiến thức đạt 67%.
   - Điểm kiểm tra (test accuracy) của học sinh tăng **25%**.
   - Chu kỳ học (learning cycle time) giảm **30%**.
3. **Độ tin cậy và Tính giải thích (Interpretability):** Nhờ việc "dịch" đường dẫn đồ thị thành lời giải thích, hệ thống đạt điểm minh bạch **4.3/5.0** do 250 giáo viên và học sinh đánh giá (tăng 34.4% độ hài lòng so với các phương pháp cũ).
4. **Hiệu suất Tính toán (Efficiency):** Dù phức tạp, nhờ chiến lược "Early fusion" qua MLP, thời gian phản hồi chỉ **98ms** và tốn **760MB** RAM (Nhanh và nhẹ hơn rất nhiều so với KGAT và M3KGR).

---

## 4. Ví dụ Trực quan (Dễ hình dung)

Hãy tưởng tượng một học sinh tên **An** đang học môn Sinh học, chủ đề *"Cấu tạo Tế bào Thực vật"*.

- **Cách các mô hình cũ (CF) làm:** Thấy An đọc bài "Tế bào", nó gợi ý thêm một bài Text PDF dài 10 trang về "Lục lạp" (Chỉ vì những người khác cũng đọc). Rất nhàm chán, An bỏ dở giữa chừng.
- **Cách mô hình Đồ thị đơn phương thức (KGAT) làm:** Nhận ra "Cấu tạo tế bào" nối với "Quang hợp". Gợi ý bài "Quang hợp". Vẫn thiếu tính cá nhân hóa theo thời gian và không đa phương thức.
- **Cách mô hình Educational MMKG (Bài báo này) làm:**
  1. **(LSTM - Động lực học):** Hệ thống nhận ra tuần này An lười đọc Text, nhưng hay xem *Video* sinh học dạng Animation (nhận diện qua Lịch sử thời gian và Phân loại nhận thức).
  2. **(MMKG - Đồ thị đa phương thức):** Đồ thị chứa điểm kiến thức "Lục lạp". Thực thể "Lục lạp" được hệ thống căn chỉnh (Align) chính xác với một *Video hoạt hình 3D về Lục lạp* và một *Bức ảnh thí nghiệm soi kính hiển vi*.
  3. **(Gợi ý):** Hệ thống đẩy Video 3D và Bức ảnh lên đầu cho An.
  4. **(Tính giải thích - Trích xuất đường dẫn DFS):** Hệ thống hiện lên thông báo cho An: *"Bởi vì tuần trước bạn đã xem [Video: Tế bào thực vật], và [Video 3D Lục lạp] này minh họa trực tiếp cho kiến thức lõi của [Bài: Quang hợp] trong sách giáo khoa, chúng tôi gợi ý tài nguyên này để giúp bạn dễ hiểu hơn"*.

$\rightarrow$ Kết quả: An hiểu bài nhanh hơn, điểm kiểm tra tăng, và hoàn toàn tin tưởng vào hệ thống gợi ý của nhà trường!
