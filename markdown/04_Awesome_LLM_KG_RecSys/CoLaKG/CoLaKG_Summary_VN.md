# Tóm tắt Chi tiết (Phân tích Chuyên sâu): CoLaKG: Comprehending Knowledge Graphs with Large Language Models for Recommender Systems

**Tên mô hình:** CoLaKG (Comprehending Knowledge Graphs with LLMs)  
**Tác giả:** Ziqiang Cui, Yunpeng Weng, Xing Tang, Fuyuan Lyu, Dugang Liu, Xiuqiang He, Chen Ma (City University of Hong Kong, Tencent, Shenzhen University)  
**Nơi công bố/Năm:** SIGIR 2025 (Từ kho lưu trữ Awesome-LLM-KG-RecSys)  
**Lĩnh vực:** Hệ thống Gợi ý dựa trên Đồ thị Tri thức (KG), Large Language Models (LLMs).

---

## 1. Bối cảnh & Điểm nghẽn (The Problem)

Đồ thị tri thức (KG) đã được chứng minh là cực kỳ hữu ích cho Hệ thống gợi ý vì khả năng kết nối các sản phẩm dựa trên các thuộc tính chung (ví dụ: cùng Đạo diễn, cùng Thể loại). Tuy nhiên, các phương pháp nhúng Đồ thị truyền thống (như KGAT, KGIN) đang đối mặt với **3 giới hạn lớn**:

1. **Missing Facts (Thiếu hụt dữ kiện):** Đồ thị tri thức thường được xây dựng thủ công hoặc tự động bởi máy móc, do đó nó luôn bị thiếu sót. Một bộ phim có thể bị thiếu mất nhãn "Khoa học viễn tưởng" trong đồ thị, khiến nó bị đứt liên kết với các phim khác.
2. **Mất mát Ngữ nghĩa (Semantic Loss):** Các thuật toán truyền thống biến tất cả chữ nghĩa (Text) trong Đồ thị thành các con số ID vô hồn. Chữ "Khoa học viễn tưởng" và chữ "Viễn tưởng" bị coi là 2 ID khác hẳn nhau, làm mất đi sự liên kết ngữ nghĩa tự nhiên.
3. **Giới hạn Cục bộ (Locality Constraint):** Các thuật toán GNN (Graph Neural Networks) chỉ truyền thông tin được từ 1 đến 3 bước nhảy (1-3 hops). Các sản phẩm nằm quá xa nhau trên đồ thị sẽ không bao giờ "nhìn thấy" nhau, dù chúng có điểm chung ẩn.

---

## 2. Ý tưởng Đột phá: Mô hình CoLaKG

CoLaKG sử dụng sức mạnh đọc hiểu và suy luận của LLM (ở đây tác giả dùng API của **DeepSeek-V2**) để "bù đắp" và "tổng hợp" thông tin từ KG ở cả hai cấp độ: Cục bộ (Local) và Toàn cục (Global).

### 2.1. Cấp độ Cục bộ: Trích xuất Đồ thị con & Prompt Engineering
Để giải quyết bài toán "Missing Facts", hệ thống làm như sau:
- Đối với mỗi Sản phẩm (Item), rút trích ra các láng giềng bậc 1 và một vài láng giềng bậc 2 xung quanh nó để tạo thành một Đồ thị con (Item-centered Subgraph).
- **Biến Đồ thị thành Text:** Chuyển đổi Đồ thị con này thành các câu văn (Prompt).
- **Nhờ LLM Suy luận:** Đưa Prompt này cho DeepSeek-V2 và hỏi: *"Dựa vào các mối quan hệ này, hãy tóm tắt và sinh ra một biểu diễn ngữ nghĩa cho Sản phẩm này"*. 
- **Kết quả:** Nhờ tri thức khổng lồ có sẵn, LLM không chỉ tóm tắt mà còn **tự động điền bù** các thuộc tính bị thiếu trong đồ thị gốc, tạo ra một đoạn Text miêu tả cực kỳ phong phú và chính xác.

### 2.2. Cấp độ Toàn cục: Truy xuất Ngữ nghĩa (Semantic Retrieval)
Để phá vỡ "Giới hạn Cục bộ" của GNN (chỉ nhìn được hàng xóm gần):
- Hệ thống dùng `RoBERTa` biến các đoạn Text (vừa được LLM sinh ra ở bước 1) thành các Vector Ngữ nghĩa.
- Thay vì đi theo đường viền của Đồ thị, hệ thống tính **Cosine Similarity** trên toàn bộ kho dữ liệu để tìm ra các Sản phẩm có ngữ nghĩa giống nhau nhất, bất kể chúng nằm cách xa nhau bao nhiêu bước trên Đồ thị.
- Tạo ra một **Đồ thị Ngữ nghĩa Item-Item (Semantic Graph)** mới nối các láng giềng ngữ nghĩa này lại (Tối ưu nhất là nối với Top 10-30 láng giềng).

### 2.3. Tích hợp và Đưa ra Dự đoán (Fusion)
- Mô hình chạy song song 2 nhánh:
  1. Nhánh 1 (Collaborative): Chạy **LightGCN** truyền thống trên đồ thị User-Item để lấy đặc trưng hành vi (ID Embedding).
  2. Nhánh 2 (Semantic): Sử dụng Mạng GCN trên Đồ thị Ngữ nghĩa (từ bước 2.2) để thu được biểu diễn Ngữ nghĩa tăng cường.
- Ánh xạ 2 vector này về chung một không gian (Mapping) và cộng gộp lại để đưa ra điểm dự đoán cuối cùng. Tối ưu bằng hàm BPR Loss.

---

## 3. Thiết lập Thí nghiệm & Kết quả (Results)

- **Datasets:** Đánh giá trên 4 bộ dữ liệu thực tế: MovieLens, Last-FM (Âm nhạc), MIND (Tin tức), và Funds (Tài chính).
- **Baselines:** Đối đầu với 12 mô hình mạnh nhất hiện nay bao gồm LightGCN, KGAT, KGIN, KGCL, KGRec, RLMRec.
- **Phần cứng & LLM:** Huấn luyện trên duy nhất 1 GPU V100. LLM sử dụng là **DeepSeek-V2 API**.
- **Kết quả cực kỳ ấn tượng:**
  - **SOTA mới:** CoLaKG đánh bại toàn bộ 12 mô hình Baseline trên cả 4 bộ dữ liệu (Chỉ số R@20 và NDCG@20 cao nhất).
  - Khắc phục **Data Sparsity:** Thí nghiệm mô phỏng việc xóa đi 1 phần Đồ thị (Missing edges) cho thấy CoLaKG hầu như không bị giảm hiệu suất, chứng minh khả năng "tự bù đắp tri thức" tuyệt vời của LLM.
  - Phân tích Ablation cho thấy cả nhánh Cục bộ (Prompt cho LLM) và nhánh Toàn cục (Semantic Retrieval) đều đóng góp lớn vào sự thành công này.

---

## 4. Ví dụ Trực quan (Dễ hình dung)

Tưởng tượng bộ phim **"Ma Trận" (Inception)** bị nhập thiếu dữ liệu vào hệ thống (Chỉ có nhãn: Đạo diễn Nolan, Diễn viên DiCapio. Bị thiếu nhãn "Khoa học Viễn tưởng").

- **Các mô hình GNN cũ (KGAT, KGIN):** Bó tay. Nó chỉ biết Inception là phim của Nolan. Nó không thể nối Inception với bộ phim *Interstellar* (vì 2 phim này bị đứt đoạn liên kết thể loại). Do bị kẹt ở "Cục bộ", nó vĩnh viễn không biết 2 phim này giống nhau.
- **Mô hình CoLaKG:**
  1. Nó lấy thông tin (Đạo diễn: Nolan, Diễn viên: DiCaprio) ném cho **DeepSeek-V2** đọc.
  2. DeepSeek-V2 bằng trí thông minh của mình sẽ trả lời: *"À, Inception của Nolan à? Phim này thuộc thể loại **Hành động, Hack não, Khoa học Viễn tưởng**, kịch bản đi vào giấc mơ rất đỉnh!"* (LLM đã tự bù đắp tri thức bị thiếu).
  3. Ở cấp độ Toàn cục, hệ thống lấy đoạn văn bản trên đem đi so sánh với toàn bộ các phim khác. Nó phát hiện ra *Interstellar* cũng có chữ "Khoa học Viễn tưởng, Hack não, Nolan".
  4. Ngay lập tức, một đường link "Ngữ nghĩa" được vẽ ra nối thẳng *Inception* và *Interstellar* với nhau. Nhờ đó, khi bạn xem Inception, hệ thống gợi ý ngay lập tức Interstellar một cách chính xác tuyệt đối!