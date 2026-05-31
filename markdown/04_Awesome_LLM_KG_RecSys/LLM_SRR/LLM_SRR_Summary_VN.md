# LLM-Powered Explanations: Unraveling Recommendations Through Subgraph Reasoning

**Tên mô hình:** LLM-SRR (LLM powered Subgraph Reasoning for explainable Recommendation)
**Tác giả:** Guangsi Shi, Xiaofeng Deng, Linhao Luo, Lijuan Xia, Lei Bao, Bei Ye, Fei Du, Shirui Pan, Yuxiao Li (Monash University, Bosch Corporate Research)
**Nơi công bố/Năm:** ACM / 2024 - 2025 (Từ kho lưu trữ Awesome-LLM-KG-RecSys)
**Lĩnh vực:** Hệ thống Gợi ý có thể giải thích (Explainable Recommendation), Suy luận Đồ thị con (Subgraph Reasoning), Bán chéo (Cross-selling).

---

## 1. Bối cảnh & Điểm nghẽn (The Problem)

Mặc dù các hệ thống gợi ý đã rất phát triển, nhưng việc cung cấp **lời giải thích hợp lý** cho người dùng vẫn là một thách thức lớn. Các hệ thống thường xuyên gặp phải hiện tượng **"Recommendation Hallucination" (Ảo giác gợi ý)**: Sinh ra một lời giải thích rất trôi chảy nhưng lại chẳng ăn nhập gì với lý do thực sự mà thuật toán chọn sản phẩm đó.

Vấn đề này đặc biệt nghiêm trọng trong bài toán **Bán chéo (Cross-selling)** ở các tập đoàn đa quốc gia (ví dụ: Tập đoàn Bosch - METC).

- Bán chéo là khi cố gắng bán một Máy khoan (thuộc ngành hàng Power Tools) cho một người vừa mua Phụ kiện ô tô (Car Accessories).
- Các thuật toán dựa trên luật (Rule-based) hoặc đường dẫn (Path-based) trên Knowledge Graph truyền thống sẽ "bó tay", vì không hề có một đường dẫn trực tiếp nào nối giữa Máy khoan và Phụ tùng ô tô trong quá khứ.

---

## 2. Ý tưởng Đột phá: Mô hình LLM-SRR

Để giải quyết bài toán trên, các nhà nghiên cứu từ Monash và Bosch đề xuất mô hình **LLM-SRR**. Đây là một quy trình kết hợp hoàn hảo giữa LLM (như ChatGPT) để đọc hiểu ngôn ngữ, và Thuật toán truyền tin trên Đồ thị con (Subgraph Message Passing) để suy luận logic.

Mô hình gồm 3 thành phần chính:

### 2.1. Information Injection (Bơm Tri thức bằng LLM)

Thay vì để Đồ thị Tri thức (KG) chỉ chứa các liên kết cứng nhắc (như `Sản phẩm A` -> `thuộc danh mục B`), mô hình dùng **LLM để đọc các đánh giá (Reviews) của khách hàng** nhằm trích xuất ra các mối quan hệ ẩn.

- Ví dụ: Khách hàng review: "Máy khoan này sửa được xe hơi".
- LLM sẽ tự động trích xuất thông tin này và tiêm (inject) vào KG như một cạnh (Edge) mới. Việc này giúp kết nối các mặt hàng vốn dĩ không thuộc cùng một ngành hàng (Cross-selling).

### 2.2. Subgraph Generation (Sinh Đồ thị con Tùy chỉnh)

Để không bị ngợp bởi kích thước khổng lồ của toàn bộ KG, hệ thống tạo ra một **Đồ thị con (Subgraph)** dành riêng cho mỗi người dùng.

- Bắt đầu từ Node Người dùng (User Node), thuật toán sẽ "lan truyền" (Diffuse) ra xung quanh qua nhiều bước (như vết dầu loang) để thu thập các Node láng giềng.
- Điểm đặc biệt là cơ chế **Attention-based Diffusion**: Thuật toán không lan truyền mù quáng mà tự động chấm điểm (scoring) các cạnh dựa trên độ tương đồng. Cạnh nào có ý nghĩa lớn đối với User hiện tại sẽ được ưu tiên giữ lại để vẽ nên Đồ thị con. Quá trình này giúp mô hình nắm bắt được "hệ sinh thái sở thích" thu nhỏ của riêng người dùng đó.

### 2.3. Reasoning Path & Explanation (Tìm Đường dẫn và Giải thích)

Sau khi thu hẹp được Đồ thị con và dự đoán ra Sản phẩm mục tiêu, hệ thống làm 2 việc cuối cùng:

1. **Trích xuất Đường dẫn (Path Extraction):** Dò ngược lại từ Sản phẩm mục tiêu về Người dùng trên Đồ thị con để lấy ra con đường có điểm Attention cao nhất. Vd: `[User A] -> Mua -> [Phụ kiện ô tô] -> Review ("Thích tự sửa xe") <- [Máy khoan]`.
2. **Sinh Lời giải thích (LLM Generation):** Quăng toàn bộ Đường dẫn này vào lại cho LLM (với một Prompt Template định sẵn). LLM sẽ dùng văn phong tự nhiên để viết ra câu giải thích: *"Vì bạn đã từng mua phụ kiện ô tô và có sở thích tự sửa chữa, chiếc Máy khoan đa năng này sẽ là một trợ thủ đắc lực cho bạn."*

---

## 3. Thiết lập Thí nghiệm & Kết quả (Results)

- **Datasets:** Sử dụng 3 bộ dữ liệu chuẩn của Amazon (Beauty, Cell Phones, Clothing) và **1 bộ dữ liệu nội bộ (Proprietary dataset) từ tập đoàn METC** chứa dữ liệu mua sắm thực tế ở nhiều kênh khác nhau.
- **Baselines:** So sánh với BPR, DKN, CKE, KGAT, PGPR, và ReMR.
- **Siêu tham số (Hyperparameters):** Chiều vector (Embedding dim) = 100, Kích thước Đồ thị con tối đa = 100 nodes.
- **Kết quả:**
  - **Đè bẹp mọi đối thủ:** Đạt hiệu suất cao nhất trên toàn bộ các chỉ số. Cải thiện trung bình **+12%** so với các mô hình SOTA hiện hành (NDCG tăng 5.36%, Recall tăng 9.33%).
  - **Chứng minh trong thực tế (METC Dataset):** Trong bài toán Bán chéo đa kênh của Bosch, hệ thống không chỉ dự đoán đúng mà còn tạo ra lời giải thích sắc bén, giúp các nhà phân tích tiếp thị (Market Analysts) hiểu được *tại sao* khách hàng lại nhảy từ ngành hàng này sang ngành hàng khác.

---

## 4. Ví dụ Trực quan (Dễ hình dung)

Bạn bước vào siêu thị điện máy và vừa mua một cái **Gạt mưa Ô tô**.

- **Hệ thống cũ:** Nó sẽ lục tung danh mục "Phụ tùng Ô tô" và gợi ý cho bạn mua thêm *Dầu nhớt*, *Thảm lót sàn*. Nếu siêu thị muốn bán chéo một cái **Máy khoan** (thuộc khu vực Công cụ cầm tay), hệ thống cũ chịu chết vì không có mũi tên nào nối "Gạt mưa" với "Máy khoan". Cùng lắm nó gợi ý mù quáng và giải thích ngớ ngẩn: *"Vì bạn mua gạt mưa nên mua máy khoan đi"*.
- **Hệ thống LLM-SRR:**
  1. LLM đọc hàng ngàn bình luận và biết rằng: Có một tập khách hàng mua phụ tùng ô tô rất hay tự chế đồ (DIY) tại garage nhà họ.
  2. Nó âm thầm vẽ một nét đứt (cập nhật Knowledge Graph) nối giữa khu "Phụ tùng Ô tô" và khu "Công cụ cầm tay" với nhãn là "Sở thích DIY".
  3. Khi bạn mua Gạt mưa, hệ thống tạo một sơ đồ tư duy (Đồ thị con) bao quanh bạn. Nó nhận ra nhánh "Sở thích DIY" đang sáng rực lên.
  4. Nó gọi Máy khoan ra, gợi ý cho bạn, và LLM viết một dòng giải thích cực ngọt: *"Dựa trên việc bạn vừa mua phụ kiện chăm sóc xe, chúng tôi thấy bạn có hứng thú với việc tự bảo dưỡng. Chiếc máy khoan Bosch này sẽ giúp bạn dễ dàng tháo lắp các chi tiết trong garage của mình!"*
