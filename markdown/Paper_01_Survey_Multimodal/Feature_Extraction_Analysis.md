# Trích Xuất Đặc Trưng (Feature Extraction) trong Hệ Thống Gợi Ý Đa Phương Thức

> **Nguồn:** Xu et al., *"A Survey on Multimodal Recommender Systems"*, IEEE Trans. Multimedia, 2025.

---

## 1. Tổng Quan

Feature Extraction là **bước đầu tiên** trong pipeline MRS. Nó nhận dữ liệu thô (ảnh sản phẩm, mô tả văn bản, âm thanh) và chuyển thành các **vector số** (embeddings) mà máy tính có thể xử lý. Mọi thứ phía sau — encoder, fusion, loss — đều phụ thuộc vào chất lượng của bước này. Nếu features kém, không bước nào phía sau bù đắp được.

**Vấn đề cốt lõi:** Dữ liệu thô rất đa dạng — ảnh là ma trận pixel, văn bản là chuỗi ký tự, âm thanh là sóng — nhưng hệ thống gợi ý cần tất cả ở **cùng một dạng** (vector số) để tính toán và so sánh được.

---

## 2. Các Phương Pháp Trích Xuất Đặc Trưng Hình Ảnh (Visual)

### 2.1. VGG *(Simonyan & Zisserman, 2014)*

- **Ý tưởng:** Xếp chồng rất nhiều lớp convolution nhỏ (filter $3 \times 3$) thay vì dùng vài lớp filter lớn. Triết lý là "sâu hơn = tốt hơn" — network 16–19 layers, sâu nhất thời điểm đó. Lấy output từ lớp fully-connected gần cuối (FC7) làm visual feature.
- **Giải quyết được gì:**

  - Chứng minh một cách thuyết phục rằng **chiều sâu** (depth) của network quyết định chất lượng feature — deep features vượt trội hẳn so với hand-crafted features (SIFT, HOG) vốn thống trị trước đó.
  - Cung cấp visual features "đủ tốt" cho nhiều tác vụ downstream, bao gồm cả recommendation.
- **Hạn chế:**

  - Cực kỳ nặng — ~138 triệu tham số, trong đó phần lớn nằm ở FC layers (không hiệu quả).
  - Feature vector chiều rất cao (4096-dim), cần thêm bước projection để giảm chiều.
  - Receptive field hữu hạn — mỗi neuron chỉ "nhìn thấy" một vùng nhỏ của ảnh, **không có global context**. Nghĩa là nó hiểu chi tiết cục bộ tốt nhưng không nắm được bố cục tổng thể.

---

### 2.2. ResNet *(He et al., 2016)*

- **Ý tưởng:** Thêm **skip connection** (đường tắt) cho phép gradient chảy thẳng từ output ngược về input mà không bị suy giảm. Cụ thể, output của mỗi block = kết quả learned + input gốc. Nếu network không học được gì hữu ích, nó ít nhất "pass through" input — không tệ hơn. Nhờ vậy có thể train network cực sâu (50, 101, thậm chí 152 layers).
- **Giải quyết được gì:**

  - **Vanishing gradient** — vấn đề khiến VGG và các CNN sâu khác không thể train quá 20 layers. Skip connection cho phép gradient vẫn đủ mạnh ở layers đầu.
  - Features chất lượng tốt hơn VGG với **ít tham số hơn** nhờ bỏ FC layers nặng, dùng Global Average Pooling thay thế.
  - Trở thành **backbone tiêu chuẩn** cho visual feature extraction trong MRS từ 2019 đến nay. Phần lớn các công trình (MMGCN, LATTICE, FREEDOM, BM3...) đều dùng ResNet-50 pre-trained trên ImageNet.
- **Hạn chế:**

  - Vẫn là CNN — receptive field bị giới hạn bởi kernel size. Không có **global context** từ đầu như Transformer. Để "nhìn" toàn bộ ảnh, cần stack rất nhiều layers.
  - Features cố định nếu chỉ dùng pre-trained mà không fine-tune — nhưng fine-tune rất tốn kém và phức tạp.

---

### 2.3. ViT — Vision Transformer *(Dosovitskiy et al., 2020)*

- **Ý tưởng:** Bỏ hoàn toàn CNN. Chia ảnh thành các mảnh (patches) $16 \times 16$ pixels, flatten mỗi mảnh thành một vector, rồi xử lý **như một chuỗi token** bằng Transformer encoder (giống xử lý câu văn). Cơ chế self-attention cho phép **mỗi patch attend tới mọi patch khác** ngay từ layer đầu tiên.
- **Giải quyết được gì:**

  - **Global context từ đầu** — khác CNN phải xếp nhiều layers mới thấy toàn ảnh, ViT cho mỗi patch "nhìn" cả ảnh ngay lập tức. Điều này đặc biệt quan trọng khi ảnh sản phẩm cần hiểu bối cảnh tổng thể (vd: chiếc áo trong outfit, món ăn trên bàn).
  - **Scalability tuyệt vời** — performance tăng mạnh khi tăng data và model size, một đặc tính mà CNN không có rõ ràng.
  - Transfer learning hiệu quả hơn CNN khi pre-train trên dataset khổng lồ (ImageNet-21K, JFT-300M).
- **Hạn chế:**

  - Yêu cầu **rất nhiều dữ liệu** pre-training. Trên dataset nhỏ, ViT thua CNN vì thiếu inductive bias (locality, translation equivariance) mà CNN có sẵn.
  - Self-attention tốn bộ nhớ bậc hai theo số patches — ảnh lớn → rất nặng.
  - Trong MRS, chưa phổ biến bằng ResNet vì phần lớn nghiên cứu dùng provided features (đã extract sẵn từ ResNet).

---

### 2.4. Xu hướng "Provided Features"

- **Ý tưởng:** Thay vì mỗi nghiên cứu tự extract features từ raw images, cộng đồng MRS chuyển sang dùng **features đã được extract sẵn** và chuẩn hóa qua framework MMRec. Mọi người dùng cùng features → so sánh công bằng.
- **Giải quyết được gì:**

  - **Reproducibility** — kết quả có thể tái lập vì ai cũng dùng cùng input.
  - Loại bỏ vấn đề ảnh bị hỏng/thiếu trong Amazon datasets (một số ảnh bị xóa khỏi server, gây inconsistency).
  - Giảm barrier rào cản — không cần GPU mạnh để extract features, chỉ cần download file sẵn.
- **Hạn chế:**

  - Features **cố định**, không được optimize theo RS objective. Đây chính là vấn đề **"semantic gap"** — features được train để phân loại ảnh (ImageNet: chó, mèo, xe...) chứ không phải để gợi ý sản phẩm.
  - Nếu pre-trained model capture sai thông tin (vd: focus vào background thay vì sản phẩm), tất cả các nghiên cứu đều bị ảnh hưởng.

---

## 3. Các Phương Pháp Trích Xuất Đặc Trưng Văn Bản (Textual)

### 3.1. TF-IDF *(Salton & Buckley, 1988)*

- **Ý tưởng:** Đo mức độ "quan trọng" của mỗi từ trong một document. Từ xuất hiện nhiều trong document hiện tại (TF cao) nhưng ít xuất hiện trong toàn bộ corpus (IDF cao) → từ đó mang thông tin đặc trưng, trọng số cao. Kết quả là mỗi document được biểu diễn bằng một vector thưa (sparse) với chiều = kích thước từ vựng.
- **Giải quyết được gì:**

  - Cực kỳ đơn giản, không cần training, có thể giải thích trực quan.
  - Baseline tốt cho content-based filtering ở giai đoạn đầu.
- **Hạn chế:**

  - **Không hiểu ngữ nghĩa** — "tốt" và "hay" là hai từ hoàn toàn khác nhau dù nghĩa tương đương. "Bank" (ngân hàng) và "bank" (bờ sông) giống hệt nhau dù nghĩa hoàn toàn khác.
  - Sparse vector chiều rất cao (= vocab size, hàng chục ngàn) → tốn bộ nhớ, cosine similarity kém.
  - Bỏ qua hoàn toàn **thứ tự từ** — "sản phẩm tốt giá rẻ" và "giá rẻ sản phẩm tốt" giống hệt nhau.

---

### 3.2. GloVe *(Pennington et al., 2014)*

- **Ý tưởng:** Học word embedding bằng cách phân tích **ma trận đồng xuất hiện** (co-occurrence matrix) toàn cục của corpus. Nếu hai từ thường xuất hiện cùng nhau trong ngữ cảnh giống nhau, chúng sẽ có vector gần nhau. Mỗi từ → một vector dense 50–300 chiều.
- **Giải quyết được gì:**

  - Dense vector thay vì sparse → hiệu quả bộ nhớ và tính toán tốt hơn TF-IDF.
  - Capture được **quan hệ ngữ nghĩa** — vector("king") − vector("man") + vector("woman") ≈ vector("queen").
  - Pre-trained trên corpus rất lớn (Common Crawl, Wikipedia) → dùng được ngay.
- **Hạn chế:**

  - **Static embedding** — mỗi từ luôn có cùng một vector bất kể ngữ cảnh. "Bank" (ngân hàng) trong câu "I went to the bank" và "bank" (bờ sông) trong "river bank" có cùng vector. Đây là hạn chế nghiêm trọng khi mô tả sản phẩm chứa nhiều từ đa nghĩa.
  - Chỉ ở word-level — không có sẵn cách gộp thành sentence embedding (phải tự average hoặc dùng thêm model).

---

### 3.3. BERT *(Devlin et al., 2018)*

- **Ý tưởng:** Transformer encoder **hai chiều** (bidirectional) — khi xử lý một từ, nó xem xét đồng thời **tất cả từ bên trái và bên phải**. Pre-training bằng hai tác vụ: (1) Masked Language Modeling — che ngẫu nhiên 15% từ rồi đoán, buộc model hiểu ngữ cảnh sâu; (2) Next Sentence Prediction — đoán hai câu có liên tiếp không. Kết quả: cùng một từ cho embedding **khác nhau tùy ngữ cảnh**.
- **Giải quyết được gì:**

  - **Context-aware** — giải quyết triệt để vấn đề polysemy (đa nghĩa) mà GloVe/Word2Vec không làm được. "Apple" trong "Apple iPhone" vs "apple pie" → embedding khác nhau.
  - **Bidirectional** — hiểu ngữ cảnh tốt hơn hẳn so với các model đọc một chiều (GPT-1, ELMo forward).
  - Pre-trained knowledge cực kỳ phong phú — transfer sang downstream tasks hiệu quả.
- **Hạn chế:**

  - Token CLS (thường dùng làm sentence embedding) **không được tối ưu** cho sentence similarity — nó được train cho Next Sentence Prediction, một tác vụ khác hoàn toàn.
  - Nặng (110M params) → inference chậm khi cần encode hàng triệu item descriptions.
  - Để so sánh N items cần $O(N^2)$ forward passes nếu dùng cross-encoder — không scalable.

---

### 3.4. Sentence-Transformer *(Reimers & Gurevych, 2019)* ← **Chuẩn hiện tại trong MRS**

- **Ý tưởng:** Fine-tune BERT bằng **Siamese network** — đưa hai câu qua cùng một BERT encoder, lấy hai embeddings, rồi train sao cho: câu giống nghĩa → cosine similarity cao, câu khác nghĩa → cosine similarity thấp. Kết quả: mỗi câu → một vector mà **khoảng cách cosine phản ánh đúng mức độ tương đồng ngữ nghĩa**.
- **Giải quyết được gì:**

  - **Semantically meaningful sentence embeddings** — hai mô tả sản phẩm nói về cùng một thứ nhưng dùng từ khác → vẫn có cosine similarity cao. BERT raw không làm được điều này (CLS token của hai câu paraphrase có thể rất xa nhau).
  - **Scalable** — encode mỗi câu một lần, sau đó so sánh pairwise chỉ cần tính cosine. So với BERT cross-encoder, nhanh hơn hàng ngàn lần.
  - Trở thành **standard de facto** cho text feature extraction trong MRS từ 2021: LATTICE, BM3, FREEDOM, MGCN, DRAGON, MENTOR, SMORE đều dùng.
- **Hạn chế:**

  - Vẫn **static sau khi encode** — embedding không thay đổi theo RS context. Mô tả "áo len ấm" cho embedding giống nhau dù user ở vùng nhiệt đới (không cần) hay ôn đới (rất cần).
  - Pre-trained trên NLI/STS (Natural Language Inference / Semantic Textual Similarity), **không phải recommendation** → semantic gap vẫn tồn tại.

---

## 4. Các Nghiên Cứu Tiêu Biểu

### 4.1. VBPR *(AAAI 2016)* — He & McAuley

- **Ý tưởng:** Lần đầu tiên tích hợp **deep CNN features** (lấy từ VGG) vào Collaborative Filtering. Mỗi item có hai loại embedding: (1) ID embedding truyền thống học từ interactions, và (2) visual embedding project từ CNN features. Điểm dự đoán = tổng hai thành phần này.
- **Giải quyết được gì:**

  - **Mở ra lĩnh vực "visually-aware recommendation"** — chứng minh rằng thêm visual information thực sự cải thiện đáng kể so với MF thuần.
  - Cold-start items (chưa có interaction) vẫn có visual embedding → có thể gợi ý dựa trên ngoại hình tương đồng.
- **Hạn chế:**

  - CNN features **cố định** từ VGG pre-trained trên ImageNet — features cho mục đích phân loại ảnh (chó, mèo, xe) không nhất thiết tốt cho recommendation (user thích kiểu dáng nào).
  - Hoàn toàn bỏ qua textual modality.
  - Linear projection (nhân ma trận) để chuyển 4096-dim → thấp hơn — hạn chế khả năng biểu diễn phi tuyến.

---

### 4.2. DVBPR *(ICDM 2017)* — Kang et al.

- **Ý tưởng:** Thay vì dùng CNN features cố định như VBPR, **fine-tune toàn bộ CNN end-to-end** cùng với BPR recommendation objective. CNN vừa extract features vừa học ngay cái gì quan trọng cho recommendation.
- **Giải quyết được gì:**

  - Chứng minh **task-specific fine-tuning vượt trội generic features** — CNN được dạy: "không chỉ nhận diện vật thể, mà hãy nhận diện cái khiến user thích sản phẩm này hơn sản phẩm kia".
  - Mở ra ý tưởng end-to-end learning trong MRS, ảnh hưởng đến các nghiên cứu sau.
- **Hạn chế:**

  - **Training cực kỳ tốn kém** — fine-tune CNN trên mỗi dataset mới cần GPU nhiều giờ/ngày.
  - **Khó reproduce** — phụ thuộc raw images, mà Amazon images hay bị thiếu/hỏng.
  - Chỉ visual, bỏ qua text — giống VBPR.

---

### 4.3. LATTICE *(MM 2021)* — Zhang et al.

- **Ý tưởng:** Lần đầu dùng **Sentence-Transformer** làm textual feature extractor chuẩn trong MRS. Sau đó, dùng cosine similarity giữa các feature vectors để xây dựng **item-item graph** — hai items có visual/textual features tương đồng sẽ được kết nối. Graph này bổ sung thông tin ngữ nghĩa mà user-item interactions không có.
- **Giải quyết được gì:**

  - **Chuẩn hóa** textual feature extraction — từ LATTICE trở đi gần như mọi nghiên cứu MRS đều dùng Sentence-Transformer, tạo nên benchmark thống nhất.
  - Quality features → quality I-I graph → quality recommendation. Chất lượng features ảnh hưởng trực tiếp thấy rõ lên cấu trúc graph.
  - **Trở thành backbone** được kế thừa bởi FREEDOM, BM3, MGCN, HCGCN và nhiều công trình khác.
- **Hạn chế:**

  - Sentence-Transformer pre-trained trên NLI/STS — **không tối ưu cho RS context**. "Áo thun cotton trắng" và "áo sơ mi cotton trắng" có thể rất gần trong embedding space, nhưng user thích thứ nhất không nhất thiết thích thứ hai.
  - Features cố định sau khi extract, không adapt theo user behavior.

---

### 4.4. LLMRec *(WSDM 2024)* — Wei et al.

- **Ý tưởng:** Items có mô tả nghèo nàn hoặc thiếu hẳn → dùng **Large Language Models (GPT series)** để augment. Ba chiến lược: (1) LLM suy luận sở thích ẩn của user từ lịch sử tương tác, (2) LLM bổ sung attributes cho items ít thông tin, (3) LLM tạo pseudo interaction pairs cho cold-start items. Kết quả augmented text → encode bằng Sentence-Transformer.
- **Giải quyết được gì:**

  - **Sparse text descriptions** — nhiều items trên Amazon chỉ có tiêu đề ngắn, không có mô tả chi tiết. LLM generate mô tả phong phú → text features chất lượng cao hơn.
  - **Cold-start** — items/users mới không có lịch sử → LLM suy luận từ thông tin có sẵn (category, attributes) để tạo pseudo signals.
  - Ý tưởng dùng LLM làm data augmentor (không phải model chính) rất pragmatic — tận dụng sức mạnh LLM mà không cần deploy LLM trong inference.
- **Hạn chế:**

  - **Chi phí inference LLM rất cao** — augment hàng triệu items với GPT-4 tốn rất nhiều API cost.
  - **Hallucination** — LLM có thể generate thông tin sai hoàn toàn, gây nhiễu thay vì giúp ích. Vd: LLM nói sản phẩm có tính năng mà thực tế không có.
  - Phụ thuộc vào chất lượng prompt — prompt khác nhau cho kết quả rất khác nhau, chưa có cách tối ưu hóa tự động.

---

### 4.5. PromptMM *(WWW 2024)* — Wei et al.

- **Ý tưởng:** Nhận thấy **semantic gap** — BERT/ViT được train cho NLP/CV tasks, không phải RS → features không tối ưu cho recommendation. Thay vì fine-tune toàn bộ (tốn kém), thêm **learnable soft prompts** (các vectors có thể learn) vào input của encoder. Prompts đóng vai trò "chỉ dẫn" cho encoder hướng attention vào thông tin relevant cho RS. Sau đó dùng knowledge distillation: multimodal teacher → collaborative student.
- **Giải quyết được gì:**

  - **Semantic gap** giữa general-purpose extractor và RS-specific needs — prompts "dạy" encoder focus vào features hữu ích cho recommendation.
  - **Nhẹ hơn rất nhiều so với full fine-tuning** — chỉ train vài trăm prompt parameters thay vì 110M+ parameters của BERT. Efficient và practical.
  - Knowledge distillation transfer RS knowledge vào features.
- **Hạn chế:**

  - Hiệu quả phụ thuộc vào thiết kế prompt (prompt length, initialization) — chưa có lý thuyết rõ ràng về cách chọn.
  - Teacher model cần đủ tốt — nếu teacher kém thì distillation không giúp gì.
  - Thêm complexity vào training pipeline.

---

### 4.6. MENTOR *(AAAI 2025)* — Xu et al.

- **Ý tưởng:** Vấn đề lớn: visual features và textual features nằm trong **không gian khác nhau** — không so sánh trực tiếp được. MENTOR không chỉ extract features mà còn **align chúng** qua multi-level self-supervised learning: (1) align visual ↔ textual embeddings của cùng item (cross-modal), (2) align modality representations với collaborative representations (cross-domain).
- **Giải quyết được gì:**

  - **Modality misalignment** — sau khi align, visual embedding và textual embedding của cùng một item sẽ gần nhau trong shared space, cho phép fusion hiệu quả hơn.
  - Multi-level alignment đảm bảo **cả semantic consistency** (cùng item → cùng nghĩa) **lẫn behavioral consistency** (items mà users tương tác giống nhau → gần nhau).
  - Framework toàn diện — giải quyết nhiều vấn đề đồng thời.
- **Hạn chế:**

  - **Rất nhiều hyperparameters** — trọng số mỗi level, temperature cho InfoNCE, thresholds cho graph construction... tuning phức tạp.
  - Training cost cao do nhiều auxiliary SSL objectives.
  - Phức tạp hóa pipeline — khó debug khi có vấn đề.

---

## 5. Tổng Hợp: Vấn Đề, Giải Pháp, và Hạn Chế Còn Lại

- **Features cho general tasks ≠ RS tasks** → DVBPR (fine-tune end-to-end), PromptMM (prompt-tuning). Nhưng: fine-tune tốn kém, prompt design thiếu lý thuyết.
- **Static features không thay đổi theo context** → DVBPR (end-to-end). Nhưng: chi phí cao, khó reproduce.
- **Semantic gap (extractor ↔ RS)** → PromptMM (prompt-tuning), MENTOR (multi-level alignment). Nhưng: vẫn phụ thuộc pre-trained backbone.
- **Mô tả sản phẩm thưa thớt, thiếu** → LLMRec (LLM augmentation). Nhưng: hallucination, API cost.
- **Modality misalignment** → MENTOR (cross-modal SSL). Nhưng: hyperparameter phức tạp.
- **Cold start** → LLMRec (LLM suy luận). Nhưng: LLM có thể sai.
- **Reproducibility** → Provided features (MMRec). Nhưng: không optimize cho RS.

---

## 6. Hướng Nghiên Cứu Tương Lai

- **Unified Extraction + Encoding** — End-to-end training để feature extractor được optimize trực tiếp theo RS objective, thay vì pipeline tách rời hiện tại.
- **LLM-as-Extractor** — Dùng Multimodal LLMs (GPT-4V, LLaVA) làm primary extractor thay vì chỉ augmentor. Thách thức: inference cost, latency.
- **Domain-Adaptive Pre-training** — Pre-train encoder trên RS-specific data (product images, reviews) thay vì ImageNet/Wikipedia → features phù hợp hơn.
- **Richer Modalities** — Audio, video, behavioral sequences, location — hiện tại bài báo tập trung chủ yếu vào visual + textual.

---

*Tổng hợp từ: Xu et al., "A Survey on Multimodal Recommender Systems", IEEE Trans. Multimedia, 2025.*
