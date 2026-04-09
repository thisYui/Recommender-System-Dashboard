# Bộ Mã Hóa (Encoder) trong Hệ Thống Gợi Ý Đa Phương Thức

> **Nguồn:** Xu et al., *"A Survey on Multimodal Recommender Systems"*, IEEE Trans. Multimedia, 2025.

---

## 1. Tổng Quan

Encoder là **bước thứ hai** trong pipeline MRS. Nó nhận modality embeddings (từ Feature Extraction) cùng dữ liệu tương tác (user đã click/mua/xem gì), rồi học ra **biểu diễn sở thích** của mỗi user và **biểu diễn đặc tính** của mỗi item. Mục tiêu cuối cùng: dự đoán user nào sẽ thích item nào.

**Sự khác biệt với Feature Extraction:**
- Feature Extraction hỏi: "Ảnh/text này **chứa gì**?" (content understanding)
- Encoder hỏi: "Dựa trên lịch sử tương tác, user này **thích gì**?" (preference learning)
- Feature Extraction dùng pre-trained knowledge (ImageNet, NLP corpus). Encoder **học trực tiếp từ dữ liệu tương tác** của hệ thống.

---

## 2. Hai Phương Pháp Encoder Chính

### 2.1. MF-based Encoder (Phân tích ma trận)

- **Ý tưởng:** Coi bài toán recommendation như bài toán **hoàn thiện ma trận** (matrix completion). Ma trận tương tác $R$ (user × item) chủ yếu là ô trống (user chưa tương tác). MF phân rã $R$ thành tích hai ma trận nhỏ hơn: $R \approx U \cdot I^\top$, trong đó mỗi user và item được biểu diễn bằng một vector d-chiều. Dự đoán = tích vô hướng giữa user vector và item vector.

  Khi tích hợp multimodal, có 2 cách:
  - **Early Fusion:** Gộp tất cả modality features thành một item embedding rồi factorize.
  - **Late Fusion:** Factorize riêng từng modality rồi aggregate kết quả.

- **Giải quyết được gì:**
  - Đơn giản, dễ hiểu, dễ triển khai. Training nhanh, ít tài nguyên.
  - Dễ tích hợp multimodal features bằng cách nối (concatenation) hoặc cộng (element-wise addition) vào item embeddings.
  - Đủ tốt cho baseline và nhiều ứng dụng thực tế.

- **Hạn chế (quan trọng — đây là lý do Graph-based encoder ra đời):**
  - Chỉ capture **quan hệ bậc 1** (first-order) — tức chỉ biết user A đã tương tác với item 1, user B đã tương tác với item 2. Nhưng **không** suy luận được rằng "user A và user B có sở thích giống nhau vì cả hai đều thích item 3" — đây là quan hệ **bậc cao** (high-order).
  - Ví dụ cụ thể: User A mua *áo thun trắng* và *quần jean xanh*. User B mua *quần jean xanh* và *giày sneaker*. MF thấy A và B chỉ có 1 item chung → similarity thấp. Nhưng thực tế, qua mạng lưới items, A rất có thể thích sneaker mà B mua. MF không nắm được điều này.
  - Kém với dữ liệu thưa thớt — user/item ít tương tác → latent vector gần như random.

---

### 2.2. Graph-based Encoder (Mạng đồ thị)

- **Ý tưởng:** Biểu diễn toàn bộ hệ thống dưới dạng **đồ thị** — users và items là nodes, tương tác là edges. Dùng **Graph Convolutional Network (GCN)** để mỗi node aggregate thông tin từ neighbors: user cập nhật embedding dựa trên items đã tương tác, item cập nhật embedding dựa trên users đã tương tác. Stack $L$ layers GCN = aggregate thông tin từ $L$-hop neighbors → capture được quan hệ bậc cao.

- **Giải quyết được gì:**
  - **High-order connectivity** — quay lại ví dụ trên: với 2-layer GCN, thông tin từ sneaker chảy qua user B → quần jean → user A. Giờ A "biết" về sneaker mà không cần tương tác trực tiếp.
  - Biểu diễn phong phú hơn MF rất nhiều — mỗi node tích hợp thông tin từ toàn bộ neighborhood.

- **Hạn chế:**
  - Phức tạp hơn — cần lưu graph, propagation tốn kém.
  - **Over-smoothing** — quá nhiều layers → mọi node embedding trở nên giống nhau vì aggregate quá nhiều lần. Thường giới hạn 2–3 layers.
  - Scalability — với hàng triệu users/items, graph rất lớn.

---

### LightGCN *(He et al., SIGIR 2020)* — Backbone tiêu chuẩn

- **Ý tưởng:** GCN gốc (Kipf & Welling, 2017) có weight matrix $W$ và activation function $\sigma$ ở mỗi layer. LightGCN phát hiện rằng trong RS, **cả hai đều không hữu ích — thậm chí gây hại**: weight matrix gây overfitting, activation function làm gradient khó train. Bỏ hết → chỉ còn neighborhood aggregation thuần túy. Final embedding = trung bình tất cả layer outputs.

- **Giải quyết được gì:**
  - Performance **tốt hơn** GCN đầy đủ cho recommendation — ít params = ít overfit.
  - Đơn giản hóa tối đa → dễ implement, dễ extend.
  - Trở thành **backbone tiêu chuẩn** — MMGCN, LATTICE, FREEDOM, MENTOR... đều dùng LightGCN hoặc biến thể.

- **Hạn chế:**
  - Chỉ dùng U-I bipartite graph — chưa khai thác I-I hay U-U relations.
  - Đơn giản quá → có thể thiếu expressiveness cho data phức tạp.

---

### Homogeneous Graph Enhancement — Thêm I-I và U-U

Ngoài U-I graph (user-item), nhiều encoder thêm:

- **Item-Item (I-I) Graph** — Kết nối items có features tương đồng (visual, textual). Ý nghĩa: hai sản phẩm trông giống nhau hoặc mô tả giống nhau → nên có biểu diễn gần nhau, dù không có chung user nào. Giúp capture **latent semantic correlations** giữa items.

- **User-User (U-U) Graph** — Kết nối users có hành vi/sở thích tương tự. Ý nghĩa: users thích cùng loại items → nên có biểu diễn gần nhau. Đặc biệt hữu ích cho users mới (ít interactions) — "mượn" thông tin từ users tương đồng.

Cả hai đều cần **symmetric normalization** để chống high-degree node bias: node kết nối quá nhiều sẽ chi phối không cân xứng.

**Phân loại theo loại graph:**
- **U-I only:** MMGCN, MGAT, GRCN
- **U-I + I-I:** LATTICE, FREEDOM, BM3, HCGCN
- **U-I + U-U:** DualGNN, MMGCL
- **U-I + U-U + I-I:** DRAGON, LGMRec, MENTOR

---

## 3. Các Nghiên Cứu Tiêu Biểu — MF-based

### 3.1. VBPR *(AAAI 2016)* — He & McAuley

- **Ý tưởng:** Mở rộng MF truyền thống bằng cách thêm **visual channel**. Mỗi item có hai embedding: (1) ID embedding học từ interactions (như MF thông thường), (2) visual embedding project từ CNN features. Điểm dự đoán = tổng hai thành phần. User cũng có hai bộ latent factors tương ứng.

- **Giải quyết được gì:**
  - **Mở ra lĩnh vực visually-aware recommendation** — lần đầu chứng minh visual signal thực sự hữu ích, cải thiện đáng kể so với MF thuần, đặc biệt trên fashion/product domains.
  - Cold-start items không có interaction vẫn có visual embedding → system vẫn có thể gợi ý dựa trên ngoại hình tương đồng.
  - Framework đơn giản, dễ hiểu, dễ reproduce → trở thành baseline phổ biến.

- **Hạn chế:**
  - MF-based → chỉ bậc 1, không khai thác high-order connectivity.
  - Visual features cố định (VGG pre-trained, không fine-tune) — features cho classification ≠ features cho recommendation.
  - Bỏ qua hoàn toàn textual modality.
  - Linear projection hạn chế — quan hệ phi tuyến giữa visual features và preferences không được capture.

---

### 3.2. ACF *(SIGIR 2017)* — Chen et al.

- **Ý tưởng:** Trong MF truyền thống, mọi items trong lịch sử user đều có **trọng số bằng nhau** khi tính user representation — điều này phi lý (user click nhầm ≠ user mua hàng). ACF đề xuất **attention 2 cấp**: (1) **Item-level attention** — học trọng số mỗi item trong lịch sử tương tác (items quan trọng hơn → trọng số cao hơn), (2) **Component-level attention** — trong mỗi ảnh, learn trọng số mỗi vùng (region nào user thực sự quan tâm).

- **Giải quyết được gì:**
  - **Lần đầu áp dụng attention trong CF** → ảnh hưởng lớn đến thiết kế các encoder sau.
  - Giải quyết unequal contribution — không phải mọi interaction đều quan trọng như nhau.
  - Component-level attention cho phép hiểu **user quan tâm phần nào** của sản phẩm (vd: chỉ quan tâm phần cổ áo, không phải toàn bộ) → tăng interpretability.

- **Hạn chế:**
  - Vẫn MF-based → bậc 1.
  - Attention mechanism tăng computational cost.
  - Component-level chỉ cho visual — text không có Region-level attention tương tự.

---

### 3.3. EliMRec *(MM 2022)* — Liu et al.

- **Ý tưởng:** Dùng **causal inference** để phát hiện và loại bỏ **single-modal bias**. Vấn đề: một số items popular vì ảnh đẹp (viral trên social media), không phải vì chất lượng thật → user interactions bị confound bởi visual attractiveness. EliMRec dùng do-calculus (interventional framework) để tách genuine preference khỏi spurious correlations gây ra bởi individual modalities.

- **Giải quyết được gì:**
  - **Unimodal bias** — khi ảnh đẹp tạo popularity bias khiến model recommend items dựa trên ngoại hình thay vì relevance thực sự. Causal framework loại bỏ confounding effect.
  - Tổng quát hơn — framework áp dụng được cho bất kỳ modality nào đang gây bias, không chỉ visual.

- **Hạn chế:**
  - Cần xác định đúng **causal graph** — giả định nhân quả có thể sai trong thực tế, dẫn đến remove nhầm signal hữu ích.
  - Tính toán interventional distribution phức tạp.
  - Khó validate — không thể chạy RCT (randomized controlled trial) trong production RS.

---

## 4. Các Nghiên Cứu Tiêu Biểu — Graph-based

### 4.1. MMGCN *(MM 2019)* — Wei et al.

- **Ý tưởng:** Xây dựng **GCN riêng biệt cho mỗi modality** — visual graph, textual graph, audio graph. Mỗi modality có U-I graph riêng → message passing riêng → modality-specific representations. Cuối cùng aggregate tất cả + cộng ID embedding.

- **Giải quyết được gì:**
  - **Đưa graph-based encoder vào MRS lần đầu tiên** một cách có hệ thống.
  - Mỗi modality **học khác nhau** — visual graph có thể có patterns khác text graph (vd: items similar visual nhưng khác text). Tách riêng tránh modalities can thiệp lẫn nhau.
  - High-order connectivity qua GCN — vượt trội so với MF-based methods trước đó.

- **Hạn chế (3 vấn đề lớn, mỗi vấn đề sinh ra một loạt nghiên cứu giải quyết):**
  - **Modality isolation** — mỗi modality GCN propagate hoàn toàn cô lập, miss cross-modal interactions. Giải quyết sau bởi: EgoGCN (2022).
  - **Chỉ U-I graph** — chưa khai thác I-I hay U-U relations. Giải quyết sau bởi: LATTICE (I-I, 2021), DualGNN (U-U, 2021).
  - **Noisy interactions** — implicit feedback (clicks) không đáng tin cậy, gây noise trong graph. Giải quyết sau bởi: GRCN (2020), FREEDOM (2023).

---

### 4.2. GRCN *(MM 2020)* — Wei et al.

- **Ý tưởng:** U-I graph xây từ implicit feedback (clicks, views) chứa rất nhiều **noisy edges** — user click nhưng không thực sự thích (misclick, curiosity browsing, forced exposure). GRCN tự động **identify và prune noisy edges** bằng confidence scoring: mỗi edge được gán một confidence score, edges dưới threshold bị loại bỏ trước khi GCN propagation.

- **Giải quyết được gì:**
  - **Graph noise** — implicit feedback vốn dĩ noisy. GRCN là đầu tiên chỉ ra rằng **làm sạch graph trước khi train** quan trọng hơn là thiết kế GCN phức tạp hơn.
  - Purified graph → cleaner signal → biểu diễn chính xác hơn, đặc biệt trên datasets có interaction density thấp.

- **Hạn chế:**
  - Chỉ denoise **U-I graph** — I-I và U-U graphs (nếu dùng) cũng có thể noisy nhưng chưa xét.
  - Confidence scoring thêm overhead tính toán.
  - Có thể **cắt nhầm cạnh hữu ích** — user thực sự thích item nhưng confidence score thấp (vd: user mới, item ít popular).

---

### 4.3. DualGNN *(TMM 2021)* — Wang et al.

- **Ý tưởng:** MMGCN chỉ có U-I graph → miss thông tin tương đồng giữa users. DualGNN thêm **User-User Graph** song song: users thích items tương đồng về visual/text → kết nối. Dual propagation: (1) U-I graph → collaborative signal, (2) U-U graph → user preference patterns ẩn. Hai representations được fuse ở cuối.

- **Giải quyết được gì:**
  - **Users ít tương tác** (cold-start users) có thể "mượn" thông tin từ users tương đồng qua U-U graph → biểu diễn tốt hơn nhiều.
  - Phát hiện **mẫu sở thích ẩn** — hai users không chung item nào nhưng thích cùng style → U-U graph kết nối họ.
  - Đặc biệt hữu ích cho domains có user diversity cao (fashion, music).

- **Hạn chế:**
  - Xây U-U graph cần tính **pairwise similarity** $O(|U|^2)$ — rất tốn với hàng triệu users.
  - Chưa có I-I graph — chỉ giải quyết user-side, item-side vẫn thiếu.
  - U-U similarity metric (cosine trên aggregated features) có thể noisy nếu features không tốt.

---

### 4.4. LATTICE *(MM 2021)* — Zhang et al.

- **Ý tưởng:** Vấn đề: U-I graph chỉ kết nối items qua users chung — hai items tương đồng nhưng không có user chung sẽ không bao giờ trao đổi thông tin. LATTICE xây **Item-Item Graph dựa trên modality similarity**: tính cosine similarity giữa visual/textual features của mọi cặp items, giữ lại top-k neighbors. Mỗi modality có I-I graph riêng, rồi dùng **learnable attention** để merge. GCN propagation trên fused I-I graph tạo ra enriched item embeddings, sau đó đưa vào U-I graph.

- **Giải quyết được gì:**
  - **Latent semantic correlations** giữa items — hai sản phẩm trông giống nhau nhưng không có chung user nào vẫn được kết nối. Đặc biệt quan trọng cho long-tail items (ít interactions).
  - Dynamic modality weighting qua attention — tự học modality nào quan trọng hơn cho item relationship (visual quan trọng hơn cho fashion, text quan trọng hơn cho books).
  - **Ảnh hưởng khổng lồ** — trở thành backbone cho hàng loạt nghiên cứu sau 2021: FREEDOM, MGCN, BM3, HCGCN, DRAGON đều kế thừa ý tưởng I-I graph.

- **Hạn chế:**
  - I-I graph kNN construction tốn $O(|I|^2)$ pairwise comparisons — không scalable cho hàng triệu items.
  - Graph **cố định** dựa trên pre-extracted features — nếu features kém (semantic gap) thì graph sai vĩnh viễn.
  - I-I graph được **cập nhật mỗi epoch** → training cost cao, instability.

---

### 4.5. MMGCL *(SIGIR 2022)* — Yi et al.

- **Ý tưởng:** Micro-video recommendation có **data rất sparse** (users xem hàng triệu videos nhưng interactions per user rất ít). MMGCL tích hợp **Self-supervised Learning (SSL)** vào graph encoder: tạo augmented views bằng (1) **modality masking** — che ngẫu nhiên một modality (vd: bỏ visual, chỉ giữ text+audio), (2) **edge dropout** — bỏ ngẫu nhiên cạnh trong graph. Contrastive learning: cùng node qua hai augmented views nên gần nhau (positive), node khác nên xa (negative).

- **Giải quyết được gì:**
  - **Data sparsity** — SSL tạo thêm supervisory signal từ chính cấu trúc dữ liệu, không cần labels thêm. Hiệu quả đặc biệt rõ trên datasets thưa.
  - **Cross-modal robustness** — modality masking buộc model không phụ thuộc quá nhiều vào một modality → robust hơn khi modality bị noise hoặc missing.
  - Innovative negative sampling strategy — tránh sampling false negatives (items tương đồng bị coi là negative).

- **Hạn chế:**
  - Augmentation strategy (mask modality nào, dropout bao nhiêu %) cần tuning cẩn thận — mask quá nhiều → mất info, quá ít → không đủ augmentation.
  - Contrastive learning **nhạy cảm** với batch size và temperature parameter.

---

### 4.6. EgoGCN *(MM 2022)* — Chen et al.

- **Ý tưởng:** Vấn đề MMGCN: mỗi modality GCN propagate **hoàn toàn cô lập** — visual graph không biết text graph đang nói gì. Thông tin chỉ được merge ở cuối (late fusion). EgoGCN đề xuất **cross-modal message passing**: khi aggregate neighbors của một node, **trọng số cạnh phụ thuộc vào thông tin từ tất cả modalities**, không chỉ modality hiện tại. Nghĩa là visual propagation "tham khảo" text information để quyết định neighbor nào quan trọng.

- **Giải quyết được gì:**
  - **Modality isolation** — phá vỡ sự cô lập giữa các modality graphs. Cho phép thông tin liên phương thức chảy ngay trong quá trình propagation, không phải chờ đến fusion cuối.
  - Ví dụ: hai items có visual giống nhau nhưng text rất khác → trong MMGCN, visual graph vẫn truyền thông tin mạnh giữa chúng. Trong EgoGCN, text signal **điều chỉnh** trọng số visual propagation → cân nhắc cả semantic alignment.

- **Hạn chế:**
  - Edge modulation tăng tính toán đáng kể — mỗi edge cần attend tới tất cả modalities.
  - Cross-modal signal có thể **gây nhiễu** nếu modalities không aligned — visual nói giống, text nói khác → confusing.

---

### 4.7. FREEDOM *(MM 2023)* — Zhou & Shen

- **Ý tưởng:** Hai insights quan trọng: (1) I-I graph (từ LATTICE) xây từ pre-extracted features → **features không thay đổi giữa các epochs** → graph cũng không cần update → **đóng băng (freeze)** hoàn toàn là hợp lý và tiết kiệm cost. (2) U-I graph có noise → cần **denoise** bằng adaptive edge dropping (edges với confidence thấp bị giảm trọng số hoặc loại bỏ).

- **Giải quyết được gì:**
  - **Insight #1 rất elegant:** LATTICE rebuild I-I graph mỗi epoch → tốn thời gian + gây instability. FREEDOM chỉ build một lần rồi freeze → **giảm >50% training time**, kết quả thậm chí tốt hơn vì ổn định hơn. Đây là bài học quan trọng: đôi khi **đơn giản lại tốt hơn**.
  - **Insight #2:** Denoise U-I graph tiếp tục cải thiện — noise reduction trên cả hai loại graph.
  - Trở thành **top SOTA 2023–2024**, một trong những papers được cite nhiều nhất trong MRS.

- **Hạn chế:**
  - Frozen I-I graph **phụ thuộc chất lượng pre-extracted features** — nếu ResNet/Sentence-Transformer features kém cho domain này, graph sai nhưng không thể sửa trong training.
  - Denoising strategy (ngưỡng drop, tỷ lệ) cần tuning.
  - Giả định I-I graph ổn định chỉ đúng khi dùng fixed pre-extracted features — nếu encoder cập nhật features (end-to-end), giả định này sai.

---

### 4.8. MGCN *(MM 2023)* — Yu et al.

- **Ý tưởng:** Modality features (từ ResNet, Sentence-Transformer) chứa **noise không liên quan đến recommendation**. Vd: ảnh sản phẩm có background đẹp — CNN "thấy" background nhưng user mua vì sản phẩm. MGCN dùng **item behavior information** (user interaction patterns) để "làm sạch" features: features không nhất quán với cách users tương tác → giảm trọng số. Sau đó model user preference per modality dựa trên cleaned features.

- **Giải quyết được gì:**
  - **Modality noise** — loại bỏ thông tin trong features mà không relevant đến RS, chỉ dựa trên behavior signal (không cần human annotation).
  - User preference per modality — hiểu user A quan tâm visual hơn (mua vì ảnh đẹp), user B quan tâm text hơn (đọc reviews kỹ) → personalized at modality level.

- **Hạn chế:**
  - Behavior information **cũng có thể noisy** (implicit feedback) → dùng signal noisy để clean signal noisy là circular.
  - Cleaning có thể vô tình loại bỏ useful signal — edge case items có behavior unusual.

---

### 4.9. LGMRec *(AAAI 2024)* — Guo et al.

- **Ý tưởng:** Vấn đề: GCN thông thường chỉ capture **local topology** — thông tin giới hạn trong L-hop neighborhood. Items/users ở xa trong graph không giao tiếp được. LGMRec kết hợp 2 cấp: (1) **Local** — GCN trên U-I graph nắm topological nuances cục bộ, (2) **Global** — **Hypergraph learning** nắm global dependencies. Trong hypergraph, mỗi hyperedge kết nối **nhóm** users/items (thay vì chỉ 2 nodes) → capture group-level patterns.

- **Giải quyết được gì:**
  - **Long-tail items** — items ít tương tác có rất ít neighbors trong U-I graph → GCN propagation kém. Hypergraph kết nối chúng với nhóm rộng hơn → biểu diễn tốt hơn.
  - Local + Global = comprehensive — local tốt cho items popular (nhiều neighbors), global tốt cho items sparse.

- **Hạn chế:**
  - **Hypergraph construction** tốn kém — cần xác định hyperedges hợp lý, thường dùng clustering.
  - Hai nhánh (local + global) tăng complexity — cần chiến lược fusing hợp lý.
  - Hypergraph có thể introduce noise nếu construction không tốt.

---

### 4.10. DiffMM *(MM 2024)* — Jiang et al.

- **Ý tưởng:** Áp dụng **Diffusion Probabilistic Models** (vốn nổi tiếng trong image generation) vào graph encoder. Forward process: **thêm noise** dần vào user representations qua nhiều bước. Reverse process: **modality-aware denoising network** học khôi phục representations clean từ noisy version. Ý tưởng: quá trình denoise buộc model hiểu **deep structure** thay vì memorize surface patterns.

- **Giải quyết được gì:**
  - **Robust representations** — giống denoising autoencoders, thêm noise rồi khôi phục giúp model generalize tốt hơn, ít overfit.
  - **Uncertainty modeling** tự nhiên — diffusion model sinh ra distribution thay vì point estimate, phản ánh independently levels of confidence.
  - Hướng nghiên cứu hoàn toàn mới — kết hợp generative models + graph-based RS.

- **Hạn chế:**
  - Diffusion cần **nhiều steps** (forward T steps + reverse T steps) → training rất chậm so với GCN thông thường.
  - Noise schedule, number of steps T → hyperparameters cần tuning kỹ.
  - Chưa validate trên billion-scale data — scalability chưa rõ.
  - Concept khó hiểu → khó reproduce.

---

### 4.11. MENTOR *(AAAI 2025)* — Xu et al.

- **Ý tưởng:** Framework **toàn diện nhất** — dùng đồng thời U-I, U-U, I-I graphs, và thêm **multi-level self-supervised learning** 3 levels: (1) Cross-modal alignment — kéo visual và textual embeddings của cùng item lại gần nhau, (2) Modality-interaction alignment — align modality representations với collaborative representations (đảm bảo modality info không đi lệch khỏi interaction patterns), (3) Graph-level structural consistency — đảm bảo cấu trúc đồ thị nhất quán qua các views.

- **Giải quyết được gì:**
  - **Modality misalignment** — visual và textual features nằm trong không gian hoàn toàn khác → Level 1 kéo chúng vào shared space.
  - **Information loss khi align** — nghiên cứu trước (vd: MMSSL) khi align modalities thường mất interaction signal → Level 2 bảo toàn interaction info.
  - **Tích hợp đầy đủ** tất cả loại graphs + SSL → state-of-the-art performance.
  - Cùng nhóm tác giả với bài survey → phản ánh best practices mới nhất.

- **Hạn chế:**
  - **Phức tạp nhất** trong tất cả methods — rất nhiều hyperparameters: trọng số cho mỗi level SSL, temperature, graph construction thresholds, GCN depth cho từng graph, ...
  - Training pipeline cực kỳ complex → khó debug, khó reproduce.
  - Chi phí tính toán cao nhất — 3 types of graphs + 3 levels of SSL.
  - Câu hỏi mở: performance gain đến từ đâu? Framework đúng hay chỉ vì nhiều components hơn?

---

## 5. Tổng Hợp: Chuỗi Giải Quyết Vấn Đề

Nhìn tổng thể, mỗi nghiên cứu mới **giải quyết hạn chế của nghiên cứu trước**, tạo thành chuỗi tiến hóa:

- **MF chỉ bậc 1** → MMGCN dùng GCN → capture high-order
- **MMGCN chỉ U-I** → LATTICE thêm I-I graph, DualGNN thêm U-U graph
- **MMGCN cô lập modalities** → EgoGCN cross-modal message passing
- **Graph có noise** → GRCN prune noisy edges, FREEDOM freeze + denoise
- **Modality features có noise** → MGCN behavior-guided cleaning
- **Data sparse** → MMGCL dùng SSL tạo thêm signal
- **Long-tail items** → LGMRec dùng hypergraph global
- **Deterministic representations** → DiffMM dùng diffusion probabilistic
- **Modality misalignment** → MENTOR multi-level SSL alignment
- **Unimodal bias** → EliMRec causal inference

---

## 6. Hướng Nghiên Cứu Tương Lai

- **Unified Extraction + Encoding** — Hiện tại hai bước tách rời gây semantic gap. Cần end-to-end models optimize cả hai cùng lúc.

- **LLM-enhanced Encoding** — Multimodal LLMs (GPT-4V, LLaVA) có thể encode cả ảnh + text trong một pass. Thách thức: inference cost, latency.

- **Adaptive Graph Construction** — Graph hiện tại build offline trước training. Cần dynamic graph cập nhật real-time theo user behavior.

- **Scalable Graph Encoders** — GCN gặp khó với billion-scale graphs. Cần graph sampling, subgraph training, mini-batch GCN.

- **Richer Modalities** — Audio, video, behavioral sequences, location — cần graph structures linh hoạt hơn.

---

*Tổng hợp từ: Xu et al., "A Survey on Multimodal Recommender Systems", IEEE Trans. Multimedia, 2025.*
