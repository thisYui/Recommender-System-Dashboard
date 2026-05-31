# Dung Hợp Đa Phương Thức (Multimodal Fusion) trong Hệ Thống Gợi Ý Đa Phương Thức

> **Nguồn:** Xu et al., *"A Survey on Multimodal Recommender Systems"*, IEEE Trans. Multimedia, 2025.

---

## 1. Tổng Quan

Multimodal Fusion là **bước thứ ba** trong pipeline MRS — quyết định **cách kết hợp** thông tin từ nhiều phương thức (visual, textual, audio...) lại với nhau.

Pipeline tổng thể:
- **Feature Extraction** → visual embedding, textual embedding, audio embedding (riêng biệt)
- **Encoder** → biểu diễn user/item tích hợp interactions
- **Multimodal Fusion** → **kết hợp các biểu diễn đa phương thức** thành một biểu diễn thống nhất
- **Loss Function** → tối ưu hóa

**Vấn đề cốt lõi:** Mỗi modality cung cấp một "góc nhìn" khác nhau về cùng một item. Visual nói "sản phẩm này trông thế nào", textual nói "sản phẩm này được mô tả ra sao", audio nói "sản phẩm này nghe thế nào". Nhưng hệ thống cần **một** đánh giá duy nhất: user này có thích item này không? Fusion quyết định cách gộp các góc nhìn lại — **gộp khi nào, gộp như thế nào** — và điều này ảnh hưởng rất lớn đến hiệu quả tổng thể.

Bài báo phân loại fusion theo **2 góc độ** song song:
1. **Thời điểm fusion** (khi nào gộp): Early vs Late
2. **Chiến lược fusion** (gộp bằng cách nào): Element-wise vs Concatenation, Heuristic vs Attentive

---

## 2. Góc Độ Thời Điểm: Early Fusion vs Late Fusion

### 2.1. Early Fusion (Dung hợp sớm)

- **Ý tưởng:** Gộp tất cả modality features **trước khi** đưa vào encoder. Nghĩa là encoder chỉ thấy **một** item embedding đã được fuse, không thấy từng modality riêng lẻ.

  Cụ thể: $\bar{E} = \text{Encoder}(U, \text{Aggr}(I_m))$ — aggregate tất cả modality embeddings thành một item embedding duy nhất, rồi mới encode.

- **Giải quyết được gì:**
  - **Cross-modal correlations được capture sớm** — encoder có thể học được rằng visual feature X kết hợp với textual feature Y tạo ra meaning Z mà không feature nào đơn lẻ có được.
  - Đơn giản hóa encoder — chỉ cần thiết kế một encoder duy nhất cho fused representation.
  - Hiệu quả khi modalities **bổ sung cho nhau** mạnh (nghĩa là kết hợp sớm giúp hiểu sâu hơn).

- **Hạn chế:**
  - **Mất đặc trưng riêng của từng modality** — khi gộp quá sớm, encoder không thể khai thác những đặc điểm mà chỉ visual hoặc chỉ textual mới có. Ví dụ: visual texture quan trọng cho fashion recommendation nhưng bị "pha loãng" khi gộp với textual features.
  - **Noise amplification** — nếu một modality noisy (vd: text mô tả không chính xác), noise bị trộn vào ngay từ đầu, ảnh hưởng toàn bộ quá trình encoding.
  - Khó phân bổ contribution — không biết modality nào đóng góp bao nhiêu vào prediction cuối cùng.

- **Công trình tiêu biểu:** VBPR, DVBPR, GraphCAR, VMCF, LATTICE, HCGCN, ACF, GRCN, BCCL, MCDRec

---

### 2.2. Late Fusion (Dung hợp muộn)

- **Ý tưởng:** Mỗi modality đi qua **encoder riêng biệt** trước (hoặc cùng encoder nhưng trên graph riêng). Chỉ **sau khi** encode xong, kết quả mới được gộp lại.

  Cụ thể: $\bar{E} = \text{Aggr}(\text{Encoder}(U, I_m))$ — encode từng modality riêng, rồi mới aggregate.

- **Giải quyết được gì:**
  - **Bảo toàn đặc trưng riêng** — mỗi modality được xử lý "chuyên biệt" trước khi gộp. Visual encoder học những gì visual-specific, text encoder học những gì text-specific. Không bị pha loãng.
  - Linh hoạt hơn — có thể dùng **architecture khác nhau** cho từng modality (GCN cho visual, Transformer cho text...).
  - Dễ **debug và analyze** — có thể kiểm tra contribution của từng modality riêng.

- **Hạn chế:**
  - **Miss cross-modal interactions** — vì mỗi modality xử lý cô lập, mô hình **không capture được tương quan giữa modalities** trong quá trình encoding. Ví dụ: visual nói "áo đỏ" nhưng text nói "áo thể thao" → gộp muộn không thấy "áo đỏ thể thao" như một khái niệm thống nhất.
  - Encoding tốn kém hơn nếu dùng encoder riêng cho mỗi modality.
  - Chiến lược aggregate cuối cùng (mean, weighted sum, attention) ảnh hưởng rất lớn nhưng khó chọn tối ưu.

- **Công trình tiêu biểu:** MMGCN, JRL, DualGNN, FREEDOM, MGCN, DRAGON, LGMRec, DiffMM, MENTOR, EgoGCN, MMGCL, PromptMM

---

### 2.3. So sánh Early vs Late Fusion

- **Early tốt khi:** Modalities bổ sung mạnh cho nhau, noise thấp, muốn đơn giản.
- **Late tốt khi:** Modalities có đặc trưng rất khác nhau, cần linh hoạt, muốn analyze từng modality.
- **Xu hướng hiện tại:** **Late Fusion thống trị** (MMGCN 2019 trở đi) — đặc biệt với graph-based encoders, mỗi modality có graph riêng. Nhưng các công trình mới bắt đầu **kết hợp cả hai** (vd: MENTOR tách riêng rồi align lại).

---

## 3. Góc Độ Chiến Lược: Gộp Bằng Cách Nào

### 3.1. Concatenation (Nối)

- **Ý tưởng:** Đơn giản nhất — nối các modality embeddings lại thành một vector dài hơn. Nếu visual embedding $d_v$-dim và text embedding $d_t$-dim → fused embedding $(d_v + d_t)$-dim. Thường đi kèm MLP để project xuống kích thước mong muốn.

- **Giải quyết được gì:**
  - **Bảo toàn toàn bộ thông tin** — không mất gì khi nối, mỗi chiều vẫn giữ nguyên giá trị.
  - Cực kỳ đơn giản, không cần thiết kế gì thêm.
  - MLP phía sau có thể learn non-linear interactions giữa modalities.

- **Hạn chế:**
  - Vector output rất dài nếu nhiều modalities → tăng memory và computation.
  - **Không có explicit interaction** giữa modalities tại điểm fusion — MLP phải "tự" learn interactions, không có structural bias.
  - Yêu cầu tất cả modalities phải **luôn có mặt** — nếu thiếu một modality, vector bị cắt ngắn.

---

### 3.2. Element-wise Operations (Phép toán từng phần tử)

- **Ý tưởng:** Kết hợp embeddings bằng phép cộng (addition), nhân (multiplication), hoặc trung bình (mean) **từng phần tử**. Yêu cầu tất cả embeddings cùng kích thước $d$. Output vẫn $d$-dim.

  Ví dụ: $\mathbf{e}_{\text{fused}} = \mathbf{e}_v + \mathbf{e}_t$ (element-wise addition) hoặc $\mathbf{e}_{\text{fused}} = \mathbf{e}_v \odot \mathbf{e}_t$ (element-wise multiplication).

- **Giải quyết được gì:**
  - **Tích hợp sâu hơn** concatenation — element-wise multiplication tạo **multiplicative interactions** (nếu visual dim i và text dim i cùng cao → fused dim i rất cao; nếu một cái thấp → fused thấp). Đây là dạng gating tự nhiên.
  - Output kích thước **không đổi** dù bao nhiêu modalities → scalable.
  - Đơn giản, không thêm parameters.

- **Hạn chế:**
  - **Khuếch đại noise** — nếu một modality noisy, element-wise add/multiply truyền noise trực tiếp vào output, không có cơ chế lọc.
  - Yêu cầu tất cả embeddings **cùng kích thước** — cần projection trước nếu kích thước khác nhau.
  - **Equal contribution** mặc định — cộng mean coi mọi modality bằng nhau, nhưng thực tế visual có thể quan trọng hơn text cho fashion, và ngược lại cho books.

---

### 3.3. Heuristic Strategies (Chiến lược dựa trên quy tắc cố định)

- **Ý tưởng:** Gán **trọng số cố định** cho mỗi modality khi gộp, dựa trên domain knowledge hoặc hyperparameter tuning. Ví dụ: $\mathbf{e}_{\text{fused}} = 0.6 \cdot \mathbf{e}_v + 0.3 \cdot \mathbf{e}_t + 0.1 \cdot \mathbf{e}_a$.

  "Heuristic" ở đây có 2 biến thể:
  - **Equal weights** — mọi modality trọng số bằng nhau (đặc biệt thường dùng trong element-wise sum/mean).
  - **Distinct weights** — trọng số khác nhau nhưng **cố định** trong suốt training, được chọn qua grid search hoặc domain expertise.

- **Giải quyết được gì:**
  - Đơn giản, không thêm parameters, dễ hiểu và dễ debug.
  - Domain expert có thể inject knowledge — "fashion recommendation cần visual nhiều hơn text".
  - Reproducible — không có randomness từ learned weights.

- **Hạn chế (quan trọng — chính xác là vấn đề mà Attentive fusion giải quyết):**
  - **Trọng số cố định cho mọi item** — nhưng thực tế, modality importance **thay đổi theo context**: visual quan trọng hơn cho item áo quần (user nhìn ảnh), nhưng textual quan trọng hơn cho sách (user đọc review). Heuristic không phân biệt được.
  - **Cố định trong suốt training** — dù model đã "hiểu" rằng visual không giúp gì cho item X, vẫn phải dùng trọng số visual cũ.
  - Grid search tốn kém — combinatorial explosion khi nhiều modalities.

- **Công trình tiêu biểu:** VBPR, DVBPR, VMCF, MMGCN, DualGNN, FREEDOM, DRAGON, LGMRec, DiffMM

---

### 3.4. Attentive Strategies (Chiến lược dựa trên attention)

- **Ý tưởng:** Thay vì trọng số cố định, dùng **attention mechanism** để learn trọng số **động** cho mỗi modality, phụ thuộc vào input hiện tại. Mô hình tự quyết định: "Với item này và user này, visual quan trọng hơn hay text quan trọng hơn?"

  Ví dụ: $\alpha_m = \text{softmax}(W \cdot \mathbf{e}_m)$, rồi $\mathbf{e}_{\text{fused}} = \sum_m \alpha_m \cdot \mathbf{e}_m$.

  Trọng số $\alpha_m$ thay đổi theo từng item (item-specific attention) hoặc từng cặp user-item (user-item-specific attention).

- **Giải quyết được gì:**
  - **Context-dependent modality weighting** — modality nào hữu ích hơn cho item/user hiện tại sẽ được trọng số cao hơn. Fashion items → visual weight cao; book items → text weight cao. Điều này heuristic không làm được.
  - **Adaptive** — trọng số tự điều chỉnh trong training, không cần manual tuning.
  - **Interpretability** — attention weights cho biết model đang "nhìn" vào modality nào, hữu ích cho debugging và explanation.

- **Hạn chế:**
  - **Chi phí tính toán cao hơn** — cần compute attention weights cho mỗi item hoặc mỗi cặp user-item.
  - **Thêm complexity vào training** — attention layers thêm parameters cần learn, dễ overfit trên small datasets.
  - **Attention collapse** — đôi khi attention "collapse" về một modality (vd: luôn coi visual quan trọng nhất) → mất ý nghĩa multimodal. Cần regularization.

- **Công trình tiêu biểu dùng attention trong Early Fusion:** ACF, GRCN, LATTICE, BCCL, MCDRec, MKGAT
- **Công trình tiêu biểu dùng attention trong Late Fusion:** PAMD, A2BM2GL, LLMRec, HHFAN, SMORE

---

## 4. Các Chiến Lược Fusion Nâng Cao

Ngoài 4 chiến lược cơ bản trên, các công trình gần đây phát triển thêm nhiều chiến lược tinh vi hơn:

### 4.1. Product-of-Experts (PoE)

- **Ý tưởng:** Thay vì gộp embeddings, gộp **distributions** (phân phối xác suất). Mỗi modality output một distribution (mean, variance), rồi fuse bằng cách nhân distributions: $p(z | G_v, G_t) \propto p(z | G_v) \times p(z | G_t)$. Kết quả: distribution mới "đồng ý" bởi cả hai modalities.

- **Giải quyết được gì:**
  - Fuse ở mức **probabilistic** — nếu visual nói "chắc chắn item này thuộc loại A" và text nói "chắc chắn loại A" → confidence rất cao. Nếu không đồng ý → uncertainty cao → model biết nó không chắc.
  - Uncertainty modeling tự nhiên.

- **Hạn chế:**
  - Cần mỗi modality encoder output distribution (mean + variance) → VAE-based architecture.
  - PoE có xu hướng overconfident khi cả hai modalities sai cùng hướng.

- **Công trình tiêu biểu:** MVGAE (2021)

---

### 4.2. Auxiliary Alignment Tasks (Tác vụ căn chỉnh phụ trợ)

- **Ý tưởng:** Không fuse trực tiếp mà dùng **auxiliary SSL tasks** để align modalities trước. Các modality representations được kéo gần nhau qua contrastive learning trước khi (hoặc đồng thời với) quá trình fusion chính.

- **Giải quyết được gì:**
  - **Modality misalignment** — visual và textual embeddings nằm trong không gian khác nhau. Alignment đưa chúng vào shared space → fusion hiệu quả hơn rất nhiều.
  - Grade of alignment controllable qua loss weight.

- **Hạn chế:**
  - Thêm auxiliary losses → training phức tạp hơn, nhiều hyperparameters.
  - Alignment quá mạnh có thể **mất modality-specific information** (mọi thứ trở nên giống nhau).

- **Công trình tiêu biểu:** MMSSL, PromptMM, MENTOR, GUME, ADDVAE, DGVAE, SAND, VMoSE

---

### 4.3. Causal Fusion

- **Ý tưởng:** Fuse modalities từ **góc nhìn nhân quả** — phân tích xem modality nào **thực sự gây ra** (cause) user preference, thay vì chỉ correlated. Dùng intervention framework (do-calculus) hoặc Average Treatment Effect (ATE) để đo đóng góp nhân quả của mỗi modality.

- **Giải quyết được gì:**
  - **Spurious correlations** — visual modality popular vì ảnh đẹp (confounding factor), không phải vì user thực sự thích. Causal fusion loại bỏ confounders.
  - Robust hơn khi distribution shift — pattern nhân quả ổn định hơn correlation.

- **Hạn chế:**
  - Xác định causal graph khó và thường chủ quan.
  - Computational cost cao hơn standard fusion.

- **Công trình tiêu biểu:** EliMRec (causal perspective, 2022), CKD (ATE strategy, 2024)

---

### 4.4. Spectral Fusion (Fusion trong miền tần số)

- **Ý tưởng:** Chiếu modality features vào **miền tần số** (frequency domain) rồi fuse trong spectral space, nơi các patterns global và local tách biệt rõ ràng hơn.

- **Giải quyết được gì:**
  - Low-frequency components capture **global patterns** (trend chung), high-frequency capture **fine-grained details** → fuse có chọn lọc theo frequency band.
  - Angle hoàn toàn mới — tránh limitations của spatial fusion truyền thống.

- **Hạn chế:**
  - Concept non-trivial, khó interpret.
  - Cần thiết kế spectral projection phù hợp.

- **Công trình tiêu biểu:** SMORE (2025)

---

## 5. Tương Tác Giữa Timing và Strategy

Bài báo nhấn mạnh: timing và strategy **không độc lập** mà tương tác hiệp đồng. Mỗi combination tạo ra đặc tính khác nhau:

- **Early + Heuristic** (VBPR, DVBPR) — Đơn giản nhất: gộp sớm bằng cách cộng hoặc nối, trọng số cố định. Baseline.

- **Early + Attentive** (LATTICE, ACF, GRCN) — Gộp sớm nhưng dùng attention chọn modality. Hiệu quả trung bình-cao.

- **Late + Heuristic** (MMGCN, FREEDOM, DiffMM) — Encode riêng rồi gộp bằng mean/sum. Đơn giản mà hiệu quả cao nhờ bảo toàn modality-specific features.

- **Late + Attentive** (PAMD, LLMRec, HHFAN) — Encode riêng, gộp dynamic. Flexibility cao nhất nhưng complexity cũng cao nhất.

- **Late + Auxiliary Alignment** (MENTOR, PromptMM, MMSSL) — Encode riêng, align qua SSL, rồi fuse. **Xu hướng mới nhất**, giải quyết misalignment.

---

## 6. Hàm Mất Mát (Loss Functions) — Gắn Chặt với Fusion

Fusion không thể tách rời Loss Function — chính loss quyết định mô hình **học gì** từ fused representations. Bài báo chia loss thành 2 loại:

### 6.1. Supervised Learning Losses (Tác vụ chính)

#### Pointwise Loss

- **Ý tưởng:** Đo khoảng cách giữa điểm dự đoán $\hat{r}_{u,i}$ và nhãn thực tế $r_{u,i}$ cho **từng cặp user-item riêng lẻ**.
  - **MSE (Mean Squared Error):** $\frac{1}{|D|}\sum (r_{u,i} - \hat{r}_{u,i})^2$ — kéo dự đoán gần nhãn thực.
  - **Cross-Entropy:** Coi recommendation như binary classification (có tương tác vs không) — phân phối dự đoán gần phân phối thực.

- **Giải quyết được gì:** Đơn giản, trực tiếp, dễ optimize.

- **Hạn chế:** Coi mọi cặp user-item **độc lập** — không capture được rằng "user thích item A **hơn** item B". Kém hiệu quả khi data sparse (quá nhiều negative pairs).

#### Pairwise Loss

- **Ý tưởng:** Thay vì đo cá nhân, so sánh **cặp**: item dương (user đã tương tác) nên có score cao hơn item âm (chưa tương tác).
  - **BPR (Bayesian Personalized Ranking):** $\sum -\log \sigma(\hat{r}_{u,i^+} - \hat{r}_{u,i^-})$ — maximize margin giữa positive và negative items.
  - **Hinge Loss:** $\max(0, 1 - r_{u,i} \cdot \hat{r}_{u,i})$ — tạo margin cứng.

- **Giải quyết được gì:**
  - **Learning-to-rank** — phù hợp hơn cho RS (mục tiêu là xếp hạng, không phải dự đoán chính xác rating).
  - Xử lý data sparse tốt hơn — chỉ cần "positive tốt hơn negative", không cần absolute score.

- **Hạn chế:** Phụ thuộc **negative sampling strategy** — chọn negative items nào ảnh hưởng rất lớn.

- **Xu hướng:** BPR là **dominant loss** trong MRS hiện đại — phần lớn papers dùng BPR.

---

### 6.2. Self-supervised Learning Losses (Tác vụ phụ trợ)

#### Feature-based SSL

- **Ý tưởng:** Tạo augmented views bằng cách **nhiễu loạn features** (feature dropout, thêm noise, dùng MLP biến đổi). Hai views của cùng node nên gần nhau (positive pair), views của nodes khác nên xa (negative pair). Đặc biệt: **hai modalities khác nhau** có thể tự nhiên tạo thành hai views.

- **Giải quyết được gì:**
  - **Data sparsity** — tạo thêm supervisory signal miễn phí.
  - **Modality alignment** — nếu coi visual và textual như hai views, contrastive learning kéo chúng gần nhau → alignment.
  - **Robustness** — model học features bất biến trước perturbation.

- **Hạn chế:** Augmentation strategy phải phù hợp — perturbation quá mạnh → mất information, quá yếu → không giúp gì.

- **Công trình tiêu biểu:** PAMD (disentangled), BCCL (bias-constrained), MGCN, LGMRec, PromptMM, SAND

---

#### Structure-based SSL

- **Ý tưởng:** Tạo augmented views bằng cách **nhiễu loạn cấu trúc graph** — edge dropout, node dropout, subgraph sampling. Hai views cùng graph nên tạo embeddings tương tự.

- **Giải quyết được gì:**
  - **Graph robustness** — model học biểu diễn ổn định dù graph bị thay đổi nhẹ → chống graph noise.
  - Capture **essential structural patterns** thay vì memorize specific edges.

- **Hạn chế:** Dropout ratio cần tuning — quá nhiều thì graph structure bị phá, quá ít thì không đủ augmentation.

- **Công trình tiêu biểu:** MMGCL (modality masking + edge dropout), HCGCN, SLMRec, A2BM2GL

---

#### Mixed SSL (Feature + Structure)

- **Ý tưởng:** Kết hợp **cả feature-based và structure-based** SSL — nhiễu loạn đồng thời features và graph structure → augmented views phong phú hơn.

- **Giải quyết được gì:**
  - **Comprehensive augmentation** — capture robustness ở cả mức feature lẫn structure.
  - Strongest supervisory signal — kết hợp ưu điểm của cả hai.

- **Hạn chế:** Nhiều auxiliary losses → training phức tạp, nhiều hyperparameters (trade-off giữa feature SSL, structure SSL, supervised loss).

- **Công trình tiêu biểu:** BM3 (2023), MMSSL (2023), MENTOR (2025)

---

#### Loss Functions — Hai loại SSL loss phổ biến

- **InfoNCE:** Maximize tỉ lệ positive pair score / tổng tất cả pair scores. Hiệu quả, scaling tốt, nhưng nhạy cảm với batch size (batch lớn → nhiều negatives → tốt hơn nhưng tốn memory).

- **Jensen-Shannon Divergence:** Gán nhãn 1 cho positive pairs, 0 cho negative pairs, rồi dùng binary cross-entropy. Đơn giản hơn InfoNCE, ít nhạy cảm với batch size.

---

## 7. Các Nghiên Cứu Tiêu Biểu về Fusion

### 7.1. MMGCN *(MM 2019)* — Wei et al. — Late + Heuristic

- **Ý tưởng:** Mỗi modality có GCN riêng, encode riêng, rồi **gộp bằng mean/sum** (heuristic, equal weights) ở cuối. Đây là paper đầu tiên thiết lập paradigm "late fusion + per-modality graph encoder".

- **Giải quyết được gì:** Chứng minh late fusion hiệu quả hơn early fusion cho graph-based MRS — mỗi modality cần encoding riêng.

- **Hạn chế:** Equal weights cho mọi modality và mọi item — phi lý vì modality importance thay đổi theo context.

---

### 7.2. LATTICE *(MM 2021)* — Zhang et al. — Early + Attentive

- **Ý tưởng:** Xây I-I graph riêng cho mỗi modality, rồi dùng **learned attention** để gộp thành một fused I-I graph **trước khi** đưa vào GCN encoder. Attention tự learn: "visual similarity hay text similarity quan trọng hơn cho item relationship?"

- **Giải quyết được gì:** Dynamic modality weighting — quan trọng vì ảnh hưởng fashion items khác books items. Trở thành baseline fusion pattern.

- **Hạn chế:** Attention weights learned ở **item-graph level**, không phải per-user — tất cả users cùng shared attention weights.

---

### 7.3. FREEDOM *(MM 2023)* — Zhou & Shen — Late + Heuristic

- **Ý tưởng:** Late fusion đơn giản — encode riêng từng modality (với frozen I-I graph + denoised U-I graph), rồi **gộp bằng sum** (heuristic). Key insight: **fusion strategy không cần phức tạp** nếu encoder đủ tốt.

- **Giải quyết được gì:** Chứng minh rằng đầu tư vào encoder quality (freeze + denoise) quan trọng hơn fusion complexity. Simple sum works surprisingly well.

- **Hạn chế:** Sum fusion vẫn coi mọi modality bằng nhau — performance ceiling khi modality contributions rất asymmetric.

---

### 7.4. MENTOR *(AAAI 2025)* — Xu et al. — Late + Auxiliary Alignment

- **Ý tưởng:** Late fusion kết hợp **multi-level auxiliary alignment**: (1) align visual ↔ textual embeddings (feature-based SSL), (2) align modality repr. ↔ collaborative repr. (ensures alignment không mất interaction info), (3) structure-based SSL cho graph robustness. Fusion cuối = sum sau khi aligned.

- **Giải quyết được gì:** **Misalignment** — vấn đề lớn nhất của late fusion truyền thống: encode riêng → embeddings ở không gian khác → sum/mean không có ý nghĩa. Alignment đưa chúng vào shared space trước → sum trở nên meaningful.

- **Hạn chế:** Rất nhiều loss terms, training phức tạp.

---

### 7.5. EliMRec *(MM 2022)* — Liu et al. — Causal Fusion

- **Ý tưởng:** Fuse modalities từ **góc nhìn nhân quả** — loại bỏ spurious correlations mà individual modalities gây ra (vd: visual attractiveness → popularity bias).

- **Giải quyết được gì:** **Unimodal bias** — khi ảnh đẹp làm model overestimate một item.

- **Hạn chế:** Causal assumptions chủ quan, khó validate.

---

### 7.6. MVGAE *(TMM 2021)* — Yi & Chen — Product-of-Experts Fusion

- **Ý tưởng:** Fuse bằng nhân distributions (Product-of-Experts) thay vì cộng/nối embeddings. Mỗi modality output phân phối Gaussian, PoE tạo joint distribution.

- **Giải quyết được gì:** **Uncertainty-aware** fusion — mô hình biết khi nào nó chắc chắn (cả hai modalities đồng ý) và khi nào không (modalities bất đồng).

- **Hạn chế:** VAE training complexity, PoE có thể overconfident.

---

### 7.7. SMORE *(2025)* — Spectral Fusion

- **Ý tưởng:** Chiếu multimodal features vào miền tần số, fuse trong spectral space. Low-frequency = global trends, high-frequency = fine-grained.

- **Giải quyết được gì:** Góc nhìn hoàn toàn mới, tránh limitations của spatial-domain fusion.

- **Hạn chế:** Non-trivial concept, chưa được validate rộng rãi.

---

## 8. Tổng Hợp: Chuỗi Tiến Hóa

- **2016–2018 (VBPR, DVBPR)** — Early + Heuristic: Gộp đơn giản, trọng số cố định → đủ cho MF-based.

- **2019–2020 (MMGCN, GRCN)** — Late + Heuristic: Chuyển sang graph-based encode riêng rồi sum → bảo toàn modality-specific features.

- **2021 (LATTICE, DualGNN)** — Early/Late + Attentive: Thêm attention chọn modality → dynamic weighting.

- **2022 (MMGCL, EgoGCN, EliMRec)** — Late + SSL/Causal: SSL tạo alignment, causal loại bias → fusion chất lượng hơn.

- **2023 (FREEDOM, MGCN, BM3)** — Late + Simple but Clean: Simple sum works if graphs are well-designed (freeze + denoise).

- **2024–2025 (MENTOR, SMORE, DiffMM)** — Late + Multi-level Alignment / Spectral: Alignment trước fusion + uncertainty modeling + new domains (frequency).

**Insight tổng quát:** Fusion **không phải bottleneck** — chất lượng encoder và alignment quantifies mattered more. Simple fusion (sum/mean) with good encoder and alignment outperforms complex fusion with poor encoder.

---

## 9. Hướng Nghiên Cứu Tương Lai

- **Adaptive Timing** — Không cần chọn early hay late cố định, mà let model learn khi nào fuse sớm, khi nào fuse muộn, tùy context.

- **Cross-modal Pre-fusion** — Cho phép modalities tương tác (ngay trong encoder) thay vì hoàn toàn cô lập (late) hoặc hoàn toàn trộn (early). EgoGCN đi theo hướng này.

- **Missing Modality Robustness** — Thực tế nhiều items thiếu một hoặc nhiều modalities. Fusion cần robust khi input incomplete.

- **Scalable Attention** — Attention cho mỗi cặp user-item-modality rất tốn → cần efficient approximations.

- **Fusion for New Modalities** — Audio, video, behavioral sequences cần fusion strategies khác visual+text truyền thống.

---

*Tổng hợp từ: Xu et al., "A Survey on Multimodal Recommender Systems: Recent Advances and Future Directions", IEEE Trans. Multimedia, 2025.*
