# Phân Tích Chi Tiết: Trích Xuất Đặc Trưng (Feature Extraction) trong Hệ Thống Gợi Ý Đa Phương Thức

> **Nguồn tham khảo chính:** Xu et al., *"A Survey on Multimodal Recommender Systems: Recent Advances and Future Directions"*, IEEE Transaction on Multimedia, 2025.

---

## 1. Tổng Quan về Feature Extraction trong MRS

### 1.1. Vị Trí trong Pipeline MRS

Feature Extraction là **bước đầu tiên và nền tảng** trong pipeline của Hệ thống Gợi ý Đa phương thức (Multimodal Recommender Systems - MRS). Toàn bộ pipeline gồm 4 bước:

```
[Raw Data] → Feature Extraction → Encoder → Multimodal Fusion → Loss Function → [Recommendations]
```

Nếu Feature Extraction thất bại hoặc kém chất lượng, toàn bộ hệ thống phía sau sẽ không thể bù đắp được.

### 1.2. Định Nghĩa và Mục Tiêu

> *"Trích xuất đặc trưng là một quá trình quan trọng nhằm biểu diễn các đặc trưng kênh có số chiều thấp, có thể diễn giải được bằng cách sử dụng các kỹ thuật nhúng (embedding)."*  
> — Xu et al., 2025

**Mục tiêu cốt lõi:**
- Chuyển đổi dữ liệu thô (hình ảnh, văn bản, âm thanh...) thành các **vector embedding** có số chiều thấp hơn
- Giữ lại thông tin ngữ nghĩa quan trọng trong khi loại bỏ nhiễu
- Tạo ra các đặc trưng **có thể diễn giải và tính toán** được cho các bước tiếp theo

### 1.3. Các Phương Thức Dữ Liệu Phổ Biến

Hầu hết các dataset trong MRS đều bao gồm ít nhất 3 phương thức chính:

| Phương thức | Ví dụ nguồn dữ liệu | Mô hình trích xuất tiêu biểu |
|-------------|---------------------|------------------------------|
| **Tương tác (Interaction)** | Lịch sử click, đánh giá, mua hàng | Ma trận tương tác R |
| **Hình ảnh (Visual)** | Amazon, Netflix, TikTok | ResNet, ViT, VGG, Inception |
| **Văn bản (Textual)** | Mô tả, đánh giá, tiêu đề | BERT, Sentence-Transformer, GloVe |
| **Âm thanh (Audio)** | TikTok, Spotify | LSTM, GRU |
| **Video** | TikTok, YouTube | Kết hợp visual + audio |

---

## 2. Feature Extraction cho Phương Thức Hình Ảnh (Visual)

### 2.1. Giai Đoạn 1: Các Mô Hình CNN Truyền Thống (2016–2020)

Các nghiên cứu ban đầu chủ yếu dựa vào các kiến trúc **Convolutional Neural Networks (CNN)**:

#### VGG (Simonyan & Zisserman, 2014) — [ref 77]
- Kiến trúc rất sâu với các lớp conv 3×3 xếp chồng
- Đặc trưng: Trích xuất feature map từ các lớp fully-connected cuối (FC6, FC7, FC8)
- Ưu điểm: Đơn giản, feature chất lượng tốt
- Nhược điểm: Rất nặng về tham số (~138M params)
- **Sử dụng trong MRS:** VBPR (2016), DVBPR (2017), ACF (2017)

#### Inception / GoogLeNet (Szegedy et al., 2015) — [ref 78]
- Kiến trúc dạng nhánh song song (Inception module) với nhiều kích thước conv khác nhau
- Ưu điểm: Nhẹ hơn VGG, hiệu quả tốt hơn về computational cost
- Đặc trưng: Feature từ pool layer trước softmax
- **Sử dụng trong MRS:** Một số công trình giai đoạn 2017–2019

#### Caffe (Jia et al., 2014) — [ref 79]
- Không phải là kiến trúc model mà là **framework** để train & extract features
- Hỗ trợ import VGG, AlexNet pre-trained trên ImageNet
- Phổ biến trong giai đoạn đầu của deep learning-based RS

#### ResNet (He et al., 2016) — [ref 16]
- **Bước đột phá lớn nhất** với cơ chế Skip Connection / Residual Block
- Giải quyết vấn đề **vanishing gradient** khi train network rất sâu (50, 101, 152 layers)
- Feature: Global average pooling sau conv layer cuối → vector 2048-dim
- Ưu điểm: Rất ổn định khi fine-tune, generalizes tốt
- **Sử dụng trong MRS:** MMGCN (2019), LATTICE (2021), BM3 (2023), và phần lớn các công trình sau 2019

### 2.2. Giai Đoạn 2: Vision Transformer (ViT) (2020–nay)

#### ViT — Vision Transformer (Dosovitskiy et al., 2020) — [ref 17]
- **Cách mạng hóa** Computer Vision bằng cơ chế Self-Attention (Transformer)
- Chia ảnh thành patches 16×16 pixels, flatten thành sequence, xử lý như token text
- Ưu điểm vượt trội so với CNN:
  - **Global context:** Mỗi patch "nhìn thấy" toàn bộ ảnh ngay từ layer đầu
  - **Scalability:** Hiệu suất tăng mạnh theo data và model size
  - **Transfer learning:** Pre-train trên dataset khổng lồ (JFT-300M, ImageNet-21K)
- **Output:** CLS token embedding (768-dim hoặc 1024-dim tùy variant)
- **Sử dụng trong MRS:** LGMRec (2024), PromptMM (2024), MENTOR (2025)

### 2.3. Xu Hướng Hiện Tại: "Provided Features"

> *"Các cách tiếp cận gần đây đã chuyển hướng sang việc sử dụng các đặc trưng được cung cấp sẵn trong phương thức hình ảnh để tránh các vấn đề liên quan đến việc xử lý thủ công."*

Do vấn đề **chất lượng dữ liệu** (một số hình ảnh bị hỏng hoặc thiếu trong Amazon datasets), và để đảm bảo tính **tái lập (reproducibility)**, xu hướng hiện tại là:

1. **Dùng pre-extracted features** được cung cấp sẵn bởi các framework như **MMRec**
2. Không tự trích xuất lại từ raw images
3. Cho phép các thí nghiệm được kiểm soát tốt hơn và dễ so sánh hơn

**MMRec** (https://github.com/enoche/MMRec.git) — Framework mã nguồn mở tiêu chuẩn hóa phương pháp trích xuất đặc trưng cho cả visual lẫn textual modality.

---

## 3. Feature Extraction cho Phương Thức Văn Bản (Textual)

### 3.1. Giai Đoạn 1: Phương Pháp Thống Kê Truyền Thống

#### TF-IDF (Salton & Buckley, 1988) — [ref 80]
- **Term Frequency–Inverse Document Frequency**
- Biểu diễn văn bản dạng sparse vector dựa trên tần suất từ
- Ưu điểm: Đơn giản, không cần training, giải thích được
- Nhược điểm: Không nắm bắt ngữ nghĩa, không xử lý từ đồng nghĩa
- **Áp dụng:** Giai đoạn đầu của content-based filtering

#### GloVe (Pennington et al., 2014) — [ref 82]
- **Global Vectors for Word Representation**
- Pre-trained trên co-occurrence statistics của corpus lớn (Common Crawl, Wikipedia)
- Output: Static word embeddings (không context-dependent)
- Kích thước: 50, 100, 200, 300 dimensions
- **Áp dụng:** MMGCN (2019), DualGNN (2021)

### 3.2. Giai Đoạn 2: Neural Language Models

#### Word2Vec (Mikolov et al., 2013) — [ref 83]
- Hai variant: **CBOW** (predict center word from context) và **Skip-gram** (predict context from center word)
- Học được distributed representations: từ có ngữ nghĩa tương đồng → vector gần nhau
- Hiện tượng nổi tiếng: `king - man + woman ≈ queen`
- Nhược điểm: Static embedding, không phân biệt đa nghĩa

#### GRU — Gated Recurrent Unit (Cho et al., 2014) — [ref 21]
- Phiên bản đơn giản hơn của LSTM
- Xử lý text theo sequence → nắm bắt thứ tự từ
- Output: Hidden state cuối cùng làm văn bản representation
- **Áp dụng trong MRS:** Đặc biệt hữu ích cho âm thanh và sequence data

#### PV-DM & PV-DBOW (Le & Mikolov, 2014) — [ref 81]
- **Paragraph Vector (Doc2Vec)** — Học embedding cho toàn bộ đoạn văn bản
- PV-DM: Distributed Memory (context + document predict word)
- PV-DBOW: Distributed Bag of Words (document predict words)
- Khắc phục hạn chế của Word2Vec khi cần embed cả đoạn văn

### 3.3. Giai Đoạn 3: Transformer-based Pre-trained Models (2018–nay)

#### BERT (Devlin et al., 2018) — [ref 18]
- **Bidirectional Encoder Representations from Transformers**
- Pre-training với 2 task: **Masked Language Modeling (MLM)** + **Next Sentence Prediction (NSP)**
- Đặc điểm quan trọng:
  - **Context-aware embeddings:** Cùng một từ → embedding khác nhau tùy ngữ cảnh
  - **Bidirectional attention:** Xem xét cả context trái và phải cùng lúc
  - **Ít cần fine-tune:** Pre-trained kiến thức có thể transfer sang nhiều task
- Output: CLS token [768-dim] hoặc token embedding trung bình
- **Sử dụng trong MRS:** MMSSL (2023), BCCL (2023), LLMRec (2024), PromptMM (2024), DiffMM (2024), VMoSE (2024)

#### Sentence-BERT / Sentence-Transformer (Reimers & Gurevych, 2019) — [ref 19]
- Biến thể của BERT, được fine-tune đặc biệt cho **sentence similarity tasks**
- Sử dụng Siamese/triplet network để học cặp câu
- Ưu điểm so với BERT raw:
  - Tạo ra **semantically meaningful sentence embeddings** tốt hơn
  - Cosine similarity giữa 2 embedding phản ánh tốt hơn semantic similarity
  - Nhanh hơn nhiều khi cần compute similarity cho nhiều cặp câu
- **Sử dụng rộng rãi trong MRS hiện đại:**
  - LATTICE (2021), FREEDOM (2023), BM3 (2023), MGCN (2023), DRAGON (2023), MENTOR (2025), SMORE (2025)
  - → Trở thành **standard de facto** cho text feature extraction

#### Sentence2Vec (Arora et al., 2017) — [ref 84]
- Phương pháp đơn giản: Weighted average của word vectors (TF-IDF weighted)
- Sau đó remove principal component (thường chứa nhiễu grammar)
- Ưu điểm: Cực kỳ nhanh, surprisingly competitive

### 3.4. Cơ Chế Chú Ý (Attention Mechanisms) trong Feature Extraction

Ngoài các mô hình pre-trained, **attention mechanisms** cũng được sử dụng trực tiếp trong quá trình trích xuất đặc trưng để chọn lọc thông tin quan trọng:

- **Self-attention**: Mỗi token/pixel "attend" tới tất cả các token/pixel khác
- **Cross-modal attention**: Cho phép visual features attend tới textual features và ngược lại

---

## 4. Feature Extraction cho Phương Thức Âm Thanh và Video

### LSTM (Hochreiter & Schmidhuber, 1997) — [ref 20]
- **Long Short-Term Memory** — Xử lý chuỗi thời gian
- Có cơ chế Input Gate, Forget Gate, Output Gate để kiểm soát thông tin
- **Áp dụng cho âm thanh:** Xử lý Mel-spectrogram hoặc MFCC features theo frame

### GRU (Cho et al., 2014) — [ref 21]
- Phiên bản nhẹ hơn LSTM với Update Gate và Reset Gate
- Tốc độ training nhanh hơn với hiệu năng tương đương

**Lưu ý:** Bài báo tập trung chủ yếu vào visual và textual. Audio/video modality ít được khảo sát sâu hơn trong bài này.

---

## 5. Phân Tích Xu Hướng Feature Extraction qua Các Năm

Bảng tóm tắt sự tiến hóa của Feature Extraction trong các mô hình MRS tiêu biểu:

| Giai đoạn | Năm | Visual | Textual | Mô hình tiêu biểu |
|-----------|-----|--------|---------|-------------------|
| **Early** | 2016–2018 | VGG, Caffe, CNN pre-trained | TF-IDF | VBPR, DVBPR, GraphCAR |
| **Mid** | 2019–2020 | ResNet-50 pre-trained | GloVe, Word2Vec, GRU | MMGCN, DualGNN, LATTICE |
| **Modern** | 2021–2022 | ResNet / Provided features | BERT | SLMRec, MMSSL, BCCL |
| **Current** | 2023–2025 | Provided features (ResNet/ViT) | **Sentence-Transformer** | FREEDOM, MENTOR, SMORE |

**Nhận xét quan trọng:**
- Visual: Chuyển từ **tự extract (VGG → ResNet)** sang **dùng provided features**
- Textual: Chuyển từ **statistical (TF-IDF)** → **word-level (GloVe, Word2Vec)** → **sentence-level (Sentence-Transformer)**
- **Sentence-Transformer đã trở thành chuẩn mực** cho phần lớn các nghiên cứu SOTA từ 2021 trở đi

---

## 6. Hạn Chế và Vấn Đề Hiện Tại

### 6.1. Sự Phân Tách Giữa Feature Extraction và Learning

> *"Các mô hình hiện tại trong lĩnh vực MRS thường tách biệt việc trích xuất đặc trưng và mã hóa biểu diễn thành hai quá trình riêng biệt... quy trình tách biệt này dẫn đến nhiễu đa phương thức vốn có, tạo ra sự mất kết nối giữa các đặc trưng được trích xuất và việc mã hóa tiếp theo của chúng."*

Vấn đề này gây ra **"multimodal noise"** — đặc trưng được extract theo mục tiêu general (ImageNet classification, NLP text matching) không phải mục tiêu của recommendation.

### 6.2. Mất Cân Bằng Phương Thức

Một số nghiên cứu chỉ tập trung vào một phương thức:
- **Chỉ Visual:** VBPR, DVBPR, VMCF, ACF, AMR
- **Chỉ Textual:** ADDVAE
→ Bỏ lỡ thông tin bổ trợ từ phương thức còn lại

### 6.3. Chất Lượng Dữ Liệu Đầu Vào

- Hình ảnh bị hỏng/thiếu trong Amazon datasets
- Văn bản có thể thiếu, ngắn, hoặc nhiễu
- Không đồng đều giữa các items về chất lượng modality

---

## 7. Các Bài Báo Liên Quan và Cải Tiến Chi Tiết

### 7.1. VBPR (2016) — Nền Tảng của Visual Feature trong RS

**Tên đầy đủ:** "VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback"  
**Tác giả:** Ruining He, Julian McAuley  
**Venue:** AAAI 2016  
**Vấn đề giải quyết:** Khai thác visual signals trong recommendation với implicit feedback  
**Đóng góp về Feature Extraction:**
- Lần đầu tiên sử dụng **deep CNN features** (VGG/Caffe pre-trained trên ImageNet) trong collaborative filtering
- Feature: FC7 activation của CNN → 4096-dim, được project xuống 20-dim
- Kết hợp visual embedding với latent factors của MF
**Hạn chế:** Không sử dụng textual features, CNN features static (không fine-tune)

---

### 7.2. DVBPR (2017) — End-to-end Visual Feature Learning

**Tên đầy đủ:** "Visually-aware Fashion Recommendation and Design with Generative Image Models"  
**Tác giả:** Wang-Cheng Kang, Chen Fang, Zhaowen Wang, Julian McAuley  
**Venue:** ICDM 2017  
**Đóng góp về Feature Extraction:**
- **Cải tiến lớn:** Thay vì dùng fixed CNN features, **fine-tune CNN end-to-end** cùng với recommendation objective
- Sử dụng CNN deeper (ResNet-style) để học visual patterns phù hợp với fashion recommendation
- Thêm Generative Image Model để synthesize new fashion items
**Bài học:** Fine-tuning feature extractor theo task-specific objective tốt hơn fixed pre-trained features

---

### 7.3. BERT-based Feature Extraction trong MRS

**Ví dụ: MMSSL (Wei et al., 2023)**  
**Tên đầy đủ:** "Multi-modal Self-supervised Learning for Recommendation"  
**Venue:** WWW 2023  
**Đóng góp về Feature Extraction:**
- Sử dụng **BERT** để encode text descriptions của items
- Kết hợp với visual features (ResNet)
- **Cải tiến quan trọng:** Áp dụng SSL (self-supervised learning) để tăng chất lượng feature representation
- Adversarial perturbations trên feature space để tăng cường robustness

---

### 7.4. Sentence-Transformer as Standard — LATTICE, FREEDOM, BM3

**LATTICE (2021)** — Zhang et al., ACM MM 2021  
- *"Mining Latent Structures for Multimedia Recommendation"*
- Lần đầu sử dụng **Sentence-Transformer** làm chuẩn cho text feature extraction trong MRS
- Build item-item graphs dựa trên semantic similarity của Sentence-Transformer embeddings
- Feature chất lượng cao → graph structure chính xác hơn → recommendation tốt hơn

**FREEDOM (2023)** — Zhou & Shen, ACM MM 2023  
- *"A Tale of Two Graphs: Freezing and Denoising Graph Structures for Multimodal Recommendation"*
- Tiếp tục dùng Sentence-Transformer cho text
- Cải tiến: **Freeze item-item graph** (built từ pre-extracted features) và chỉ fine-tune user-item graph
- Giải quyết vấn đề noise trong graph structure

**BM3 (2023)** — Zhou et al., WWW 2023  
- *"Bootstrap Latent Representations for Multi-modal Recommendation"*
- Dùng Sentence-Transformer + ResNet (provided features)
- Cải tiến: Thay sampling âm bằng **dropout strategy** để SSL tốt hơn
- Đơn giản hóa pipeline mà vẫn đạt SOTA

---

### 7.5. LLMRec (2024) — LLMs Augmenting Feature Extraction

**Tên đầy đủ:** "LLMRec: Large Language Models with Graph Augmentation for Recommendation"  
**Tác giả:** Wei et al.  
**Venue:** WSDM 2024  
**Đóng góp cách mạng về Feature Extraction:**
- **Sử dụng LLMs (GPT-4 family)** để augment user/item profiles
- LLM generate rich textual descriptions → sau đó encode bằng text encoder
- 3 chiến lược augmentation:
  1. **User profile enhancement:** LLM suy luận sở thích tiềm ẩn của user
  2. **Item attribute enrichment:** LLM bổ sung attributes cho items nghèo thông tin  
  3. **Synthetic interaction pairs:** LLM tạo pseudo user-item pairs
- **Giải quyết vấn đề:** Sparse textual features và cold-start
- **Hạn chế mới:** Chi phí LLM inference cao, hallucination

---

### 7.6. PromptMM (2024) — Prompt-based Feature Alignment

**Tên đầy đủ:** "PromptMM: Multi-modal Knowledge Distillation for Recommendation with Prompt-Tuning"  
**Venue:** WWW 2024  
**Đóng góp về Feature Extraction:**
- Nhận ra **semantic gap** giữa general-purpose text/visual encoders và RS-specific signals
- Sử dụng **learnable prompts** để adapt pre-trained multimodal encoders (BERT, ViT) sang RS context
- Knowledge distillation từ multimodal teacher → collaborative student
- **Giải quyết:** Sự mất kết nối giữa feature extraction và downstream recommendation

---

### 7.7. MENTOR (2025) — Multi-level Feature Alignment

**Tên đầy đủ:** "MENTOR: Multi-level Self-supervised Learning for Multimodal Recommendation"  
**Tác giả:** Xu et al. (cùng nhóm với bài survey)  
**Venue:** AAAI 2025  
**Đóng góp về Feature Extraction & Alignment:**
- Multi-level alignment giữa các modalities + tương tác người dùng
- Sử dụng cross-modal self-supervised learning để căn chỉnh visual và textual features
- Alignment ở nhiều granularity: word-level, sentence-level, item-level
- **Cải tiến:** Không chỉ extract features mà còn học cách **align chúng** tốt hơn

---

### 7.8. SMORE (2025) — Spectrum-based Feature Representation

**Tên đầy đủ:** "Spectrum-based Modality Representation Fusion Graph Convolutional Network for Multimodal Recommendation"  
**Venue:** WSDM 2025  
**Đóng góp về Feature Representation:**
- Hướng tiếp cận hoàn toàn mới: **chiếu features vào miền tần số (frequency domain)**
- Tận dụng không gian phổ để phân tích và fusion các modality features
- Low-frequency components → global semantic patterns
- High-frequency components → local detail patterns
- **Giải quyết:** Vấn đề noise trong feature space và alignment giữa các modalities

---

## 8. Tổng Hợp: Các Vấn Đề và Giải Pháp Được Nghiên Cứu

| Vấn đề | Mô tả | Các Công Trình Giải Quyết |
|--------|-------|--------------------------|
| **Feature quality gap** | Pre-trained features cho general tasks ≠ RS tasks | DVBPR, PromptMM, MENTOR |
| **Sparse/missing visual** | Nhiều ảnh bị thiếu/hỏng trong datasets | Xu hướng "provided features", MMRec |
| **Unimodal bias** | Chỉ dùng 1 modality, bỏ qua modality còn lại | MMGCN, DualGNN, BM3 |
| **Static features** | Features không cập nhật theo context RS | DVBPR (fine-tune), LLMRec (augment) |
| **Semantic gap** | Khoảng cách giữa encoder output và RS semantics | PromptMM (prompt-tuning), MENTOR |
| **Modality noise** | Features chứa noise không liên quan đến RS | FREEDOM (freeze+denoise), MGCN |
| **Cold start** | Items mới thiếu interaction data | LLMRec (LLM augmentation) |
| **Modality imbalance** | Các modalities không đồng đều về thông tin | CKD (modality-balanced learning) |

---

## 9. Hướng Nghiên Cứu Tương Lai trong Feature Extraction

### 9.1. Unified Extraction-Encoding Model
> *"Có một nhu cầu cấp thiết đối với một mô hình hợp nhất tích hợp các quá trình [extraction và encoding] này một cách gắn kết hơn."*

Thay vì pipeline 2 bước tách biệt, nghiên cứu cần:
- End-to-end training từ raw data → recommendation
- Feature extractor được tối ưu hóa **đồng thời** với RS objective

### 9.2. Richer Modality Diversity
- Âm thanh, mùi hương (olfactory), cử động (kinesthetic)
- Metadata, social signals, sequential patterns
- Đặc biệt cần thêm models xử lý **audio-visual** kết hợp

### 9.3. LLM-enhanced Feature Extraction
- Sử dụng LLMs (GPT-4, Gemini) không chỉ như data augmentor mà như **primary feature extractor**
- Multi-modal LLMs (LLaVA, GPT-4V) để xử lý cả image + text cùng lúc

### 9.4. Domain-Adaptive Pre-training
- Pre-train visual/text encoders **trên RS-specific data** thay vì dùng ImageNet/Wikipedia models
- Có thể tận dụng phong phú user-item interactions để self-supervise

---

## 10. Công Cụ và Framework Thực Tiễn

### MMRec
- **Link:** https://github.com/enoche/MMRec.git
- **Mục đích:** Chuẩn hóa feature extraction cho cả visual và textual trong MRS
- **Đặc điểm:** Cung cấp sẵn pre-extracted features → tái lập được kết quả

### Sentence-Transformers Library
- **Link:** https://www.sbert.net/
- Cung cấp nhiều pre-trained models (all-MiniLM-L6-v2, all-mpnet-base-v2...)
- Dễ dàng extract sentence embeddings chất lượng cao

### torchvision / timm (PyTorch Image Models)
- Pre-trained ResNet, ViT và nhiều architectures khác
- Thuận tiện cho visual feature extraction

---

## 11. Tóm Tắt

Feature Extraction trong MRS là một lĩnh vực đã và đang phát triển mạnh:

1. **Visual:** Từ CNN tĩnh (VGG, ResNet) → ViT → Provided features
2. **Textual:** Từ TF-IDF → Word embeddings → BERT → **Sentence-Transformer** (chuẩn hiện tại)
3. **Thách thức chính:** Khoảng cách ngữ nghĩa giữa general pre-trained features và RS-specific signals
4. **Hướng tương lai:** Unified models, LLM-enhanced extraction, domain-adaptive pre-training

Chất lượng của feature extraction ảnh hưởng trực tiếp đến tất cả các bước sau trong pipeline MRS và là nền tảng quyết định hiệu suất của toàn bộ hệ thống.

---

*Tài liệu được tổng hợp từ: Xu et al., "A Survey on Multimodal Recommender Systems: Recent Advances and Future Directions", IEEE Transaction on Multimedia, 2025.*
