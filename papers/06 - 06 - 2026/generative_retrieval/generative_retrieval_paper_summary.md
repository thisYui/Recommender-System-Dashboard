# Summary: Generative Retrieval / Generative Recommendation trong Recommender Systems

## 1. Bức tranh chung

Generative retrieval trong recommender systems là một hướng mới chuyển bài toán recommendation từ **matching/scoring** sang **generation**. Trong recommender truyền thống, hệ thống thường dùng retrieve-and-rank: đầu tiên retrieval model lấy ra một tập candidate item bằng ANN, two-tower, matrix factorization hoặc dense retrieval; sau đó ranker model tính score và sắp xếp lại. Trong generative retrieval, mô hình không chỉ tính score cho từng item nữa, mà **trực tiếp sinh ra định danh của item cần recommend**.

Cách viết đơn giản là:

```text
Traditional retrieval:
    user/query embedding -> ANN search -> candidate items

Generative retrieval:
    user history/query -> generative model -> generated item identifier
```

Điểm mấu chốt của generative retrieval không nằm ở việc dùng Transformer hay LLM một cách đơn thuần, mà nằm ở câu hỏi: **item được biểu diễn bằng token như thế nào để model có thể sinh ra?** Nếu item vẫn là atomic ID, vocabulary sẽ cực lớn, không có cấu trúc ngữ nghĩa, khó generalize cho cold-start/long-tail. Vì vậy, phần lớn paper trong nhóm này xoay quanh **Semantic ID**, **item tokenization**, **hierarchical identifiers**, **collaborative tokenization**, hoặc **continuous/discrete token generation**.

Có thể xem generative retrieval như một sự thay đổi interface:

```text
Item ID / item embedding
        ↓
Semantic token sequence / hierarchical code / learned identifier
        ↓
Autoregressive or diffusion-based generation
```

Điểm mới của hướng này là recommender model không chỉ học user preference, mà còn học một **ngôn ngữ định danh item**. Nếu item tokenizer tốt, các item giống nhau hoặc liên quan nhau có thể chia sẻ token, giúp model generalize tốt hơn. Nếu tokenizer kém, model sẽ khó decode item đúng, dù backbone mạnh.

---

## 2. Paper nền: TIGER / Recommender Systems with Generative Retrieval

Paper **Recommender Systems with Generative Retrieval** là paper nền quan trọng nhất của nhóm này. Paper đề xuất TIGER, trong đó mỗi item được biểu diễn bằng một tuple Semantic ID. Semantic ID được tạo từ content embedding của item, sau đó quantize thành chuỗi codeword. User history cũng được viết dưới dạng chuỗi Semantic IDs, và Transformer seq2seq được train để sinh Semantic ID của next item.

Điểm quan trọng của TIGER là nó thay thế item ID nguyên tử bằng một chuỗi token có cấu trúc. Ví dụ, hai item cùng loại có thể chia sẻ prefix hoặc một phần Semantic ID. Nhờ vậy model có thể học được quan hệ giữa các item tương tự, thay vì xem mỗi item là một nhãn độc lập. Đây là lợi thế lớn cho cold-start hoặc item ít interaction.

Có thể xem TIGER là bước mở đầu cho toàn bộ nhánh **Semantic ID-based generative recommendation**:

```text
item content -> semantic embedding -> quantization -> Semantic ID
user history of Semantic IDs -> Transformer -> next Semantic ID
```

Đóng góp chính của TIGER là chứng minh recommendation có thể được biểu diễn như một bài toán **sequence-to-sequence generation**. Retrieval không còn cần ANN index theo kiểu truyền thống; thay vào đó, Transformer memory có thể hoạt động như một index sinh item.

Tuy nhiên, TIGER cũng mở ra nhiều vấn đề về sau: Semantic ID nên được tạo thế nào, code có cần chứa collaborative signal không, autoregressive decoding có quá chậm không, beam search có đủ hiệu quả không, và làm sao triển khai ở quy mô công nghiệp.

---

## 3. Vấn đề trung tâm: item tokenization quan trọng hơn backbone

Sau TIGER, rất nhiều paper không còn tập trung chủ yếu vào backbone Transformer, mà tập trung vào **tokenizer**. Lý do là trong generative recommendation, tokenizer quyết định vocabulary, sequence length, semantic sharing, decoding difficulty, cold-start capability và serving cost.

Một item tokenizer tốt cần cân bằng nhiều mục tiêu:

```text
1. Uniqueness:
   mỗi item cần có identifier riêng để tránh conflict.

2. Semantic sharing:
   item tương tự nên chia sẻ token để generalize tốt.

3. Collaborative alignment:
   item thường được tương tác/mua/xem cùng nhau nên có representation gần hoặc dễ dự đoán.

4. Decodability:
   chuỗi token phải dễ sinh autoregressively hoặc dễ denoise.

5. Efficiency:
   token length và beam search space phải đủ nhỏ để dùng thực tế.

6. Cold-start:
   item mới vẫn có thể được mã hóa từ content hoặc metadata.

7. Long-tail:
   item hiếm cần đủ semantic detail để phân biệt.
```

Đây là lý do các paper như **UNGER**, **CoCoRec/CCFRec**, **COSETTE/MARIUS**, **SimCIT**, **Purely Semantic Indexing**, **DIGER**, **VarLenRec**, **ReSID** hoặc **FACE** đều quay quanh một câu hỏi: **làm sao biến item thành token tốt hơn?**

---

## 4. Semantic-only tokenization chưa đủ: cần collaborative signal

Một hướng rất rõ trong các paper gần đây là phê phán việc dùng semantic content embedding thuần túy. Nếu Semantic ID được tạo từ text/image embedding của foundation model, nó có thể phản ánh similarity về nội dung, nhưng chưa chắc phản ánh similarity về hành vi người dùng.

Ví dụ:

```text
snacks và balloons
```

Hai item này không nhất thiết giống nhau về text hoặc image, nhưng có thể thường được mua cùng nhau trong bối cảnh party. Nếu tokenizer chỉ dựa vào semantic similarity, nó có thể bỏ qua quan hệ collaborative này.

### 4.1. UNGER

Paper **UNGER: Generative Recommendation with A Unified Code via Semantic and Collaborative Integration** giải quyết vấn đề này bằng cách tích hợp semantic knowledge và collaborative knowledge vào một unified code. Thay vì dùng hai code riêng cho semantic và collaborative signal, UNGER cố gắng tạo một code thống nhất. Ý tưởng này quan trọng vì dual-code có thể làm tăng chi phí lưu trữ, inference latency và không tận dụng được sự tương tác giữa semantic và collaborative information.

Ý nghĩa của UNGER:

```text
Semantic code không nên chỉ là content code.
Nó nên là unified code phản ánh cả item content và user behavior.
```

### 4.2. CoCoRec / CCFRec

Paper **Bridging Textual-Collaborative Gap through Semantic Codes for Sequential Recommendation** cũng đặt trọng tâm vào khoảng cách giữa textual semantics và collaborative semantics. Paper tạo semantic codes từ multi-view text embeddings qua vector quantization, rồi dùng code-guided semantic fusion để kết hợp text representation với collaborative sequential signal. Ngoài ra paper dùng masked code modeling và masked sequence alignment để làm representation học tốt hơn.

Ý nghĩa chính:

```text
Semantic codes có thể đóng vai trò cầu nối giữa text metadata và collaborative behavior.
```

### 4.3. COSETTE / MARIUS

Paper **Closing the Performance Gap in Generative Recommenders with Collaborative Tokenization and Efficient Modeling** chỉ ra hai nguyên nhân làm generative recommenders thua ID-based baselines: item tokenization thiếu collaborative signal và encoder-decoder architecture kém hiệu quả. Paper đề xuất COSETTE, một contrastive tokenization method tích hợp collaborative information vào item representations, đồng thời đề xuất MARIUS, một model nhẹ hơn tách timeline modeling khỏi item decoding.

Ý nghĩa:

```text
Generative recommendation muốn cạnh tranh với SASRec/ID-based models
thì tokenizer phải học collaborative signal, không chỉ content reconstruction.
```

### 4.4. FACE

Paper **FACE: A General Framework for Mapping Collaborative Filtering Embeddings into LLM Tokens** đi theo chiều ngược lại: thay vì đưa semantic embedding vào recommender, nó ánh xạ CF embeddings vào LLM tokens. CF embedding vốn mạnh về collaborative signal nhưng không có nghĩa ngôn ngữ rõ ràng. FACE dùng disentangled projection và quantized autoencoder để chuyển CF embeddings thành tokens/descriptors có thể align với textual signal.

Ý nghĩa:

```text
Collaborative signal cũng có thể được token hóa để LLM hiểu được.
```

Đây là một hướng rất đáng chú ý vì nó không xem LLM/foundation model là trung tâm tuyệt đối. Nó đưa collaborative filtering space vào language/token space.

---

## 5. ID token và semantic token nên bổ sung cho nhau

Một số paper không chọn hoàn toàn semantic token hoặc hoàn toàn ID token, mà tìm cách kết hợp cả hai.

Paper **Unified Semantic and ID Representation Learning for Deep Recommenders** lập luận rằng ID tokens và semantic tokens có vai trò bổ sung. ID token mạnh ở việc phân biệt item cụ thể, giúp memorization tốt. Semantic token mạnh ở việc biểu diễn shared transferable characteristics, giúp generalization và cold-start tốt hơn. Paper đề xuất framework học chung ID và semantic representation, đồng thời phân tích vai trò của cosine similarity và Euclidean distance trong embedding search.

Ý nghĩa của nhóm này là:

```text
Không nên cực đoan bỏ item ID hoàn toàn.
Vấn đề là thiết kế interface để ID-specific signal và semantic-shared signal cùng tồn tại.
```

Trong thực tế, đây có thể là hướng an toàn nhất. Với head items, ID signal vẫn rất mạnh. Với tail/cold-start items, semantic signal giúp generalize. Hybrid tokenization có thể đạt trade-off tốt hơn content-only hoặc ID-only.

---

## 6. Tokenizer không nên chỉ reconstruct item embedding

Nhiều Semantic ID pipeline dùng RQ-VAE hoặc vector quantization để reconstruct content embedding. Tuy nhiên, reconstruction objective chưa chắc tối ưu cho generative recommendation. Một tokenizer có reconstruction tốt có thể vẫn tạo ra chuỗi token khó predict, hoặc không phân biệt tốt các item trong retrieval.

### 6.1. SimCIT

Paper **A Simple Contrastive Framework Of Item Tokenization For Generative Recommendation** chỉ ra rằng reconstruction-based quantization có thể xung đột với retrieval objective. Retrieval quan trọng ở khả năng phân biệt item đúng khỏi item sai, không chỉ reconstruct embedding chính xác. SimCIT dùng contrastive learning cho item tokenization, kết hợp multi-modal knowledge alignment và semantic tokenization trong cùng framework.

Ý nghĩa:

```text
Item tokenization nên tối ưu discriminative/retrieval objective,
không chỉ tối ưu reconstruction loss.
```

### 6.2. Differentiable Semantic ID / DIGER

Paper **Differentiable Semantic ID for Generative Recommendation** đi xa hơn: thay vì train tokenizer độc lập rồi giữ Semantic ID cố định, paper làm Semantic ID differentiable để recommendation loss có thể cập nhật tokenizer. Vấn đề là nếu làm trực tiếp, codebook có thể collapse. DIGER dùng Gumbel noise để khuyến khích exploration ở giai đoạn đầu, sau đó dùng uncertainty decay để chuyển dần sang exploitation.

Ý nghĩa:

```text
Semantic ID nên được học cùng recommendation objective,
không nên là static indexing chỉ tối ưu content reconstruction.
```

Đây là hướng rất quan trọng về mặt research vì nó làm cho item indexing trở thành một phần end-to-end của recommender training.

---

## 7. Vấn đề conflict và uniqueness trong Semantic ID

Một vấn đề thực tế của Semantic ID là **ID conflict**: các item quá giống nhau có thể nhận cùng code. Cách đơn giản là thêm một token ngẫu nhiên hoặc non-semantic suffix để phân biệt. Nhưng điều này làm mất tính semantic, tăng search space và có thể khiến model khó predict.

Paper **Purely Semantic Indexing for LLM-based Generative Recommendation and Retrieval** giải quyết vấn đề này bằng cách tạo unique semantic-preserving IDs mà không cần append non-semantic token. Paper đề xuất các thuật toán như exhaustive candidate matching và recursive residual searching để đảm bảo uniqueness trong khi vẫn giữ semantic structure.

Ý nghĩa:

```text
Semantic ID phải vừa unique vừa semantic.
Nếu dùng non-semantic suffix để tránh conflict, model có thể mất lợi thế generalization.
```

Đây là vấn đề nhỏ nhưng rất quan trọng khi scale lên catalog lớn. Trong generative retrieval, nếu nhiều item trùng ID hoặc ID phải bổ sung suffix ngẫu nhiên, decoding sẽ trở nên khó hơn và cold-start có thể kém hơn.

---

## 8. Fixed-length Semantic ID chưa chắc tối ưu

Hầu hết Semantic ID methods dùng fixed-length code: mỗi item có cùng số lượng token. Nhưng catalog thực tế có power-law distribution. Head items có rất nhiều interaction, model có thể nhận diện chúng bằng ít token hơn. Tail items ít interaction hơn, cần nhiều semantic detail hơn để phân biệt.

Paper **Learning Variable-Length Tokenization for Generative Recommendation** nêu ra hiện tượng **Popularity-Length Paradox**: popular items thường đạt tốt với ID ngắn, còn tail items cần code dài hơn để chứa đủ semantic information. Paper đề xuất VarLenRec, học tokenization có độ dài thay đổi. Ý tưởng cốt lõi là information budget nên phân bổ theo popularity: item càng phổ biến thì cần ít semantic code hơn, item càng tail thì cần nhiều code hơn.

Ý nghĩa:

```text
Không phải mọi item cần cùng độ dài Semantic ID.
Token length nên là một phần học được của item tokenizer.
```

Đây là hướng rất đáng khai thác vì nó kết nối tokenization với long-tail recommendation và efficiency. Nó cũng gần với natural language: khái niệm phổ biến thường có từ ngắn, khái niệm hiếm thường cần mô tả dài hơn.

---

## 9. Hierarchical identifiers và tree-based generation

Một nhánh khác dùng hierarchical identifiers hoặc tree-structured item IDs. Ý tưởng là tổ chức item space thành cây, mỗi item là một path trong cây. Khi generate, model sinh từng level từ coarse đến fine. Điều này giảm search space và tạo cấu trúc phân cấp.

Paper **Efficient Optimization of Hierarchical Identifiers for Generative Recommendation** xoay quanh SEATER, một generative retrieval model dùng balanced tree-structured item identifiers và contrastive training. Paper nhấn mạnh tree construction có thể trở thành bottleneck khi số item lớn, nên đề xuất các thuật toán xây cây nhanh hơn như greedy hoặc hybrid construction, giảm mạnh thời gian build tree mà giữ chất lượng.

Ý nghĩa:

```text
Identifier design không chỉ ảnh hưởng accuracy,
mà còn ảnh hưởng training/inference efficiency và khả năng scale.
```

Hierarchical ID phù hợp với generative retrieval vì model có thể decode từng tầng, nhưng nếu cây không cân bằng hoặc xây quá chậm, hệ thống khó triển khai thực tế.

---

## 10. Continuous token và diffusion: không nhất thiết phải autoregressive discrete decoding

Phần lớn generative retrieval dùng discrete Semantic IDs và autoregressive decoding. Nhưng một số paper đặt câu hỏi: liệu phải dùng discrete tokens không? Có thể dùng continuous tokens hoặc diffusion generation không?

### 10.1. Continuous-token diffusion

Paper **Diffusion Generative Recommendation with Continuous Tokens** đề xuất dùng continuous tokens thay vì discrete quantized tokens. Lý do là quantization có thể làm mất thông tin và bị giới hạn bởi vocabulary. Framework như DeftRec/ContRec dùng denoising diffusion để xử lý user preference trong continuous domain, rồi dùng score-based retrieval để lấy recommendation.

Ý nghĩa:

```text
Generative recommendation không nhất thiết phải bị giới hạn bởi discrete tokenization.
Continuous token + diffusion có thể là hướng thay thế cho lossy quantization.
```

### 10.2. Masked Diffusion

Paper **Masked Diffusion for Generative Recommendation** phê phán autoregressive SID generation vì inference tốn kém, decode tuần tự, và có thể học quá nhiều short-context token relationships. Paper dùng masked diffusion trên SID sequences: model học phân phối sequence bằng masking noise và có thể predict nhiều token song song. Điều này giúp parallel decoding và có thể tốt hơn trong data-constrained settings.

Ý nghĩa:

```text
Generative recommendation có thể chuyển từ autoregressive decoding sang diffusion/denoising decoding.
```

Điểm này quan trọng vì một trong các hạn chế lớn của generative retrieval là latency do beam search và token-by-token decoding. Diffusion/masked generation có thể giúp sinh song song.

---

## 11. Industrial generative retrieval: từ benchmark sang hệ thống thật

Một nhóm paper rất quan trọng trong danh sách là các paper công nghiệp: **OneRec**, **PinRec**, **MTGR**, **TBGRecall**. Chúng cho thấy generative retrieval không còn chỉ là academic benchmark, mà đang được thử nghiệm/deploy trong hệ thống lớn.

### 11.1. OneRec

Paper **OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment** đề xuất thay thế pipeline retrieve-and-rank bằng một unified generative model. OneRec dùng encoder-decoder để encode user history và generate video recommendations; dùng sparse MoE để tăng model capacity; dùng session-wise generation thay vì next-item point prediction; và dùng iterative preference alignment với DPO-like objective để cải thiện quality.

Ý nghĩa:

```text
Generative recommender có thể không chỉ là retrieval stage,
mà có thể tiến tới unified retrieve-and-rank.
```

Điểm mạnh của OneRec là nó đẩy generative recommendation vào hệ thống video recommendation thực tế, nơi objective không chỉ là recall mà còn là watch-time/user engagement.

### 11.2. PinRec

Paper **PinRec: Outcome-Conditioned, Multi-Token Generative Retrieval for Industry-Scale Recommendation Systems** tập trung vào Pinterest. PinRec nhấn mạnh hai vấn đề thực tế: hệ thống recommendation có nhiều outcome metrics, và output cần đa dạng/hiệu quả. PinRec dùng outcome-conditioned generation để điều khiển trade-off giữa các metric như saves/clicks, và multi-token generation để tăng diversity và tối ưu generation.

Ý nghĩa:

```text
Generative retrieval cần controllable generation,
không chỉ generate item có xác suất cao nhất.
```

Trong hệ thống thật, mục tiêu có thể là click, save, watch-time, purchase, diversity, exploration hoặc business constraint. Outcome-conditioned generation là hướng quan trọng để generative retrieval không bị đóng khung trong next-item accuracy.

### 11.3. MTGR

Paper **MTGR: Industrial-Scale Generative Recommendation Framework in Meituan** chỉ ra một vấn đề rất thực tế: nhiều generative recommendation approach phải bỏ các cross features và feature engineering mạnh của DLRM truyền thống, dẫn đến performance giảm; scaling up không nhất thiết bù được. MTGR dựa trên HSTU architecture, giữ lại DLRM features/cross features, dùng user-level compression để tăng tốc training/inference, Group-Layer Normalization cho nhiều semantic spaces và dynamic masking để tránh leakage.

Ý nghĩa:

```text
Muốn deploy generative recommendation, không thể bỏ qua hệ feature engineering/cross feature của industrial recommender.
```

Đây là một bài học rất quan trọng: generative model không tự động thắng hệ thống truyền thống nếu mất đi các feature đã được tối ưu nhiều năm.

### 11.4. TBGRecall

Paper **TBGRecall: A Generative Retrieval Model for E-commerce Recommendation Scenarios** tập trung vào e-commerce retrieval, đặc biệt Taobao. Paper phê phán autoregressive generation vì khó generate nhiều item không có positional constraints trong cùng một session. TBGRecall đưa vào Next Session Prediction: dữ liệu được chia thành multi-session sequences, mỗi sequence gồm session token và set item tokens. Paper cũng nhấn mạnh limited historical pre-training và stochastic partial incremental training, cho thấy data recency quan trọng hơn chỉ tăng volume dữ liệu.

Ý nghĩa:

```text
Trong e-commerce retrieval, objective không chỉ là next item,
mà là next session / set of items phù hợp với request hiện tại.
```

Đây là điểm khác với nhiều benchmark sequential recommendation. E-commerce thực tế thường cần retrieve một tập candidate cho một request/session, không phải sinh đúng một item tiếp theo.

---

## 12. Token efficiency và multimodal item representation

Generative recommendation dùng LLM gặp vấn đề token cost. Nếu item description dài, user history dài, và mỗi item cần nhiều token, chi phí inference/training sẽ rất lớn.

Paper **Token-Efficient Item Representation via Images for LLM Recommender Systems** đặt vấn đề này theo hướng khác: thay vì biểu diễn item bằng text description dài, dùng image như một representation hiệu quả hơn. Paper quan sát rằng image và description có overlap thông tin lớn, nên image có thể thay cho mô tả text dài, giảm token usage nhưng vẫn giữ semantic information. Paper cũng cho thấy image-based representation có thể robust hơn với noisy descriptions.

Ý nghĩa:

```text
Item representation trong LLM recommender không nhất thiết phải là text.
Image có thể là token-efficient alternative cho long item descriptions.
```

Paper này liên hệ mạnh với hướng bạn đã hỏi trước đó: **không dùng input gốc trực tiếp**, mà chuyển sang modality/encoder khác trước khi đưa vào recommender.

---

## 13. Unified tokenization across tasks/domains

Một số paper mở rộng generative retrieval từ một domain/task sang nhiều domain/task.

### 13.1. Joint search and recommendation / unified semantic IDs

Các paper như **Semantic IDs for Joint Generative Search and Recommendation** và các hướng joint S&R cho thấy một vấn đề: Semantic ID tốt cho search chưa chắc tốt cho recommendation. Search cần query-item relevance; recommendation cần user preference và collaborative sequence. Vì vậy item tokenization phải cân bằng giữa task-specific và shared representation.

### 13.2. UniTok

Paper **Tokenize Once, Recommend Anywhere: Unified Item Tokenization for Multi-domain LLM-based Recommendation** đề xuất UniTok, một unified item tokenization framework cho nhiều domain. Thay vì train tokenizer riêng cho từng domain, UniTok dùng MoE với shared encoder, domain-specific experts và shared expert để giữ cả common knowledge lẫn domain-specific semantics. Paper cũng dùng mutual information calibration để tránh domain semantic imbalance.

Ý nghĩa:

```text
Nếu Semantic ID là vocabulary của recommender,
thì multi-domain recommendation cần một tokenizer có thể dùng chung nhưng vẫn giữ domain-specific information.
```

Đây là hướng quan trọng nếu muốn xây recommender foundation model hoặc LLM-based recommender đa miền.

---

## 14. Survey và handbook: dấu hiệu lĩnh vực đã trưởng thành nhanh

Hai paper survey/handbook trong danh sách giúp đặt toàn bộ nhóm vào bức tranh lớn.

Paper **From Principles to Applications: A Comprehensive Survey of Discrete Tokenizers in Generation, Comprehension, Recommendation, and Information Retrieval** cho thấy discrete tokenizer là interface trung tâm giữa raw data và generative models. Với recommendation, Semantic ID chính là một dạng discrete tokenizer cho item space.

Paper **Generative Recommendation with Semantic IDs: A Practitioner’s Handbook** giới thiệu GRID, một framework modular/open-source để benchmark và ablation các component của GR với SID. Điều này cho thấy lĩnh vực đang chuyển từ “mỗi paper một setup riêng” sang nhu cầu chuẩn hóa pipeline, benchmark, tokenizer, model backbone và inference strategy.

Paper **A Survey on Generative Recommendation: Data, Model, and Tasks** đặt generative recommendation trong framework rộng hơn: data, model, task. Đây là dấu hiệu rằng generative recommendation không chỉ là một trick retrieval, mà đang trở thành một paradigm mới có nhiều nhánh: LLM-based, diffusion-based, retrieval generation, conversational/explainable/personalized content generation.

---

## 15. Các nhóm vấn đề chính trong generative retrieval

Từ toàn bộ paper, có thể chia thành 7 nhóm vấn đề lớn.

### 15.1. Identifier design

Câu hỏi:

```text
Item nên được biểu diễn bằng ID, semantic code, hierarchical path, continuous token hay variable-length token?
```

Paper liên quan:

```text
TIGER
UNGER
Purely Semantic Indexing
Efficient Optimization of Hierarchical Identifiers
VarLenRec
DIGER
Unified Semantic and ID Representation Learning
```

### 15.2. Semantic-collaborative alignment

Câu hỏi:

```text
Làm sao để Semantic ID vừa chứa content semantics vừa chứa collaborative behavior?
```

Paper liên quan:

```text
UNGER
CoCoRec / CCFRec
COSETTE
FACE
SimCIT
Unified Semantic and ID Representation Learning
```

### 15.3. Tokenization objective

Câu hỏi:

```text
Tokenizer nên tối ưu reconstruction, contrastive discrimination, collaborative prediction, recommendation loss hay sequential predictability?
```

Paper liên quan:

```text
SimCIT
DIGER
COSETTE
ReSID-like directions
Practitioner’s Handbook / GRID
```

### 15.4. Decoding mechanism

Câu hỏi:

```text
Có nên dùng autoregressive decoding hay diffusion/masked parallel generation?
```

Paper liên quan:

```text
TIGER
Masked Diffusion
Continuous-token Diffusion
OneRec
PinRec
TBGRecall
```

### 15.5. Industrial scalability

Câu hỏi:

```text
Generative retrieval có thể đạt latency, throughput, feature compatibility và online gain trong hệ thống thật không?
```

Paper liên quan:

```text
OneRec
PinRec
MTGR
TBGRecall
```

### 15.6. Multi-objective controllability

Câu hỏi:

```text
Generative model có thể điều khiển output theo click, save, watch-time, diversity, exploration hoặc business metric không?
```

Paper liên quan:

```text
PinRec
OneRec
MTGR
```

### 15.7. Multi-domain / multi-task tokenization

Câu hỏi:

```text
Một tokenizer có thể dùng chung cho nhiều domain hoặc nhiều task như search + recommendation không?
```

Paper liên quan:

```text
UniTok
Semantic IDs for Joint Search and Recommendation
Generative Recommendation surveys
```

---

## 16. Điểm mới có thể khai thác nếu muốn làm đề tài

Nếu bạn chưa có idea chính, tôi đề xuất một số hướng có khả năng viết thành proposal.

### Hướng 1: Collaborative-aware Semantic ID

Ý tưởng:

```text
Tạo Semantic ID không chỉ từ content embedding,
mà từ cả content semantics và collaborative interaction graph.
```

Motivation: semantic-only tokenization thiếu behavior signal. Hướng này nối UNGER, CoCoRec, COSETTE, FACE và SimCIT.

Câu hỏi nghiên cứu:

```text
Collaborative signal nên được đưa vào tokenizer ở giai đoạn nào:
trước quantization, trong quantization, hay sau quantization?
```

### Hướng 2: Recommendation-native tokenizer

Ý tưởng:

```text
Thiết kế item tokenizer tối ưu cho next-item / next-session prediction,
không chỉ reconstruct item embedding.
```

Motivation: reconstruction loss không khớp retrieval objective. Hướng này nối SimCIT, DIGER, VarLenRec, GRID và ReSID-like papers.

Câu hỏi nghiên cứu:

```text
Semantic ID tốt cho recommendation nên được đánh giá bằng reconstruction error
hay bằng prefix predictability / retrieval likelihood / ranking utility?
```

### Hướng 3: Variable-length Semantic ID cho long-tail

Ý tưởng:

```text
Head item dùng ID ngắn; tail item dùng ID dài hơn để chứa semantic detail.
```

Motivation: catalog có power-law distribution; fixed-length code không tối ưu. Hướng này nối VarLenRec, hierarchical identifiers và long-tail/cold-start recommendation.

Câu hỏi nghiên cứu:

```text
Token length nên phụ thuộc vào popularity, content complexity hay collaborative uncertainty?
```

### Hướng 4: Generative retrieval cho next-session/set prediction

Ý tưởng:

```text
Thay vì generate một next item, generate hoặc retrieve một set/session-level candidate list.
```

Motivation: e-commerce request thường cần candidate set, không phải single next item. Hướng này nối TBGRecall, PinRec và OneRec.

Câu hỏi nghiên cứu:

```text
Generative retrieval nên sinh item theo thứ tự, sinh song song,
hay sinh session representation rồi ANN retrieval?
```

### Hướng 5: Diffusion-based generative retrieval

Ý tưởng:

```text
Thay autoregressive token-by-token decoding bằng masked diffusion hoặc continuous-token diffusion.
```

Motivation: autoregressive decoding chậm và có exposure bias. Hướng này nối Masked Diffusion và Continuous-token Diffusion.

Câu hỏi nghiên cứu:

```text
Diffusion có thể giảm latency và tăng robustness so với autoregressive SID decoding không?
```

### Hướng 6: Outcome-conditioned / controllable generative recommendation

Ý tưởng:

```text
Generate recommendation conditioned on target outcome:
click, save, purchase, watch-time, diversity, exploration.
```

Motivation: hệ thống thực tế có nhiều metric. Hướng này nối PinRec và OneRec.

Câu hỏi nghiên cứu:

```text
Làm sao đưa business objective vào generative retrieval mà không phải train nhiều model riêng?
```

### Hướng 7: Semantic ID for encoded multimodal input

Ý tưởng:

```text
Tạo Semantic ID từ image/text/audio/video/OCR/multimodal encoder
thay vì chỉ từ text embedding.
```

Motivation: item content đa modality, text có thể noisy/dài, image có thể token-efficient. Hướng này nối I-LLMRec, multimodal GR và text-as-vision papers bạn đã đọc trước.

Câu hỏi nghiên cứu:

```text
Modality nào tạo Semantic ID tốt nhất cho domain cụ thể?
```

---

## 17. Nhận xét cá nhân: hướng nào đáng ưu tiên nhất?

Nếu phải chọn hướng rõ nhất từ danh sách paper này, tôi sẽ ưu tiên:

```text
Recommendation-native item tokenization for generative retrieval.
```

Lý do là gần như mọi paper đều quay về cùng một điểm: **tokenizer/identifier là bottleneck chính**. Backbone có thể là Transformer, LLM, HSTU, MoE hoặc diffusion, nhưng nếu item tokenization không phù hợp, mô hình vẫn sẽ gặp vấn đề về generalization, decoding, latency hoặc alignment.

Có thể viết thesis/proposal theo câu sau:

```text
Generative retrieval shifts the core problem of recommendation from candidate scoring to item identifier generation.
Therefore, the design of item identifiers/tokenizers becomes a central modeling problem.
```

Sau đó triển khai theo các nhánh:

```text
- semantic vs collaborative tokenization
- fixed-length vs variable-length tokenization
- reconstruction vs recommendation-oriented objective
- autoregressive vs diffusion decoding
- single-item vs session/set generation
- benchmark vs industrial deployment
```

Đây là một góc nhìn đủ rộng để bao phủ nhiều paper, nhưng vẫn có trục chính rõ ràng.

---

## 18. Mapping nhanh từng paper trong thư mục

| Paper | Vai trò chính |
|---|---|
| **2305.05065 - Recommender Systems with Generative Retrieval / TIGER** | Paper nền mở ra Semantic ID-based generative retrieval. |
| **2502.06269 - UNGER** | Unified code tích hợp semantic và collaborative knowledge. |
| **2502.12448 - Discrete Tokenizers Survey** | Survey nền về discrete tokenizers như interface cho generative models. |
| **2502.16474 - Unified Semantic and ID Representation Learning** | Kết hợp ID token và semantic token để cân bằng memorization/generalization. |
| **2502.18965 - OneRec** | Unified retrieve-and-rank bằng generative recommender + preference alignment. |
| **2503.06238 - Token-Efficient Item Representation via Images / I-LLMRec** | Dùng image thay text dài để biểu diễn item hiệu quả hơn trong LLM recommender. |
| **2503.12183 - Bridging Textual-Collaborative Gap / CoCoRec/CCFRec** | Dùng semantic codes để nối textual metadata và collaborative behavior. |
| **2504.10507 - PinRec** | Outcome-conditioned, multi-token generative retrieval ở quy mô Pinterest. |
| **2504.12007 - Continuous-token Diffusion** | Dùng continuous token + diffusion thay discrete-only generation. |
| **2505.18654 - MTGR** | Industrial-scale GR ở Meituan, giữ DLRM/cross features và tối ưu scaling. |
| **2506.16683 - SimCIT** | Contrastive item tokenization thay reconstruction-only quantization. |
| **2507.22224 - Semantic IDs Handbook / GRID** | Framework thực hành và benchmark modular cho GR với SID. |
| **2508.11977 - TBGRecall** | Generative retrieval cho e-commerce/Taobao với Next Session Prediction. |
| **2508.14910 - COSETTE/MARIUS** | Collaborative tokenization + efficient generative modeling để thu hẹp gap với ID baselines. |
| **2509.16446 - Purely Semantic Indexing** | Unique semantic-preserving IDs không cần non-semantic suffix. |
| **2510.15729 - FACE** | Map CF embeddings thành LLM tokens để kết nối collaborative signal với language space. |
| **2510.27157 - Survey on Generative Recommendation** | Survey rộng về generative recommendation theo data-model-task. |
| **2511.12922 - UniTok** | Unified item tokenization cho multi-domain LLM-based recommendation. |
| **2511.23021 - Masked Diffusion** | Masked diffusion cho SID sequence, hỗ trợ parallel decoding. |
| **2512.18434 - Efficient Hierarchical Identifiers** | Tối ưu xây dựng hierarchical/tree identifiers hiệu quả hơn. |
| **2601.19711 - Differentiable Semantic ID / DIGER** | Cho recommendation loss update tokenizer; giảm codebook collapse bằng Gumbel exploration. |
| **2605.17779 - VarLenRec** | Học variable-length tokenization; tail items cần ID dài hơn head items. |

---

## 19. Kết luận

Generative retrieval đang chuyển recommender systems từ tư duy “score item candidates” sang “generate item identifiers”. Sự chuyển đổi này khiến **item identifier/tokenizer** trở thành thành phần trung tâm. Các paper ban đầu như TIGER chứng minh Semantic ID có thể thay atomic item ID trong generative retrieval. Các paper sau đó tập trung giải quyết những vấn đề phát sinh: semantic-collaborative gap, uniqueness conflict, fixed-length inefficiency, reconstruction-objective mismatch, autoregressive latency, multi-objective controllability và industrial deployment.

Nếu cần một câu tóm tắt toàn bộ nhóm paper, có thể viết:

```text
Generative retrieval is less about simply applying LLMs to recommendation,
and more about designing a recommendation-native item language:
a tokenization scheme that is semantic, collaborative, unique, efficient, decodable, and scalable.
```

Đây là ý tưởng chính nên dùng để dẫn dắt phần literature review. Thay vì liệt kê từng paper rời rạc, nên gom chúng quanh câu hỏi: **làm sao xây một “ngôn ngữ item” tốt cho generative recommender?**
