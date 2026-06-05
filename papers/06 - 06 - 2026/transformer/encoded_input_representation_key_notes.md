# Note bổ sung: Những điểm đáng chú ý trong hướng encoded-input representation cho recommender systems

## 1. Encoder choice có thể là contribution chính, không chỉ là preprocessing

Điểm quan trọng nhất của hướng này là encoder không chỉ là bước tạo feature phụ. Encoder quyết định **không gian biểu diễn** mà recommender model nhìn thấy. Cùng một item description có thể mang ý nghĩa rất khác nhau tùy cách encode.

Ví dụ:

```text
Wireless Mouse, 2.4G, 1600DPI, USB Receiver, Black
```

Nếu dùng text encoder thông thường, model nhìn input như chuỗi token hoặc subword. Các cụm như `2.4G`, `1600DPI`, `USB`, tên brand, model number hoặc đơn vị kỹ thuật có thể bị tách vụn thành nhiều token nhỏ, làm yếu quan hệ giữa các thuộc tính. Nếu render text thành image rồi dùng OCR/vision encoder, model nhìn input như một tín hiệu thị giác chứa bố cục, ký hiệu, số, đơn vị và cách các thuộc tính xuất hiện cùng nhau.

Vì vậy, có thể phát biểu hướng nghiên cứu mạnh hơn như sau:

```text
The representation interface between item content and recommender model is itself a modeling problem.
```

Nói cách khác, ta không chỉ hỏi “dùng Transformer nào cho recommender”, mà hỏi trước đó: **input nên được biến thành dạng gì trước khi Transformer xử lý?**

---

## 2. Khoảng trống giữa semantic similarity và collaborative similarity

Một encoder pretrained từ NLP, CV hoặc multimodal learning thường học theo **semantic similarity**. Tuy nhiên recommender system lại cần **behavioral/collaborative similarity**. Hai loại similarity này không luôn trùng nhau.

Ví dụ:

```text
snacks ↔ balloons
```

Hai item này không giống nhau nhiều về text hoặc image. Nhưng trong dữ liệu hành vi, chúng có thể thường được mua chung trong bối cảnh party. Nếu encoder chỉ học semantic similarity, nó có thể đặt hai item này rất xa nhau. Nhưng với recommender system, chúng lại có quan hệ hành vi mạnh.

Đây là điểm rất đáng khai thác: encoder cho recommender không nên chỉ học “item giống nhau về nội dung”, mà còn cần học “item liên quan nhau trong hành vi người dùng”.

Có thể viết thành luận điểm:

```text
Encoder cho recommender system cần học semantic structure và collaborative structure cùng lúc,
thay vì chỉ tối ưu semantic representation từ foundation models.
```

Điểm này cũng giải thích vì sao các Semantic ID hoặc item representation lấy trực tiếp từ foundation model chưa chắc tối ưu cho recommendation. Chúng có thể semantic-rich nhưng collaborative-weak.

---

## 3. Text-as-image không chỉ để xử lý text khác đi, mà còn để giảm modality gap

Paper text-as-vision trong generative recommendation có một ý quan trọng: OCR-text embedding có thể gần image embedding hơn so với standard text embedding. Nghĩa là khi item có cả text và image, việc render text thành ảnh rồi dùng OCR/vision encoder có thể giúp không gian biểu diễn của text và image gần nhau hơn.

Điều này tạo ra một hướng rất đáng chú ý:

```text
Text-as-image có thể là cầu nối giữa textual modality và visual modality trong multimodal recommendation.
```

Nếu dùng standard text encoder cho mô tả sản phẩm và image encoder cho ảnh sản phẩm, hai embedding có thể nằm trong hai không gian hình học khác nhau. Khi fusion, model phải học alignment giữa hai không gian này. Nhưng nếu text cũng được đưa qua OCR/vision encoder, representation của text có thể tự nhiên gần hơn với image representation, giúp multimodal fusion ổn định hơn.

Vì vậy, text-as-image không chỉ giải quyết vấn đề tokenization của item description, mà còn có thể giúp **giảm modality gap** trong multimodal recommender systems.

---

## 4. Semantic ID không chỉ là mã hóa item, mà là vocabulary mới của generative recommender

Trong generative recommendation, Semantic ID không chỉ là một feature biểu diễn item. Nó là chuỗi token mà model phải sinh ra. Điều này làm Semantic ID khác với embedding thông thường.

Với embedding retrieval, item vector chỉ cần nằm đúng vị trí trong vector space. Nhưng với generative retrieval, Semantic ID phải được decode token-by-token. Vì vậy, một Semantic ID tốt cần đồng thời thỏa nhiều điều kiện:

```text
- compact
- semantically meaningful
- dễ decode autoregressively
- share được information giữa các item tương tự
- hỗ trợ item mới và long-tail item
- giảm uncertainty ở từng bước sinh token
```

Đây là điểm rất quan trọng. Nếu Semantic ID được tạo bằng quantizer không phù hợp, model có thể gặp khó khi sinh code sequence, dù embedding ban đầu có vẻ tốt. Do đó, trong generative recommendation, tokenizer/quantizer không còn là bước kỹ thuật phụ; nó là một phần trực tiếp của modeling.

Một hướng khai thác tốt là:

```text
Semantic ID tokenization should be optimized not only for semantic reconstruction,
but also for autoregressive predictability in recommendation.
```

---

## 5. Có thể khai thác hướng encoder sau encoder

Một điểm mới trong các paper gần đây là: ngay cả khi đã có Semantic ID, đưa SID token trực tiếp vào LLM vẫn chưa chắc đủ. SID token có nghĩa phụ thuộc vào prefix context. Cùng một code ở level sau có thể mang ý nghĩa khác nhau dưới các prefix khác nhau.

Ví dụ:

```text
prefix A + code 505 -> sneakers
prefix B + code 505 -> vintage art
```

Nếu dùng flat token embedding, token `505` luôn có cùng embedding, dù ý nghĩa của nó thay đổi theo prefix. Điều này khiến LLM phải tự học cấu trúc phân cấp của SID từ dữ liệu, rất tốn kém ở quy mô lớn.

Vì vậy, có thể xem SID như một modality riêng cần encoder riêng, tương tự như image cần vision encoder hoặc audio cần audio encoder. Pipeline khi đó không chỉ là:

```text
raw item -> Semantic ID -> LLM
```

mà có thể là:

```text
raw item content -> content encoder -> Semantic ID tokenizer -> SID encoder -> generative recommender
```

Đây là hướng rất đáng chú ý vì nó mở rộng ý tưởng “không dùng input gốc trực tiếp” thành nhiều tầng biểu diễn. Mỗi tầng encoder giải quyết một vấn đề khác nhau: content encoder học semantic/modal information; tokenizer tạo discrete representation; SID encoder contextualize code theo prefix; generative model học sequence recommendation.

---

## 6. Cold-start và long-tail là nơi hướng này dễ chứng minh giá trị nhất

Nếu làm thực nghiệm hoặc viết proposal, không nên chỉ nhấn mạnh overall Recall/NDCG. Hướng encoded-input representation có giá trị rõ nhất ở các nhóm item mà ID embedding truyền thống yếu:

```text
- new items
- cold-start items
- long-tail items
- low-interaction items
- noisy-description items
- attribute-heavy items
- multimodal items
```

Random item ID mạnh với item phổ biến vì model có nhiều interaction để học embedding. Nhưng với item mới hoặc hiếm, random ID gần như không mang thông tin. Ngược lại, nếu item được encode từ text, image, metadata hoặc multimodal content, model có thể generalize từ các item tương tự.

Vì vậy, contribution nên viết theo hướng:

```text
Encoded item representation improves generalization on cold-start and long-tail items
while preserving competitive overall recommendation quality.
```

Nếu chỉ báo cáo average metric trên toàn dataset, lợi ích của hướng này có thể bị che bởi các head items. Cần tách riêng các slice như item popularity bucket, item age, number of interactions, modality completeness, hoặc description noise level.

---

## 7. Trade-off lớn: memorization vs generalization

Một điểm cần cẩn thận là không nên phát biểu rằng “bỏ item ID là luôn tốt hơn”. Item ID embedding có ưu điểm lớn: nó giúp model memorize item-specific behavior. Với các item phổ biến, có nhiều interaction, memorization là tín hiệu rất mạnh.

Content encoder hoặc Semantic ID giúp generalization tốt hơn, nhưng có thể mất thông tin riêng của từng item. Nếu hai item có content rất giống nhau nhưng user behavior khác nhau, content-only representation có thể không đủ.

Vì vậy, hướng hợp lý hơn là tìm biểu diễn trung gian cân bằng hai yếu tố:

```text
memorization ↔ generalization
```

Một số cách cân bằng:

```text
- dùng Semantic ID thay vì content embedding thuần túy
- kết hợp atomic ID với Semantic ID
- dùng prefix/sub-piece hashing của Semantic ID
- thêm residual item-specific embedding cho head items
- dùng content encoder cho cold-start/long-tail, ID embedding cho head items
```

Điểm nên nhấn mạnh:

```text
Mục tiêu không phải là loại bỏ item ID bằng mọi giá,
mà là thiết kế representation interface giúp model vừa nhớ được item cụ thể,
vừa generalize được sang item mới và item hiếm.
```

---

## 8. Có thể kết hợp với hướng time prediction trước đó

Hướng encoded-input representation có thể đứng riêng, nhưng cũng có thể kết hợp với hướng time prediction trước đó.

Hướng trước tập trung vào đổi target:

```text
predict item -> predict time / distribution / intensity curve
```

Hướng này tập trung vào đổi input representation:

```text
raw item ID/text/image -> encoded item representation
```

Kết hợp lại, có thể tạo một hướng mạnh hơn:

```text
Predict purchase timing using transformed or encoded multimodal item representations.
```

Ví dụ:

```text
item description -> text-as-image -> OCR encoder -> item representation
user history + encoded item -> purchase time distribution
```

Hoặc:

```text
image/text/audio/video -> multimodal Semantic ID
user sequence of Semantic IDs -> predict next item and next time
```

Khi đó novelty nằm ở cả hai phía:

```text
Input side:
    không dùng raw item ID/text trực tiếp, mà dùng encoded representation.

Output side:
    không chỉ predict item, mà predict time distribution hoặc item-time pair.
```

Đây là một hướng có thể rất mạnh nếu muốn kết nối hai nhóm paper bạn đang đọc.

---

## 9. Các câu hỏi nghiên cứu đáng viết

Một số research questions phù hợp với hướng này:

```text
RQ1: Text-as-image/OCR embeddings có tốt hơn standard text embeddings cho item descriptions nhiều ký hiệu, số, đơn vị và abbreviation không?

RQ2: Semantic IDs tạo từ modality nào tốt nhất cho generative recommendation: text, image, OCR-text, multimodal, hay collaborative encoder?

RQ3: Early fusion hay late fusion tốt hơn khi tạo Semantic IDs từ nhiều modality?

RQ4: Recsys-native encoder có vượt foundation encoder trong cold-start/long-tail recommendation không?

RQ5: SID token có cần encoder riêng không, hay flat token embedding là đủ?

RQ6: Representation nào cân bằng tốt nhất giữa memorization và generalization?

RQ7: Encoder/tokenizer nên được tối ưu theo semantic similarity, collaborative similarity, hay sequential predictability?

RQ8: Với item có nhiều modality, modality nào đóng góp nhiều nhất trong từng domain: e-commerce, video, music, news, short-form content?

RQ9: Text-as-image có giúp giảm modality gap giữa text embedding và image embedding trong multimodal recommendation không?

RQ10: Encoded representation có cải thiện khả năng recommend item mới chưa có interaction history không?
```

---

## 10. Một hướng nên ưu tiên nếu muốn làm proposal

Hướng rõ và dễ bảo vệ nhất là:

```text
Text-as-vision and recsys-native Semantic IDs for generative recommendation.
```

Lý do là hướng này có paper nền rõ, có khoảng trống cụ thể, và dễ giải thích. Có thể lập luận như sau:

Item descriptions trong recommender systems, đặc biệt là e-commerce, không giống văn bản tự nhiên. Chúng thường là chuỗi thuộc tính ngắn, nhiều số, đơn vị, brand, ký hiệu kỹ thuật và model code. Standard text encoders được tối ưu cho natural language nên có thể không biểu diễn tốt loại input này. Text-as-image/OCR encoder có thể giữ tốt hơn cấu trúc attribute-centric của item descriptions. Embedding thu được có thể dùng để tạo Semantic IDs cho generative recommendation.

Sau đó mở rộng thêm:

Semantic ID tokenizer không nên chỉ tối ưu semantic reconstruction, mà cần tối ưu sequential predictability cho recommender model. Vì trong generative recommendation, Semantic ID là target sequence mà model phải decode. Do đó, encoder/tokenizer nên được thiết kế theo mục tiêu recommendation, không chỉ mượn trực tiếp từ NLP/CV.

Có thể viết contribution theo hướng:

```text
1. We study text-as-vision item representation for generative recommendation,
   targeting symbolic and attribute-centric product descriptions.

2. We construct Semantic IDs from OCR-based visual text embeddings
   and compare them against standard text-encoder-based Semantic IDs.

3. We analyze the effect of representation choice on cold-start, long-tail,
   noisy-description and multimodal item recommendation.

4. We investigate whether recsys-native tokenization improves autoregressive
   predictability of Semantic ID sequences.
```

---

## 11. Một formulation tổng quát

Có thể mô tả bài toán bằng công thức đơn giản:

```text
x_i = raw item input
z_i = E(x_i)
s_i = Q(z_i)
```

Trong đó:

```text
x_i: raw item input, ví dụ text, image, audio, video, metadata
E: encoder hoặc modality transformer
z_i: continuous encoded representation
Q: tokenizer/quantizer
s_i: Semantic ID hoặc discrete token sequence
```

Sau đó generative recommender học:

```text
P(s_next | s_history, user_context)
```

Nếu không dùng discrete SID, có thể học trực tiếp:

```text
score(u, i) = f(UserEncoder(history), E(x_i))
```

Nếu kết hợp với hướng time prediction:

```text
P(time | user_history, E(x_i))
```

hoặc:

```text
λ_user,item(t | E_text(x_i), E_image(x_i), E_audio(x_i), E_video(x_i))
```

Formulation này giúp bạn trình bày hướng nghiên cứu một cách tổng quát: trọng tâm không phải backbone recommender, mà là cách biến raw input thành representation mà recommender có thể học hiệu quả.

---

## 12. Baseline nên so sánh

Nếu làm nghiên cứu, nên so sánh nhiều dạng representation:

```text
1. Atomic item ID
2. Standard text encoder embedding
3. Text-as-image / OCR-text embedding
4. Image encoder embedding
5. Multimodal encoder embedding
6. Content-derived Semantic ID
7. Collaborative or recsys-native Semantic ID
8. Hybrid ID + Semantic ID
9. Prefix-aware SID encoder
```

Các metric nên tách riêng:

```text
- overall Recall/NDCG
- cold-start Recall/NDCG
- long-tail Recall/NDCG
- new item performance
- noisy text robustness
- multimodal missingness robustness
- decoding accuracy for Semantic IDs
- prefix-level SID prediction accuracy
- serving cost / latency
```

Điểm quan trọng: nếu paper của bạn nói về representation, evaluation cũng phải chứng minh representation đó tốt ở các slice mà representation matters, không chỉ ở average performance.

---

## 13. Rủi ro và điểm cần tránh

### 13.1. Không nên nói encoder càng mạnh càng tốt

Encoder mạnh trong NLP/CV chưa chắc phù hợp cho recommender. Cần nhấn mạnh downstream alignment.

### 13.2. Không nên bỏ qua serving cost

Vision/OCR/video/audio encoders có thể tốn compute. Trong production, cần precompute item embeddings hoặc Semantic IDs.

### 13.3. Không nên chỉ so sánh overall metric

Lợi ích của encoded representation thường nằm ở cold-start, long-tail, noisy hoặc multimodal cases.

### 13.4. Không nên xem Semantic ID chỉ là compression

Semantic ID trong generative recommendation là target vocabulary. Nó ảnh hưởng đến cả training và decoding.

### 13.5. Không nên fusion multimodal một cách đơn giản

Early fusion có thể gây modality dominance; late fusion cần alignment. Đây là trade-off cần phân tích.

---

## 14. Kết luận

Điểm đáng chú ý nhất của hướng này là: **input representation đang trở thành một thành phần modeling chính của recommender systems hiện đại**. Trước đây, nghiên cứu thường tập trung vào backbone: matrix factorization, two-tower, GNN, RNN, Transformer, LLM. Nhưng với generative recommendation và multimodal recommendation, cách item được encode/tokenize có thể quyết định mạnh đến khả năng generalization, cold-start, long-tail retrieval và multimodal fusion.

Có thể tóm tắt hướng này bằng một câu:

```text
Thay vì cải tiến recommender backbone trực tiếp,
ta cải tiến cách raw item input được biến đổi thành representation hoặc token sequence
mà recommender backbone sử dụng.
```

Hướng đáng ưu tiên nhất là kết hợp **text-as-image/OCR representation** với **Semantic ID learning** cho generative recommendation, sau đó mở rộng sang **recsys-native tokenizer** và **prefix-aware SID encoder**. Đây là hướng có nền tảng paper rõ, có khoảng trống cụ thể, và có thể liên kết tốt với các vấn đề lớn trong recommender systems: cold-start, long-tail, multimodal fusion, representation alignment và generative retrieval.
