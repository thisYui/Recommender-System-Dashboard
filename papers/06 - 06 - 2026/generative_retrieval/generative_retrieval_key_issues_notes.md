# Note bổ sung: Các vấn đề cần chú ý trong Generative Retrieval / Generative Recommendation

## 1. Generative retrieval chưa chắc thay thế được retrieve-and-rank

Generative retrieval thường được trình bày như một hướng có khả năng thay thế pipeline retrieve-and-rank truyền thống. Tuy nhiên, khi xét trong hệ thống thực tế, retrieve-and-rank vẫn rất mạnh vì nó nhanh, ổn định, dễ scale và dễ kiểm soát. Các mô hình retrieval truyền thống như two-tower, dense retrieval, ANN search hoặc candidate generation bằng embedding index đã được tối ưu rất lâu cho latency và throughput.

Ngược lại, generative retrieval linh hoạt hơn vì có thể trực tiếp sinh item identifiers, nhưng nó cũng có nhiều ràng buộc mới. Mô hình phải decode token, xử lý beam search hoặc constrained decoding, kiểm soát output hợp lệ, và đảm bảo latency đủ thấp. Vì vậy không nên viết rằng generative retrieval chắc chắn thay thế hoàn toàn retrieve-and-rank. Cách viết hợp lý hơn là:

```text
Generative retrieval có tiềm năng thống nhất retrieval và ranking,
nhưng hiện vẫn phải giải quyết latency, decoding, controllability và feature compatibility.
```

Các paper công nghiệp như OneRec, PinRec, MTGR và TBGRecall đáng chú ý vì chúng không chỉ đánh giá trên benchmark, mà cố gắng trả lời câu hỏi thực tế hơn: làm sao đưa generative retrieval vào production system.

---

## 2. Bottleneck thật sự là item language, không phải model backbone

Một nhầm lẫn phổ biến là nghĩ rằng generative retrieval chủ yếu là “dùng LLM hoặc Transformer lớn hơn cho recommendation”. Tuy nhiên, trong generative retrieval, bottleneck thường không chỉ nằm ở backbone, mà nằm ở **ngôn ngữ định danh item**.

Vì model phải sinh item identifier, item cần được biểu diễn thành một “ngôn ngữ” mà model có thể học và sinh ra. Item language này có thể là Semantic ID, hierarchical ID, collaborative token, continuous token hoặc variable-length token. Nếu item language kém, backbone mạnh cũng khó tạo kết quả tốt.

Một item language tốt cần có nhiều tính chất cùng lúc:

```text
- đủ semantic để share information giữa item tương tự
- đủ collaborative để phản ánh hành vi user
- đủ unique để phân biệt từng item
- đủ ngắn để decode nhanh
- đủ dễ predict để model học được
- đủ linh hoạt cho item mới và cold-start
- đủ chi tiết cho long-tail item
```

Đây là lý do nhiều paper mới tập trung vào collaborative tokenization, differentiable Semantic ID, variable-length tokenization, purely semantic indexing và hierarchical identifiers. Có thể xem đây là trục chính của generative retrieval: **thiết kế một item language phù hợp với recommendation**.

---

## 3. Autoregressive decoding là vấn đề lớn

Nhiều generative recommender sinh item identifier theo kiểu autoregressive:

```text
token_1 -> token_2 -> token_3 -> token_4 -> item
```

Cách này tự nhiên với Transformer/LLM, nhưng có vấn đề lớn về latency. Mỗi token phụ thuộc vào token trước, nên decoding phải diễn ra tuần tự. Nếu dùng beam search để tăng khả năng tìm đúng item, chi phí inference còn cao hơn nữa.

Trong production recommender systems, latency là ràng buộc rất nghiêm trọng. Một hệ thống retrieval có thể cần trả về candidate trong vài chục mili-giây. Nếu generative retrieval decode quá chậm, nó khó thay thế hoặc bổ sung cho retrieval truyền thống.

Vì vậy, cần chú ý các hướng thay thế hoặc tối ưu:

```text
- masked diffusion hoặc parallel decoding
- continuous-token diffusion
- tree/hierarchical decoding
- candidate pruning
- constrained decoding
- session-level generation thay vì item-level generation
- multi-token generation để sinh nhiều candidate hiệu quả hơn
```

Các paper về masked diffusion, continuous-token diffusion và hierarchical identifiers không chỉ là biến thể mô hình, mà còn là cách xử lý bottleneck inference của generative retrieval.

---

## 4. Exposure bias và error propagation trong sinh token

Generative retrieval có vấn đề riêng mà ranking truyền thống không gặp rõ như vậy: nếu model sinh sai token đầu, các token sau có thể sai theo. Đây là error propagation trong quá trình generation.

Ví dụ:

```text
SID đúng:     12 -> 45 -> 7 -> 90
SID model sinh: 13 -> ...
```

Chỉ cần sai token đầu, model đã đi sang một nhánh item khác. Dù các token sau được sinh hợp lý theo prefix sai, full item vẫn sai. Điều này làm việc đánh giá token-level trở nên rất quan trọng.

Vì vậy khi phân tích generative retrieval, không nên chỉ nhìn Recall@K cuối cùng. Cần xem thêm:

```text
- prefix accuracy
- deepest-level SID accuracy
- full-SID exact match
- token-level calibration
- beam search sensitivity
- constrained decoding effectiveness
- lỗi xảy ra ở level nào của Semantic ID
```

Nếu lỗi tập trung ở token đầu, vấn đề có thể nằm ở coarse semantic partition. Nếu lỗi tập trung ở token cuối, vấn đề có thể nằm ở khả năng phân biệt item gần nhau hoặc conflict trong Semantic ID.

---

## 5. Semantic ID conflict và semantic drift

Semantic ID có thể gặp hai vấn đề thực tế: **conflict** và **drift**.

Conflict xảy ra khi nhiều item khác nhau được gán cùng một Semantic ID. Điều này thường xuất hiện nếu các item rất giống nhau hoặc quantizer không đủ capacity. Một cách xử lý đơn giản là thêm random suffix hoặc non-semantic token để phân biệt. Tuy nhiên, cách này làm mất tính semantic của identifier, tăng search space và khiến model khó predict hơn.

Drift xảy ra khi catalog và hành vi user thay đổi theo thời gian. Item mới được thêm vào, item cũ biến mất, trend thay đổi, collaborative relation thay đổi. Semantic ID được tạo ở thời điểm hiện tại có thể không còn tối ưu sau vài tháng. Đây là vấn đề quan trọng trong hệ thống thực tế nhưng dễ bị bỏ qua trong benchmark offline.

Các câu hỏi cần chú ý:

```text
- Update Semantic ID như thế nào?
- Có cần re-tokenize toàn bộ catalog không?
- Item mới được gán code ra sao?
- Code cũ có bị mất ý nghĩa không?
- Model có cần continual training không?
- Nếu re-tokenize, làm sao giữ compatibility với model đã train?
```

Đây là điểm có thể phát triển thành một hướng nghiên cứu riêng: dynamic item tokenization hoặc continual Semantic ID learning.

---

## 6. Tokenizer cố định hay tokenizer học end-to-end?

Có hai cách thiết kế tokenizer trong generative recommendation.

Cách thứ nhất là **static tokenizer**:

```text
Tạo Semantic ID trước -> cố định Semantic ID -> train recommender model
```

Cách này dễ triển khai, ổn định, dễ cache và dễ kiểm soát. Tuy nhiên, nó có thể không tối ưu cho recommendation objective vì tokenizer thường được học bằng reconstruction loss hoặc semantic similarity.

Cách thứ hai là **differentiable/end-to-end tokenizer**:

```text
Tokenizer được cập nhật bởi recommendation loss
```

Cách này có tiềm năng align tốt hơn với downstream recommendation, nhưng khó train hơn. Nó có thể gặp codebook collapse, instability, chi phí training cao hoặc khó maintain trong production.

Đây là một trục phân loại tốt trong literature review:

```text
content-based static SID
collaborative-aware SID
contrastive SID
differentiable SID
variable-length SID
recsys-native SID
```

Nếu viết proposal, có thể đặt câu hỏi: Semantic ID nên là một index cố định hay một phần học được của recommender model?

---

## 7. Đánh giá generative retrieval không nên chỉ dùng Recall/NDCG

Recall@K và NDCG@K vẫn cần, vì chúng cho biết chất lượng recommendation cuối cùng. Nhưng với generative retrieval, chỉ hai metric này là chưa đủ. Lý do là generative retrieval có các vấn đề riêng về tokenization, decoding, conflict, latency và controllability.

Nên bổ sung các metric như:

```text
- SID exact match
- prefix-level accuracy
- token-level entropy
- token-level calibration
- decoding latency
- beam size sensitivity
- conflict rate
- cold-start performance
- long-tail performance
- diversity
- controllability theo outcome
- online engagement nếu có
```

Ví dụ, hai model có Recall@20 gần nhau nhưng một model cần beam size lớn gấp 10 lần thì chưa chắc model đó tốt hơn trong production. Hoặc một model đạt overall Recall cao nhưng kém ở cold-start/long-tail thì chưa chắc phù hợp với mục tiêu của Semantic ID.

Vì vậy, evaluation của generative retrieval nên tách rõ ba lớp:

```text
1. Recommendation quality:
   Recall, NDCG, Hit Rate, MRR.

2. Identifier/generation quality:
   SID accuracy, prefix accuracy, conflict rate, token entropy.

3. System quality:
   latency, memory, serving cost, beam size, update cost.
```

---

## 8. Generative retrieval có thể yếu trong feature-rich industrial recommender

Nhiều paper academic đơn giản hóa recommender input thành user history sequence. Tuy nhiên, hệ thống công nghiệp thường có rất nhiều feature:

```text
user profile
context
time
location
device
query
session
real-time behavior
cross features
business constraints
exposure history
ranking features
```

Nếu generative retrieval chỉ dùng item sequence, nó có thể mất nhiều feature quan trọng mà DLRM/ranker truyền thống đang sử dụng. Đây là lý do một số paper công nghiệp như MTGR nhấn mạnh việc giữ lại DLRM features hoặc cross features khi chuyển sang generative framework.

Điểm cần viết cẩn thận:

```text
Generative retrieval không chỉ cần sinh item tốt,
mà còn phải tương thích với feature engineering và serving constraints của hệ thống thật.
```

Nếu không giải quyết feature compatibility, generative retrieval có thể tốt trên benchmark sequential recommendation nhưng yếu khi so với production recommender đầy đủ feature.

---

## 9. Multi-objective recommendation là vấn đề lớn

Trong thực tế, recommender system không chỉ tối ưu click hoặc next-item likelihood. Nó có thể tối ưu nhiều mục tiêu cùng lúc:

```text
click
purchase
watch time
save
long-term retention
diversity
freshness
creator fairness
business value
user satisfaction
```

Nếu generative retrieval chỉ học next-item likelihood, output có thể không align với mục tiêu thật của hệ thống. Đây là lý do các hướng như outcome-conditioned generation, preference alignment, DPO-like training và multi-token generation rất đáng chú ý.

Một câu có thể dùng trong report:

```text
The next frontier of generative retrieval is not only generating relevant items,
but generating controllable items under different target outcomes.
```

Với cách nhìn này, generative recommender không chỉ sinh item có xác suất cao, mà còn có thể sinh item theo điều kiện: tối ưu click, tối ưu purchase, tăng diversity, hoặc phục vụ mục tiêu dài hạn.

---

## 10. Ba hướng đáng tập trung nhất

Nếu cần chọn ba hướng quan trọng nhất để phát triển idea từ nhóm paper generative retrieval, có thể ưu tiên ba hướng sau.

### 10.1. Recommendation-native tokenizer

Đây là hướng mạnh nhất vì nó nằm ở trung tâm của generative retrieval. Mục tiêu là không chỉ tạo Semantic ID từ content embedding, mà thiết kế tokenizer tối ưu cho recommendation objective.

```text
Không chỉ hỏi “item giống nhau về semantic không?”,
mà hỏi “item có dễ được dự đoán đúng trong recommender model không?”.
```

Hướng này bao phủ các paper về UNGER, CoCoRec, COSETTE, SimCIT, DIGER và VarLenRec.

### 10.2. Efficient decoding

Generative retrieval muốn deploy thật thì phải giải quyết latency và decoding cost.

```text
autoregressive decoding vs masked diffusion vs hierarchical decoding
```

Đây là hướng thực tế, rõ vấn đề, dễ liên hệ với production. Nó cũng giúp giải thích vì sao masked diffusion, continuous-token diffusion và tree-based identifiers quan trọng.

### 10.3. Semantic-collaborative alignment

Đây là khoảng trống lý thuyết và thực nghiệm lớn:

```text
semantic similarity ≠ collaborative similarity
```

Nếu giải quyết tốt, model có thể vừa generalize tốt cho cold-start/long-tail, vừa giữ được behavior signal quan trọng cho recommendation.

---

## 11. Một hướng proposal khả thi

Một hướng proposal có thể viết như sau:

```text
Recommendation-Native Semantic Tokenization for Generative Retrieval
```

Ý tưởng chính:

```text
Generative retrieval chuyển bài toán recommendation từ scoring item sang generating item identifiers.
Do đó, item tokenizer trở thành bottleneck trung tâm.
Tuy nhiên, các tokenizer hiện tại thường dựa quá nhiều vào semantic reconstruction,
trong khi recommendation cần collaborative alignment, decoding efficiency và cold-start generalization.
```

Các câu hỏi nghiên cứu:

```text
RQ1: Semantic ID nên được tối ưu bằng reconstruction, contrastive learning,
collaborative prediction hay recommendation loss?

RQ2: Làm sao giảm gap giữa semantic similarity và collaborative similarity trong item tokenization?

RQ3: Tokenizer có nên tạo fixed-length code hay variable-length code theo popularity/uncertainty?

RQ4: Làm sao đánh giá tokenizer không chỉ bằng Recall/NDCG mà còn bằng prefix accuracy,
conflict rate và decoding latency?

RQ5: Tokenizer có thể update theo catalog drift mà không cần retrain toàn bộ model không?
```

Contribution có thể viết:

```text
1. We identify item tokenization as the central bottleneck of generative retrieval.

2. We propose a recommendation-native tokenizer that integrates semantic and collaborative signals.

3. We optimize the tokenizer not only for reconstruction quality but also for autoregressive predictability.

4. We evaluate the tokenizer on overall, cold-start and long-tail recommendation,
   as well as identifier-level metrics such as prefix accuracy and conflict rate.

5. We analyze the trade-off between decoding efficiency, semantic sharing and item uniqueness.
```

---

## 12. Kết luận

Generative retrieval không chỉ là dùng generative model để sinh item. Bản chất của nó là thiết kế một **item language** có thể được model học, sinh, kiểm soát và mở rộng ở quy mô recommender thực tế.

Có thể giữ câu sau làm ý chính:

```text
Generative retrieval is not merely about applying generative models to recommendation;
it is about designing a recommendation-native item language that is semantic,
collaborative, unique, efficient, decodable and scalable.
```

Nếu viết tiếp literature review hoặc proposal, nên xoay quanh năm vấn đề chính:

```text
1. Tokenizer / Semantic ID design
2. Efficient decoding
3. Semantic-collaborative alignment
4. Industrial feature and serving constraints
5. Evaluation beyond Recall/NDCG
```

Đây là các điểm quyết định generative retrieval có thể vượt qua hoặc bổ sung tốt cho retrieve-and-rank truyền thống hay không.
