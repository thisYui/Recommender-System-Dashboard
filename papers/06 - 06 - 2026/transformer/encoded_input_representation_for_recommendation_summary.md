# Hướng nghiên cứu: Không dùng input gốc trực tiếp, mà encode qua modality/encoder khác trước khi đưa vào recommender model

## 1. Ý tưởng trung tâm

Hướng này tập trung vào một câu hỏi khác so với time-aware recommendation: **liệu recommender system có cần dùng input gốc theo đúng modality ban đầu hay không?** Thay vì đưa trực tiếp item ID, text description, image, video, audio hoặc metadata vào model, ta có thể chuyển input qua một **encoder trung gian** hoặc một **biểu diễn thay thế** rồi mới dùng biểu diễn đó cho recommendation.

Cách nhìn này có thể viết ngắn gọn như sau:

```text
raw input -> encoder / tokenizer / modality transformation -> representation / tokens -> recommender model
```

Trong đó `raw input` có thể là text, item ID, image, audio, video, structured metadata hoặc multimodal content. Phần quan trọng nằm ở bước giữa: thay vì để model recommendation học trực tiếp từ input gốc, ta thiết kế hoặc tận dụng một encoder để chuyển input sang dạng phù hợp hơn. Encoder đó có thể là vision encoder, OCR encoder, text encoder, multimodal encoder, VQ/RQ-VAE tokenizer, Semantic ID generator, hoặc một encoder được thiết kế riêng cho recommender systems.

Một ví dụ rõ nhất là nhóm paper **text-as-image**. Thay vì xem text là chuỗi token và xử lý bằng text tokenizer, người ta render text thành ảnh rồi dùng vision model hoặc OCR-based vision encoder để xử lý. Với cách này, text không còn là token sequence theo nghĩa truyền thống, mà trở thành visual signal. Điểm đáng chú ý là mô hình không còn phụ thuộc hoàn toàn vào vocabulary, subword tokenizer, hoặc assumptions của pretrained language model. Điều này đặc biệt hữu ích khi text có nhiều ký hiệu, số, đơn vị, viết tắt, định dạng lạ, hoặc không giống ngôn ngữ tự nhiên chuẩn.

Từ ví dụ text-as-image, có thể mở rộng thành một hướng tổng quát hơn: **đổi không gian biểu diễn của input trước khi recommendation model học**. Item không nhất thiết phải là atomic ID; text không nhất thiết phải đi qua text encoder; multimodal content không nhất thiết phải fusion trực tiếp; Semantic IDs không nhất thiết chỉ là flat tokens. Mỗi loại input đều có thể được chuyển qua một encoder phù hợp hơn với bản chất dữ liệu và mục tiêu recommendation.

---

## 2. Vì sao hướng này đáng khai thác trong recommender systems?

Recommender systems truyền thống phụ thuộc rất nhiều vào item ID embedding. Cách này mạnh về memorization: nếu một item có nhiều interaction, model có thể học rất tốt ID đó. Nhưng item ID gần như không có nghĩa ngữ nghĩa. Hai item rất giống nhau nhưng có ID hoàn toàn khác sẽ không tự động chia sẻ thông tin. Điều này gây vấn đề lớn với long-tail item, cold-start item, item mới, và các corpus cực lớn.

Ngược lại, nếu dùng content encoder như text encoder, image encoder hoặc multimodal encoder, model có thể generalize tốt hơn vì item tương tự sẽ có embedding gần nhau. Tuy nhiên, content embedding thuần túy cũng có vấn đề: nó có thể thiếu khả năng memorization, hoặc không align tốt với collaborative signal. Một item có thể giống về nội dung nhưng không giống về hành vi người dùng. Ví dụ, “snacks” và “balloons” không giống nhau về semantic text/image, nhưng có thể thường được mua chung trong bối cảnh party. Vì vậy, hướng này không chỉ là “dùng encoder mạnh hơn”, mà là tìm cách encode input sao cho vừa giữ được semantic information, vừa phù hợp với recommendation objective.

Các paper được upload cho thấy nhiều biến thể của cùng một ý tưởng lớn:

```text
Không dùng raw ID/input trực tiếp.
Thay vào đó, tạo một representation trung gian giàu cấu trúc hơn:
- text-as-image representation
- OCR-text embedding
- content-derived Semantic ID
- multimodal Semantic ID
- recsys-native Semantic ID
- prefix-conditioned SID encoding
- direct multimodal item representation
```

Điểm chung là các paper đều xem **representation/tokenization/encoding** như một phần cốt lõi của recommender system, không phải bước tiền xử lý phụ.

---

## 3. Paper nền: Language Modelling with Pixels

Paper **Language Modelling with Pixels** đề xuất PIXEL, một mô hình language modeling không xử lý text bằng vocabulary/subword token như BERT, mà render text thành ảnh rồi dùng Vision Transformer kiểu masked autoencoder để học representation. Ý tưởng chính là loại bỏ vocabulary bottleneck. Với mô hình text truyền thống, mọi input phải đi qua một vocabulary hữu hạn: word, character, byte hoặc subword. Điều này tạo ra vấn đề với ngôn ngữ mới, script mới, noise, emoji, ký hiệu, code-switching, hoặc những dạng text không nằm tốt trong vocabulary.

PIXEL giải quyết bằng cách biến text thành ảnh. Text renderer vẽ chuỗi text lên một canvas RGB, sau đó chia ảnh thành patch và đưa vào ViT encoder. Pretraining không dự đoán token, mà reconstruct pixels của các masked patches. Khi finetune, decoder được thay bằng classification head hoặc task-specific head.

Điểm quan trọng đối với hướng nghiên cứu của bạn là: paper này chứng minh **input không nhất thiết phải được xử lý theo modality gốc**. Text có thể được xử lý như ảnh. Nếu text-as-image có thể giúp language model vượt qua vocabulary bottleneck và robust hơn với unseen script/noisy text, thì trong recommender systems, item description cũng có thể được xử lý không chỉ bằng text encoder truyền thống mà bằng OCR/vision encoder.

Đây là cơ sở rất mạnh cho lập luận:

```text
Representation space matters.
Changing the encoder or modality used to represent input can change generalization, robustness, and downstream behavior.
```

---

## 4. Paper gần nhất với hướng text-as-image trong recommendation: When Text-as-Vision Meets Semantic IDs

Paper **When Text-as-Vision Meets Semantic IDs in Generative Recommendation** là paper match trực tiếp nhất với hướng bạn đang nói. Paper này đặt vấn đề trong Generative Recommendation, nơi item được biểu diễn bằng Semantic IDs. Thông thường, Semantic IDs được tạo bằng cách dùng pretrained text encoder để encode item description, sau đó quantize embedding thành các discrete codes. Tuy nhiên, item description trong recommendation không giống văn bản tự nhiên chuẩn. Nó thường là dạng attribute-centric, nhiều số, đơn vị, ký hiệu, viết tắt, brand, model number, cấu hình sản phẩm.

Ví dụ một mô tả như:

```text
Wireless Mouse, 2.4G, 1600DPI, USB Receiver, Black
```

không phải một câu tự nhiên hoàn chỉnh. Text encoder truyền thống có thể tokenize rời rạc các ký hiệu như `2.4G`, `1600DPI`, `USB`, làm mất quan hệ thuộc tính. Paper lập luận rằng với loại dữ liệu này, render text thành ảnh rồi dùng OCR-based vision encoder có thể giữ tốt hơn cấu trúc biểu diễn.

Điểm đóng góp chính là paper không chỉ dùng text-as-vision cho NLP, mà đưa trực tiếp vào **Semantic ID learning cho recommendation**. Item description được render thành ảnh, sau đó OCR-based encoder tạo embedding, rồi embedding này được dùng để sinh Semantic IDs. Kết quả cho thấy OCR-text embedding có thể match hoặc vượt standard text embedding trong cả unimodal và multimodal generative recommendation.

Điểm rất đáng khai thác là paper cũng nói về **geometry alignment giữa text và image embedding**. Standard text embedding và image embedding có thể nằm trong những không gian hình học khác nhau, khiến multimodal fusion khó ổn định. OCR-text embedding, vì cũng được sinh từ vision/OCR encoder, có thể gần image embedding hơn về cấu trúc hình học. Điều này làm giảm modality gap trong multimodal recommendation. Nói cách khác, text-as-vision không chỉ giải quyết text tokenization, mà còn giúp fusion text-image dễ hơn.

Hướng này có thể được mở rộng thành:

```text
For item representation in recommendation, choosing the encoder/modality transformation may be as important as choosing the recommendation backbone.
```

---

## 5. Generative Retrieval và Semantic ID: đổi item ID thành token sequence có nghĩa

Paper **Recommender Systems with Generative Retrieval** giới thiệu TIGER, một framework generative retrieval cho sequential recommendation. Thay vì dùng ANN retrieval với query/item embeddings hoặc predict atomic item ID, TIGER biểu diễn mỗi item bằng một tuple Semantic ID, tức một chuỗi discrete tokens được tạo từ content embedding của item. User history cũng được biểu diễn bằng chuỗi Semantic IDs, sau đó Transformer seq2seq được train để autoregressively decode Semantic ID của item tiếp theo.

Điểm quan trọng là item không còn là một ID nguyên tử rời rạc. Một item như “orange shoes, brand X” và “orange shoes, brand Y” có thể chia sẻ một phần Semantic ID nếu chúng gần nhau về nội dung. Nhờ đó, model có thể transfer knowledge giữa các item tương tự. Đây là thay đổi rất lớn so với item ID embedding, nơi mỗi item là một token riêng không có cấu trúc ngữ nghĩa.

Với hướng của bạn, TIGER là paper nền cho luận điểm:

```text
Item representation can be transformed from atomic ID into a learned semantic token sequence.
The downstream recommender no longer consumes raw item IDs, but consumes encoded/discretized semantic representations.
```

Điểm này gần với NLP: giống như từ được token hóa thành subword, item cũng có thể được token hóa thành semantic codewords. Nhưng khác với text tokenizer, item tokenizer được tạo từ content embedding hoặc collaborative/semantic representation.

---

## 6. Semantic IDs for ranking: cân bằng memorization và generalization

Paper **Better Generalization with Semantic IDs: A Case Study in Ranking for Recommendations** nghiên cứu việc thay random item IDs trong ranking model bằng Semantic IDs ở quy mô công nghiệp, cụ thể là YouTube recommendation. Paper chỉ ra một điểm quan trọng: nếu thay item ID bằng content embedding trực tiếp, model có thể giảm chất lượng vì mất khả năng memorization. Ngược lại, random item ID có memorization tốt nhưng generalization kém cho item mới và long-tail item. Semantic ID được xem như một điểm cân bằng giữa hai cực này.

Semantic IDs trong paper được tạo từ frozen content embeddings bằng RQ-VAE. Chúng là các discrete item representations có tính hierarchical và compact. Paper đề xuất cách đưa Semantic IDs vào ranking model thông qua hashing các sub-pieces của Semantic ID sequences. Một điểm thú vị là họ dùng SentencePiece, một kỹ thuật tokenizer phổ biến trong LLM, để học các đoạn con của SID sequence thay vì chỉ dùng n-gram thủ công. Kết quả cho thấy Semantic IDs có thể thay thế video IDs, cải thiện generalization cho new/long-tail items mà không hy sinh overall quality.

Paper này quan trọng vì nó nhấn mạnh rằng representation trung gian phải giải quyết đồng thời hai nhu cầu:

```text
Memorization: nhớ được item cụ thể khi có nhiều data.
Generalization: chia sẻ thông tin cho item mới, item hiếm, item tương tự.
```

Đây là một luận điểm lớn nếu bạn muốn viết hướng nghiên cứu. Một encoder tốt trong recommender system không chỉ là encoder có embedding “semantic” đẹp, mà phải phù hợp với cả collaborative behavior và khả năng serving ở quy mô lớn.

---

## 7. Multimodal Generative Recommendation: encoder nào, modality nào cũng ảnh hưởng mạnh

Paper **Beyond Unimodal Boundaries: Generative Recommendation with Multimodal Semantics** nghiên cứu Multimodal Generative Recommendation. Paper lập luận rằng các hệ Generative Recommendation trước đó thường dùng semantic ID từ một modality, thường là text, trong khi dữ liệu thực tế thường multimodal: text, image, video, audio, metadata. Vì Semantic ID phụ thuộc mạnh vào modality và encoder được chọn, việc chỉ dùng một modality có thể bỏ sót nhiều thông tin.

Paper so sánh early fusion và late fusion. Early fusion dùng một multimodal encoder để encode nhiều modality cùng lúc, tạo một Semantic ID chung. Vấn đề là một modality có thể dominate, làm mất thông tin từ modality khác. Late fusion tạo Semantic IDs riêng cho từng modality rồi kết hợp sau, giúp giữ thông tin riêng biệt hơn, nhưng lại gặp vấn đề correspondence giữa các modality. Paper đề xuất MGR-LF++, một late fusion framework có contrastive modality alignment và special tokens để phân biệt ID sequence của từng modality.

Paper này rất quan trọng cho hướng của bạn vì nó cho thấy:

```text
Không chỉ có câu hỏi “có dùng encoder hay không”,
mà còn có câu hỏi “dùng encoder nào, modality nào, fusion ở đâu, và token nào đại diện cho modality nào”.
```

Nếu item có text, image, audio, video, metadata, mỗi modality có thể tạo ra một view khác nhau về item. Encoder không chỉ nén dữ liệu, mà còn quyết định semantic space của item. Một video có thể được encode theo visual style, audio mood, transcript content hoặc user engagement pattern. Mỗi encoder sẽ tạo ra một recommendation signal khác nhau.

---

## 8. Learning item representations directly from multimodal features

Paper **Learning Item Representations Directly from Multimodal Features for Effective Recommendation** đặt vấn đề với paradigm phổ biến trong multimodal recommendation: kết hợp item ID embedding với multimodal feature. Paper cho rằng các phương pháp này thường xem multimodal feature như thông tin phụ trợ cho ID embedding. Qua phân tích empirical và theoretical, paper chỉ ra có gradient bias trong quá trình học: multimodal feature component học nhanh và có informational gain mạnh hơn, trong khi ID embedding có thể học chậm hoặc suboptimal.

Từ đó, paper đề xuất LIRDRec, một mô hình học item representation trực tiếp từ multimodal features, không dựa vào item ID embeddings. LIRDRec dùng modality-specific encoders, multimodal transformation mechanism và progressive weight copying fusion để tổng hợp thông tin từ các modality khác nhau. Paper cũng khai thác MLLM/LLM bằng cách chuyển item image thành text rồi extract semantic embeddings.

Paper này hỗ trợ một hướng khác với Semantic ID: không nhất thiết phải discretize item thành tokens; có thể trực tiếp học representation từ encoded multimodal features. Điểm chung vẫn là không dùng raw ID làm trung tâm. Item representation được sinh từ feature encoder. Điều này đặc biệt hợp với cold-start recommendation, vì item mới vẫn có text/image/video/metadata để encode.

Hướng khai thác từ paper này:

```text
Should item ID embeddings remain the backbone of recommender systems,
or should item representations be generated directly from multimodal encoders?
```

---

## 9. Semantic IDs cho joint search và recommendation

Paper **Semantic IDs for Joint Generative Search and Recommendation** nghiên cứu cách xây dựng Semantic IDs dùng chung cho cả search và recommendation trong một generative model. Vấn đề là search và recommendation có objective khác nhau. Search cần align query-item relevance, còn recommendation cần align user preference và sequential/collaborative behavior. Nếu Semantic ID được tạo từ embedding tối ưu cho một task, nó có thể không generalize tốt sang task còn lại.

Paper so sánh nhiều chiến lược tạo Semantic IDs: task-specific, cross-task, multi-task, fused embeddings, separate token spaces, unified token spaces. Kết quả cho thấy dùng bi-encoder được fine-tune trên cả search và recommendation, sau đó tạo unified Semantic ID space, là trade-off tốt cho joint model.

Ý nghĩa với hướng nghiên cứu của bạn là: encoder không chỉ cần phù hợp với modality, mà còn phải phù hợp với **task objective**. Một Semantic ID tốt cho search chưa chắc tốt cho recommendation. Một image encoder tốt cho classification chưa chắc tốt cho collaborative recommendation. Một text encoder tốt cho sentence similarity chưa chắc tốt cho product recommendation.

Vì vậy, khi nói “dùng encoder khác thay input gốc”, cần bổ sung rằng encoder nên được đánh giá theo downstream task chứ không chỉ theo semantic quality.

---

## 10. ReSID: recsys-native encoding thay vì phụ thuộc vào LLM/foundation model

Paper **Rethinking Generative Recommender Tokenizer: Recsys-Native Encoding and Semantic Quantization Beyond LLMs** phê phán pipeline Semantic ID phổ biến: dùng foundation model để tạo item embedding, rồi dùng generic quantization như RQ-VAE hoặc hierarchical k-means để tạo Semantic IDs. Paper cho rằng pipeline này bị mismatch với mục tiêu generative recommendation.

Có hai vấn đề chính. Thứ nhất, embedding từ foundation model thường tối ưu semantic similarity, không nhất thiết align với collaborative prediction. Items thường được mua chung có thể không giống nhau về semantic. Thứ hai, quantization hiện tại thường không tối ưu cho autoregressive decoding. Một chuỗi Semantic ID tốt không chỉ cần reconstruct item tốt, mà còn phải dễ predict token-by-token. Nếu các code level có prefix-conditional uncertainty cao, generative model sẽ khó decode đúng item.

ReSID đề xuất hướng recsys-native hơn: Field-Aware Masked AutoEncoding để học item representation từ structured features theo hướng predictive-sufficient cho recommendation, và Globally Aligned Orthogonal Quantization để tạo SID sequence compact, giảm ambiguity và giảm uncertainty khi decode.

Paper này rất đáng khai thác vì nó đưa hướng nghiên cứu từ:

```text
Use existing foundation encoder for recommendation.
```

sang:

```text
Design encoders/tokenizers natively for recommendation objectives.
```

Đây là điểm quan trọng nếu muốn phát triển hướng riêng. Không chỉ thử image/audio/video/text encoder có sẵn, mà có thể đặt câu hỏi: **encoder nào phù hợp nhất cho recommender system?** Có thể cần một encoder không giống hoàn toàn NLP, CV hay audio encoder, mà được thiết kế riêng cho collaborative prediction, sequential predictability và item retrieval.

---

## 11. PrefixMem: Semantic IDs cũng cần encoder riêng như vision/audio

Paper **LLMs Need Encoders for Semantic IDs Too** đưa ra một lập luận rất gần với hướng của bạn. Trong multimodal LLMs, image không được đưa vào LLM bằng raw pixels; nó đi qua vision encoder. Audio cũng không chỉ là raw waveform; nó có audio codec/tokenizer/depth transformer. Paper lập luận rằng Semantic IDs trong generative recommendation cũng là một modality đặc biệt, và cũng cần encoder riêng.

Vấn đề là SID token có nghĩa phụ thuộc vào prefix context. Cùng một code ở level sau có thể mang nghĩa khác nhau nếu prefix khác. Nhưng nhiều hệ hiện tại lại đưa SID tokens vào LLM như flat vocabulary tokens. Điều này giống như bắt LLM tự học toàn bộ cấu trúc hierarchical của SID từ đầu, rất tốn data và khó ở billion-scale corpus.

PrefixMem giải quyết bằng một SID encoder dựa trên prefix n-gram memory tables. Với mỗi SID level, encoder tạo prefix-conditioned vector và cộng vào token embedding. Nhờ đó, cùng một token có representation khác nhau tùy prefix. Paper cho thấy cải thiện đáng kể SID accuracy và retrieval recall.

Paper này là bằng chứng rất rõ cho ý tưởng:

```text
Ngay cả khi đã có Semantic IDs, việc đưa tokens trực tiếp vào LLM vẫn chưa đủ.
Một representation trung gian hoặc encoder chuyên biệt vẫn có thể cần thiết.
```

Đây là hướng mở rộng hay: không chỉ encode raw item thành SID, mà còn encode SID sequence trước khi đưa vào generative model. Tức là có nhiều tầng encoding:

```text
raw item content -> Semantic ID tokenizer -> SID encoder -> LLM/generative recommender
```

---

## 12. Tóm tắt mapping các paper với hướng nghiên cứu

| Paper | Vai trò trong hướng nghiên cứu |
|---|---|
| **Language Modelling with Pixels** | Paper nền cho ý tưởng đổi modality: text được render thành image và xử lý bằng ViT, tránh vocabulary bottleneck. |
| **When Text-as-Vision Meets Semantic IDs** | Match trực tiếp với recommendation: item text được render thành image/OCR representation để tạo Semantic IDs tốt hơn cho GR. |
| **Recommender Systems with Generative Retrieval / TIGER** | Đổi item ID nguyên tử thành Semantic ID sequence, cho phép generative retrieval và generalization. |
| **Better Generalization with Semantic IDs** | Dùng Semantic IDs thay video IDs trong ranking ở quy mô công nghiệp, cân bằng memorization và generalization. |
| **Beyond Unimodal Boundaries** | Cho thấy modality choice và fusion strategy ảnh hưởng mạnh đến Generative Recommendation. |
| **LIRDRec** | Học item representation trực tiếp từ multimodal features thay vì dựa vào ID embeddings. |
| **Semantic IDs for Joint Search and Recommendation** | Encoder/Semantic ID cần align với nhiều task objective, không chỉ một task riêng lẻ. |
| **ReSID** | Đề xuất recsys-native encoder/tokenizer thay vì phụ thuộc hoàn toàn vào foundation model/LLM embeddings. |
| **PrefixMem** | SID tokens cũng cần encoder riêng vì meaning phụ thuộc prefix context, giống image/audio cần encoder trong multimodal LLM. |

---

## 13. Các hướng khai thác có thể viết thành proposal

### 13.1. Input transformation for recommendation

Hướng tổng quát nhất là nghiên cứu ảnh hưởng của việc biến đổi input trước khi đưa vào recommender:

```text
raw text -> text encoder
raw text -> rendered image -> OCR/vision encoder
image -> vision encoder
video -> video encoder
audio -> audio encoder
metadata -> field-aware encoder
item ID -> Semantic ID tokenizer
SID sequence -> prefix-conditioned SID encoder
```

Research question:

```text
Biểu diễn nào giúp recommender model generalize tốt nhất cho cold-start, long-tail và multimodal items?
```

### 13.2. Text-as-image for item representation

Đây là nhánh cụ thể và rõ ràng nhất. Nhiều item description trong e-commerce không phải văn bản tự nhiên, mà là chuỗi thuộc tính. Có thể render chúng thành image và dùng OCR/vision encoder để lấy embedding. Sau đó embedding được dùng cho:

```text
ranking model
sequential recommendation
Semantic ID generation
generative retrieval
cold-start recommendation
```

Điểm mới có thể là phân tích loại item nào hưởng lợi nhiều nhất: sản phẩm nhiều số/ký hiệu, item có thông số kỹ thuật, item đa ngôn ngữ, item có noisy title, item có brand/model code.

### 13.3. Modality-specific Semantic IDs

Thay vì tạo một Semantic ID duy nhất từ text, có thể tạo nhiều SID theo modality:

```text
SID_text
SID_image
SID_audio
SID_video
SID_metadata
SID_collaborative
```

Sau đó model học cách fuse hoặc decode các SID này. Đây là hướng liên quan đến MGR-LF++.

Một câu hỏi hay:

```text
Nên fuse modalities trước khi quantize hay sau khi quantize?
```

Early fusion có thể mất thông tin modality yếu; late fusion giữ thông tin riêng nhưng cần alignment. Đây là khoảng trống đáng khai thác.

### 13.4. Recsys-native tokenizer

Một hướng sâu hơn là không dùng encoder có sẵn từ NLP/CV, mà thiết kế encoder/tokenizer riêng cho recommendation. Thay vì tối ưu semantic similarity, encoder tối ưu:

```text
collaborative predictability
sequential predictability
low prefix uncertainty
cold-start transfer
retrieval efficiency
```

Đây là hướng ReSID gợi mở. Nó đặc biệt đáng chú ý vì Semantic ID không chỉ là representation, mà còn là **vocabulary của generative recommender**. Nếu tokenization sai, autoregressive model sẽ khó học.

### 13.5. Encoder for encoded tokens

PrefixMem mở ra ý tưởng rằng sau khi đã tokenize item thành SID, vẫn cần một encoder khác để contextualize token. Tức là không chỉ encode raw item, mà còn encode token sequence:

```text
Semantic ID tokens -> prefix-aware SID encoder -> LLM
```

Điều này có thể mở rộng cho nhiều dạng token khác: audio codec tokens, video tokens, image patch tokens, product attribute tokens. Câu hỏi là:

```text
Khi nào flat token embedding là không đủ?
Khi nào cần encoder chuyên biệt cho token type đó?
```

---

## 14. So sánh với hướng time prediction trước đó

Hai hướng này có thể được giữ riêng, nhưng cũng có thể kết hợp. Hướng trước tập trung vào **đổi target**: từ item sang time. Hướng này tập trung vào **đổi input representation**: từ raw input sang encoded/transformed representation.

Nếu kết hợp, có thể viết thành:

```text
Item-conditioned time prediction using encoded multimodal item representations.
```

Tức là model không predict thời điểm mua dựa trên item ID thô, mà dựa trên item representation được sinh từ text-as-image, image/audio/video encoders, Semantic IDs hoặc recsys-native tokenizer.

Ví dụ:

```text
raw item description -> rendered image -> OCR encoder -> Semantic ID
user history + candidate item SID -> time distribution
```

Hoặc:

```text
image/text/video/audio -> multimodal item encoder -> item component representation
user history + item representation -> purchase timing curve
```

Sự kết hợp này khá mạnh vì nó vừa có novelty về output, vừa có novelty về input representation.

---

## 15. Các điểm cần cẩn thận

### 15.1. Encoder mạnh chưa chắc tốt cho recommendation

Một encoder tốt cho NLP/CV chưa chắc tạo embedding tốt cho recommendation. Semantic similarity và collaborative similarity không giống nhau. Ví dụ, snacks và balloons khác semantic nhưng có thể cùng xuất hiện trong party context. Vì vậy cần đánh giá encoder theo downstream recommendation metric.

### 15.2. Content-only representation có thể mất memorization

Nếu bỏ item ID hoàn toàn, model có thể generalize tốt hơn nhưng mất khả năng nhớ các item phổ biến/có đặc điểm riêng. Semantic ID hoặc hybrid design có thể là compromise tốt hơn.

### 15.3. Multimodal fusion dễ bị modality dominance

Nếu fusion quá sớm, một modality mạnh có thể lấn át modality khác. Nếu fusion quá muộn, cần alignment tốt giữa các modality. Đây là trade-off quan trọng.

### 15.4. Tokenization ảnh hưởng đến autoregressive decoding

Trong generative recommendation, SID sequence không chỉ là mã hóa item. Nó là target sequence mà model phải decode. Vì vậy tokenization cần tối ưu cả reconstruction/semantic quality lẫn predictability.

### 15.5. Serving cost và latency

Vision encoder, video encoder, audio encoder, MLLM hoặc OCR encoder có thể tốn kém. Trong production, có thể cần precompute item representations, cache Semantic IDs, hoặc dùng lightweight encoder.

---

## 16. Bộ contribution có thể viết

Một proposal theo hướng này có thể viết contribution như sau:

```text
1. We study input representation transformation for recommendation,
   where raw item inputs are not consumed directly but encoded through modality-specific or task-specific encoders.

2. We analyze how different encoders, including text encoders, OCR/vision encoders, multimodal encoders,
   and Semantic ID tokenizers, affect recommendation generalization and cold-start performance.

3. We propose a unified framework that represents items through encoded semantic tokens or multimodal embeddings
   before feeding them into sequential/generative recommendation models.

4. We investigate whether text-as-image and OCR-based encoders provide better item representations
   for symbolic, attribute-centric product descriptions than standard text encoders.

5. We discuss how recsys-native encoders/tokenizers can better align semantic representation with collaborative
   and sequential recommendation objectives.
```

---

## 17. Một formulation gợi ý

Có thể mô tả mô hình tổng quát như sau:

```text
x_i = raw item input
z_i = E(x_i)
y_i = Q(z_i)
```

Trong đó:

```text
E = encoder / modality transformer
Q = quantizer / tokenizer / Semantic ID generator
z_i = continuous representation
y_i = discrete semantic token sequence
```

Sau đó recommender model học:

```text
P(y_next | y_history, user_context)
```

hoặc nếu dùng continuous representation:

```text
score(user, item) = f(user_history_encoder(E(x_history)), E(x_item))
```

Nếu kết hợp với hướng time prediction:

```text
P(time | user_history, E(x_item))
```

hoặc:

```text
λ_user,item(t | E_text_as_image(x_item), E_image(x_item), E_audio(x_item), E_video(x_item))
```

---

## 18. Kết luận định hướng

Nhóm paper này cho thấy một xu hướng rõ ràng: phần **encoding/tokenization của item** đang trở thành một thành phần trung tâm của recommender systems hiện đại. Trước đây, recommender systems thường tập trung vào backbone như MF, two-tower, GRU, Transformer, GNN hoặc LLM. Nhưng trong generative recommendation và multimodal recommendation, cách biểu diễn item có thể quyết định mạnh đến khả năng generalization, cold-start, long-tail retrieval, multimodal fusion và autoregressive decoding.

Hướng bạn đang xét có thể phát biểu là:

```text
Thay vì cải tiến recommender backbone trực tiếp,
ta cải tiến cách raw input được biến đổi thành representation/token mà recommender backbone sử dụng.
```

Paper text-as-image chứng minh text có thể được encode qua vision space. Paper Semantic ID chứng minh item ID có thể được thay bằng semantic token sequence. Paper multimodal GR chứng minh modality choice và fusion strategy ảnh hưởng mạnh đến kết quả. Paper ReSID và PrefixMem cho thấy encoding/tokenization nên được thiết kế riêng cho recommendation, không chỉ mượn từ NLP/CV.

Đây là một hướng rất đáng khai thác vì nó nằm ở giao điểm của representation learning, multimodal learning, generative recommendation và industrial retrieval. Nếu phát triển tiếp, có thể tập trung vào một trong hai nhánh: **text-as-image/OCR representation cho item descriptions**, hoặc **recsys-native semantic tokenizer/encoder cho generative recommendation**. Nhánh đầu dễ giải thích và có cơ sở trực tiếp từ paper 2601.14697; nhánh thứ hai sâu hơn và có tiềm năng research mạnh hơn nếu muốn đóng góp về mô hình.
