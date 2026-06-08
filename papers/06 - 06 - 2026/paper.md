## Nhận xét chi tiết về xu hướng nghiên cứu Recommender Systems giai đoạn 01/2025–06/2026

Dựa trên tập hơn 1,600 paper về **Recommender Systems** trong giai đoạn từ tháng 01/2025 đến 06/2026, có thể thấy lĩnh vực này đang dịch chuyển mạnh từ các mô hình recommendation truyền thống sang các hướng tiếp cận dựa trên **LLM**, **foundation model**, **generative model**, **multimodal learning** và **trustworthy AI**.

### Bảng 1. Các cụm xu hướng nổi bật trong corpus

| Cụm xu hướng | Số paper gần đúng | Nội dung chính |
|---|---:|---|
| **LLM / Foundation Model / GenAI for RS** | ~380+ | Dùng LLM để hiểu user profile, item description, review, metadata, sinh recommendation, rerank hoặc giải thích đề xuất. |
| **Sequential / Temporal / Session-based RS** | ~280 | Mô hình hóa chuỗi hành vi của user, next-item prediction, session-based recommendation và preference thay đổi theo thời gian. |
| **Generative / Diffusion Recommendation** | ~260 | Chuyển recommendation từ bài toán scoring/ranking sang bài toán sinh đề xuất, sinh item representation hoặc sinh user preference. |
| **Multimodal RS** | ~200 | Kết hợp text, image, video, audio, metadata và interaction log để biểu diễn user/item tốt hơn. |
| **Trustworthy RS** | ~180 | Tập trung vào fairness, bias, privacy, explainability, robustness, calibration và độ tin cậy khi triển khai. |
| **Graph / GNN / Knowledge Graph** | ~170 | Khai thác user-item graph, item-item graph, social graph hoặc knowledge graph để học quan hệ phức tạp. |
| **RAG / Retrieval / Ranking / Reranking** | ~145 | Kết hợp retrieval, RAG, reranking và LLM để tăng chất lượng đề xuất và giảm hallucination. |
| **Agentic / Conversational RS** | ~140 | Xây dựng recommender dạng agent hoặc hội thoại, có khả năng hỏi lại, ghi nhớ, lập kế hoạch và giải thích. |
| **Efficiency / Scalability** | ~125 | Tối ưu chi phí, tốc độ inference, training, serving, compression và deployment cho RS quy mô lớn. |
| **Cold-start / Long-tail / Sparsity** | ~65 | Giải quyết thiếu dữ liệu user/item, item mới, user mới, sparse interaction và long-tail recommendation. |

---

## 1. LLM / Foundation Model / GenAI for Recommender Systems

Nhóm này là xu hướng lớn nhất trong corpus. Các paper thuộc nhóm này thường không chỉ dùng mô hình recommendation truyền thống để học embedding user-item, mà tận dụng **Large Language Models** hoặc **foundation models** để hiểu sâu hơn về ngữ cảnh recommendation.

### Nhóm này đang làm gì?

Các nghiên cứu thường tập trung vào các bài toán sau:

| Hướng nhỏ | Đang làm về cái gì? |
|---|---|
| **LLM as feature extractor** | Dùng LLM để biến item description, review, profile, title, metadata thành embedding giàu ngữ nghĩa hơn. |
| **LLM as recommender** | Đưa lịch sử user và candidate items vào prompt, sau đó yêu cầu LLM sinh hoặc chọn item phù hợp. |
| **LLM as reranker** | Mô hình truyền thống sinh candidate trước, sau đó LLM sắp xếp lại danh sách đề xuất. |
| **LLM for explanation** | Dùng LLM để sinh lời giải thích tự nhiên: vì sao item này được đề xuất cho user. |
| **LLM for user profiling** | Tóm tắt sở thích user từ lịch sử tương tác, review, click, rating hoặc hội thoại. |
| **LLM for cold-start** | Dùng thông tin text/metadata để đề xuất cho user mới hoặc item mới khi chưa có nhiều interaction. |
| **LLM + small recommender model** | Kết hợp LLM với mô hình nhỏ hơn để cân bằng giữa accuracy và chi phí triển khai. |

### Ý nghĩa của trend này

Trước đây, RS chủ yếu dựa vào interaction như click, rating, purchase. Vấn đề là interaction thường sparse và khó biểu diễn sở thích phức tạp. LLM giúp hệ thống hiểu được thông tin ngữ nghĩa như mô tả sản phẩm, nội dung review, mục đích người dùng và các ràng buộc trong ngôn ngữ tự nhiên.

Nói ngắn gọn, hướng này biến recommendation từ bài toán:

> “User này giống user nào, item này giống item nào?”

thành bài toán:

> “User này thật sự đang cần gì, trong ngữ cảnh nào, và item nào phù hợp nhất?”

---

## 2. Generative / Diffusion Recommendation

Đây là một trong các hướng tăng mạnh trong giai đoạn 2025–2026. Thay vì chỉ tính điểm cho từng item rồi sắp xếp, các paper thuộc nhóm này xem recommendation như một bài toán **generation**.

### Nhóm này đang làm gì?

| Hướng nhỏ | Đang làm về cái gì? |
|---|---|
| **Generative recommendation** | Sinh trực tiếp item, danh sách item, hoặc token đại diện cho item. |
| **Semantic ID generation** | Biến item thành semantic token/ID để mô hình sinh item giống như sinh ngôn ngữ. |
| **Diffusion-based recommendation** | Dùng diffusion model để khử nhiễu preference, sinh embedding hoặc sinh candidate item. |
| **Generative reranking** | Sinh lại thứ tự item dựa trên ngữ cảnh user và candidate list. |
| **Counterfactual generation** | Sinh dữ liệu phảnfactual để giảm bias hoặc cải thiện fairness. |
| **Negative sample generation** | Sinh negative samples chất lượng cao để training mô hình ranking tốt hơn. |

### Ý nghĩa của trend này

Recommendation truyền thống thường bị giới hạn bởi candidate item có sẵn. Generative recommendation mở ra hướng mô hình có thể học phân phối sở thích của user và sinh ra item phù hợp hơn theo ngữ cảnh.

Đặc biệt, diffusion model được quan tâm vì có khả năng mô hình hóa phân phối phức tạp, khử nhiễu dữ liệu sparse và tạo biểu diễn ổn định hơn cho user-item preference.

---

## 3. Sequential / Temporal / Session-based Recommendation

Đây vẫn là nhóm rất quan trọng trong RS. Các paper thuộc nhóm này tập trung vào việc hiểu **thứ tự hành vi** và **sự thay đổi sở thích theo thời gian**.

### Nhóm này đang làm gì?

| Hướng nhỏ | Đang làm về cái gì? |
|---|---|
| **Sequential recommendation** | Dự đoán item tiếp theo dựa trên chuỗi hành vi trước đó của user. |
| **Session-based recommendation** | Đề xuất trong một phiên ngắn, ví dụ user vừa xem 5 sản phẩm thì nên gợi ý gì tiếp. |
| **Temporal recommendation** | Mô hình hóa yếu tố thời gian: giờ, ngày, mùa, xu hướng, recency. |
| **Short-term vs long-term preference** | Tách sở thích nhất thời và sở thích dài hạn của user. |
| **Dynamic preference modeling** | Theo dõi việc sở thích user thay đổi liên tục. |
| **Lifelong sequential behavior** | Học hành vi dài hạn của user qua nhiều giai đoạn. |

### Ý nghĩa của trend này

Trong thực tế, sở thích user không cố định. Một user có thể thích laptop gaming trong hôm nay, nhưng tuần sau lại tìm laptop văn phòng. Với video, music, news hoặc e-commerce, thứ tự hành vi rất quan trọng.

Ví dụ:

- User vừa xem nhiều video nấu ăn thì recommendation tiếp theo nên liên quan đến nấu ăn.
- User vừa tìm laptop dưới 20 triệu thì hệ thống không nên đề xuất laptop 60 triệu.
- User nghe nhạc buồn vào buổi tối nhưng nghe nhạc sôi động khi tập thể dục.

Vì vậy, sequential và temporal RS vẫn là phần lõi trong các hệ thống recommendation hiện đại.

---

## 4. Multimodal Recommender Systems

Multimodal RS tăng mạnh vì nhiều item hiện nay không chỉ có ID hoặc rating, mà có nhiều loại dữ liệu khác nhau như text, image, video, audio và metadata.

### Nhóm này đang làm gì?

| Hướng nhỏ | Đang làm về cái gì? |
|---|---|
| **Text-based representation** | Dùng title, description, review, comment để biểu diễn item/user. |
| **Image-based recommendation** | Dùng ảnh sản phẩm, ảnh thời trang, ảnh món ăn để học đặc trưng thị giác. |
| **Video recommendation** | Kết hợp frame, caption, âm thanh, hành vi xem video cho short-video hoặc micro-video RS. |
| **Audio/music recommendation** | Khai thác âm thanh, lyrics, mood, genre cho music recommendation. |
| **Cross-modal alignment** | Căn chỉnh embedding giữa text-image-video để biểu diễn item thống nhất. |
| **Dynamic multimodal fusion** | Tự động quyết định modality nào quan trọng hơn trong từng trường hợp. |
| **Incomplete multimodal recommendation** | Xử lý trường hợp item thiếu ảnh, thiếu mô tả hoặc thiếu modality nào đó. |

### Ý nghĩa của trend này

Multimodal RS giúp giải quyết tốt hơn các bài toán như **cold-start item** và **sparse interaction**. Ví dụ, một sản phẩm mới chưa có lượt mua vẫn có thể được đề xuất nếu ảnh, mô tả và metadata của nó giống với sản phẩm user từng thích.

Trong các domain như fashion, food, movie, short video hoặc music, multimodal feature đặc biệt quan trọng vì sở thích user không chỉ đến từ nội dung text mà còn từ hình ảnh, âm thanh, phong cách và cảm xúc.

---

## 5. Trustworthy Recommender Systems

Nhóm này tập trung vào việc làm cho recommender system không chỉ chính xác mà còn **công bằng, minh bạch, an toàn và đáng tin cậy**.

### Nhóm này đang làm gì?

| Hướng nhỏ | Đang làm về cái gì? |
|---|---|
| **Fairness-aware recommendation** | Đảm bảo recommendation không thiên vị một nhóm user hoặc item provider. |
| **Popularity bias reduction** | Giảm việc hệ thống chỉ đề xuất item phổ biến và bỏ qua long-tail items. |
| **Gender / demographic bias** | Kiểm tra và giảm thiên vị theo giới tính, độ tuổi hoặc nhóm người dùng. |
| **Explainable recommendation** | Giải thích vì sao một item được đề xuất. |
| **Privacy-preserving recommendation** | Bảo vệ dữ liệu user, thường kết hợp federated learning hoặc differential privacy. |
| **Robust recommendation** | Chống nhiễu, attack, fake profile, adversarial behavior hoặc data poisoning. |
| **Calibration** | Đảm bảo phân phối item đề xuất phù hợp với sở thích thật của user. |

### Ý nghĩa của trend này

Một recommender system có accuracy cao vẫn có thể gây vấn đề. Ví dụ:

- Luôn đề xuất item phổ biến, làm item mới không có cơ hội được nhìn thấy.
- Đề xuất job hoặc nội dung bị bias theo giới tính.
- Dùng dữ liệu cá nhân quá mức mà không đảm bảo privacy.
- Dễ bị tấn công bằng fake interaction.
- Không giải thích được vì sao user nhận recommendation đó.

Vì vậy, RS hiện đại không chỉ tối ưu Recall@K hoặc NDCG@K, mà còn phải xét fairness, robustness, explainability, privacy và long-term user satisfaction.

---

## 6. Graph / GNN / Knowledge Graph Recommendation

Graph-based recommendation vẫn là một hướng quan trọng, nhưng trong corpus này tỷ trọng có xu hướng giảm tương đối so với LLM và generative recommendation.

### Nhóm này đang làm gì?

| Hướng nhỏ | Đang làm về cái gì? |
|---|---|
| **User-item graph** | Biểu diễn user và item dưới dạng graph hai phía, cạnh là interaction. |
| **Item-item graph** | Học quan hệ giữa các item dựa trên đồng mua, đồng xem hoặc similarity. |
| **Social graph recommendation** | Dùng quan hệ bạn bè, follow, trust network để cải thiện recommendation. |
| **Knowledge graph recommendation** | Kết hợp entity như actor, brand, category, author, topic để hiểu quan hệ ngữ nghĩa. |
| **Graph contrastive learning** | Tạo nhiều view của graph để học embedding ổn định hơn. |
| **Heterogeneous graph** | Mô hình hóa nhiều loại node và edge khác nhau. |
| **Graph + LLM** | Dùng graph để cung cấp cấu trúc tri thức cho LLM-based recommender. |

### Ý nghĩa của trend này

Graph rất phù hợp với RS vì dữ liệu recommendation tự nhiên là mạng quan hệ giữa user, item, category, brand, tag, review và interaction. Tuy nhiên, thay vì chỉ cải tiến GNN encoder, nhiều nghiên cứu gần đây chuyển sang kết hợp graph với LLM, RAG, multimodal feature hoặc fairness.

Điều này cho thấy graph không biến mất, nhưng vai trò của nó đang chuyển từ “mô hình chính” sang “nguồn cấu trúc tri thức hỗ trợ” cho các hệ thống lớn hơn.

---

## 7. RAG / Retrieval / Ranking / Reranking

Nhóm này nổi lên cùng với LLM-based recommendation. Vì LLM không thể nhớ toàn bộ item catalog hoặc lịch sử user, các hệ thống cần retrieval để lấy thông tin liên quan trước khi sinh hoặc rerank recommendation.

### Nhóm này đang làm gì?

| Hướng nhỏ | Đang làm về cái gì? |
|---|---|
| **Candidate retrieval** | Lấy ra một tập item ứng viên từ catalog lớn trước khi ranking. |
| **RAG-enhanced recommendation** | Truy xuất user history, item knowledge, review hoặc knowledge graph để đưa vào LLM. |
| **LLM reranking** | Dùng LLM để sắp xếp lại danh sách item ứng viên. |
| **Retriever optimization** | Tối ưu retriever để lấy đúng item liên quan và giảm latency. |
| **Knowledge graph RAG** | Truy xuất thông tin từ knowledge graph để tăng tính chính xác và giải thích. |
| **Poisoning / robustness in RAG** | Nghiên cứu rủi ro dữ liệu độc hại hoặc nhiễu trong retrieval. |

### Ý nghĩa của trend này

Với catalog lớn, không thể yêu cầu LLM đánh giá tất cả item. Do đó, pipeline phổ biến là:

1. Retriever lấy candidate items.
2. Ranking model hoặc LLM reranker sắp xếp lại.
3. LLM sinh giải thích hoặc phản hồi hội thoại.

RAG giúp recommendation có căn cứ hơn, giảm hallucination và cho phép hệ thống dùng thông tin mới mà không cần train lại toàn bộ LLM.

---

## 8. Agentic / Conversational Recommender Systems

Nhóm này phát triển mạnh nhờ LLM. Thay vì recommendation là một lần gợi ý cố định, hệ thống có thể trở thành một **agent** tương tác nhiều lượt với user.

### Nhóm này đang làm gì?

| Hướng nhỏ | Đang làm về cái gì? |
|---|---|
| **Conversational recommendation** | Hệ thống hỏi đáp với user để hiểu nhu cầu trước khi đề xuất. |
| **Agent-based recommendation** | Xây dựng agent có memory, tool use, planning và reasoning. |
| **User preference elicitation** | Hỏi thêm thông tin khi preference chưa rõ. |
| **Interactive refinement** | User phản hồi “rẻ hơn”, “ít phổ biến hơn”, “gần đây hơn”, hệ thống cập nhật recommendation. |
| **Long-term memory** | Agent ghi nhớ sở thích user qua nhiều phiên. |
| **User simulation / evaluation** | Dùng LLM mô phỏng user để đánh giá conversational RS. |

### Ý nghĩa của trend này

Recommendation truyền thống thường giả định hệ thống đã biết đủ về user. Nhưng trong thực tế, nhiều nhu cầu chỉ xuất hiện trong hội thoại.

Ví dụ:

- “Tôi muốn một laptop để học AI, giá vừa phải, không quá nặng.”
- “Tôi muốn phim giống phim này nhưng nhẹ nhàng hơn.”
- “Tôi cần nhà hàng yên tĩnh cho gia đình, không quá đắt.”

Agentic/conversational RS giúp hệ thống hỏi lại, làm rõ nhu cầu, giải thích lựa chọn và điều chỉnh recommendation theo phản hồi của user.

---

## 9. Efficiency / Scalability

Khi LLM, multimodal model và generative model được đưa vào RS, chi phí tính toán tăng mạnh. Vì vậy, nhiều paper tập trung vào hiệu quả triển khai.

### Nhóm này đang làm gì?

| Hướng nhỏ | Đang làm về cái gì? |
|---|---|
| **Efficient inference** | Giảm chi phí khi dùng LLM hoặc model lớn để recommend. |
| **Early exit** | Cho phép mô hình dừng sớm nếu đã đủ tự tin. |
| **Model compression** | Nén model để giảm memory và latency. |
| **Knowledge distillation** | Dùng model lớn dạy model nhỏ hơn. |
| **On-device recommendation** | Chạy một phần recommendation trên thiết bị người dùng. |
| **Streaming retrieval** | Cập nhật index/retriever theo thời gian thực. |
| **Scalable training infrastructure** | Tối ưu training và serving cho hệ thống quy mô lớn. |

### Ý nghĩa của trend này

Trong môi trường production, một model recommendation không chỉ cần chính xác mà còn phải nhanh, rẻ và ổn định. LLM-based RS có thể cho kết quả tốt nhưng nếu inference quá chậm hoặc quá đắt thì khó triển khai.

Do đó, các nghiên cứu về efficiency thường cố gắng cân bằng giữa:

- Accuracy
- Latency
- Memory
- Throughput
- Cost
- Scalability

---

## 10. Cold-start / Long-tail / Sparsity

Đây là vấn đề kinh điển của RS nhưng vẫn tiếp tục được nghiên cứu, đặc biệt trong bối cảnh LLM và multimodal model.

### Nhóm này đang làm gì?

| Hướng nhỏ | Đang làm về cái gì? |
|---|---|
| **User cold-start** | Đề xuất cho user mới khi chưa có nhiều lịch sử tương tác. |
| **Item cold-start** | Đề xuất item mới khi chưa có click, rating hoặc purchase. |
| **Sparse interaction** | Xử lý dữ liệu user-item rất thưa. |
| **Long-tail recommendation** | Tăng cơ hội xuất hiện cho item ít phổ biến. |
| **Content-enhanced recommendation** | Dùng text, image, metadata để bù cho thiếu interaction. |
| **LLM for cold-start** | Dùng LLM suy luận từ description/profile thay vì phụ thuộc hoàn toàn vào interaction. |

### Ý nghĩa của trend này

Cold-start và long-tail là vấn đề rất thực tế. Một nền tảng e-commerce, movie, music hoặc news luôn có item mới, user mới và nhiều item ít tương tác. Nếu chỉ dựa vào collaborative filtering, hệ thống dễ bỏ qua các item này.

LLM và multimodal model giúp giảm vấn đề này vì chúng có thể hiểu nội dung item ngay cả khi chưa có interaction.

---

## Bảng 2. Thay đổi tỷ trọng chủ đề theo thời gian

| Chủ đề | 2025 H1 | 2025 H2 | 2026 Jan–May | Xu hướng |
|---|---:|---:|---:|---|
| **LLM / Foundation Model / GenAI** | 24.4% | 23.3% | 21.2% | Vẫn là hướng lớn nhất, nhưng có dấu hiệu bão hòa tương đối. |
| **Generative / Diffusion Recommendation** | 12.3% | 15.6% | 19.6% | Tăng mạnh, đặc biệt trong năm 2026. |
| **Sequential / Temporal Recommendation** | 16.1% | 14.4% | 16.8% | Ổn định, tiếp tục là hướng lõi của RS. |
| **Graph / GNN / Knowledge Graph** | 13.2% | 7.5% | 6.2% | Giảm tỷ trọng, không còn là trend độc lập nổi bật nhất. |
| **Trustworthy RS** | 11.2% | 10.8% | 11.0% | Ổn định, phản ánh nhu cầu triển khai RS an toàn và công bằng. |
| **Agentic / Conversational RS** | 5.9% | 6.9% | 8.8% | Tăng nhờ sự phát triển của LLM và interactive systems. |
| **Efficiency / Scalability** | 6.2% | 8.2% | 8.8% | Tăng do nhu cầu triển khai LLM4Rec hiệu quả hơn. |
| **Cold-start / Long-tail / Sparsity** | 2.9% | 4.3% | 5.2% | Tăng nhẹ, thường gắn với LLM và multimodal recommendation. |

> Lưu ý: Tháng 06/2026 có ít paper hơn các tháng trước trong corpus, nên khi phân tích xu hướng theo thời gian nên ưu tiên so sánh đến **05/2026** để tránh sai lệch.

---

## Tổng kết nhận xét

Tổng thể, Recommender Systems trong giai đoạn 2025–2026 đang đi theo bốn hướng lớn.

| Hướng lớn | Nội dung chính |
|---|---|
| **LLM / Foundation Model-based RS** | Tăng khả năng hiểu ngữ cảnh, user profile, item content và sinh giải thích tự nhiên. |
| **Generative Recommendation** | Chuyển từ scoring/ranking item sang sinh recommendation, semantic item ID hoặc preference representation. |
| **Conversational / Agentic RS** | Phát triển recommender có khả năng tương tác, hỏi lại, ghi nhớ và điều chỉnh đề xuất theo phản hồi. |
| **Trustworthy and Practical RS** | Mở rộng mục tiêu từ accuracy sang fairness, privacy, robustness, explainability, efficiency và scalability. |

Như vậy, xu hướng hiện tại không còn chỉ tập trung vào việc cải thiện một vài điểm Recall@K hoặc NDCG@K. Thay vào đó, các nghiên cứu đang cố gắng xây dựng recommender systems có khả năng hiểu ngữ cảnh tốt hơn, tận dụng dữ liệu đa phương thức, tương tác tự nhiên với người dùng, sinh đề xuất linh hoạt và đảm bảo tính đáng tin cậy khi triển khai thực tế.

Nếu lựa chọn hướng nghiên cứu theo mức độ thời sự, các chủ đề tiềm năng nhất gồm:

1. **LLM-based recommender systems**
2. **RAG-enhanced recommendation**
3. **Generative / diffusion-based recommendation**
4. **Conversational and agentic recommender systems**
5. **Trustworthy LLM-based recommendation**
6. **Multimodal recommendation for cold-start and sparse data**