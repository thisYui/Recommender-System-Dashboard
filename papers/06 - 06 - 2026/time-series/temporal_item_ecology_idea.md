# Temporal Item Ecology — Quan hệ cạnh tranh, hợp tác và cộng sinh giữa các item trong recommender systems

## 1. Động cơ

Trong recommender systems truyền thống, item thường được nhìn qua hai lăng kính chính. Lăng kính thứ nhất là **collaborative filtering**, tức item giống nhau nếu chúng được tương tác bởi các nhóm user giống nhau, hoặc thường xuất hiện gần nhau trong lịch sử hành vi của user. Lăng kính thứ hai là **content/semantic similarity**, tức item giống nhau nếu chúng có category, mô tả, hình ảnh, thương hiệu hoặc thuộc tính tương tự.

Tuy nhiên, hai lăng kính này có thể bỏ sót một loại quan hệ khác: **quan hệ hành vi theo thời gian**. Hai item có thể không liên quan rõ ràng theo collaborative filtering, không cùng category, không có nhiều user overlap, nhưng vẫn có **biểu đồ nhu cầu theo thời gian giống nhau**. Ngược lại, hai item có thể thường xuất hiện cùng user nhưng lại có tác động cạnh tranh hoặc thay thế nhau theo thời gian.

Điều này gợi ý một cách nhìn mới: item trong recommender systems có thể được xem như các “thực thể” trong một hệ sinh thái. Trong hệ sinh thái đó, item không tồn tại độc lập, mà có thể cạnh tranh, hỗ trợ, kích hoạt, thay thế hoặc cộng sinh với nhau. Tương tự sinh học, nơi số lượng loài ăn thịt và ăn cỏ ảnh hưởng lẫn nhau qua chu kỳ, các item cũng có thể ảnh hưởng đến demand, exposure, attention hoặc purchase probability của nhau theo thời gian.

Ý tưởng trung tâm:

```text
Collaborative filtering học quan hệ ai-tương-tác-với-gì.
Content-based learning học item-nói-về-cái-gì.
Temporal item ecology học item-ảnh-hưởng-lẫn-nhau-theo-thời-gian-như-thế-nào.
```

---

## 2. Vấn đề với collaborative filtering truyền thống

Collaborative filtering mạnh ở việc học quan hệ dựa trên user overlap. Nếu nhiều user cùng tương tác item A và item B, hệ thống sẽ xem A và B có liên quan. Nhưng CF có một giới hạn: nó thường không mô hình hóa rõ **dynamics theo thời gian** giữa item.

Ví dụ, hai item có thể không có nhiều user overlap:

```text
Item A: áo khoác mùa đông
Item B: máy sưởi mini
```

Chúng có thể thuộc category khác nhau và không thường được mua bởi cùng user trong cùng một sequence. Nhưng demand của cả hai đều tăng vào mùa lạnh. Như vậy, chúng có thể không gần trong collaborative space nhưng lại gần trong **temporal demand space**.

Ngược lại, hai item có thể gần trong CF vì cùng được nhiều user xem, nhưng về mặt kinh tế/hành vi lại là **substitute**. Ví dụ:

```text
Item A: laptop gaming model X
Item B: laptop gaming model Y
```

User có thể xem cả hai, nhưng khi mua A thì xác suất mua B giảm. Trong CF, A và B có thể gần nhau; nhưng trong temporal causal/demand dynamics, chúng có thể cạnh tranh.

Vì vậy, CF không đủ để trả lời các câu hỏi như:

```text
- Item nào có demand curve giống nhau?
- Item nào kích hoạt nhu cầu của item khác sau một khoảng trễ?
- Item nào cạnh tranh attention hoặc purchase với item khác?
- Item nào tăng cùng nhau vì cùng chịu ảnh hưởng bởi một latent context?
- Item nào thay thế nhau dù có user overlap cao?
```

---

## 3. Temporal similarity: khi hai item không liên quan theo CF nhưng giống nhau theo thời gian

Nếu hai item có biểu đồ thời gian giống nhau, không nên kết luận ngay rằng chúng có quan hệ collaborative. Quan hệ đúng hơn nên gọi là:

```text
temporal behavioral similarity
```

hoặc:

```text
time-pattern similarity
```

Đây là similarity trong **time-response space**, không phải trong user-interaction space.

Ví dụ:

```text
Item A: kem chống nắng
Item B: kính râm
Item C: vali du lịch
```

Ba item này có thể không luôn cùng user, không cùng category hẹp, nhưng có thể cùng tăng vào mùa hè hoặc mùa du lịch. Điều này cho thấy chúng chia sẻ một **latent temporal context**: mùa hè, du lịch, thời tiết nắng, kỳ nghỉ.

Một ví dụ khác:

```text
Item A: áo mưa
Item B: ô
Item C: túi chống nước
```

Các item này có thể cùng tăng vào mùa mưa. Nếu CF không phát hiện rõ quan hệ vì user overlap thấp, temporal curve vẫn có thể cho thấy chúng thuộc cùng một “temporal niche”.

Do đó, biểu đồ thời gian giống nhau có thể là dấu hiệu của một tầng quan hệ cao hơn:

```text
shared latent temporal driver
```

Các latent driver có thể gồm:

```text
mùa
thời tiết
dịp lễ
ngày lương
promotion
trend xã hội
kỳ thi
năm học
sự kiện thể thao
lịch du lịch
chu kỳ tiêu dùng
```

---

## 4. Quan hệ sinh thái giữa các item

Có thể mượn khái niệm từ sinh học để phân loại quan hệ giữa các item. Không phải item nào giống nhau theo thời gian cũng có cùng loại quan hệ. Có thể chia thành các nhóm sau.

### 4.1. Cạnh tranh / competition

Hai item cạnh tranh khi sự tăng demand hoặc purchase của item này làm giảm khả năng user chọn item kia. Đây thường là quan hệ giữa các sản phẩm thay thế.

Ví dụ:

```text
iPhone case mẫu A vs iPhone case mẫu B
Laptop gaming A vs laptop gaming B
Tai nghe Sony vs tai nghe Bose
```

Trong CF, các item này có thể gần nhau vì user cùng xem hoặc so sánh. Nhưng trong purchase dynamics, chúng có thể cạnh tranh vì user thường chỉ mua một trong hai.

Biểu hiện dữ liệu có thể là:

```text
- cùng tăng exposure hoặc view
- nhưng purchase của A tăng thì purchase của B giảm
- hoặc conversion của B giảm sau khi user mua A
```

Có thể mô hình hóa bằng interaction coefficient âm:

```text
effect(A -> B) < 0
```

### 4.2. Bổ trợ / complementarity

Hai item bổ trợ khi event của item A làm tăng xác suất event của item B sau một khoảng trễ.

Ví dụ:

```text
mua điện thoại -> mua ốp lưng
mua máy in -> mua mực in
mua máy ảnh -> mua thẻ nhớ
mua laptop -> mua chuột / túi chống sốc
```

Đây là quan hệ rất quan trọng cho cross-sell. Nó không nhất thiết xuất hiện như similarity cùng thời điểm, mà có thể xuất hiện như **lagged temporal effect**.

Biểu hiện dữ liệu:

```text
event(A, t) làm tăng intensity của event(B, t + Δ)
```

Có thể mô hình hóa bằng Hawkes process hoặc time-lag attention:

```text
effect(A at time t -> B at time t + Δ) > 0
```

### 4.3. Cộng sinh / mutualism

Hai item có quan hệ cộng sinh khi chúng cùng tăng vì hỗ trợ lẫn nhau hoặc cùng tạo ra một package/experience.

Ví dụ:

```text
máy chơi game + tay cầm phụ
bàn phím cơ + keycap
máy pha cà phê + hạt cà phê
lều cắm trại + túi ngủ
```

Khác với complementarity một chiều, cộng sinh có thể hai chiều hoặc cùng phụ thuộc vào một usage context.

Biểu hiện dữ liệu:

```text
A tăng làm B tăng
B tăng cũng làm A tăng
hoặc cả A và B cùng tăng trong một usage context
```

Có thể xem đây là positive bidirectional relationship:

```text
effect(A -> B) > 0
effect(B -> A) > 0
```

### 4.4. Ký sinh / parasitic relation

Một item có thể hưởng lợi từ item khác mà không tạo lợi ích ngược lại. Trong recommendation, đây có thể là item ăn theo trend, ăn theo traffic của item chính.

Ví dụ:

```text
phụ kiện rẻ ăn theo sản phẩm flagship
hàng nhái/hàng tương tự ăn theo brand nổi
sản phẩm nhỏ ăn theo search traffic của sản phẩm lớn
```

Biểu hiện dữ liệu:

```text
A tăng kéo B tăng
nhưng B tăng không kéo A tăng
```

Có thể mô hình hóa như positive one-way effect:

```text
effect(A -> B) > 0
effect(B -> A) ≈ 0
```

### 4.5. Chuỗi thức ăn / cascading relation

Một event có thể kích hoạt chuỗi nhiều item theo thời gian.

Ví dụ:

```text
mua nhà -> mua nội thất -> mua đồ trang trí -> mua đồ vệ sinh
mua laptop -> mua chuột -> mua bàn phím -> mua màn hình
đăng ký gym -> mua giày thể thao -> mua bình nước -> mua protein
```

Đây là dạng multi-step temporal dependency. Nó gần với sinh thái hơn vì nhu cầu không chỉ giữa hai item, mà lan truyền qua một mạng item.

Biểu hiện dữ liệu:

```text
A at t
-> B at t + Δ1
-> C at t + Δ2
-> D at t + Δ3
```

---

## 5. Cần phân biệt temporal similarity và causal interaction

Một điểm rất quan trọng: hai item có curve giống nhau chưa chắc item này ảnh hưởng item kia. Có thể chúng chỉ cùng chịu ảnh hưởng bởi một yếu tố bên ngoài.

Ví dụ:

```text
áo khoác và máy sưởi cùng tăng vào mùa lạnh
```

Điều này không có nghĩa mua áo khoác làm user mua máy sưởi. Có thể cả hai cùng do thời tiết lạnh kích hoạt.

Vì vậy cần phân biệt:

```text
temporal correlation:
    A và B cùng tăng/giảm theo thời gian

temporal causation/influence:
    event của A làm thay đổi xác suất event của B sau đó

shared latent context:
    A và B cùng bị điều khiển bởi context C
```

Ba loại này khác nhau về ý nghĩa mô hình hóa.

Một cách viết đúng:

```text
Temporal curve similarity reveals possible latent context similarity,
but additional modeling is needed to distinguish direct item-item influence
from shared external temporal drivers.
```

Do đó, nếu muốn nghiêm túc, cần tách các thành phần:

```text
global trend
category trend
seasonality
promotion effect
item-specific residual curve
```

Sau đó mới so sánh residual curve giữa item.

Công thức đơn giản:

```text
raw_item_curve
= global_curve
+ category_curve
+ seasonal_curve
+ promotion_curve
+ item_specific_residual
```

Temporal relationship giữa item nên được học trên phần residual hoặc phần đã control context, nếu không sẽ dễ nhầm global effect thành item relation.

---

## 6. Ý tưởng mô hình: Temporal Item Ecology Graph

Một hướng rõ ràng là xây dựng một graph mới gọi là:

```text
Temporal Item Ecology Graph
```

Trong graph này:

```text
node = item
edge = temporal relationship giữa item
```

Edge có thể có loại:

```text
competition
complementarity
mutualism
one-way trigger
substitution
shared temporal context
```

Edge cũng có thể có trọng số theo thời gian:

```text
A -> B with lag 3 days
A -> B with lag 2 weeks
A competes with B during promotion period
A and B co-peak in December
```

Khác với item-item graph truyền thống dựa trên co-click/co-purchase, Temporal Item Ecology Graph dựa trên **time dynamics**.

Có thể tạo nhiều loại edge:

```text
1. Synchronous edge:
   A và B cùng tăng trong cùng time window.

2. Lagged edge:
   A tăng trước, B tăng sau Δ time.

3. Negative edge:
   A tăng thì B giảm.

4. Conditional edge:
   A ảnh hưởng B chỉ trong một context, ví dụ mùa hè hoặc promotion.

5. Residual edge:
   A và B có correlation sau khi loại bỏ global/category trend.
```

---

## 7. Cách học temporal item embedding

Một cách khai thác đơn giản là tạo embedding cho mỗi item từ time series của nó.

Với mỗi item:

```text
x_i = [demand_i(t1), demand_i(t2), ..., demand_i(tT)]
```

Sau đó encode bằng:

```text
Fourier features
seasonality decomposition
temporal CNN
Transformer time-series encoder
state-space model
autoencoder time-series
foundation time-series encoder
```

Kết quả:

```text
z_i_temporal = TemporalEncoder(x_i)
```

Embedding cuối của item có thể là:

```text
z_i = z_i_collaborative + z_i_semantic + z_i_temporal
```

Hoặc dùng fusion có trọng số:

```text
z_i = Fusion(z_i_collaborative, z_i_semantic, z_i_temporal)
```

Điểm quan trọng là temporal embedding không thay thế collaborative embedding. Nó bổ sung một chiều mới:

```text
collaborative embedding:
    ai tương tác với item?

semantic embedding:
    item là gì?

temporal embedding:
    item trở nên phù hợp khi nào?
```

---

## 8. Cách học quan hệ cạnh tranh/hợp tác

Có thể học quan hệ item-item theo nhiều cách.

### 8.1. Correlation trên residual demand curve

Đầu tiên tính demand curve cho mỗi item, sau đó loại bỏ global/category effect:

```text
residual_i(t) = demand_i(t) - global(t) - category_i(t)
```

Sau đó đo correlation hoặc dynamic time warping giữa residual curves:

```text
sim_temporal(i, j) = corr(residual_i, residual_j)
```

Nếu correlation dương cao:

```text
i và j có shared temporal behavior
```

Nếu correlation âm:

```text
i và j có thể cạnh tranh hoặc thay thế theo thời gian
```

### 8.2. Lagged cross-correlation

Để phát hiện quan hệ kích hoạt có độ trễ:

```text
corr(demand_A(t), demand_B(t + Δ))
```

Nếu cao tại Δ = 7 ngày, có thể A kích hoạt B sau khoảng 7 ngày.

Ví dụ:

```text
mua điện thoại -> mua ốp lưng sau 1-3 ngày
mua máy in -> mua mực sau 30-60 ngày
```

### 8.3. Hawkes process / temporal point process

Dùng temporal point process để học ảnh hưởng của event item này lên intensity của item khác:

```text
λ_B(t) = base_B(t) + sum over A events α_A,B * decay(t - t_A)
```

Trong đó:

```text
α_A,B > 0: A kích hoạt B
α_A,B < 0: A ức chế/cạnh tranh B
α_A,B ≈ 0: không có quan hệ trực tiếp
```

Đây là cách rất hợp với ý tưởng sinh học vì nó mô hình hóa interaction giữa các loài/item theo thời gian.

### 8.4. Granger causality / causal discovery

Nếu có time series đủ dài, có thể kiểm tra liệu quá khứ của A có giúp dự đoán B tốt hơn chỉ dùng quá khứ của B hay không.

```text
A Granger-causes B
nếu history của A cải thiện dự đoán demand của B.
```

Cách này không chứng minh causal thật tuyệt đối, nhưng giúp phát hiện directional temporal influence.

### 8.5. Graph neural network trên temporal ecology graph

Sau khi có graph cạnh tranh/hợp tác, dùng GNN để propagate information:

```text
item representation = GNN(collaborative graph + temporal ecology graph)
```

Có thể dùng edge type khác nhau:

```text
competition edge
complement edge
mutualism edge
trigger edge
shared-context edge
```

---

## 9. Liên hệ với bài toán predict time

Ý tưởng này rất hợp với hướng **item-conditioned time prediction** trước đó. Nếu muốn dự đoán item B sẽ được mua khi nào, ta không chỉ nhìn user history và B, mà còn nhìn những item khác trong “hệ sinh thái” của B.

Ví dụ:

```text
User vừa mua điện thoại.
Candidate item: ốp lưng.
Temporal ecology graph biết điện thoại -> ốp lưng có lag 1-3 ngày.
Model dự đoán purchase probability curve của ốp lưng tăng trong vài ngày tới.
```

Hoặc:

```text
Candidate item: áo khoác.
Temporal graph biết áo khoác cùng niche mùa lạnh với máy sưởi, khăn choàng.
Nếu các item cùng niche đang tăng demand, curve của áo khoác cũng được điều chỉnh tăng.
```

Công thức có thể viết:

```text
P(t_event | user, item_i, history, ecology_neighbors(i))
```

hoặc intensity:

```text
λ_u,i(t)
= base_u,i(t)
+ temporal_context_effect(i, t)
+ sum_j interaction_effect(j -> i, lag)
```

Trong đó các item `j` là item user đã tương tác hoặc item trong temporal ecology graph.

---

## 10. Liên hệ với generative retrieval

Ý tưởng temporal ecology cũng có thể kết hợp với generative retrieval. Hiện nhiều Semantic ID tập trung vào semantic/collaborative structure. Nhưng nếu item có temporal behavior riêng, có thể tạo thêm:

```text
Temporal Semantic ID
```

hoặc:

```text
Ecology-aware Semantic ID
```

Thay vì token hóa item chỉ dựa trên text/image/collaborative embedding, có thể thêm temporal embedding hoặc ecology graph embedding:

```text
item representation
= semantic embedding
+ collaborative embedding
+ temporal behavior embedding
+ ecology graph embedding
```

Sau đó quantize thành Semantic ID.

Điểm mới:

```text
Items that share latent temporal contexts or temporal interaction patterns
may share part of their Semantic IDs,
even if they are not semantically or collaboratively close.
```

Điều này giúp generative recommender học được các item cùng mùa, cùng event, cùng lifecycle hoặc cùng usage context.

---

## 11. Các câu hỏi nghiên cứu

Một số research questions có thể viết:

### RQ1

Hai item không gần trong collaborative filtering nhưng có temporal curve giống nhau có tạo ra một dạng item similarity hữu ích cho recommendation không?

### RQ2

Temporal item similarity phản ánh shared latent context, direct item-item influence hay chỉ là global trend?

### RQ3

Việc thêm temporal item embedding vào semantic/collaborative item representation có cải thiện cold-start, long-tail hoặc time-aware recommendation không?

### RQ4

Có thể học được quan hệ cạnh tranh và bổ trợ giữa item từ event-time data không?

### RQ5

Temporal ecology graph có giúp dự đoán purchase timing tốt hơn so với chỉ dùng user history và item embedding không?

### RQ6

Item tokenizer trong generative retrieval có nên tích hợp temporal behavior embedding để tạo Semantic ID không?

---

## 12. Evaluation gợi ý

Không nên chỉ đánh giá bằng Recall/NDCG. Vì ý tưởng này liên quan đến temporal relation, cần thêm metric.

### 12.1. Recommendation quality

```text
Recall@K
NDCG@K
MRR
Hit Rate
```

### 12.2. Time prediction quality

```text
time MAE/RMSE
negative log-likelihood của event time
time-window hit rate
calibration của purchase probability curve
```

### 12.3. Temporal relation quality

```text
correlation trên residual curves
lag prediction accuracy
edge sign accuracy nếu có label substitute/complement
Granger causality score
stability của temporal clusters qua thời gian
```

### 12.4. Business/behavior metric

```text
cross-sell lift
repeat purchase lift
seasonal recommendation lift
diversity theo temporal niche
cold-start item exposure
```

---

## 13. Rủi ro và điểm cần cẩn thận

### 13.1. Nhầm global trend thành item relation

Nếu toàn hệ thống tăng traffic vào cuối năm, nhiều item sẽ cùng tăng. Không được kết luận chúng có quan hệ sinh thái trực tiếp. Cần loại bỏ global/category trend.

### 13.2. Data sparsity ở tail items

Item ít interaction có curve nhiễu. Cần smoothing, aggregation theo category, hoặc Bayesian estimation.

### 13.3. Correlation không phải causation

Hai item cùng tăng không có nghĩa item này gây ra item kia. Cần tách shared context và directional effect.

### 13.4. Exposure bias

Nếu hệ thống recommend item A nhiều hơn item B, demand curve của A tăng có thể do exposure, không phải nhu cầu tự nhiên. Nếu có impression log, nên control exposure.

### 13.5. Promotion confounding

Promotion có thể làm nhiều item tăng cùng lúc. Cần biết campaign/promotion context nếu có.

### 13.6. Temporal granularity

Ngày, tuần, tháng cho ra kết luận khác nhau. Một số item có cycle theo giờ; một số theo mùa. Cần chọn granularity theo domain.

---

## 14. Một formulation đề xuất

Tên hướng có thể là:

```text
Temporal Item Ecology for Recommender Systems
```

Hoặc:

```text
Ecology-aware Time-sensitive Recommendation
```

Formulation:

```text
Given:
    user-item-event logs with timestamps
    optional item content/category metadata

Learn:
    item temporal behavior embeddings
    item-item temporal interaction graph
    event-time prediction model

Goal:
    improve time-aware recommendation, purchase timing prediction,
    cross-sell timing and cold-start generalization.
```

Model tổng quát:

```text
z_i^sem = SemanticEncoder(item_i)
z_i^cf  = CollaborativeEncoder(item_i)
z_i^tmp = TemporalEncoder(demand_curve_i)
G_tmp   = TemporalEcologyGraph(items)

z_i = Fusion(z_i^sem, z_i^cf, z_i^tmp, GNN(G_tmp, i))

P(t | u, i, history) = TimePredictionHead(UserEncoder(history), z_i)
```

Nếu dùng point process:

```text
λ_u,i(t)
= μ_u,i(t)
+ Σ_j α_j,i · g(t - t_j)
+ context_i(t)
```

Trong đó:

```text
α_j,i > 0: item j kích hoạt item i
α_j,i < 0: item j cạnh tranh/ức chế item i
α_j,i ≈ 0: không có ảnh hưởng trực tiếp
```

---

## 15. Đóng góp có thể viết

Một bộ contribution khả thi:

```text
1. We introduce Temporal Item Ecology, a view of recommender systems where items are connected not only by user overlap or content similarity, but also by temporal demand dynamics.

2. We define multiple ecological item relations, including competition, complementarity, mutualism, one-way triggering and shared latent temporal context.

3. We propose to learn temporal item embeddings from residual demand curves after removing global and category-level trends.

4. We construct a temporal ecology graph using synchronous and lagged temporal relationships between items.

5. We integrate the temporal ecology graph into item-conditioned event-time prediction to improve purchase timing and time-aware recommendation.

6. We discuss how ecology-aware item representations can be used to enhance Semantic ID learning in generative retrieval.
```

---

## 16. Kết luận

Ý tưởng chính là: item trong recommender systems không chỉ giống nhau vì được cùng user tương tác hoặc vì có nội dung giống nhau. Chúng còn có thể giống nhau vì cùng phản ứng với thời gian, cùng chịu ảnh hưởng bởi latent context, hoặc trực tiếp ảnh hưởng lẫn nhau như các thực thể trong một hệ sinh thái.

Có thể tóm tắt bằng câu:

```text
Collaborative filtering captures who-consumes-what similarity.
Content modeling captures what-items-are similarity.
Temporal item ecology captures when-and-how-items-influence-each-other similarity.
```

Đây là một hướng đáng khai thác vì nó mở thêm một chiều biểu diễn item: **temporal behavior and interaction dynamics**. Chiều này có thể giúp phát hiện các quan hệ mà CF truyền thống chưa thấy, đặc biệt trong các bài toán time-aware recommendation, purchase timing prediction, cross-sell timing, cold-start và generative retrieval với Semantic IDs.
