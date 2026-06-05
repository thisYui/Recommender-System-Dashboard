# Note mở rộng: Những điểm đáng khai thác cho hướng item-conditioned time prediction trong recommender systems

## 1. Trục nghiên cứu đáng chú ý nhất

Hướng nghiên cứu này không nên được trình bày đơn giản là “thêm time feature vào recommender system”. Nếu viết như vậy thì sẽ bị trùng với rất nhiều paper time-aware recommendation đã có. Điểm đáng chú ý hơn là **đổi vai trò của time trong bài toán**. Trong nhiều recommender systems, time thường là input feature để hỗ trợ predict item. Ở hướng này, time nên được đưa thành **target chính** hoặc ít nhất là một biến đồng mục tiêu với item.

Cách phát biểu mạnh hơn là:

```text
Thay vì học user sẽ mua item nào tại một thời điểm cho trước,
ta học một item hoặc nhóm item sẽ trở nên phù hợp với user vào thời điểm nào.
```

Với phát biểu này, bài toán chuyển từ:

```text
P(item | user, history, current_time)
```

sang:

```text
P(time | user, item, history, context)
```

hoặc:

```text
λ_user,item(t)
```

trong đó `λ_user,item(t)` là intensity curve biểu diễn mức độ có khả năng user tương tác/mua item tại thời điểm `t`.

Điểm mạnh của cách đặt bài toán này là nó phù hợp với các tình huống thực tế như purchase timing, replenishment, seasonal recommendation, promotion timing, cross-sell timing, notification scheduling và active recommendation. Trong các tình huống này, doanh nghiệp thường đã có candidate item hoặc campaign item; câu hỏi quan trọng không phải luôn là “gợi ý gì”, mà là “gợi ý khi nào”.

---

## 2. Điểm khác biệt nên nhấn mạnh với các paper time-aware hiện có

Các paper như TimelyRec và TLSRec chứng minh time có ảnh hưởng mạnh đến preference, nhưng chúng vẫn chủ yếu phục vụ mục tiêu recommend item. TimelyRec học periodic pattern và evolving pattern để cải thiện timely recommendation. TLSRec dùng time lag để điều chỉnh đóng góp của long-term và short-term preference. Các paper này rất quan trọng để làm nền, nhưng hướng của ta nên nhấn mạnh rằng:

```text
Các paper time-aware hiện tại thường dùng time để cải thiện item ranking.
Hướng này dùng item và item components để dự đoán phân phối thời gian.
```

Nói cách khác, đây là một sự đảo chiều:

```text
Time-aware item recommendation:
    time -> item

Item-conditioned time prediction:
    item -> time
```

Dù hai hướng có liên quan, bài toán thứ hai có mục tiêu, output và evaluation khác. Output không phải là top-K item list, mà có thể là time bucket distribution, purchase probability curve, hazard curve hoặc intensity function.

---

## 3. Vì sao item decomposition là điểm đáng khai thác

Nếu chỉ dùng item ID để predict time, bài toán dễ bị sparse. Nhiều item có ít lịch sử, nhiều user chưa từng mua item đó, và rất khó học được temporal pattern ổn định cho từng user-item pair. Vì vậy item nên được phân rã thành các thành phần có ý nghĩa hành vi và thời gian.

Một item có thể được phân rã thành:

```text
category
brand
price segment
consumable / durable
seasonal sensitivity
promotion sensitivity
complementary relation
substitute relation
lifecycle stage
repeat-purchase cycle
visual/text semantic embedding
```

Ví dụ, một chiếc “ốp lưng iPhone” không chỉ là item ID. Nó có category là accessory, có relation với smartphone, thường phát sinh sau khi user mua hoặc quan tâm đến điện thoại, và có time lag ngắn. Một sản phẩm như “sữa rửa mặt” lại có repeat-purchase cycle. Một sản phẩm như “áo khoác” có seasonal pattern. Một khóa học online có thể phụ thuộc vào lịch học, kỳ thi hoặc giai đoạn mục tiêu cá nhân.

Điểm khai thác ở đây là model có thể học temporal pattern ở cấp component thay vì chỉ học ở cấp item ID. Điều này giúp generalize tốt hơn sang item mới hoặc item ít dữ liệu. Đây có thể là điểm mới quan trọng nếu viết proposal:

```text
We model item-conditioned purchase timing by decomposing items into temporally meaningful components,
allowing the model to transfer timing patterns across semantically or behaviorally related items.
```

---

## 4. Các loại time signal nên tách riêng

Một lỗi dễ gặp là gom tất cả thông tin thời gian vào một embedding duy nhất. Literature hiện tại cho thấy time không đồng nhất. Nên tách time thành nhiều loại signal:

### 4.1. Absolute time

Đây là thời điểm tuyệt đối như ngày, tháng, mùa, ngày trong tuần, giờ trong ngày. Nó hữu ích cho seasonal và periodic behavior. Ví dụ, food delivery có pattern theo giờ ăn; fashion có pattern theo mùa; entertainment có pattern theo cuối tuần.

### 4.2. Relative time / time lag

Đây là khoảng cách từ event trước đến thời điểm hiện tại, hoặc từ một trigger event đến target event. Nó rất quan trọng cho cross-sell và follow-up recommendation. Ví dụ, sau khi mua điện thoại, user có thể mua phụ kiện trong vài ngày; sau khi mua mỹ phẩm, user có thể mua lại sau vài tuần.

### 4.3. Inter-event interval

Đây là khoảng cách giữa các lần user tương tác/mua cùng item hoặc cùng category. Nó phù hợp với replenishment và repeated consumption. Ví dụ, user có thể mua lại cà phê, sữa rửa mặt hoặc thực phẩm theo chu kỳ.

### 4.4. Event-triggered time

Một số item không có chu kỳ cố định mà được kích hoạt bởi event khác. Ví dụ, user mua laptop có thể kéo theo chuột, túi chống sốc, phần mềm hoặc màn hình phụ. Đây là nơi Hawkes process hoặc attention over historical events rất phù hợp.

### 4.5. External/event time

Promotion, holiday, payday, weather, semester, exam season, release date hoặc campaign period có thể làm thay đổi demand curve. Nếu dataset có context ngoài, đây là hướng rất đáng khai thác.

---

## 5. Output nên là curve/distribution, không phải một timestamp

Một điểm quan trọng là không nên ép model chỉ predict một thời điểm duy nhất. Hành vi mua hàng có uncertainty cao, nên output nên là distribution hoặc curve.

Có bốn dạng output đáng cân nhắc:

### 5.1. Time bucket distribution

Chia tương lai thành các bucket:

```text
0-1 ngày
1-3 ngày
3-7 ngày
7-14 ngày
14-30 ngày
>30 ngày
```

Model predict:

```text
P(bucket | user, item, history)
```

Ưu điểm là dễ implement, dễ evaluate, dễ dùng với classification model. Nhược điểm là mất độ mịn thời gian.

### 5.2. Purchase probability curve

Model xuất một vector xác suất theo horizon:

```text
[p_day1, p_day2, ..., p_day30]
```

Đây là dạng rất trực quan, dễ vẽ, dễ giải thích. Có thể dùng cho ranking thời điểm tốt nhất hoặc chọn window gửi recommendation.

### 5.3. Survival / hazard curve

Model học:

```text
S(t) = xác suất user chưa mua item đến thời điểm t
h(t) = hazard mua item tại thời điểm t nếu trước đó chưa mua
```

Dạng này phù hợp nếu bài toán là time-to-event. Nó cũng xử lý được censored data, tức là nhiều trường hợp user chưa mua trong quan sát nhưng không có nghĩa là sẽ không bao giờ mua.

### 5.4. Continuous-time intensity curve

Model học:

```text
λ_user,item(t)
```

Đây là dạng mạnh nhất về mặt nghiên cứu, đặc biệt nếu event xảy ra không đều. Nó phù hợp với temporal point process, Hawkes process hoặc neural point process. Output là một intensity curve liên tục theo thời gian.

---

## 6. Censored data là vấn đề rất quan trọng

Trong bài toán predict time-to-purchase, không phải mọi sample đều có target time rõ ràng. Có rất nhiều trường hợp user chưa mua item trong khoảng quan sát. Không nên gán các trường hợp này là negative tuyệt đối, vì có thể user sẽ mua sau khi dataset kết thúc. Đây gọi là censored data.

Ví dụ:

```text
User xem item A vào ngày 1.
Dataset kết thúc ngày 30.
User chưa mua item A trong 30 ngày.
```

Ta không thể kết luận user sẽ không bao giờ mua item A. Chỉ có thể nói rằng event chưa xảy ra trong window quan sát. Nếu dùng classification bình thường và gán nhãn negative cứng, model có thể học sai.

Đây là lý do survival analysis hoặc point process rất đáng cân nhắc. Chúng có framework tự nhiên để xử lý censored observations. Nếu làm bản đơn giản, có thể dùng window-based labeling như:

```text
positive nếu mua trong 30 ngày
negative nếu không mua trong 30 ngày
```

nhưng cần ghi rõ đây là approximation và có nguy cơ label bias.

---

## 7. Negative sampling cần thiết kế cẩn thận

Nếu target là item-conditioned time prediction, ta cần tạo sample dạng:

```text
(user, candidate_item, history) -> purchase_time_distribution
```

Vấn đề là số lượng user-item không mua là cực lớn. Nếu random negative quá dễ, model sẽ học phân biệt item không liên quan thay vì học timing. Vì vậy nên có nhiều loại negative:

### 7.1. Random negative

Chọn item user không mua. Loại này dễ nhưng thường quá đơn giản.

### 7.2. Category-level hard negative

Chọn item cùng category với item đã mua nhưng user không mua. Loại này giúp model học timing trong các item tương tự.

### 7.3. Temporal hard negative

Chọn item user có thể quan tâm nhưng chưa đến thời điểm mua. Đây là loại quan trọng nhất cho bài toán này.

### 7.4. Exposure-aware negative

Nếu có log impression/exposure, chỉ xem item đã được user thấy nhưng không mua là negative mạnh hơn. Nếu không có exposure, cần cẩn thận vì user không mua có thể do chưa từng thấy item.

Điểm cần nhấn mạnh: bài toán này không chỉ là “user có mua item không”, mà là “item đó có đúng thời điểm không”. Do đó negative sampling nên tạo ra các trường hợp sai về timing, không chỉ sai về item.

---

## 8. Evaluation nên khác với recommender system truyền thống

Nếu chỉ dùng HR@K hoặc NDCG@K cho item ranking thì không đánh giá đúng mục tiêu. Cần thêm metric cho time prediction.

Một số evaluation phù hợp:

### 8.1. Time MAE / RMSE

Dùng khi model predict một expected time hoặc peak time:

```text
|predicted_time - actual_time|
```

Dễ hiểu nhưng không đánh giá tốt uncertainty.

### 8.2. Negative log-likelihood

Dùng nếu model output distribution hoặc intensity:

```text
-log P(actual_time | user, item, history)
```

Đây là metric tốt hơn cho probabilistic prediction.

### 8.3. Calibration

Nếu model nói xác suất mua trong 7 ngày là 0.3, thì trong nhóm sample như vậy, khoảng 30% có thật sự mua trong 7 ngày không? Calibration rất quan trọng nếu output dùng cho business decision.

### 8.4. Top-time-window hit rate

Chọn top window mà model dự đoán xác suất cao nhất, kiểm tra event có rơi vào window đó không.

Ví dụ:

```text
Model chọn window ngày 5-7.
User thực sự mua ngày 6.
=> hit
```

### 8.5. Joint item-time metric

Nếu vẫn cần recommend cả item và time, dùng metric kiểu item-time hit. Một prediction chỉ đúng khi vừa đúng item vừa đúng time window.

### 8.6. Utility-aware metric

Trong active recommendation, không phải càng sớm càng tốt. Gửi quá sớm có thể bị ignore; gửi quá muộn có thể mất cơ hội. Có thể định nghĩa utility theo khoảng cách thời gian:

```text
utility = exp(-|predicted_time - actual_time| / τ)
```

hoặc utility có penalty nếu gửi quá nhiều notification.

---

## 9. Các baseline nên có

Để proposal thuyết phục, nên có baseline từ đơn giản đến mạnh.

### 9.1. Popularity-by-time baseline

Dự đoán thời điểm dựa trên distribution phổ biến của category/item. Ví dụ, item category này thường được mua vào cuối tuần hoặc tháng 12.

### 9.2. User periodicity baseline

Dự đoán dựa trên routine của user: user thường mua vào giờ/ngày nào.

### 9.3. Item/category repeat interval baseline

Dự đoán dựa trên chu kỳ mua lại trung bình của item/category.

### 9.4. Sequential recommender + post-hoc time model

Dùng sequential recommender predict item trước, sau đó dùng một model khác predict time. Baseline này giúp so với hướng joint hoặc item-conditioned.

### 9.5. Survival model

Dùng DeepSurv/DeepHit hoặc discrete-time survival làm baseline time-to-event.

### 9.6. Temporal point process

Dùng Hawkes hoặc neural point process làm baseline mạnh nhất nếu dữ liệu event-time đủ tốt.

### 9.7. Transformer time-bucket classifier

Dùng Transformer encoder cho user history + item embedding, output time bucket distribution. Đây là baseline hiện đại nhưng dễ implement hơn point process.

---

## 10. Có thể kết hợp với foundation model time series không?

Có, nhưng cần cẩn thận khi trình bày. Các foundation model time series như Chronos, TimesFM, Lag-Llama thường mạnh trong forecasting chuỗi số theo thời gian. Tuy nhiên, recommender system có item ID, user ID, sparse event, categorical metadata và implicit feedback. Vì vậy không thể nói đơn giản là đưa RS vào time-series foundation model là xong.

Có hai cách hợp lý hơn:

### 10.1. Dùng foundation model cho aggregated demand

Dự đoán demand curve ở cấp item/category/brand:

```text
daily demand of item/category
```

Sau đó dùng recommender model cá nhân hóa cho từng user.

Cách này phù hợp nếu dữ liệu cá nhân quá sparse nhưng có demand aggregate tốt.

### 10.2. Dùng time-series representation làm feature

Foundation model có thể encode trend/seasonality của item hoặc category thành embedding, rồi embedding này được đưa vào item-conditioned time prediction model.

Ví dụ:

```text
category demand history -> time-series foundation model -> temporal demand embedding
```

Sau đó:

```text
(user history, item embedding, temporal demand embedding) -> P(time | user, item)
```

Đây là cách kết hợp an toàn và hợp lý hơn so với việc bắt foundation model trực tiếp predict user-item purchase event.

---

## 11. Có thể khai thác generative modeling

Một hướng hiện đại là xem future interaction như một quá trình sinh event:

```text
generate next event time
generate next item
generate next context
```

Hoặc với hướng đảo lại:

```text
given user and item, generate plausible future interaction times
```

Diffusion model, VAE hoặc normalizing flow có thể được dùng để học distribution phức tạp của time. PASRec là paper đáng chú ý vì dùng diffusion-based sequential recommendation và nhấn mạnh joint ToI-IoI. Nếu viết hướng nghiên cứu hiện đại, có thể đề xuất:

```text
Item-conditioned temporal diffusion model for purchase timing prediction
```

Trong đó model sinh nhiều sample thời điểm tương lai cho một user-item pair, sau đó tạo distribution hoặc curve từ các sample này. Điểm mạnh là có thể mô hình hóa uncertainty và multi-modal timing. Ví dụ, một item có thể có hai peak: cuối tuần và ngày lương.

---

## 12. Multi-modal item decomposition cũng đáng khai thác

Nếu item có text, image hoặc metadata phong phú, có thể dùng encoder để phân rã item tốt hơn. Đây là hướng liên quan đến ý tưởng “item sẽ được phân rã thành các thành phần”.

Ví dụ:

```text
item title/description -> text embedding
item image -> visual embedding
category/brand/price -> structured embedding
historical co-purchase -> collaborative embedding
```

Sau đó model học component nào ảnh hưởng đến timing. Ví dụ, visual style có thể liên quan đến seasonality trong thời trang; text keyword như “gift”, “winter”, “school”, “festival” có thể liên quan đến thời điểm mua; price segment có thể liên quan đến payday hoặc promotion.

Điểm đáng khai thác là explainability:

```text
Model dự đoán item này phù hợp vào tháng 12 vì component seasonal/winter cao.
Model dự đoán phụ kiện này phù hợp sau 3 ngày vì nó complementary với điện thoại user vừa mua.
```

---

## 13. Phân biệt ba bài toán gần nhau

Để tránh bị lẫn, nên phân biệt rõ ba bài toán:

### 13.1. Time-aware recommendation

```text
Input: user, history, current_time
Output: item ranking
Question: Bây giờ nên recommend gì?
```

Đây là hướng truyền thống hơn.

### 13.2. Active recommendation / when-and-what

```text
Input: user, history
Output: next time + item
Question: Khi nào nên chủ động recommend và recommend gì?
```

PASRec và notification papers nằm gần hướng này.

### 13.3. Item-conditioned time prediction

```text
Input: user, history, candidate item
Output: time distribution / intensity curve
Question: Item này nên được recommend/mua vào lúc nào?
```

Đây là hướng bạn đang thiên về. Cần giữ thuật ngữ này xuyên suốt để tránh bị reviewer hiểu nhầm là chỉ làm time-aware recommendation thông thường.

---

## 14. Pipeline nghiên cứu gợi ý

Một pipeline khả thi:

### Step 1: Construct event sequences

Tạo chuỗi event theo user:

```text
(user_id, item_id, timestamp, event_type, context)
```

### Step 2: Build item components

Tạo item representation từ:

```text
ID embedding
category
brand
price
text/image embedding
co-purchase relation
seasonal profile
repeat-purchase profile
```

### Step 3: Generate training pairs

Với mỗi user, chọn candidate item từ:

```text
item đã mua trong tương lai
item cùng category
item complementary
item exposed but not purchased
negative sampled item
```

Label là:

```text
time-to-purchase
time bucket
future probability curve
censored indicator
```

### Step 4: Train model

Model input:

```text
user history encoder
candidate item component encoder
temporal context encoder
```

Model output:

```text
time bucket distribution
hoặc hazard curve
hoặc intensity curve
```

### Step 5: Evaluate

Dùng:

```text
NLL
time-window hit rate
MAE/RMSE for predicted peak time
calibration
joint item-time hit
business utility metric
```

### Step 6: Optional active delivery layer

Nếu muốn gợi ý chủ động:

```text
time prediction model -> candidate timing
policy model/RL -> send or not send
```

---

## 15. Các câu hỏi nghiên cứu có thể viết

Một số research questions có thể dùng trong proposal:

### RQ1

Item-conditioned time prediction có cải thiện khả năng xác định thời điểm tương tác/mua so với time-aware item recommendation truyền thống không?

### RQ2

Việc phân rã item thành category, relation, semantic và temporal components có giúp generalize sang item sparse hoặc cold-start tốt hơn không?

### RQ3

Loại temporal signal nào quan trọng nhất cho từng nhóm item: periodicity, time lag, inter-event interval, event-triggered effect hay external trend?

### RQ4

Output dạng distribution/curve có calibrated tốt hơn output dạng single timestamp không?

### RQ5

Khi kết hợp với active notification policy, time prediction có giúp giảm số lần gửi nhưng vẫn giữ hoặc tăng engagement không?

### RQ6

Temporal point process, survival model và Transformer time-bucket classifier khác nhau thế nào về accuracy, calibration và interpretability?

---

## 16. Rủi ro và điểm cần cẩn thận

### 16.1. Không có exposure log

Nếu chỉ có purchase log, ta không biết user không mua vì không thích hay vì chưa từng thấy item. Đây là bias lớn trong recommendation.

### 16.2. Label bị censoring

User chưa mua trong dataset không đồng nghĩa với negative thật. Cần xử lý bằng survival/censoring hoặc window labeling rõ ràng.

### 16.3. Time pattern có thể là population-level chứ không phải personal-level

Một số item có seasonality mạnh ở toàn hệ thống, nhưng không cá nhân hóa. Cần tách global item/category temporal demand khỏi personal timing.

### 16.4. Data sparsity ở user-item level

User-item pair thường rất sparse. Đây là lý do item decomposition và sharing qua category/component là quan trọng.

### 16.5. Evaluation dễ bị leakage

Nếu dùng feature như item popularity trong tương lai hoặc demand curve tính trên toàn dataset, có thể leak thông tin tương lai. Cần split theo thời gian và chỉ dùng history trước cutoff.

### 16.6. Predict đúng time không đồng nghĩa nên gửi notification

User có thể mua tự nhiên mà không cần can thiệp. Nếu gửi notification sai cách có thể gây giảm long-term value. Vì vậy notification/RL nên được xem là tầng quyết định riêng.

---

## 17. Điểm có thể viết thành đóng góp chính

Một bộ contribution có thể viết như sau:

```text
1. We reformulate time-aware recommendation as item-conditioned time prediction,
   where the model predicts when a candidate item is likely to become relevant to a user.

2. We propose to decompose items into temporally meaningful components,
   enabling the transfer of timing patterns across sparse and cold-start items.

3. We model the output as a time distribution/intensity curve rather than a single timestamp,
   capturing uncertainty and multi-modal purchase timing.

4. We design evaluation protocols for purchase timing prediction,
   including time-window hit rate, likelihood, calibration and utility-aware metrics.

5. We discuss how the predicted timing distribution can be integrated with an active recommendation
   or notification policy to optimize long-term user experience.
```

---

## 18. Một hướng mô hình cụ thể có thể đề xuất

Tên tạm thời:

```text
ICTPRec: Item-Conditioned Time Prediction for Recommendation
```

Kiến trúc:

```text
User History Encoder:
    Transformer / GRU / attention over historical interactions

Item Component Encoder:
    ID embedding + metadata embedding + text/image embedding + relation embedding

Temporal Encoder:
    absolute time embedding
    relative time lag embedding
    periodic encoding
    event-triggered decay encoding

Fusion:
    cross-attention between candidate item and user history

Prediction Head:
    time bucket distribution
    hazard curve
    or neural intensity function
```

Output:

```text
P(time_bucket | user, candidate_item, history)
```

hoặc:

```text
λ_user,item(t)
```

Training objective:

```text
cross-entropy for time bucket
negative log-likelihood for survival/point process
calibration regularization
contrastive loss for correct item-time pairs
```

Có thể thêm auxiliary task:

```text
predict whether user will buy item within horizon
predict item category
predict next interaction time
predict repeat interval
```

Auxiliary tasks giúp model học representation ổn định hơn.

---

## 19. Kết luận ngắn

Điểm đáng khai thác nhất là biến time thành target, không chỉ là feature. Literature hiện tại đã có nhiều bằng chứng rằng time ảnh hưởng mạnh đến recommendation, nhưng vẫn còn khoảng trống ở bài toán **item-conditioned time prediction**: cho một user và một candidate item, dự đoán phân phối thời điểm item đó trở nên phù hợp. Hướng này có thể kết nối tự nhiên với temporal point process, survival analysis, probabilistic forecasting, diffusion-based recommendation và notification policy.

Nếu triển khai tốt, hướng này có thể đóng góp cả về mặt mô hình lẫn ứng dụng: mô hình không chỉ biết user có thể thích gì, mà còn biết khi nào item đó nên được đưa ra để tối đa hóa khả năng tương tác và giảm rủi ro làm phiền user.
