# Abstract và định hướng đề xuất

## Abstract

Recommender systems are commonly designed to answer the question of **what item should be recommended to a user**. However, in many real-world scenarios, the item or action to be promoted is not always an open prediction target. It may already be determined by business objectives, campaign constraints, inventory priorities, or a separate recommendation module. In such settings, the central problem shifts from predicting both **what** and **when** to predicting the most effective time to trigger a user event for a given item or action.

This study proposes an **action-conditioned next-time prediction framework** for timestamped user behavior data. Instead of jointly modeling the next item and the next interaction time, the proposed direction isolates the timing component and treats the item or action as an external condition. Given a user's historical interaction sequence and a selected target action, the objective is to estimate when the user is most likely to generate the desired future event. This formulation reduces potential error propagation from multi-stage item-time prediction and allows the model to focus more directly on temporal user dynamics.

The research begins with a multi-dataset feasibility analysis over timestamped interaction logs. All datasets are normalized into a unified event schema consisting of `user_id`, `item_id`, `event_type`, and `timestamp`. From these logs, next-time targets are constructed using the time interval between consecutive user events. Initial statistical and lightweight machine learning baselines are used to evaluate whether temporal signals exist in each dataset. Since direct supervised regression on exact time intervals is often unstable due to long-tailed and sparse behavior patterns, the study further motivates a transition toward time-to-event modeling, such as discrete-time hazard modeling, survival analysis, and temporal point process methods.

The proposed framework reframes recommendation timing as an event probability or intensity estimation problem. Rather than predicting a single exact `delta_t`, the model estimates the likelihood of a user event over future time windows conditioned on the selected item or action and the user's temporal history. This enables the system to choose a trigger time that maximizes expected event probability or business utility. By focusing specifically on the timing layer, this research provides a practical intermediate step between simple next-time prediction baselines and more complex joint time-item recommendation models.

## Tóm tắt

Các hệ thống gợi ý thường được thiết kế để trả lời câu hỏi người dùng nên được gợi ý item nào. Tuy nhiên, trong nhiều bối cảnh thực tế, item hoặc hành động cần được đưa đến người dùng không nhất thiết là một biến cần dự đoán hoàn toàn bởi mô hình. Chúng có thể đã được xác định trước bởi mục tiêu kinh doanh, chiến dịch quảng bá, ràng buộc tồn kho hoặc một tầng recommender độc lập. Khi đó, vấn đề quan trọng không còn chỉ là “gợi ý gì”, mà là “nên gợi ý vào thời điểm nào” để người dùng có khả năng phát sinh tương tác cao nhất.

Nghiên cứu này đề xuất hướng tiếp cận **dự đoán thời điểm tương tác có điều kiện theo hành động** cho dữ liệu hành vi người dùng có dấu thời gian. Thay vì mô hình hóa đồng thời item tiếp theo và thời điểm tương tác tiếp theo, nghiên cứu tách riêng thành phần thời gian và xem item hoặc action là một điều kiện đầu vào đã được chọn trước. Với lịch sử tương tác của người dùng và một item/action mục tiêu, bài toán được đặt ra là ước lượng thời điểm hoặc khoảng thời gian mà người dùng có khả năng phát sinh event mong muốn cao nhất.

Hướng tiếp cận này bắt đầu bằng việc đánh giá tính khả thi trên nhiều tập dữ liệu hành vi có timestamp. Dữ liệu được chuẩn hóa về schema sự kiện chung gồm `user_id`, `item_id`, `event_type` và `timestamp`. Từ đó, target thời gian ban đầu được xây dựng bằng khoảng cách giữa hai sự kiện liên tiếp của người dùng. Các baseline thống kê và mô hình học máy nhẹ được sử dụng để kiểm tra liệu dữ liệu có chứa tín hiệu đủ mạnh cho bài toán dự đoán thời gian hay không. Do việc hồi quy trực tiếp trên `delta_t` thường gặp khó khăn bởi phân phối lệch, long-tail và dữ liệu thưa, nghiên cứu tiếp tục định hướng bài toán sang mô hình hóa time-to-event, bao gồm discrete-time hazard modeling, survival analysis và temporal point process.

Thay vì dự đoán một giá trị thời gian chính xác duy nhất, framework đề xuất ước lượng xác suất hoặc cường độ người dùng phát sinh event trong các khoảng thời gian tương lai, có điều kiện theo item/action đã chọn và lịch sử hành vi của người dùng. Từ đó, hệ thống có thể lựa chọn thời điểm kích hoạt action sao cho tối đa hóa xác suất event hoặc lợi ích kỳ vọng. Bằng cách tập trung riêng vào tầng timing, nghiên cứu cung cấp một bước trung gian thực dụng giữa các baseline dự đoán thời gian đơn giản và các mô hình gợi ý phức tạp kết hợp đồng thời item và thời gian.

## Lí giải đề xuất

Hướng nghiên cứu này xuất phát từ hạn chế của bài toán supervised learning trực tiếp trên target thời gian. Nếu đặt bài toán dưới dạng:

```text
X → delta_t_seconds
```

mô hình phải dự đoán chính xác khoảng thời gian từ event hiện tại đến event tiếp theo. Tuy nhiên, trong dữ liệu hành vi người dùng, `delta_t_seconds` thường có phân phối rất lệch, long-tail mạnh và nhiễu cao. Một số event xảy ra sau vài giây hoặc vài phút, trong khi một số event khác có thể cách nhau nhiều ngày hoặc nhiều tuần. Điều này làm cho các mô hình regression thông thường dễ có MAE hoặc RMSE rất cao, đồng thời kết quả khó diễn giải về mặt hệ thống.

Vì vậy, thay vì xem bài toán là dự đoán một giá trị thời gian chính xác, hướng đề xuất chuyển sang tư duy **time-to-event modeling**. Mô hình không nhất thiết phải trả lời chính xác user sẽ quay lại sau bao nhiêu giây. Thay vào đó, mô hình ước lượng xác suất hoặc cường độ user phát sinh event trong các khoảng thời gian tương lai. Cách đặt bài toán này phù hợp hơn với bản chất của dữ liệu hành vi, vì trong thực tế hệ thống thường cần biết khoảng thời gian nào có khả năng user phản hồi cao nhất, chứ không nhất thiết cần một timestamp tuyệt đối chính xác.

Một điểm quan trọng của đề xuất là tách riêng bài toán **timing prediction** khỏi bài toán **item prediction**. Một số nghiên cứu hiện đại đặt bài toán theo hướng joint modeling, tức là vừa dự đoán item tiếp theo vừa dự đoán thời điểm tương tác tiếp theo. Tuy nhiên, trong nhiều hệ thống thực tế, item hoặc action không nhất thiết phải được sinh hoàn toàn bởi mô hình. Chúng có thể được xác định trước bởi business rule, campaign objective, inventory constraint hoặc một recommendation layer độc lập. Ví dụ, hệ thống có thể đã biết cần đẩy một voucher, một sản phẩm tồn kho, một playlist, một artist mới hoặc một notification cụ thể. Khi đó, câu hỏi quan trọng hơn không phải là “recommend item nào?”, mà là “nên đưa item/action này đến user vào thời điểm nào?”.

Cách tách này cũng giúp giảm rủi ro **error propagation**. Nếu hệ thống dự đoán đồng thời item và time, sai số ở phần item prediction có thể ảnh hưởng đến time prediction, hoặc ngược lại. Nếu mô hình dự đoán sai item, thời điểm được dự đoán cho item đó có thể không còn ý nghĩa. Nếu mô hình dự đoán sai time trước, bước chọn item tại thời điểm đó cũng có thể bị lệch. Trong khi đó, khi item/action đã được chọn trước, mô hình có thể tập trung toàn bộ năng lực vào việc học temporal dynamics của user, tức là nhịp độ, chu kỳ, độ gần đây và khả năng quay lại của user theo thời gian.

Do đó, bài toán được định vị lại thành:

```text
Given:
    user history H_u(t)
    selected item/action a
    current time t

Predict:
    probability or intensity that user u will generate target event
    in future time windows
```

Nói cách khác, nghiên cứu không hỏi:

```text
User sẽ tương tác với item nào và khi nào?
```

mà hỏi:

```text
Với item/action đã chọn, khi nào nên kích hoạt để user có khả năng phát sinh event cao nhất?
```

Đây là một hướng thực dụng hơn trong bối cảnh recommender system hoặc campaign system. Nó phù hợp với các tình huống mà business đã có mục tiêu rõ ràng, còn hệ thống cần tối ưu thời điểm tiếp cận user. Vì vậy, hướng này có thể được gọi là:

```text
Action-Conditioned Event Timing Prediction
```

hoặc:

```text
Business-Conditioned Next-Time Prediction for Recommender Systems
```

## Pipeline đề xuất

Pipeline tổng quát của nghiên cứu có thể được mô tả như sau:

```text
Raw timestamped user behavior data
→ Normalize events into a common schema
→ Select target event type and business-conditioned item/action
→ Build user temporal history
→ Construct next-time target or future observation windows
→ Analyze target distribution and dataset feasibility
→ Train statistical and lightweight supervised baselines
→ Reformulate the task as time-to-event prediction
→ Estimate event probability or intensity over future time windows
→ Choose trigger time by maximizing event probability or expected utility
→ Evaluate timing quality, calibration, and business-oriented event lift
```

Ở bước đầu, dữ liệu từ nhiều nguồn khác nhau được chuẩn hóa về một schema chung:

```text
user_id
item_id
event_type
timestamp
```

Schema này giúp thống nhất các dataset có bản chất khác nhau như nghe nhạc, thương mại điện tử hoặc click behavior. Sau đó, các event được sắp xếp theo user và timestamp để tạo target ban đầu:

```text
delta_t_seconds = next_timestamp - current_timestamp
```

Target này được dùng cho bước baseline ban đầu. Mục tiêu của baseline không phải là đạt kết quả mạnh nhất, mà là kiểm tra xem dữ liệu có chứa tín hiệu thời gian đủ rõ hay không. Nếu baseline thống kê như `user_median` đã rất khó vượt qua, hoặc nếu regression trực tiếp cho sai số quá lớn, điều đó cho thấy cần chuyển sang formulation phù hợp hơn.

Formulation mạnh hơn là chia tương lai thành các time window hoặc risk interval. Thay vì dự đoán một giá trị `delta_t_seconds`, mô hình dự đoán xác suất event trong từng khoảng thời gian:

```text
P(event in next 1 hour)
P(event in next 6 hours)
P(event in next 24 hours)
P(event in next 7 days)
```

Từ đó, hệ thống có thể chọn thời điểm có xác suất event cao nhất hoặc có expected utility cao nhất. Đây là cách tiếp cận gần với discrete-time hazard modeling, survival analysis và temporal point process.

## Hướng mô hình hóa

Ba nhóm mô hình phù hợp với đề xuất này là **discrete-time hazard model**, **survival analysis** và **temporal point process**.

Discrete-time hazard model là lựa chọn dễ triển khai nhất. Mô hình chia thời gian tương lai thành các bucket và dự đoán xác suất user phát sinh event trong từng bucket, với điều kiện user chưa phát sinh event trước đó. Cách này vẫn có thể dùng các mô hình học máy quen thuộc như logistic regression, gradient boosting hoặc neural network nhẹ, nhưng formulation có ý nghĩa time-to-event rõ ràng hơn classification bucket thông thường.

Survival analysis là hướng thống kê chính quy hơn cho bài toán time-to-event. Thay vì chỉ dự đoán nhãn bucket, survival model học xác suất user chưa phát sinh event cho đến thời điểm `t`, hoặc hazard tại thời điểm `t`. Hướng này đặc biệt phù hợp nếu dữ liệu có censoring, tức là có những user chưa phát sinh event tiếp theo trong khoảng quan sát.

Temporal point process là hướng gần nhất với các paper mạnh như Hawkes process. Mô hình học intensity function:

```text
λ_u,a(t | H_u)
```

Trong đó `u` là user, `a` là item/action đã chọn, `t` là thời điểm cần đánh giá, và `H_u` là lịch sử hành vi của user. Intensity càng cao thì khả năng user phát sinh event tại thời điểm đó càng lớn. Nếu muốn mô hình hóa hiệu ứng lặp lại, có thể thêm self-excitation; nếu muốn mô hình hóa thói quen theo ngày hoặc tuần, có thể thêm thành phần chu kỳ.

Một dạng trực quan của intensity có thể là:

```text
λ_u,a(t) =
    base preference
    + recency effect
    + frequency effect
    + periodicity effect
    + user-level temporal behavior
```

Với hướng này, mô hình không chỉ học “user thường quay lại sau bao lâu”, mà còn học “xác suất user phản hồi thay đổi như thế nào theo thời gian”.

## Đóng góp kỳ vọng

Đóng góp chính của nghiên cứu không nằm ở việc đề xuất ngay một mô hình SOTA cho joint item-time recommendation. Thay vào đó, nghiên cứu đóng góp một cách đặt bài toán thực dụng hơn: tách riêng **timing layer** khỏi **item selection layer** trong recommender systems.

Cụ thể, nghiên cứu có ba đóng góp kỳ vọng. Thứ nhất, nó đánh giá tính khả thi của next-time prediction trên nhiều dataset hành vi có timestamp, thay vì chỉ kiểm tra trên một dataset đơn lẻ. Thứ hai, nó chỉ ra hạn chế của supervised regression trực tiếp trên `delta_t_seconds` trong dữ liệu long-tail và sparse. Thứ ba, nó đề xuất chuyển bài toán sang action-conditioned time-to-event modeling, nơi hệ thống dự đoán xác suất hoặc intensity của event theo thời gian cho một item/action đã được chọn trước.

Cách tiếp cận này tạo ra một bước trung gian hợp lý giữa baseline đơn giản và các mô hình temporal recommendation phức tạp. Nếu baseline cho thấy dữ liệu có tín hiệu thời gian, nghiên cứu có thể mở rộng sang hazard modeling, survival analysis hoặc point process. Nếu một số dataset không có tín hiệu đủ mạnh, điều đó cũng là kết quả có giá trị vì nó cho thấy không phải mọi timestamped recommender dataset đều phù hợp cho next-time prediction.

## Kết luận định hướng

Tóm lại, hướng nghiên cứu đề xuất không cố gắng giải quyết đồng thời toàn bộ bài toán “recommend what and when”. Thay vào đó, nó tập trung vào một lớp quan trọng nhưng thường bị trộn lẫn với item prediction: **khi nào nên kích hoạt một item/action đã được chọn**. Đây là một hướng phù hợp với nhiều hệ thống recommendation thực tế, nơi item có thể đến từ business rule, campaign objective hoặc một recommender module khác.

Bằng cách tập trung vào timing, nghiên cứu có thể phân tích sâu hơn temporal behavior của user, giảm ảnh hưởng của error propagation từ item prediction, và xây dựng một pipeline rõ ràng từ feasibility analysis, baseline modeling đến time-to-event modeling. Hướng này vừa có giá trị thực tiễn cho hệ thống recommendation/campaign, vừa có nền tảng học thuật từ survival analysis, hazard modeling và temporal point process.