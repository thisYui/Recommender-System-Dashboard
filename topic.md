# Research Gaps in Temporal and Cyclic Bias in Recommender Systems

## 1. Temporal Bias and Fairness

Một khoảng trống quan trọng trong recommender system là mối quan hệ giữa **temporal bias** và **fairness**. Hiện nay đã có nhiều nghiên cứu về fairness trong recommender system, cũng như nhiều nghiên cứu về time-aware recommendation. Tuy nhiên, vẫn còn ít nghiên cứu tập trung vào câu hỏi: hệ thống recommendation có còn công bằng hay không khi thay đổi thời điểm đưa ra gợi ý.

Một hệ thống có thể đạt fairness tốt nếu đánh giá trung bình trên toàn bộ dữ liệu, nhưng khi tách theo thời gian, sự thiên lệch có thể xuất hiện rõ hơn. Ví dụ, vào buổi sáng hệ thống có thể ưu tiên một nhóm item nhất định, trong khi vào buổi tối lại ưu tiên nhóm item khác. Trong mùa sale hoặc dịp đặc biệt, các brand phổ biến có thể được tăng exposure, còn các item nhỏ hoặc item long-tail bị giảm khả năng xuất hiện.

Vấn đề này có thể được gọi là:

```text
Temporal Fairness in Recommender Systems
```

hoặc:

```text
Cyclic Bias in Time-Aware Recommendation
```

Ý tưởng chính là đánh giá fairness không chỉ trên toàn bộ dataset, mà còn theo từng time bucket như giờ trong ngày, ngày trong tuần, tháng, mùa hoặc sự kiện đặc biệt.

---

## 2. Temporal Robustness of LLM-based Recommender Systems

LLM-based recommender system đang là một xu hướng lớn trong các nghiên cứu hiện tại. LLM có thể giúp hệ thống hiểu ngữ nghĩa, hiểu intent của user, xử lý review/comment, tạo explanation và cá nhân hóa recommendation tốt hơn. Tuy nhiên, một khoảng trống còn mở là tính ổn định của LLM recommender khi thay đổi ngữ cảnh thời gian.

Ví dụ, với cùng một user profile và cùng một nhu cầu:

```text
Recommend a laptop for a student.
```

nếu thêm các time context khác nhau:

```text
It is back-to-school season.
It is Black Friday.
It is exam season.
It is late at night.
```

LLM có thể đưa ra các recommendation khác nhau. Vấn đề cần phân tích là sự thay đổi đó có thật sự hợp lý theo ngữ cảnh hay chỉ là một dạng temporal bias hoặc seasonal stereotype.

Hướng nghiên cứu này có thể được đặt tên là:

```text
Temporal Robustness of LLM-based Recommender Systems
```

hoặc:

```text
Time-Context Sensitivity in LLM Recommendations
```

Mục tiêu là kiểm tra xem LLM recommender có quá nhạy với các mô tả thời gian hay không, và liệu sự nhạy cảm đó có tạo ra bias trong recommendation hay không.

---

## 3. Distinguishing Temporal Personalization from Temporal Bias

Không phải mọi thay đổi recommendation theo thời gian đều là xấu. Trong nhiều trường hợp, việc hệ thống thay đổi recommendation theo thời điểm là hợp lý. Ví dụ, buổi sáng hệ thống gợi ý cà phê, buổi trưa gợi ý món ăn chính, buổi tối gợi ý phim dài hoặc nội dung giải trí. Đây có thể được xem là **temporal personalization**.

Tuy nhiên, nếu hệ thống liên tục ưu tiên một nhóm item, brand hoặc user group tại một thời điểm nhất định mà không có bằng chứng rõ ràng từ preference thật của user, thì đó có thể trở thành **temporal bias**.

Khoảng trống ở đây là cách phân biệt giữa hai khái niệm:

```text
Temporal Personalization
```

và:

```text
Temporal Bias
```

Một hướng nghiên cứu tốt là xây dựng tiêu chí đánh giá để trả lời câu hỏi:

```text
Khi nào recommendation thay đổi theo thời gian là personalization hợp lý?
Khi nào sự thay đổi đó trở thành cyclic bias?
```

Tên hướng có thể là:

```text
Distinguishing Temporal Personalization from Temporal Bias
```

Đây là một vấn đề quan trọng vì nếu đánh giá quá đơn giản, ta có thể nhầm personalization hợp lý thành bias, hoặc ngược lại, bỏ qua bias thật vì cho rằng đó chỉ là personalization.

---

## 4. Temporal Exposure Diversity

Một khoảng trống khác là **exposure diversity theo thời gian**. Nhiều recommender system tối ưu click, rating, purchase hoặc ranking accuracy, nhưng chưa đánh giá đầy đủ xem exposure của các item có bị lệch theo chu kỳ thời gian hay không.

Ví dụ, trong e-commerce, vào các dịp sale lớn, hệ thống có thể liên tục đẩy các brand lớn lên đầu kết quả recommendation. Khi đó, các seller nhỏ, item mới hoặc long-tail item có thể gần như không có cơ hội xuất hiện. Điều này tạo ra vấn đề không chỉ cho user, mà còn cho item provider, seller hoặc content creator.

Hướng này có thể được gọi là:

```text
Temporal Exposure Bias
```

hoặc:

```text
Time-Aware Exposure Fairness
```

Các chỉ số có thể phân tích gồm:

```text
Exposure share by item group
Exposure share by brand
Long-tail item visibility
Category diversity across time buckets
Popularity bias across seasons or events
```

Ý tưởng chính là không chỉ đánh giá hệ thống recommend đúng hay sai, mà còn đánh giá item nào được nhìn thấy nhiều hơn ở từng thời điểm.

---

## 5. Benchmarking Cyclic Bias in Recommender Systems

Một khó khăn lớn khi nghiên cứu cyclic bias là thiếu benchmark và evaluation protocol rõ ràng. Nhiều dataset recommender chỉ có user, item, rating và timestamp, nhưng không có đầy đủ thông tin về context như mùa, sự kiện, campaign, thời tiết, brand group hoặc exposure log.

Để nghiên cứu cyclic bias, dataset lý tưởng nên có các thông tin sau:

```text
user_id
item_id
timestamp
interaction type
item category
item brand / provider group
ranking position or exposure log
hour of day
day of week
month / season
event or holiday flag
```

Một hướng nghiên cứu khả thi là xây dựng protocol audit thay vì xây model quá phức tạp. Quy trình có thể gồm:

```text
1. Chia dữ liệu thành các time bucket
2. Train hoặc lấy output từ recommender system
3. Đánh giá accuracy theo từng time bucket
4. Đo exposure diversity theo từng time bucket
5. Đo popularity bias hoặc brand bias theo từng time bucket
6. So sánh temporal personalization và temporal bias
```

Tên hướng có thể là:

```text
Benchmarking Cyclic Bias in Recommender Systems
```

hoặc:

```text
Evaluation Protocol for Temporal Bias in Recommender Systems
```

Hướng này phù hợp với project nghiên cứu vì có thể triển khai bằng các dataset phổ biến như MovieLens, Amazon Review, Yelp hoặc các dataset có timestamp. Điểm mạnh của hướng này là tập trung vào audit và evaluation, không bắt buộc phải xây dựng một recommender model quá phức tạp.
