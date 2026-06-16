# Paper 3: When and What to Recommend: Joint Modeling of Timing and Content for Active Sequential Recommendation

## Mục đích đọc paper

Paper này đại diện cho hướng hiện đại hơn của time-aware recommendation. Thay vì chỉ dự đoán item tiếp theo hoặc chỉ dự đoán thời gian tiếp theo, paper đặt bài toán **active sequential recommendation**, trong đó hệ thống phải quyết định cả **khi nào nên recommend** và **nên recommend item nào**. Đây là hướng rất gần với mục tiêu dài hạn của project nếu muốn đi từ next-time prediction sang joint time-item recommendation.

Paper đề xuất mô hình **PASRec**. Ý tưởng chính là mô hình hóa đồng thời **Time of Interest** và **Item of Interest**. Time of Interest là thời điểm user có khả năng tương tác tiếp theo. Item of Interest là item phù hợp tại thời điểm đó. Nếu project hiện tại đang kiểm tra khả năng dự đoán `delta_t_seconds`, thì paper này có thể được xem là hướng mở rộng: sau khi biết khi nào user quay lại, hệ thống cần biết item nào nên được đề xuất tại thời điểm đó.

## Bài toán paper giải quyết

Paper cho rằng phần lớn sequential recommender hiện tại là passive recommender. Nghĩa là hệ thống chỉ recommend khi user chủ động mở ứng dụng hoặc truy cập service. Trong thực tế, hệ thống có thể muốn chủ động gửi recommendation hoặc notification vào thời điểm user có khả năng quan tâm. Nhưng nếu gửi sai thời điểm, recommendation có thể bị xem là spam. Nếu đúng thời điểm nhưng item không phù hợp, recommendation cũng không có giá trị.

Vì vậy, paper đặt bài toán dưới dạng joint modeling giữa ToI và IoI. Một cách trực quan, pipeline không nên tách thành hai bước độc lập hoàn toàn như “dự đoán thời gian trước, rồi dự đoán item sau”, vì sai số ở bước dự đoán thời gian có thể lan sang bước dự đoán item. Paper gọi đây là single point of failure. PASRec được thiết kế để giảm vấn đề đó bằng cách học quan hệ giữa timing và content trong cùng một framework.

## Dataset được sử dụng

Paper sử dụng **5 benchmark datasets**. Ba dataset đầu tiên đến từ Amazon 2014 review data, gồm các category **Beauty**, **Sports** và **Toys**. Hai dataset còn lại là **MovieLens-10M** và **MovieLens-20M**. Như vậy, paper kết hợp cả e-commerce review data và movie rating data để kiểm tra khả năng generalize.

Theo bảng thống kê trong paper, Beauty có **22363 sequences**, **12101 items**, **162150 actions** và average sequence length khoảng **7.25**. Toys có **19412 sequences**, **11921 items**, **138444 actions** và average length khoảng **7.13**. Sports có **35598 sequences**, **18357 items**, **256598 actions** và average length khoảng **7.21**. MovieLens-10M có **69878 sequences**, **10027 items**, **3054340 actions** và average length khoảng **43.71**. MovieLens-20M có **138493 sequences**, **17177 items**, **6013602 actions** và average length khoảng **43.42**.

Điểm đáng chú ý là Amazon datasets có sequence ngắn hơn nhiều so với MovieLens, trong khi sparsity của tất cả dataset đều rất cao. Điều này phản ánh một khó khăn phổ biến của sequential recommendation: dữ liệu user-item cực kỳ sparse, và nhiều user không có chuỗi tương tác dài.

## Tiền xử lý và data construction

Paper xử lý tất cả user-item interactions như **implicit positive feedback**. Các interaction không quan sát được được xem như negative. Đây là cách làm phổ biến trong sequential recommendation, đặc biệt khi dữ liệu là rating hoặc review nhưng mục tiêu là dự đoán hành vi tương tác tiếp theo.

Các interaction của mỗi user được sắp xếp theo timestamp để tạo thành historical interaction sequence. Paper lọc bỏ user hoặc item có ít hơn **5 interactions**. Sequence được giới hạn độ dài: **50** cho MovieLens và **10** cho Amazon. Các sequence ngắn hơn được padding bằng special token. Đây là một điểm rất thực dụng: thay vì để sequence dài bất kỳ, paper chuẩn hóa độ dài sequence để training bằng neural model ổn định hơn.

Về xử lý thời gian, raw timestamps được chuyển về **day-level granularity**, sau đó được shift để bắt đầu từ 0 và normalize để mô hình học temporal pattern hiệu quả hơn. Đây là chi tiết rất quan trọng cho project hiện tại. Nếu dùng nhiều dataset có timestamp khác nhau, ta cần quyết định độ phân giải thời gian: giây, phút, giờ hay ngày. PASRec chọn day-level vì các dataset như Amazon/MovieLens có hành vi không quá dày theo từng giây/phút.

Paper dùng hai chiến lược chia dữ liệu. Chiến lược thứ nhất là **8:1:1 temporal split**, tức là chia theo thứ tự thời gian thành train, validation và test. Chiến lược thứ hai là **leave-one-out**, trong đó interaction gần nhất của mỗi user dùng cho test, interaction gần nhì dùng cho validation, và phần còn lại dùng cho training. Đây là điểm rất liên quan với notebook hiện tại, vì notebook cũng đang dùng leave-last-one-out theo user để tránh leakage thời gian.

## Pipeline mô hình

Pipeline tổng quát của PASRec có thể mô tả như sau:

```text
Raw user-item interactions with timestamps
→ Convert interactions to implicit feedback
→ Sort interactions by user and timestamp
→ Filter users/items with fewer than 5 interactions
→ Truncate or pad user sequences
→ Convert timestamps to day-level, shift and normalize time
→ Encode historical item sequence and historical time sequence
→ Predict Time of Interest representation
→ Fuse predicted ToI information into user representation
→ Use diffusion-based module to generate/predict Item of Interest
→ Optimize joint objective for ToI and IoI
→ Evaluate next-item recommendation under LOO and temporal split
```

PASRec có nhiều module. Đầu tiên là phần time encoding. Paper thử nhiều cách encode thời gian, gồm sinusoidal function, Gaussian kernel function và Random Fourier Feature. Mục tiêu là biểu diễn timestamp thực tế chứ không chỉ encode vị trí thứ tự của item trong sequence. Đây là điểm khác so với nhiều sequential recommenders chỉ dùng positional embedding.

Tiếp theo là **ToI Prediction Module**. Module này ước lượng time of interest tiếp theo của user dựa trên lịch sử tương tác và thời gian. Sau đó, thông tin ToI được đưa vào representation của user để hỗ trợ dự đoán item. Điều này phản ánh giả định quan trọng của paper: user preference phụ thuộc vào thời điểm. Cùng một chuỗi item nhưng nếu thời gian giữa các event khác nhau, item tiếp theo có thể khác nhau.

Phần **IoI Prediction Module** dùng diffusion-based framework. Diffusion model được dùng để mô hình hóa phân phối phức tạp của user preference và sinh/predict item representation. Điểm khác biệt của PASRec so với các diffusion recommender trước đó là nó không chỉ dùng item sequence, mà còn đưa predicted ToI vào quá trình tạo recommendation.

## Training objective và inference

PASRec tối ưu một joint objective gồm ToI prediction loss và IoI prediction loss. Paper lập luận rằng việc tối ưu đồng thời hai loss giúp mô hình tăng mutual information giữa biểu diễn thời gian và biểu diễn item. Nói đơn giản, mô hình không học “time” và “item” như hai bài toán tách rời, mà cố gắng học mối phụ thuộc giữa chúng.

Trong inference, hệ thống cần dự đoán ToI trước hoặc tạo biểu diễn ToI cho thời điểm quan tâm, sau đó dùng thông tin đó để dự đoán IoI. Kết quả cuối cùng vẫn được đánh giá chủ yếu như sequential recommendation: model phải xếp hạng item đúng cao hơn các item ứng viên khác. Đây là điểm cần chú ý: paper nói nhiều về ToI, nhưng bảng kết quả chính vẫn tập trung vào recommendation metrics cho item.

## Evaluation, metrics và baselines

Paper đánh giá bằng **Hit Rate@K** và **NDCG@K**, cụ thể có H@5, H@10, N@5 và N@10. Đây là các metric phổ biến cho next-item recommendation. Hit Rate đo xem item đúng có xuất hiện trong top-k hay không. NDCG đo thêm vị trí của item đúng trong ranking, nên nếu item đúng nằm ở top cao hơn thì NDCG tốt hơn.

Paper so sánh với **8 baseline models**, chia thành ba nhóm. Nhóm traditional sequential recommenders gồm GRU4Rec, SASRec và BERT4Rec. Nhóm time-aware recommenders gồm TiSASRec và MEANTIME. Nhóm generative recommenders gồm DreamRec, DiffuRec và PreferDiff. PASRec được báo cáo là đạt performance tốt nhất trên hầu hết dataset và cả hai cách split.

Paper cũng thực hiện ablation study, phân tích time encoding function, tác động của ToI information và độ chính xác của ToI Prediction Module. Đây là điểm đáng học: nếu project đi hướng mới, không chỉ cần bảng performance tổng quát, mà còn cần phân tích từng thành phần để chứng minh phần thời gian thực sự giúp ích.

## Vấn đề, giới hạn và bài học cho hướng mới

Paper này rất mạnh cho motivation “when and what to recommend”, nhưng nếu project hiện tại chỉ tập trung vào `delta_t_seconds`, cần đọc cẩn thận. Metrics chính trong bảng kết quả là Hit Rate và NDCG cho item recommendation, không phải MAE/RMSE cho thời gian. Paper có phân tích accuracy của ToI prediction, nhưng mục tiêu cuối cùng vẫn là cải thiện next-item recommendation trong active setting.

Một điểm khác là PASRec sử dụng deep learning và diffusion model, nên pipeline phức tạp hơn nhiều so với baseline notebook hiện tại. Nếu project còn ở giai đoạn feasibility, chưa nên triển khai ngay PASRec. Thay vào đó, nên học cách paper tổ chức pipeline: chuẩn hóa sequence, xử lý timestamp, lọc sequence ngắn, chia temporal split/leave-one-out, đánh giá bằng metric phù hợp và phân tích ablation.

Bài học quan trọng nhất là ToI và IoI có quan hệ chặt chẽ. Nếu chỉ dự đoán item mà bỏ qua thời gian, recommendation có thể không đúng ngữ cảnh. Nếu chỉ dự đoán thời gian mà không gắn với item, hệ thống chưa tạo được recommendation hoàn chỉnh. Vì vậy, hướng nghiên cứu dài hạn có thể là: đầu tiên kiểm tra next-time prediction bằng baseline; sau đó thêm next-item prediction; cuối cùng xây dựng mô hình joint time-item.

Với notebook hiện tại, paper này nên được đặt ở phần future direction. Nó cho thấy next-time prediction không phải một bài toán phụ tách rời, mà có thể trở thành thành phần trung tâm của active sequential recommendation. Tuy nhiên, khi viết report, nên phân biệt rõ: notebook hiện tại đánh giá khả năng dự đoán thời gian, còn PASRec đánh giá khả năng recommendation khi đưa thông tin thời gian vào mô hình.
