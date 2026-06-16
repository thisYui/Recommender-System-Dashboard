# Paper 1: Point Process Based Time Sensitive Personalised Recommendation

## Mục đích đọc paper

Paper này phù hợp với hướng nghiên cứu mới về **next-time prediction** vì nó đặt recommendation trong bối cảnh thời gian. Thay vì chỉ hỏi “user sẽ thích item nào”, paper đặt thêm câu hỏi “khi nào user có khả năng quay lại hoặc tương tác lại”. Đây là điểm gần với target `delta_t_seconds = next_timestamp - current_timestamp` trong notebook hiện tại. Tuy nhiên, cần lưu ý rằng phần thực nghiệm của paper chủ yếu đánh giá chất lượng **ranking recommendation** bằng NDCG, còn phần dự đoán returning time được mô tả trong framework nhưng không được báo cáo sâu bằng MAE/RMSE.

## Bài toán paper giải quyết

Paper cho rằng nhiều recommender system truyền thống chỉ tạo recommendation theo kiểu tĩnh, dựa trên sở thích tổng quát hoặc lịch sử tương tác, nhưng không xét đến việc hành vi người dùng thường có tính lặp lại theo thời gian. Ví dụ, người dùng có thể nghe một nhóm nhạc vào một khung giờ quen thuộc, mua một số loại hàng vào những ngày nhất định, hoặc quay lại một dịch vụ sau một khoảng thời gian tương đối ổn định. Vì vậy, paper hướng đến **time-sensitive personalised recommendation**, tức là đề xuất đúng item cho đúng user tại đúng thời điểm.

Paper nêu hai mục tiêu chính. Mục tiêu thứ nhất là recommend item phù hợp cho user tại thời điểm `t`. Mục tiêu thứ hai là ước lượng returning time, tức là khi nào user có khả năng quay lại service hoặc tiếp tục hành vi lặp lại. Hai mục tiêu này làm cho paper có liên quan trực tiếp đến hướng next-time prediction, dù cách đánh giá thực nghiệm của paper vẫn thiên về ranking hơn là regression trên thời gian.

## Dataset được sử dụng

Paper sử dụng một dataset thực nghiệm chính là **Last.fm**. Dataset này gồm hành vi nghe nhạc của khoảng **1200 users**, **1000 artists** và sinh ra khoảng **48000 events**. Last.fm được chọn vì nghe nhạc là một loại hành vi có tính lặp lại tự nhiên: user có thể nghe lại cùng artist hoặc cùng nhóm artist vào một số thời điểm quen thuộc. Điều này phù hợp với giả thuyết của paper rằng user-item interactions không chỉ phụ thuộc vào sở thích tĩnh, mà còn phụ thuộc vào nhịp độ và chu kỳ tiêu dùng.

So với project hiện tại, Last.fm cũng là dataset rất hợp lý cho bài toán temporal consumption. Nếu project muốn bắt đầu bằng next-time prediction, Last.fm nên được xem là dataset ưu tiên vì timestamp của nó gần với thời điểm user thực sự nghe nhạc, thay vì chỉ là thời điểm rating như MovieLens.

## Tiền xử lý và biểu diễn dữ liệu

Paper biểu diễn dữ liệu giao dịch dưới dạng event theo thời gian. Mỗi event có dạng `(user, item, time)`, trong đó `user` là người dùng, `item` là artist hoặc item được tiêu thụ, và `time` là timestamp của tương tác. Cách biểu diễn này gần với schema chuẩn trong notebook hiện tại: `user_id`, `item_id`, `event_type`, `timestamp`.

Từ các event này, paper xây dựng lịch sử tương tác của user với item trước một thời điểm `T`. Lịch sử đó được dùng để tính intensity tại thời điểm cần recommendation. Điểm quan trọng là timestamp không bị xem như một feature phụ, mà được dùng để mô hình hóa quá trình phát sinh event. Với project hiện tại, bước tương đương là sắp xếp event theo user và timestamp, sau đó tạo target `delta_t_seconds` hoặc `time_bucket` từ cặp event liên tiếp.

Paper không trình bày chi tiết các bước cleaning, lọc user/item hoặc cách chia train-test như nhiều paper deep learning hiện đại. Vì vậy, khi học pipeline từ paper này, phần nên rút ra là cách **biểu diễn event sequence và intensity**, không phải một template preprocessing đầy đủ.

## Pipeline mô hình

Pipeline của paper có thể được hiểu là một chuỗi xử lý từ dữ liệu hành vi thô đến danh sách recommendation có xét yếu tố thời gian. Điểm quan trọng của pipeline này là paper không chỉ học user thích item nào, mà còn học **cường độ tương tác** giữa user và item tại từng thời điểm. Cường độ này được gọi là **intensity**. Nếu intensity của một cặp user-item tại thời điểm `t` càng cao, mô hình hiểu rằng user càng có khả năng tương tác với item đó tại thời điểm `t`.

```text
Raw user behavior data with timestamps
→ Represent interactions as (user, item, time)
→ Estimate intrinsic user-item preference by Hierarchical Poisson Factorization
→ Use HPF output as base intensity
→ Add Hawkes process component for temporal self-excitation
→ Add sinusoidal cyclic term and exponential decay
→ Compute user-item intensity at time t
→ Rank candidate items by intensity
→ Optionally sample next returning time using Ogata's thinning algorithm
```

Bước đầu tiên của pipeline là lấy raw user behavior data with timestamps. Đây là dữ liệu hành vi người dùng có thông tin thời gian, ví dụ một user nghe một artist tại một thời điểm cụ thể trong Last.fm. Paper biểu diễn mỗi interaction dưới dạng bộ ba (user, item, time). Cách biểu diễn này rất quan trọng vì nó chuyển bài toán recommendation từ dạng ma trận user-item tĩnh sang dạng chuỗi sự kiện theo thời gian. Nếu chỉ có (user, item), mô hình chỉ biết user đã từng tương tác với item nào. Nhưng khi có thêm time, mô hình có thể học được user tương tác vào lúc nào, tương tác có lặp lại không, và khoảng cách giữa các lần tương tác là bao lâu.

Sau khi chuẩn hóa dữ liệu thành các event có timestamp, paper dùng Hierarchical Poisson Factorization để ước lượng sở thích nền giữa user và item. Có thể hiểu HPF là thành phần học long-term preference. Ví dụ, nếu một user thường xuyên nghe nhạc rock, HPF sẽ học được rằng user đó có xu hướng thích các artist thuộc nhóm này. Nếu một artist được nhiều user tương tự nghe, HPF cũng có thể học được latent attribute của artist đó. Thành phần này chưa tập trung vào thời điểm cụ thể, mà chủ yếu trả lời câu hỏi: “về tổng thể, user này có quan tâm item này không?”. Trong framework của paper, kết quả từ HPF được dùng làm base intensity, tức là cường độ nền ban đầu cho cặp user-item.

Tuy nhiên, chỉ dùng HPF thì mô hình vẫn còn khá tĩnh. Nó biết user có thể thích item nào, nhưng chưa biết tại thời điểm nào user có khả năng quay lại tương tác. Vì vậy paper thêm Hawkes process để mô hình hóa yếu tố thời gian. Hawkes process khác với Poisson process thông thường ở chỗ nó không xem các event là độc lập hoàn toàn. Trong Hawkes process, những event đã xảy ra trong quá khứ có thể làm tăng xác suất hoặc cường độ xảy ra event trong tương lai. Đây được gọi là self-excitation. Ví dụ, nếu một user vừa nghe một artist hôm nay, khả năng user tiếp tục nghe lại artist đó trong tương lai gần có thể cao hơn so với một artist đã lâu không nghe. Paper cũng nhấn mạnh khác biệt này: Poisson process giả định event độc lập, còn Hawkes process để future evolution phụ thuộc vào toàn bộ lịch sử trước đó.

Với recommender system, self-excitation có thể hiểu là hiệu ứng “vừa tương tác xong nên còn khả năng tương tác tiếp”. Điều này rất khác với baseline thống kê đơn giản như user median hoặc global median. Baseline median chỉ học khoảng cách thời gian điển hình của user, nhưng không phân biệt rõ item nào vừa được tương tác, item nào có lịch sử lặp lại, hoặc event gần nhất có làm tăng khả năng quay lại hay không. Hawkes process giúp đưa những yếu tố này vào intensity function.

Thành phần tiếp theo là cyclic behavior. Đây là điểm đáng chú ý của paper, vì không phải mọi hành vi lặp lại đều chỉ đến từ hiệu ứng gần đây. Nhiều hành vi có tính chu kỳ. Ví dụ, một user có thể nghe nhạc vào buổi tối, mua đồ vào cuối tuần, hoặc quay lại một dịch vụ vào một khung giờ quen thuộc. Nếu chỉ dùng exponential decay, mô hình chủ yếu bắt được hiệu ứng “event càng gần thì ảnh hưởng càng mạnh, event càng xa thì ảnh hưởng càng yếu”. Nhưng nếu hành vi có chu kỳ, một event xảy ra cách đây khá lâu vẫn có thể hữu ích nếu nó rơi vào cùng một pha thời gian, chẳng hạn cùng buổi tối hoặc cùng ngày trong tuần. Vì vậy paper thêm một hàm sinusoidal kết hợp với exponential decay để vừa mô hình hóa chu kỳ, vừa làm giảm dần ảnh hưởng của event cũ theo thời gian. Abstract của paper cũng nêu rõ họ dùng Hawkes process với initial intensity dựa trên HPF, đồng thời dùng sinusoidal function kết hợp exponential effect decay để mô tả dynamic activity phù hợp với activity cycle như nghe nhạc vào một thời điểm cụ thể trong ngày.

Có thể hiểu trực quan intensity cuối cùng của một cặp user-item gồm ba phần. Phần thứ nhất là sở thích nền: user có thích item này không. Phần thứ hai là hiệu ứng lịch sử gần đây: user hoặc cộng đồng có vừa tương tác với item này không. Phần thứ ba là hiệu ứng chu kỳ: thời điểm hiện tại có giống với những thời điểm mà user thường tương tác với item này trong quá khứ không. Khi ba yếu tố này kết hợp lại, mô hình có thể cho intensity cao cho những item vừa phù hợp với sở thích dài hạn, vừa có tín hiệu tương tác gần đây, vừa xuất hiện đúng chu kỳ hành vi của user.

Sau khi tính được intensity λ_up(t) cho từng cặp user-item tại thời điểm t, paper dùng intensity này để xếp hạng candidate items. Item nào có intensity cao hơn sẽ được xếp cao hơn trong recommendation list. Vì vậy, output chính trong phần recommendation không phải là một giá trị rating tĩnh, mà là một ranking phụ thuộc vào thời điểm. Cùng một user có thể nhận được danh sách recommendation khác nhau ở các thời điểm khác nhau, vì intensity thay đổi theo lịch sử và chu kỳ thời gian. Figure 3 trong paper cũng mô tả rõ pipeline này: dữ liệu hành vi có temporal details được đưa qua HPF, sau đó model multivariate Hawkes process intensity, rồi rank predicted items at time t theo intensity.

Một nhánh khác trong framework là dự đoán next returning time. Sau khi đã có intensity function theo thời gian, về mặt lý thuyết có thể suy ra khi nào event tiếp theo có khả năng xảy ra. Paper nhắc đến việc dùng Ogata's thinning algorithm để sample thời điểm quay lại tiếp theo. Ý tưởng là thay vì trực tiếp tính một công thức đóng cho next event time, mô hình có thể mô phỏng các thời điểm event khả dĩ từ point process, sau đó dùng các sample này để ước lượng thời điểm user quay lại. Đây là phần liên quan trực tiếp đến hướng delta_t_seconds trong notebook hiện tại. Tuy nhiên, khi viết report cần lưu ý rằng paper có mô tả nhánh này trong framework, nhưng phần thực nghiệm chính lại tập trung vào ranking recommendation bằng NDCG hơn là đánh giá sai số dự đoán thời gian bằng MAE hoặc RMSE.

Tóm lại, pipeline của paper có thể được diễn giải như một mô hình kết hợp giữa recommendation tĩnh và temporal event modeling. HPF đóng vai trò học sở thích user-item dài hạn. Hawkes process thêm khả năng mô hình hóa ảnh hưởng của các event quá khứ lên tương lai. Thành phần sinusoidal giúp bắt các hành vi có chu kỳ. Intensity cuối cùng được dùng để rank item tại một thời điểm cụ thể, và cũng có thể được dùng để sample next returning time. Đây là lý do paper này phù hợp làm related work cho hướng next-time prediction, dù phần thực nghiệm của paper chưa đánh giá trực tiếp delta_t_seconds như notebook hiện tại.

## Evaluation và metrics

Phần thực nghiệm của paper dùng **NDCG** để đánh giá chất lượng ranking của recommendation list. NDCG phù hợp với top-k recommendation vì nó quan tâm đến vị trí của item đúng trong danh sách xếp hạng. Paper so sánh mô hình đề xuất **HPFCB** với baseline **Hierarchical Poisson Factorization**. Khi kích thước top-k tăng, cả HPF và HPFCB đều cải thiện, nhưng HPFCB được báo cáo là vượt baseline với khoảng cách rõ rệt.

Điểm cần ghi nhớ là paper không báo cáo rõ metric hồi quy cho returning-time prediction, chẳng hạn MAE hoặc RMSE. Dù framework có nhắc đến Ogata's thinning algorithm để sample next returning time, phần kết quả thực nghiệm chủ yếu là NDCG cho ranking. Vì vậy, nếu đưa paper này vào report, nên viết rằng paper này hỗ trợ ý tưởng **time-sensitive recommendation bằng point process**, nhưng không nên nói rằng nó đánh giá trực tiếp `delta_t_seconds` giống notebook hiện tại.

## Vấn đề, giới hạn và bài học cho hướng mới

Giới hạn đầu tiên là paper chỉ dùng một dataset thực nghiệm chính là Last.fm. Điều này làm cho kết quả có tính tập trung vào domain nghe nhạc, trong khi chưa chứng minh rộng trên e-commerce, news click hoặc nhiều domain khác. Với project hiện tại, đây là lý do nên kiểm tra thêm RetailRocket hoặc Taobao nếu muốn chứng minh hướng next-time prediction không chỉ đúng với music.

Giới hạn thứ hai là baseline so sánh khá hẹp. Paper so sánh chủ yếu với HPF, trong khi chưa so với nhiều sequential recommender mạnh hoặc time-aware deep model. Vì vậy, paper hữu ích để học ý tưởng point process và cyclic behavior, nhưng chưa đủ để làm chuẩn đánh giá mạnh cho state-of-the-art.

Giới hạn thứ ba là sự lệch giữa motivation và evaluation. Paper nói đến returning-time prediction, nhưng phần kết quả lại tập trung vào ranking/NDCG. Đây là điểm cần tránh trong project hiện tại. Nếu project của mình nói bài toán chính là predict time, thì evaluation cần có metric trực tiếp cho time như MAE trên log-delta, RMSE trên log-delta, median absolute error theo giây/giờ, hoặc accuracy/F1 nếu chuyển thành time bucket.

Bài học lớn nhất từ paper này là: nếu dữ liệu có timestamp và hành vi lặp lại, ta có thể mô hình hóa interaction như một quá trình phát sinh event theo thời gian. Baseline notebook hiện tại có thể bắt đầu bằng median/regression/classification, nhưng nếu dữ liệu cho thấy có tín hiệu, hướng nâng cao hợp lý là point process, Hawkes process, survival model hoặc một mô hình có thành phần cyclic time encoding.
