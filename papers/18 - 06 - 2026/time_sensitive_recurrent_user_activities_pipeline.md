# Paper 2: Time-Sensitive Recommendation From Recurrent User Activities

## Mục đích đọc paper

Paper *Time-Sensitive Recommendation From Recurrent User Activities* là paper nền tảng rất sát với hướng **predict next time**. Nó không chỉ làm item recommendation, mà còn đánh giá trực tiếp bài toán **returning-time prediction**, tức là dự đoán thời điểm user quay lại hoặc event tiếp theo xảy ra. Vì vậy, nếu project hiện tại xây dựng target `delta_t_seconds = next_timestamp - current_timestamp`, đây là một trong những paper quan trọng nhất để làm cơ sở lý thuyết và thực nghiệm.

## Bài toán paper giải quyết

Paper đặt ra hai câu hỏi chính trong time-sensitive recommendation. Câu hỏi thứ nhất là: tại một thời điểm `t`, nên recommend item nào cho user. Câu hỏi thứ hai là: với lịch sử tương tác đã quan sát, khi nào user sẽ quay lại hoặc tương tác tiếp theo. Hai câu hỏi này tương ứng với hai nhánh của bài toán “what to recommend” và “when to recommend”.

Điểm mạnh của paper là nó không xem thời gian chỉ như một đặc trưng phụ. Paper mô hình hóa mỗi chuỗi event theo thời gian bằng **temporal point process**, cụ thể là Hawkes process. Điều này cho phép mô hình học được cường độ phát sinh event tại từng thời điểm, từ đó vừa có thể ranking item, vừa có thể suy ra thời điểm event tiếp theo.

## Dataset được sử dụng

Paper dùng cả synthetic data và ba real datasets. Synthetic data được dùng để kiểm tra khả năng khôi phục tham số, tốc độ hội tụ và scalability của thuật toán. Ba real datasets được dùng để kiểm tra performance thực tế trên nhiều domain khác nhau.

Dataset thứ nhất là **Last.fm**, gồm music streaming logs giữa khoảng **1000 users** và **3000 artists**. Paper báo cáo khoảng **20000 observed user-artist pairs** và hơn **1 triệu events**. Đơn vị thời gian được dùng cho Last.fm là **hour**. Dataset này rất gần với project hiện tại vì nó là chuỗi tương tác nghe nhạc có timestamp thật.

Dataset thứ hai là **Tmall.com**, gồm khoảng **100K shopping events** giữa **26376 users** và **2563 stores**. Đơn vị thời gian cũng là **hour**. Tmall đại diện cho e-commerce, cho thấy bài toán recurrent activity không chỉ tồn tại trong music mà còn xuất hiện trong shopping behavior.

Dataset thứ ba là **MIMIC II**, một tập dữ liệu y tế gồm clinical visit records của ICU patients trong 7 năm. Paper lọc ra **650 patients** và **204 diseases**. Mỗi event ghi lại thời điểm một patient được diagnosed với một disease. Đơn vị thời gian trong MIMIC II là **week**. Dataset này không phải recommender system truyền thống, nhưng cho thấy framework có thể áp dụng cho nhiều loại event sequence khác nhau.

## Tiền xử lý và data construction

Pipeline dữ liệu của paper dựa trên việc nhóm event theo từng cặp user-item. Với Last.fm, mỗi cặp là `(user, artist)`. Với Tmall, mỗi cặp là `(user, store)`. Với MIMIC II, mỗi cặp có thể hiểu là `(patient, disease)`. Mỗi cặp như vậy tạo thành một event sequence theo thời gian.

Cách biểu diễn này khác với nhiều sequential recommendation thông thường, nơi người ta chỉ xem mỗi user là một chuỗi item. Ở đây, paper coi mỗi user-item pair là một temporal point process riêng. Điều này có lợi cho returning-time prediction vì mô hình có thể hỏi: với cặp `(u, i)`, lần tiếp theo user `u` tương tác với item `i` sẽ xảy ra khi nào.

Việc chọn đơn vị thời gian phụ thuộc vào domain. Last.fm và Tmall dùng hour vì hành vi nghe nhạc và mua sắm online có thể thay đổi theo giờ. MIMIC II dùng week vì medical visits diễn ra thưa hơn. Bài học cho project hiện tại là không nên chỉ nhìn `delta_t_seconds` như một con số tuyệt đối; cần diễn giải theo domain. Một sai số vài giờ có thể lớn trong music nhưng có thể nhỏ trong một domain có chu kỳ dài hơn.

## Pipeline mô hình

Pipeline của paper có thể được hiểu là một quy trình mô hình hóa chuỗi sự kiện theo thời gian. Thay vì xem recommendation như một bài toán dự đoán rating hoặc chọn item tiếp theo một cách tĩnh, paper xem mỗi lần user tương tác với item là một **event** xảy ra tại một thời điểm cụ thể. Từ đó, bài toán recommendation được chuyển thành bài toán học **intensity function** cho từng cặp user-item. Intensity này biểu diễn mức độ có khả năng xảy ra tương tác tại thời điểm `t`.

```text
Timestamped user-item interactions
→ Group events by user-item pair
→ Build event sequence T_{u,i}
→ Model each sequence with Hawkes process intensity λ_{u,i}(t)
→ Impose low-rank structure on base intensity and excitation matrices
→ Learn parameters by convex optimization
→ For recommendation: rank candidate items by λ_{u,i}(t)
→ For returning-time prediction: derive next-event density from λ_{u,i}(t)
→ Estimate next returning time by sampling
```

Bước đầu tiên là sử dụng dữ liệu tương tác có timestamp. Mỗi dòng dữ liệu có thể được hiểu là một event dạng `(user, item, time)`, trong đó `user` là người phát sinh hành vi, `item` là đối tượng được tương tác, và `time` là thời điểm xảy ra hành vi đó. Với Last.fm, event có thể là một user nghe một artist tại một thời điểm nhất định. Với Tmall, event có thể là một user tương tác với một store. Với MIMIC II, event có thể là một bệnh nhân được chẩn đoán với một disease ở một thời điểm cụ thể. Điểm chung là tất cả đều có thể biểu diễn thành chuỗi sự kiện theo thời gian.

Sau đó, paper nhóm các event theo từng cặp user-item. Với mỗi cặp `(u, i)`, toàn bộ các thời điểm mà user `u` từng tương tác với item `i` được gom lại thành một event sequence, thường ký hiệu là `T_{u,i}`. Ví dụ, nếu một user nghe cùng một artist nhiều lần, các timestamp nghe nhạc đó tạo thành một chuỗi sự kiện riêng cho cặp user-artist. Cách xây dựng này giúp mô hình không chỉ biết rằng user đã từng tương tác với item, mà còn biết user tương tác bao nhiêu lần, các lần tương tác cách nhau bao lâu, và tương tác có tính lặp lại hay không.

Thành phần cốt lõi của paper là **Low-Rank Hawkes Process**. Hawkes process là một loại temporal point process, được dùng để mô hình hóa thời điểm xảy ra các event. Khác với Poisson process thông thường, Hawkes process cho phép các event trong quá khứ ảnh hưởng đến khả năng xảy ra event trong tương lai. Nói cách khác, nếu một event vừa xảy ra, intensity của các event tương lai có thể tăng lên trong một khoảng thời gian nhất định. Đây là cơ chế **self-excitation**, rất phù hợp với recurrent activities. Ví dụ, nếu một user vừa nghe một artist hôm nay, khả năng user nghe lại artist đó trong tương lai gần có thể cao hơn so với một artist đã lâu không nghe.

Trong framework này, intensity `λ_{u,i}(t)` của một cặp user-item thường gồm hai phần chính. Phần thứ nhất là **baseline intensity**, thể hiện xu hướng tương tác nền hoặc long-term preference giữa user và item. Nếu baseline intensity cao, điều đó có nghĩa là về tổng thể user có xu hướng quay lại tương tác với item này, ngay cả khi không xét các event rất gần đây. Phần thứ hai là **triggering component**, thể hiện ảnh hưởng của các event quá khứ lên thời điểm hiện tại. Mỗi event cũ có thể tạo ra một mức kích thích lên intensity tương lai, và mức ảnh hưởng này thường giảm dần theo thời gian.

Có thể hiểu trực quan như sau: baseline intensity trả lời câu hỏi “user này về lâu dài có quan tâm item này không?”, còn triggering component trả lời câu hỏi “những tương tác gần đây có làm user có khả năng quay lại sớm hơn không?”. Hai thành phần này kết hợp lại giúp mô hình vừa bắt được sở thích dài hạn, vừa bắt được động lực ngắn hạn của hành vi lặp lại. Đây là điểm khác biệt quan trọng so với các baseline đơn giản như global median hoặc user median, vì median chỉ học khoảng cách thời gian trung bình mà không mô hình hóa trực tiếp ảnh hưởng của từng event quá khứ.

Tuy nhiên, nếu mỗi cặp user-item có một Hawkes process hoàn toàn độc lập, mô hình sẽ gặp vấn đề rất lớn về số lượng tham số. Trong recommender system, số lượng user và item thường rất lớn, trong khi mỗi user chỉ tương tác với một phần rất nhỏ item. Điều này dẫn đến dữ liệu cực kỳ thưa. Nhiều cặp user-item chỉ có một vài event, thậm chí có những cặp chưa từng xuất hiện trong training data. Nếu học riêng từng process, mô hình sẽ dễ overfit trên các cặp nhiều dữ liệu và gần như không học được gì từ các cặp ít dữ liệu.

Để giải quyết vấn đề này, paper áp dụng giả định **low-rank structure** lên các ma trận tham số của Hawkes process. Thay vì học một tham số độc lập cho từng cặp user-item, mô hình giả định rằng các user và item có thể được biểu diễn trong một không gian latent có số chiều thấp hơn. Điều này tương tự ý tưởng matrix factorization trong recommender system. Các user có hành vi giống nhau sẽ có latent representation gần nhau, và các item có pattern tương tác giống nhau cũng sẽ chia sẻ thông tin với nhau. Nhờ vậy, mô hình có thể chuyển tri thức từ các cặp user-item có nhiều quan sát sang các cặp có ít quan sát hơn.

Low-rank structure là một điểm rất quan trọng đối với bài toán next-time prediction. Trong dữ liệu thực tế, không phải user nào cũng có chuỗi tương tác dài. Nhiều user chỉ có vài event, nhiều item chỉ xuất hiện rất ít lần, và phân phối số lượng interaction thường rất lệch. Nếu chỉ dùng thống kê riêng từng user hoặc từng item, mô hình sẽ không ổn định với các trường hợp ít dữ liệu. Low-rank modeling giúp chia sẻ thông tin trên toàn bộ ma trận user-item, từ đó cải thiện khả năng generalize. Đây là một bài học trực tiếp cho project hiện tại: nếu muốn đi xa hơn baseline, cần có cơ chế xử lý sparsity thay vì chỉ dựa vào lịch sử riêng của từng user.

Sau khi định nghĩa intensity và low-rank structure, paper học tham số của mô hình thông qua tối ưu hóa. Mục tiêu học là tìm các tham số sao cho likelihood của các event quan sát được trong training data là cao nhất, đồng thời vẫn kiểm soát độ phức tạp của mô hình. Vì mô hình có nhiều tham số liên quan đến user, item, baseline intensity và excitation, paper sử dụng hướng tối ưu hóa convex để học tham số hiệu quả hơn. Điều này cho thấy pipeline của paper không chỉ là một ý tưởng mô hình hóa, mà còn quan tâm đến khả năng học được trên dữ liệu lớn và thưa.

Sau khi học xong mô hình, paper sử dụng intensity `λ_{u,i}(t)` cho hai nhiệm vụ. Nhiệm vụ thứ nhất là **item recommendation**. Tại một thời điểm `t`, với một user `u`, mô hình tính intensity cho các item ứng viên. Item nào có intensity cao hơn sẽ được xếp hạng cao hơn. Như vậy, danh sách recommendation không phải là cố định cho user, mà thay đổi theo thời điểm. Cùng một user có thể được recommend các item khác nhau vào các thời điểm khác nhau, vì intensity phụ thuộc vào lịch sử tương tác và thời điểm hiện tại.

Nhiệm vụ thứ hai là **returning-time prediction**. Khi đã có intensity function theo thời gian, mô hình có thể suy ra phân phối xác suất của thời điểm event tiếp theo. Nói đơn giản, intensity cho biết tại mỗi thời điểm tương lai khả năng xảy ra event là cao hay thấp. Từ intensity này, paper xây dựng next-event density để dự đoán khi nào user có thể quay lại tương tác. Vì kỳ vọng của thời điểm tiếp theo không phải lúc nào cũng có công thức đóng dễ tính, paper dùng sampling để ước lượng next returning time. Đây là phần liên quan trực tiếp nhất đến target `delta_t_seconds` trong project hiện tại.

Nếu liên hệ với notebook hiện tại, pipeline của paper này có thể xem là phiên bản nâng cao hơn của bài toán đang làm. Notebook hiện tại xây dựng target:

```text
delta_t_seconds = next_timestamp - current_timestamp
```

sau đó thử các baseline như user median, regression hoặc classification theo time bucket. Trong khi đó, paper này mô hình hóa toàn bộ quá trình phát sinh event bằng Hawkes process. Thay vì dự đoán trực tiếp một giá trị `delta_t`, mô hình học intensity theo thời gian, rồi suy ra thời điểm event tiếp theo từ intensity đó. Đây là khác biệt lớn về cách tiếp cận: notebook hiện tại là supervised prediction đơn giản, còn paper là temporal point process modeling.

Bài học quan trọng nhất từ pipeline này là next-time prediction không nên chỉ dựa vào khoảng cách thời gian trung bình. Trong dữ liệu hành vi, event quá khứ có thể tạo ra ảnh hưởng ngắn hạn, user-item có long-term preference khác nhau, và dữ liệu thường rất thưa. Low-Rank Hawkes Process giải quyết đồng thời ba vấn đề này: baseline intensity cho long-term preference, triggering component cho ảnh hưởng của event quá khứ, và low-rank structure cho sparsity/generalization. Đây là lý do paper này là một trong những related work quan trọng nhất cho hướng nghiên cứu next-time prediction.

Tóm lại, pipeline của paper bắt đầu từ timestamped user-item interactions, gom các event thành chuỗi theo từng cặp user-item, dùng Hawkes process để mô hình hóa intensity, áp dụng low-rank structure để chia sẻ thông tin giữa các cặp thưa dữ liệu, sau đó dùng intensity cho hai mục tiêu: xếp hạng item và dự đoán thời điểm quay lại. So với project hiện tại, paper này cho thấy một hướng phát triển rõ ràng: sau khi baseline chứng minh rằng target thời gian có tín hiệu học được, bước tiếp theo có thể là chuyển từ regression/classification đơn giản sang temporal point process hoặc survival-based modeling.

## Prediction và inference

Với item recommendation, mô hình tính intensity `λ_{u,i}(t)` cho các item ứng viên tại thời điểm `t`. Sau đó, item được xếp hạng theo intensity. Item có intensity cao hơn được xem là có khả năng được user tương tác hơn tại thời điểm đó.

Với returning-time prediction, mô hình dùng intensity để xác định phân phối thời gian của event tiếp theo. Trực giác là nếu intensity tăng mạnh ở một khoảng thời gian nào đó, khả năng event tiếp theo xuất hiện trong khoảng đó cũng cao hơn. Do kỳ vọng của next-event time thường khó tính trực tiếp, paper dùng sampling để ước lượng thời điểm quay lại.

Cách này khác với notebook baseline hiện tại. Notebook baseline trực tiếp tạo target `delta_t_seconds` rồi học regression/classification. Paper này không học `delta_t` như một nhãn hồi quy thông thường, mà học intensity function sinh ra event, sau đó suy ra next time từ quá trình sinh event đó.

## Evaluation, metrics và baselines

Paper đánh giá hai task. Với item recommendation, paper dùng **MAE của predicted ranking**. Với returning-time prediction, paper dùng **MAE của predicted returning time**. Ngoài ra, paper dùng **quantile plot** dựa trên time-change theorem để kiểm tra xem point process fit event pattern tốt đến đâu. Paper cũng đánh giá scalability bằng runtime khi số lượng entries/events tăng.

Các baseline gồm **Poisson process**, **Rayleigh distribution**, **SVD**, **temporal tensor factorization** và **STiC**. Poisson process là baseline quan trọng vì nó giống phiên bản đơn giản hơn của Hawkes, chỉ có base intensity và không có self-exciting history effect. SVD đại diện cho matrix factorization tĩnh, không mô hình hóa temporal dynamics. Tensor factorization thêm chiều thời gian bằng cách rời rạc hóa time slot. STiC là baseline liên quan đến returning-time prediction nhưng chỉ được dùng cho time prediction task.

Kết quả chính của paper cho thấy Hawkes-based model thường tốt hơn các baseline trên cả item recommendation và returning-time prediction. Đặc biệt, paper nhấn mạnh rằng history có ích: mô hình có xét ảnh hưởng của toàn bộ event history tốt hơn các mô hình chỉ dựa vào average inter-event gap hoặc time interval rời rạc.

## Vấn đề, giới hạn và bài học cho hướng mới

Paper này mạnh hơn paper HPFCB ở chỗ nó đánh giá trực tiếp returning-time prediction bằng MAE. Vì vậy, nếu cần một paper để biện minh rằng next-time prediction là bài toán hợp lệ trong recommender systems, đây là paper nên ưu tiên.

Tuy nhiên, pipeline của paper cũng khá nặng nếu muốn reproduce đầy đủ. Mô hình yêu cầu temporal point process, low-rank optimization, kernel, sampling và tuning nhiều tham số. Với project hiện tại, việc bắt đầu bằng baseline regression/classification là hợp lý hơn. Sau khi baseline cho thấy có tín hiệu, mới nên cân nhắc triển khai Hawkes/survival model.

Một điểm cần chú ý là paper group event theo user-item pair. Trong notebook hiện tại, nếu target được tạo theo chuỗi user tổng quát, tức là dự đoán event tiếp theo của user bất kể item nào, thì formulation sẽ khác. Paper này phù hợp nhất nếu muốn hỏi: “khi nào user sẽ quay lại với item/artist/store/disease cụ thể?”. Nếu project muốn hỏi rộng hơn “khi nào user sẽ có bất kỳ event tiếp theo nào?”, thì cần điều chỉnh formulation.

Bài học quan trọng nhất là evaluation phải tách rõ hai task. Nếu làm item recommendation thì dùng ranking metrics. Nếu làm next-time prediction thì cần metric trực tiếp trên thời gian. Đây là điểm notebook hiện tại nên giữ rõ: regression dùng MAE/RMSE/log error, classification dùng accuracy/F1/balanced accuracy theo time bucket, và ranking metrics chỉ nên dùng khi có task next-item recommendation.
