# Time-aware Recommendation: Chuyển từ “mua gì tại thời điểm này” sang “sản phẩm này sẽ được mua khi nào”

## 1. Ý tưởng trung tâm

Phần lớn recommender systems truyền thống được xây dựng quanh câu hỏi: **tại thời điểm hiện tại, nên gợi ý item nào cho user?** Với cách đặt bài toán này, time thường chỉ đóng vai trò như một feature phụ: timestamp, thứ trong tuần, giờ trong ngày, khoảng cách giữa hai lần tương tác, hoặc vị trí của item trong chuỗi hành vi. Mục tiêu cuối cùng vẫn là predict item, còn time chỉ giúp mô hình hiểu preference tốt hơn.

Hướng đang xét ở đây đảo lại trọng tâm của bài toán. Thay vì hỏi *“ở thời điểm này user sẽ mua gì?”*, ta hỏi *“với một sản phẩm hoặc một nhóm sản phẩm cụ thể, khả năng user sẽ mua nó vào thời điểm nào là cao nhất?”*. Nói cách khác, item không còn chỉ là nhãn cần dự đoán, mà trở thành điều kiện đầu vào hoặc một phần của trạng thái; còn time trở thành biến mục tiêu chính cần mô hình hóa. Đây là một cách nhìn gần với **returning-time prediction**, **time-of-interest prediction**, **active recommendation**, và **temporal point process** hơn là next-item recommendation thông thường.

Cách đặt bài toán này phù hợp với các tình huống mà hệ thống không chỉ muốn biết user thích item nào, mà còn muốn biết **khi nào nên can thiệp**. Ví dụ, nếu user có xu hướng mua phụ kiện điện thoại vài ngày sau khi mua điện thoại, hoặc mua áo khoác theo mùa, hoặc quay lại một loại dịch vụ theo chu kỳ, thì câu hỏi quan trọng không chỉ là “item nào phù hợp”, mà là “đến thời điểm nào thì item đó trở nên phù hợp nhất”. Khi đó, mục tiêu của model có thể được viết theo hướng ước lượng phân phối thời gian:

```text
P(time | user, item, history, context)
```

thay vì chỉ ước lượng:

```text
P(item | user, history, current_time)
```

hoặc trong dạng joint:

```text
P(item, time | user, history)
```

Từ góc nhìn này, item có thể được phân rã thành các thành phần như category, brand, price range, chức năng, độ thay thế/bổ sung với các item trước đó, seasonal attribute, hoặc metadata mô tả vòng đời tiêu dùng. Sau đó model học xem từng thành phần đó tạo ra tín hiệu thời gian như thế nào. Ví dụ, “áo khoác” có thể mang tín hiệu mùa; “phụ kiện điện thoại” có thể mang tín hiệu ngắn hạn sau một giao dịch chính; “sản phẩm tiêu hao” có thể mang tín hiệu chu kỳ mua lại; “sản phẩm khuyến mãi” có thể mang tín hiệu trend hoặc event. Vì vậy, bài toán không còn là matching user-item tĩnh, mà là modeling một **temporal demand curve** cho từng user-item hoặc user-item-component.

---

## 2. Vì sao time nên được xem là feature chính, không phải feature phụ?

Các paper time-aware recommendation cho thấy timestamp không chỉ là metadata ghi lại thời điểm xảy ra interaction. Time phản ánh routine, seasonality, periodicity, recency, trend, và cả trạng thái hiện tại của user. Nếu hai user có cùng chuỗi item nhưng thời điểm tương tác khác nhau, next item và next interaction time có thể khác nhau. Điều này nghĩa là thứ tự item chưa đủ để mô hình hóa preference; khoảng cách thời gian, thời điểm tuyệt đối, chu kỳ hành vi và bối cảnh thời gian đều có thể thay đổi ý nghĩa của cùng một interaction sequence.

TimelyRec là một paper rất phù hợp để chứng minh luận điểm này. Paper này cho rằng các nghiên cứu trước thường xem time như một loại feature đơn nhất, trong khi temporal preference thực tế là không đồng nhất. Một preference có thể tăng theo chu kỳ, ví dụ giờ trong ngày, ngày trong tuần, tháng trong năm; hoặc có thể thay đổi theo các sự kiện gần đây như user vừa mua item liên quan, hoặc item đang chịu ảnh hưởng bởi promotion/trend. TimelyRec vì vậy tách time thành hai nhóm pattern: **periodic pattern** và **evolving pattern**. Periodic pattern lại được chia theo nhiều granularity như hour, day of week, date, month; còn evolving pattern liên quan đến recent interactions và temporal trend của item.

Điểm quan trọng của TimelyRec đối với hướng nghiên cứu này là paper không chỉ dùng time để tăng accuracy của item recommendation, mà còn đưa ra evaluation cho **item-timing recommendation**. Đây là cầu nối trực tiếp với ý tưởng “item đó phù hợp ở thời điểm nào”. Trong item-timing recommendation, mô hình không chỉ cần rank đúng item, mà còn cần gắn item đó với thời điểm phù hợp. Điều này chứng minh rằng time có thể được đưa từ vai trò feature phụ lên thành một phần của mục tiêu dự đoán. Với hướng đề xuất của ta, TimelyRec có thể được xem như bằng chứng rằng item và time nên được mô hình hóa cùng nhau, nhưng ta đẩy xa hơn bằng cách đặt **time prediction** thành mục tiêu trung tâm.

TLSRec bổ sung một góc nhìn khác: không phải mọi interaction trước đó đều có cùng ảnh hưởng lên hành vi hiện tại. Khoảng cách thời gian giữa hành vi gần nhất và thời điểm dự đoán có thể quyết định mức độ nên tin vào short-term preference hay long-term preference. Paper này đề xuất neural time gate để kết hợp long-term và short-term preference ở mức dimension-wise, thay vì dùng một hệ số scalar thô. Điều này quan trọng vì trong hướng “predict time for item”, ta cũng cần một cơ chế tương tự: với một item cụ thể, model cần biết nên dựa vào tín hiệu ngắn hạn hay dài hạn. Ví dụ, phụ kiện sau một giao dịch lớn thường phụ thuộc nhiều vào short-term signal; sản phẩm theo sở thích bền vững hoặc sản phẩm tiêu hao định kỳ lại phụ thuộc nhiều hơn vào long-term temporal routine.

---

## 3. Nhóm paper trực tiếp gần nhất với ý tưởng “sản phẩm này sẽ được mua khi nào”

Nhóm paper gần nhất với ý tưởng này là các paper dùng **temporal point process**, đặc biệt là Hawkes process, vì chúng mô hình hóa trực tiếp thời điểm xảy ra event tiếp theo. Trong recommender systems, một event có thể được hiểu là user tương tác với item, mua item, nghe bài hát, quay lại dịch vụ, hoặc phát sinh nhu cầu với một loại sản phẩm. Khi đó, thay vì chỉ học score user-item, mô hình học một **intensity function** theo thời gian. Intensity càng cao tại một thời điểm thì xác suất event xảy ra quanh thời điểm đó càng lớn.

Paper *Time-Sensitive Recommendation From Recurrent User Activities* là cơ sở rất mạnh cho hướng này. Paper đặt ra hai câu hỏi: làm sao recommend item đúng thời điểm, và làm sao dự đoán returning time tiếp theo của user với một service/item. Đây gần như là formulation gốc cho việc chuyển từ “predict item” sang “predict time”. Paper mô hình hóa mỗi user-item pair như một chuỗi event theo thời gian và dùng self-exciting point process kết hợp low-rank model để học temporal dynamics trên số lượng lớn user-item pairs. Về mặt ý tưởng, nếu xem “mua sản phẩm X” là một recurrent event, thì model có thể học khi nào user có khả năng quay lại hoặc mua lại sản phẩm đó.

Đóng góp quan trọng của paper này là kết nối giữa **Hawkes process** và **low-rank recommendation model**. Hawkes process cho phép một event trước đó làm tăng intensity của event sau đó, tức là phản ánh hiệu ứng kích hoạt theo thời gian. Low-rank structure giúp chia sẻ thông tin giữa nhiều user-item pairs, tránh việc mỗi cặp user-item phải học một time process riêng biệt quá sparse. Đây chính là cơ sở kỹ thuật cho hướng phân rã item thành các thành phần: nếu item được biểu diễn bằng latent factor hoặc component embedding, thì time intensity của một item có thể được suy ra không chỉ từ lịch sử của chính item đó, mà còn từ các item tương tự hoặc các component tương tự.

Paper *Point Process based time sensitive personalised recommendation* tiếp tục hướng này bằng cách dùng Hawkes process với initial intensity được cá nhân hóa từ Hierarchical Poisson Factorization, đồng thời thêm sinusoidal function và exponential decay để mô hình hóa chu kỳ hoạt động. Paper này rất hợp với ý tưởng “time là biến mục tiêu”, vì nó giải quyết đồng thời hai vấn đề: recommend đúng item đúng user đúng lúc, và ước lượng khi nào user sẽ quay lại service/product sau các hành vi lặp lại. Phần sinusoidal thể hiện nhịp chu kỳ, còn exponential decay thể hiện ảnh hưởng giảm dần của các event trong quá khứ. Nếu áp dụng vào bài toán mua hàng, một sản phẩm có thể có cả tín hiệu chu kỳ, như mùa hoặc ngày lương, và tín hiệu decay, như nhu cầu phụ kiện giảm dần sau khi mua sản phẩm chính.

PASRec là paper hiện đại nhất và cũng gần nhất với cách viết joint formulation. Paper này phân biệt **Time of Interest (ToI)** và **Item of Interest (IoI)**, đồng thời cho rằng active recommendation cần dự đoán cả hai. Công thức trong paper có dạng:

```text
P(item, time | user history) = P(time | user history) · P(item | time, user history)
```

Cách phân rã này vẫn bắt đầu bằng dự đoán time rồi mới dự đoán item theo time. Tuy nhiên, nó rất hữu ích cho hướng của ta vì nó khẳng định rằng time và item không nên tách rời. Nếu ToI sai, IoI cũng dễ sai; nếu item không được condition theo ToI, recommendation có thể đúng về mặt preference tổng quát nhưng sai về thời điểm. Hướng đề xuất có thể đảo lại decomposition thành:

```text
P(item, time | user history) = P(item | user history) · P(time | item, user history)
```

hoặc trực tiếp học:

```text
P(time | user, item, decomposed_item_features, history)
```

Điểm khác biệt là PASRec tập trung vào active sequential recommendation: dự đoán thời điểm user sẽ tương tác tiếp theo và item phù hợp tại thời điểm đó. Hướng của ta tập trung hơn vào item-conditioned time prediction: với một sản phẩm hoặc một nhóm sản phẩm đã được chọn, dự đoán phân phối thời điểm user có khả năng mua. Vì vậy, PASRec là paper match rất mạnh ở cấp formulation joint ToI-IoI, nhưng hướng đề xuất có thể được trình bày như một biến thể đảo chiều hoặc mở rộng: từ “when and what to recommend” sang “for this product, when is it most likely to be needed”.

---

## 4. Notification papers: cơ sở cho active delivery và quyết định can thiệp đúng lúc

Các paper về notification không nhất thiết dự đoán “sản phẩm nào sẽ được mua khi nào”, nhưng chúng là cơ sở quan trọng cho phần **active recommendation** và **decision timing**. Khi model đã dự đoán được một sản phẩm có xác suất mua cao ở một thời điểm nào đó, hệ thống vẫn cần quyết định có nên gửi gợi ý, gửi khi nào, gửi bao nhiêu lần, và tránh làm phiền user ra sao. Đây là nơi các paper notification/RL trở thành nền tảng.

Paper của Twitter, *Should I send this notification?*, cho thấy một recommender system myopic thường tối ưu phản hồi tức thời, ví dụ open/click ngay sau notification. Vấn đề là hệ thống như vậy có xu hướng luôn gửi nếu còn một xác suất nhỏ user sẽ mở, nhưng hậu quả tiêu cực như user khó chịu, tắt notification, hoặc quen với việc bỏ qua notification lại xuất hiện trong tương lai. Paper này dùng model-based reinforcement learning để mô hình hóa ảnh hưởng của việc gửi notification lên hành vi tương lai của user trong horizon nhiều ngày. Kết quả thực nghiệm A/B test cho thấy có thể gửi ít notification hơn, đạt open rate cao hơn, trong khi vẫn giữ mức engagement tương đương baseline heuristic.

Ý nghĩa của paper này với hướng đề xuất là: dự đoán đúng thời điểm mua chưa đủ; hệ thống còn cần học **policy can thiệp**. Nếu time model dự đoán rằng user có xác suất mua cao vào ngày thứ 7 sau khi xem sản phẩm, thì notification policy phải quyết định có gửi vào ngày đó hay không, gửi trước bao lâu, và có nên hoãn nếu user vừa nhận quá nhiều thông báo. Do đó, mô hình time prediction có thể đóng vai trò candidate generator hoặc state feature cho một policy layer phía trên.

Paper của LinkedIn, *Multi-objective Optimization of Notifications Using Offline Reinforcement Learning*, mở rộng luận điểm này trong bối cảnh near-real-time notification. Paper mô hình hóa quyết định notification như một Markov Decision Process, trong đó reward có nhiều mục tiêu: engagement với notification, site visit, long-term engagement, guardrail về volume hoặc disable. Điểm đáng chú ý là notification feedback thường bị delay vài giờ hoặc vài ngày, và hiệu quả của một notification không thể gán hoàn toàn cho một hành động đơn lẻ mà phụ thuộc vào chuỗi quyết định trước đó. Đây là vấn đề rất gần với recommendation theo thời gian: hành vi mua một sản phẩm thường không chỉ là kết quả của một impression, mà là kết quả của một chuỗi exposure, browsing, prior purchases, seasonality và timing.

Paper LinkedIn cũng nhấn mạnh môi trường non-uniform discrete time: hệ thống không ra quyết định ở các bước thời gian đều nhau, mà tại các timestamp bất thường khi notification candidate xuất hiện. Đây là cơ sở tốt để mô hình hóa hệ thống thực tế: interaction, session, purchase, notification, promotion đều xảy ra không đều theo thời gian. Nếu hướng nghiên cứu dùng point process hoặc time-to-event model, nó tự nhiên hơn so với việc ép dữ liệu vào các time step đều nhau.

---

## 5. Cách diễn giải item decomposition trong hướng này

Để chuyển từ “predict item” sang “predict time for item”, cần tránh xem item chỉ là một ID rời rạc. Nếu item chỉ là ID, bài toán rất sparse: nhiều item ít interaction, nhiều user chưa từng mua item đó, và time pattern khó học. Vì vậy item nên được phân rã thành các thành phần có ý nghĩa thời gian.

Một sản phẩm có thể được phân rã theo category, brand, price segment, consumable/non-consumable, complementary relation, substitute relation, seasonal sensitivity, promotion sensitivity, và lifecycle stage. Ví dụ, “ốp lưng iPhone” không chỉ là một item ID, mà còn là accessory, liên quan đến smartphone, có thể có time lag ngắn sau khi user mua hoặc xem iPhone. “Sữa rửa mặt” là consumable, có thể có purchase cycle. “Áo khoác” là seasonal item, chịu ảnh hưởng bởi tháng, thời tiết hoặc mùa. “Khóa học online” có thể chịu ảnh hưởng bởi đầu năm học, kỳ thi, hoặc mục tiêu cá nhân dài hạn.

Với cách phân rã này, model có thể học các loại temporal signal khác nhau. Component dạng seasonal học periodic pattern. Component dạng complementary học delay sau event kích hoạt. Component dạng consumable học inter-purchase interval. Component dạng trend/promotion học burst ngắn hạn. Component dạng personal routine học giờ/ngày/tháng mà user thường hoạt động. Đây là điểm nối giữa TimelyRec, TLSRec, PASRec và các point-process papers: TimelyRec cung cấp cách nghĩ về periodic/evolving temporal pattern; TLSRec cung cấp cơ chế time-lag để cân bằng short-term và long-term signal; PASRec cung cấp joint ToI-IoI formulation; Hawkes/point-process papers cung cấp mô hình trực tiếp cho event time.

---

## 6. Formulation đề xuất

Một formulation phù hợp cho hướng này là xem mỗi event mua hàng như một tuple:

```text
(user, item, timestamp, context)
```

Sau đó item được mã hóa thành:

```text
item = f(category, brand, price, attributes, relation_to_history, seasonal_profile, promotion_state)
```

Mục tiêu của model là học phân phối thời gian hoặc intensity function:

```text
λ_u,i(t) = intensity(user u mua item i tại thời điểm t | history, context)
```

hoặc dưới dạng xác suất rời rạc:

```text
P(t_bucket | user, item_components, history, context)
```

Trong đó `t_bucket` có thể là hour-of-day, day-of-week, day-after-event, week, month, hoặc một khoảng thời gian tương lai như 1 ngày, 3 ngày, 7 ngày, 30 ngày. Nếu muốn dự đoán time liên tục, temporal point process phù hợp hơn. Nếu muốn đưa vào pipeline recommender hiện có, có thể dùng discrete time bucket classification hoặc survival/time-to-event prediction.

Một kiến trúc có thể gồm bốn phần. Thứ nhất là **User History Encoder**, mã hóa chuỗi interaction của user. Thứ hai là **Item Component Encoder**, mã hóa item không chỉ bằng ID mà bằng các thành phần có ý nghĩa thời gian. Thứ ba là **Temporal Pattern Encoder**, học periodicity, recency, time lag, seasonality và trend. Thứ tư là **Time Prediction Head**, xuất ra phân phối thời gian hoặc intensity curve cho item đó. Nếu cần active recommendation, phía sau có thể thêm **Policy Layer** kiểu RL để quyết định có nên gửi recommendation/notification ở thời điểm model dự đoán hay không.

---

## 7. Mapping từng paper với ý tưởng nghiên cứu

| Paper | Vai trò trong lập luận |
|---|---|
| **Time-Sensitive Recommendation From Recurrent User Activities** | Paper match trực tiếp nhất với hướng predict time. Mô hình hóa user-item event sequence bằng Hawkes process và low-rank structure, giải quyết cả item-at-time recommendation và returning-time prediction. |
| **Point Process based time sensitive personalised recommendation** | Củng cố hướng dùng point process cho returning time; kết hợp HPF, Hawkes process, sinusoidal cycle và exponential decay để học hành vi lặp lại theo thời gian. |
| **PASRec: When and What to Recommend** | Cung cấp formulation joint giữa Time of Interest và Item of Interest. Rất gần với hướng đề xuất, nhưng paper thiên về active next interaction; hướng của ta đảo lại thành item-conditioned time prediction. |
| **TimelyRec** | Chứng minh time là heterogeneous signal gồm periodic và evolving pattern. Hỗ trợ luận điểm rằng time nên được mô hình hóa như thành phần chính, không chỉ là timestamp phụ. |
| **TLSRec** | Chứng minh time lag ảnh hưởng đến cách kết hợp long-term và short-term preference. Hữu ích khi thiết kế model dự đoán thời điểm mua cho item có tính ngắn hạn/dài hạn khác nhau. |
| **Should I send this notification?** | Cơ sở cho tầng decision/policy sau khi đã dự đoán thời điểm. Cho thấy gửi đúng lúc phải tối ưu long-term value, không chỉ click/open tức thời. |
| **Multi-objective Optimization of Notifications Using Offline RL** | Cơ sở cho hệ thống active delivery thực tế: delayed reward, non-uniform decision time, nhiều mục tiêu và guardrails. Hữu ích nếu biến time prediction thành hành động recommendation/notification. |

---

## 8. Điểm mới có thể nhấn mạnh

Điểm mới của hướng này không phải là “dùng time trong recommender system”, vì các paper time-aware đã làm nhiều. Điểm mới nằm ở việc **đổi biến mục tiêu**: thay vì dùng time để giúp predict item, ta dùng item và item components để predict time. Nói cách khác, time không chỉ là context mà trở thành target. Điều này phù hợp với các bài toán như purchase timing, replenishment recommendation, seasonal demand prediction, active promotion, và lifecycle-aware recommendation.

So với TimelyRec hoặc PASRec, hướng này có thể được trình bày là đi sâu hơn vào nhánh **item-conditioned temporal prediction**. TimelyRec hỏi: user tại timestamp `t` có khả năng tương tác item nào? PASRec hỏi: next Time of Interest và Item of Interest là gì? Hướng này hỏi: với item hoặc item group `i`, phân phối thời điểm user `u` sẽ mua/tương tác là gì? Câu hỏi này đặc biệt hữu ích khi doanh nghiệp đã có candidate item từ inventory, campaign, hoặc product lifecycle, và cần biết thời điểm tối ưu để recommend.

So với Hawkes/point-process papers, hướng này có thể mở rộng bằng cách phân rã item thành component-level temporal signals thay vì chỉ học user-item pair intensity. Điều này giúp giảm sparsity và tăng khả năng generalize sang item mới hoặc item ít dữ liệu. Ví dụ, một sản phẩm mới chưa có nhiều lịch sử mua vẫn có thể kế thừa temporal pattern từ category, brand, price range hoặc quan hệ bổ sung với các item đã mua trước đó.

---

## 9. Kết luận định hướng

Tổng hợp các paper cho thấy time là một thành phần rất quan trọng trong recommender systems, nhưng có nhiều cấp độ sử dụng khác nhau. Ở cấp thấp, time là feature phụ như timestamp hoặc time interval. Ở cấp cao hơn, time được mô hình hóa thành periodic/evolving pattern như trong TimelyRec, hoặc thành time lag điều khiển long-term/short-term fusion như trong TLSRec. Ở cấp gần nhất với hướng nghiên cứu này, time trở thành đối tượng dự đoán thông qua returning-time prediction, Time of Interest, hoặc temporal point process như trong Hawkes-based recommendation và PASRec.

Vì vậy, hướng “sản phẩm này sẽ được mua ở thời điểm nào” là có cơ sở tốt từ literature. Các paper point process chứng minh có thể dự đoán returning time của user-item/service event. PASRec chứng minh time và item nên được mô hình hóa chung thay vì tách rời. TimelyRec và TLSRec chứng minh time chứa nhiều pattern quan trọng và ảnh hưởng mạnh đến preference. Các paper notification/RL chứng minh rằng khi đã dự đoán được thời điểm, hệ thống vẫn cần một tầng quyết định để active delivery đúng lúc và không gây hại cho long-term user experience.

Một hướng mô hình hợp lý là xây dựng **item-conditioned time prediction model**: đầu vào gồm user history, item decomposition, temporal context và event history; đầu ra là time distribution hoặc intensity curve cho khả năng user mua/tương tác với item trong tương lai. Sau đó, nếu triển khai thực tế, mô hình này có thể kết hợp với policy layer để quyết định thời điểm gửi recommendation/notification. Đây là cách biến time từ một feature hỗ trợ thành mục tiêu dự đoán trung tâm của recommender system.
