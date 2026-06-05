# Tóm tắt các paper về Time-aware / Time-series Recommendation

**Chủ đề đọc:** time series trong recommender systems, đặc biệt là các hướng xem **thời gian / timestamp / time lag / next interaction time** như một feature chính để quyết định **gợi ý cái gì** và/hoặc **gợi ý khi nào**.

## 0. Nhận xét tổng quan

Các paper trong thư mục có thể chia thành 4 nhóm chính:

| Nhóm | Ý tưởng chính | Paper tiêu biểu | Vai trò của time |
|---|---|---|---|
| Temporal point process / Hawkes process | Mô hình hóa mỗi tương tác user-item như một chuỗi sự kiện theo thời gian | Du et al. 2015; Abbas et al. 2023 | Time là biến liên tục để tính intensity, return time, và item tại thời điểm cụ thể |
| Time-aware sequential recommendation | Dùng timestamp, periodicity, time-lag để học preference thay đổi theo thời gian | TimelyRec 2021; TLSRec 2022 | Time là feature phụ nhưng rất quan trọng để điều kiện hóa user preference |
| Notification / active decision | Quyết định có nên gửi notification tại thời điểm hiện tại không | Twitter 2022; LinkedIn 2022 | Time nằm trong state, horizon, delayed reward, frequency/spacing |
| Joint timing-content active recommendation | Dự đoán đồng thời **Time of Interest** và **Item of Interest** | PASRec 2025 | Time là target chính, không chỉ là feature đầu vào |

Điểm chung của các paper là chúng đều phản đối cách recommender truyền thống chỉ học từ thứ tự item hoặc rating tĩnh. Thay vào đó, chúng cho rằng hành vi người dùng có **chu kỳ**, **độ trễ**, **xu hướng gần đây**, **cường độ quay lại**, và **tác động dài hạn theo thời gian**.

---

## 1. Time-Sensitive Recommendation From Recurrent User Activities  
**File:** `NIPS-2015-time-sensitive-recommendation-from-recurrent-user-activities-Paper.pdf`  
**Tác giả:** Nan Du, Yichen Wang, Niao He, Le Song  
**Năm:** 2015

### Nội dung chính

Paper này đặt bài toán recommender theo hướng **recurrent event modeling**. Thay vì chỉ hỏi “user thích item nào?”, paper hỏi hai câu quan trọng hơn:

1. Tại một thời điểm `t`, nên recommend item nào cho user?
2. Khi nào user sẽ quay lại sử dụng một service/item?

Mỗi cặp `(user, item)` được xem như một chuỗi sự kiện theo thời gian. Ví dụ, user nghe một artist nhiều lần, mua ở một store nhiều lần, hoặc bệnh nhân được chẩn đoán cùng một bệnh nhiều lần. Các lần xuất hiện này không độc lập, mà có thể có tính **self-exciting**: một tương tác trong quá khứ làm tăng khả năng xảy ra tương tác tiếp theo trong tương lai gần.

### Time được dùng như feature chính thế nào?

Time không chỉ được dùng như “thứ tự interaction”, mà là biến liên tục trong **temporal point process**. Mỗi user-item pair có một **conditional intensity function** biểu diễn xác suất xảy ra event tiếp theo tại thời điểm hiện tại, dựa trên lịch sử trước đó.

Nói đơn giản:

- lịch sử tương tác càng gần thì ảnh hưởng càng mạnh;
- tương tác cũ dần mất ảnh hưởng theo decay kernel;
- model có thể dự đoán cả **item tại thời điểm t** và **returning time tiếp theo**.

### Đóng góp chính

- Kết nối **self-exciting Hawkes process** với **low-rank matrix modeling** để mô hình hóa lượng lớn chuỗi user-item-event.
- Đưa ra công thức tối ưu convex cho bài toán low-rank point process.
- Phát triển thuật toán học hiệu quả, có thể scale tới hàng triệu user-item pair và hàng trăm triệu event.
- Chứng minh mô hình hoạt động tốt trên cả synthetic data và real data như Last.fm, Tmall, MIMIC II.
- Mở rộng được với context khác như profile, text, spatial feature.

### Ý nghĩa với hướng “time là feature chính”

Đây là paper nền tảng cho hướng **recommendation theo thời điểm**. Nếu muốn biến RS thành bài toán time-series/event prediction, paper này rất quan trọng vì nó xem mỗi user-item interaction là một event process chứ không chỉ là một row trong interaction matrix.

---

## 2. Point Process based time sensitive personalised recommendation  
**File:** `1-s2.0-S1877050923001576-main.pdf`  
**Tác giả:** Khushnood Abbas, Dong Shi, Asif Khan  
**Năm:** 2023

### Nội dung chính

Paper này tiếp tục hướng point-process cho time-sensitive recommendation. Mục tiêu là recommend đúng item cho đúng user tại đúng thời điểm, đồng thời ước lượng khi nào user sẽ quay lại service hoặc item sau các hành vi lặp lại.

Paper tập trung vào hiện tượng hành vi người dùng có chu kỳ, ví dụ nghe nhạc vào một khung giờ cố định trong ngày hoặc sử dụng dịch vụ theo ngày trong tuần.

### Time được dùng như feature chính thế nào?

Paper dùng **Hawkes process** để biểu diễn dynamic activity theo thời gian. Điểm đáng chú ý là intensity function gồm:

- **initial intensity** cá nhân hóa bằng Hierarchical Poisson Factorization;
- thành phần **sinusoidal function** để bắt chu kỳ thời gian;
- thành phần **exponential decay** để mô hình hóa ảnh hưởng giảm dần của hành vi quá khứ.

Vì vậy time được biểu diễn theo hai mặt:

1. **Cyclic time:** hành vi lặp lại theo giờ/ngày/tuần.
2. **Recent influence:** event gần đây ảnh hưởng mạnh hơn event xa.

### Đóng góp chính

- Kết hợp Hawkes process với Hierarchical Poisson Factorization cho implicit feedback recommendation.
- Đưa thành phần sinusoidal vào để học chu kỳ hành vi người dùng.
- Kết hợp exponential decay để mô hình hóa ảnh hưởng động của event quá khứ.
- Có khả năng dự đoán cả recommendation tại thời điểm phù hợp và thời gian user quay lại.
- Nhấn mạnh tính diễn giải tốt hơn so với nhiều deep neural network model.

### Ý nghĩa với hướng “time là feature chính”

Paper này phù hợp nếu bạn muốn dùng **time-cycle feature** như hour-of-day, day-of-week, periodicity vào RS. Tuy nhiên, so với paper 2015, paper này thiên về cải tiến mô hình cường độ bằng thành phần chu kỳ và HPF hơn là framework tối ưu tổng quát.

---

## 3. Learning Heterogeneous Temporal Patterns of User Preference for Timely Recommendation — TimelyRec  
**File:** `2104.14200v1(1).pdf`  
**Tác giả:** Junsu Cho, Dongmin Hyun, SeongKu Kang, Hwanjo Yu  
**Năm:** 2021

### Nội dung chính

Paper này cho rằng các phương pháp time-aware trước đó thường xem time như một feature đơn nhất. Điều này chưa đủ, vì preference theo thời gian có nhiều kiểu khác nhau.

Paper chia temporal pattern thành hai nhóm:

1. **Periodic pattern:** preference tăng theo chu kỳ, ví dụ theo giờ, thứ trong tuần, ngày trong tháng, tháng trong năm.
2. **Evolving pattern:** preference thay đổi do sự kiện gần đây, ví dụ user vừa mua item liên quan hoặc có promotion làm item trở thành trend.

Từ đó, paper đề xuất **TimelyRec**, một model học đồng thời hai loại temporal pattern này.

### Time được dùng như feature chính thế nào?

TimelyRec dùng hai encoder theo dạng cascade:

#### Multi-Aspect Time Encoder — MATE

MATE mã hóa timestamp theo nhiều granularity:

- month;
- day of week;
- date;
- hour.

Điểm quan trọng là MATE không chỉ one-hot time rồi concatenate, mà dùng attention để học:

- granularity nào quan trọng với user nào;
- chu kỳ có thể lệch nhẹ, không tuyệt đối đúng một ngày/giờ;
- pattern thời gian mang tính cá nhân hóa.

#### Time-Aware History Encoder — TAHE

TAHE học evolving pattern từ lịch sử tương tác gần đây. Nó dùng time-based attention để ưu tiên các interaction trong quá khứ có temporal pattern giống target time.

Ví dụ, nếu target time là tối thứ sáu, model sẽ chú ý hơn đến những hành vi quá khứ cũng xảy ra vào tối thứ sáu hoặc có pattern tương tự.

### Đóng góp chính

- Định nghĩa rõ các đặc trưng của temporal pattern trong preference: periodicity đa mức, irregularity, personalization, recent interaction effect, item trend.
- Đề xuất TimelyRec gồm MATE và TAHE để học cả periodic pattern và evolving pattern.
- Đề xuất attention module mới: gradual attention và time-based attention.
- Đưa ra evaluation scenario cho **item-timing recommendation**, tức đánh giá đồng thời item và thời điểm recommend.
- Thực nghiệm cho thấy TimelyRec vượt nhiều time-aware baseline, đặc biệt trong item-timing recommendation.

### Ý nghĩa với hướng “time là feature chính”

Đây là paper rất sát với hướng bạn đang tìm: **time không chỉ là timestamp phụ**, mà được encode thành nhiều aspect rồi dùng để điều kiện hóa recommendation. Nếu bạn muốn thử “encode time theo nhiều hệ khác nhau rồi đưa vào model”, TimelyRec là một baseline/ý tưởng rất mạnh.

---

## 4. Should I send this notification? Optimizing push notifications decision making by modeling the future  
**File:** `2202.08812v1.pdf`  
**Tác giả:** Conor O’Brien, Huasen Wu, Shaodan Zhai, Dalin Guo, Wenzhe Shi, Jonathan J. Hunt  
**Năm:** 2022  
**Bối cảnh:** Twitter push notification

### Nội dung chính

Paper này không tập trung vào recommend item truyền thống, mà tập trung vào quyết định **có nên gửi notification tại thời điểm hiện tại hay không**.

Vấn đề là nhiều recommender tối ưu myopic: nếu notification có xác suất được mở > 0 thì hệ thống có xu hướng gửi. Nhưng gửi quá nhiều hoặc gửi không đúng lúc có thể gây tác hại dài hạn: user bỏ qua notification, tắt notification, hoặc giảm engagement.

Paper đề xuất dùng **model-based reinforcement learning** để tối ưu long-term value thay vì chỉ tối ưu immediate open/click.

### Time được dùng như feature chính thế nào?

Time xuất hiện ở ba mức:

1. **Decision time:** tại mỗi thời điểm có candidate notification, hệ thống quyết định send hoặc not send.
2. **Future horizon:** reward không chỉ tính ngay lúc notification được mở, mà tính tác động trong nhiều ngày.
3. **User state over time:** hành vi mở/bỏ qua notification trước đó làm thay đổi trạng thái user và ảnh hưởng quyết định tương lai.

Time ở đây không phải timestamp encoding kiểu hour/day, mà là **sequential decision horizon**: mỗi hành động hiện tại ảnh hưởng chuỗi hành vi sau này.

### Đóng góp chính

- Chỉ ra notification recommendation không nên tối ưu myopic open rate.
- Mô hình hóa tác động của notification lên hành vi tương lai của user.
- Dùng model-based RL để quyết định send/not-send.
- Thử nghiệm A/B trên hệ thống mạng xã hội lớn.
- Kết quả: gửi ít notification hơn, open rate cao hơn, nhưng vẫn giữ mức engagement tương đương baseline heuristic.

### Ý nghĩa với hướng “time là feature chính”

Paper này hữu ích nếu bạn quan tâm đến **active recommendation** hoặc **push recommendation**. Time không chỉ giúp predict item, mà giúp quyết định **khi nào không nên recommend** để tránh gây nhiễu và giảm long-term value.

---

## 5. Multi-objective Optimization of Notifications Using Offline Reinforcement Learning  
**File:** `2207.03029v1.pdf`  
**Tác giả:** Prakruthi Prabhakar, Yiping Yuan, Guangyu Yang, Wensheng Sun, Ajith Muralidharan  
**Năm:** 2022  
**Bối cảnh:** LinkedIn notification system

### Nội dung chính

Paper này cũng thuộc hướng notification decision, nhưng tập trung vào **near-real-time notification system** và **multi-objective optimization**.

Thay vì chỉ tối ưu click/open, hệ thống phải cân bằng nhiều mục tiêu:

- engagement trên notification;
- site visit;
- long-term engagement;
- guardrail như notification volume hoặc disables;
- các objective kinh doanh khác.

Paper mô hình hóa notification decision như một **Markov Decision Process** và dùng **offline reinforcement learning** để học policy từ logged data.

### Time được dùng như feature chính thế nào?

Time đóng vai trò trong:

1. **Near-real-time stream decision:** notification đến theo stream và phải quyết định gần như ngay lập tức.
2. **Time discretization:** thời gian được rời rạc hóa để xây dựng MDP state/action/reward.
3. **Delayed reward:** user có thể phản hồi sau vài giờ hoặc vài ngày.
4. **Notification spacing/frequency:** quyết định hiện tại ảnh hưởng đến trải nghiệm và reward trong tương lai.

### Đóng góp chính

- Formulate bài toán near-real-time notification decision thành MDP.
- Đề xuất end-to-end offline RL framework cho sequential notification decision.
- Dùng Double DQN và Conservative Q-Learning để giảm distribution shift và Q-value overestimation trong offline training.
- Tích hợp multi-objective reward engineering để tối ưu nhiều mục tiêu cùng lúc.
- Trình bày hệ thống triển khai thực tế và đánh giá bằng offline + online experiments.

### Ý nghĩa với hướng “time là feature chính”

Paper này cho thấy với recommendation thực tế, time không chỉ nằm ở input item sequence mà còn nằm ở **decision process**: hệ thống phải xét lịch sử gửi notification, khoảng cách giữa các lần gửi, reward trễ, và long-term user experience.

---

## 6. Time Lag Aware Sequential Recommendation — TLSRec  
**File:** `2208.04760v1.pdf`  
**Tác giả:** Lihua Chen, Ning Yang, Philip S. Yu  
**Năm:** 2022

### Nội dung chính

Paper này giải quyết sequential recommendation bằng cách tách preference thành:

1. **Long-term preference:** sở thích ổn định của user qua nhiều session.
2. **Short-term preference:** sở thích dao động trong session gần đây.

Vấn đề của nhiều model trước đó là fuse long-term và short-term preference bằng scalar weight đơn giản. TLSRec cho rằng cách này quá thô, vì mỗi chiều/aspect trong embedding có thể cần trọng số khác nhau.

### Time được dùng như feature chính thế nào?

Điểm chính là **time lag**: khoảng cách thời gian giữa thời điểm hiện tại và hành vi cuối cùng của user.

TLSRec dùng **neural time gate** để quyết định mức độ đóng góp của long-term và short-term preference dựa trên time lag.

Trực giác:

- Nếu time lag ngắn: hành vi gần đây còn rất liên quan, nên short-term preference quan trọng hơn.
- Nếu time lag dài: session gần nhất có thể đã bớt liên quan, nên long-term preference quan trọng hơn.

Khác với scalar gate, TLSRec dùng gating vector để fuse ở mức aspect/dimension của embedding.

### Đóng góp chính

- Đề xuất TLSRec cho sequential recommendation có xét time lag.
- Dùng hierarchical self-attention để học preference ở hai time scale: local/session-level và global/cross-session.
- Đề xuất neural time gate để fuse long-term và short-term preference theo time lag.
- Fusion diễn ra ở mức fine-grained embedding dimension thay vì scalar weight.
- Thực nghiệm trên real datasets cho thấy TLSRec hiệu quả hơn các baseline.

### Ý nghĩa với hướng “time là feature chính”

Paper này đặc biệt hữu ích nếu bạn muốn biến **khoảng cách thời gian giữa các interaction** thành feature chính. Nó không cần dự đoán thời điểm tiếp theo, nhưng dùng time lag để thay đổi cách biểu diễn user preference.

---

## 7. When and What to Recommend: Joint Modeling of Timing and Content for Active Sequential Recommendation — PASRec  
**File:** `2511.18717v1.pdf`  
**Tác giả:** Jin Chai, Xiaoxiao Ma, Jian Yang, Jia Wu  
**Năm:** 2025

### Nội dung chính

Paper này đặt bài toán active sequential recommendation: hệ thống không chỉ đợi user mở app rồi recommend, mà phải chủ động dự đoán:

1. **Time of Interest — ToI:** khi nào user sẽ quan tâm/tương tác tiếp.
2. **Item of Interest — IoI:** item nào nên recommend tại thời điểm đó.

Paper cho rằng cách làm naive là dự đoán ToI trước rồi dùng ToI để dự đoán IoI. Nhưng nếu ToI sai, lỗi sẽ lan sang IoI, gây single point of failure.

PASRec đề xuất joint modeling giữa timing và content bằng diffusion-based sequential recommendation.

### Time được dùng như feature chính thế nào?

Time ở đây vừa là input vừa là target:

- lịch sử interaction time được model cùng với item sequence;
- next ToI được dự đoán như một thành phần mục tiêu;
- IoI được điều kiện hóa theo ToI;
- training objective được thiết kế để tăng dependency/mutual information giữa ToI và IoI.

Điểm mới là time không còn là side feature. Time trở thành một nửa của bài toán prediction: **when + what**.

### Đóng góp chính

- Đề xuất PASRec cho active sequential recommendation.
- Joint modeling giữa Time of Interest và Item of Interest.
- Dùng diffusion model để học phân phối phức tạp của user preference.
- Thiết kế objective để tăng mutual information giữa ToI và IoI, giảm rủi ro single point of failure.
- Phân tích lý thuyết qua ELBO, cho thấy tích hợp predicted ToI có thể tạo lower bound chặt hơn so với diffusion recommender truyền thống.
- Thực nghiệm trên 5 benchmark datasets và 8 baseline, trong cả leave-one-out và temporal split.

### Ý nghĩa với hướng “time là feature chính”

Đây là paper gần nhất với ý tưởng “đưa recommendation về bài toán dự đoán thời điểm + item”. Nếu bạn muốn nghiên cứu **predict time-item jointly** hoặc dùng time-series/foundation model để dự đoán thời điểm recommend, PASRec là paper rất quan trọng.

---

## 8. So sánh nhanh các paper theo cách dùng time

| Paper | Bài toán chính | Time là input? | Time là target? | Dạng time chính | Output |
|---|---:|---:|---:|---|---|
| Du et al. 2015 | Time-sensitive recurrent recommendation | Có | Có | Continuous event time, Hawkes intensity | item at time `t`, next return time |
| Abbas et al. 2023 | Point-process personalized recommendation | Có | Có | Cyclic time + event decay | item đúng thời điểm, return time |
| TimelyRec 2021 | Timely item / item-timing recommendation | Có | Một phần | month/day/date/hour + history time | item, item-timing |
| Twitter 2022 | Push notification send/not-send | Có | Không trực tiếp | decision horizon, future behavior | send hoặc not send |
| LinkedIn 2022 | Multi-objective notification optimization | Có | Không trực tiếp | near-real-time stream, delayed reward | notification decision policy |
| TLSRec 2022 | Sequential recommendation | Có | Không | time lag since last behavior | next item |
| PASRec 2025 | Active sequential recommendation | Có | Có | historical timestamp + next ToI | next time + next item |

---

## 9. Paper nào quan trọng nhất cho hướng nghiên cứu của bạn?

Nếu mục tiêu của bạn là **time ảnh hưởng như một feature chính trong recommender systems**, nên ưu tiên đọc theo thứ tự sau:

### Ưu tiên 1 — trực tiếp nhất

1. **TimelyRec 2021**  
   Vì paper này phân tích rất rõ các loại temporal pattern: periodic, irregular, personalized, evolving. Nó cũng đưa ra cách encode time đa granularity.

2. **PASRec 2025**  
   Vì paper này đi xa hơn: không chỉ dùng time làm feature mà dự đoán cả **Time of Interest** và **Item of Interest**.

3. **Du et al. 2015**  
   Vì đây là nền tảng point-process cho recurrent user activity, rất phù hợp nếu muốn nhìn RS như time-series/event prediction.

### Ưu tiên 2 — bổ sung modeling idea

4. **TLSRec 2022**  
   Dùng time lag để điều chỉnh long-term/short-term preference. Rất hữu ích nếu data của bạn có khoảng cách thời gian giữa interaction.

5. **Abbas et al. 2023**  
   Bổ sung hướng cyclic time + Hawkes + HPF. Có thể dùng làm paper liên quan cho periodic behavior.

### Ưu tiên 3 — nếu quan tâm active/push recommendation

6. **Twitter 2022**  
   Hữu ích nếu hệ thống có hành động “send/not send” và cần tối ưu long-term value.

7. **LinkedIn 2022**  
   Hữu ích nếu bạn muốn mô hình hóa notification decision ở quy mô production, nhiều objective, reward trễ.

---

## 10. Gợi ý hướng phát triển từ các paper

Từ nhóm paper này, có thể hình thành một hướng nghiên cứu như sau:

> Thay vì chỉ dùng sequence item `i1, i2, ..., in`, ta dùng chuỗi interaction có timestamp `(i1, t1), (i2, t2), ..., (in, tn)` và học representation trong đó thời gian là thành phần chính. Model cần học được chu kỳ hành vi, khoảng cách giữa các interaction, ảnh hưởng của sự kiện gần đây, và có thể dự đoán cả item lẫn thời điểm recommend.

Một hướng cụ thể có thể là:

1. Encode timestamp theo nhiều hệ:
   - absolute time: hour, weekday, month;
   - relative time: time lag, inter-event interval;
   - periodic encoding: sine/cosine;
   - event intensity: Hawkes/time-to-next-event;
   - learned time embedding hoặc time-series foundation model embedding.

2. So sánh các cách đưa time vào recommender:
   - concatenate time embedding với item embedding;
   - attention bias theo time interval;
   - time-aware gating giữa long-term/short-term preference;
   - joint prediction `p(item, time | history)`;
   - active recommendation: send/not-send tại predicted time.

3. Đánh giá bằng hai loại metric:
   - item recommendation: HR@K, NDCG@K, Recall@K;
   - timing recommendation: MAE/RMSE của next time, hoặc hit đúng item trong đúng time window.

---

## 11. Kết luận ngắn

Các paper này cho thấy “time” trong recommender systems có thể được dùng ở nhiều cấp độ:

- **Feature phụ:** thêm hour/day/month vào model.
- **Feature điều kiện hóa preference:** cùng một user-item nhưng ở thời điểm khác nhau có score khác nhau.
- **Sequence dynamics:** khoảng cách thời gian quyết định short-term hay long-term preference quan trọng hơn.
- **Event process:** tương tác user-item là chuỗi event có intensity theo thời gian.
- **Target prediction:** model phải dự đoán khi nào user sẽ tương tác tiếp.
- **Decision process:** hệ thống phải quyết định khi nào nên gửi hoặc không gửi recommendation.

Trong số đó, hướng mạnh nhất cho research hiện tại là kết hợp **time-aware sequential recommendation** với **joint what-and-when prediction**, tức không chỉ recommend item mà còn dự đoán thời điểm item đó có khả năng được user quan tâm nhất.
