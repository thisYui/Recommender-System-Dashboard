# Phân Tích Chuyên Sâu: Advances in Temporal Point Processes (Phần 2 - Ứng dụng)

**Nguồn:** Trích xuất từ mã nguồn LaTeX của bài báo "Advances in Temporal Point Processes: Bayesian, Neural, and LLM Approaches" (Zhou et al., Tháng 1/2025, arXiv:2501.14291).

Temporal Point Processes (TPPs) không chỉ là những phương trình toán học khô khan. Chúng giải quyết những bài toán mà các mô hình Machine Learning thông thường bó tay: Dự đoán thời gian thực. Theo Section 6 của bài báo, các ứng dụng của TPP được chia thành 2 nhánh khổng lồ: **Event Prediction** (Dự đoán sự kiện) và **Causal Discovery** (Khám phá nhân quả).

---

## 1. Ứng dụng 1: Event Prediction (Dự đoán Sự kiện)

Mục tiêu cốt lõi của nhánh này là: Dựa vào lịch sử, dự đoán **thời điểm (Timing)**, **tần suất (Frequency)**, và **loại (Type)** của sự kiện tiếp theo.

Các lĩnh vực ứng dụng bao gồm:
- **Mạng xã hội (Social Networks):** Khi nào một tin tức giả (Fake news) sẽ lan truyền (Retweet dynamics)? Khi nào một tài khoản bot sẽ spam?
- **Dịch tễ học (Epidemiology):** Dự đoán thời điểm và vị trí bùng phát số ca nhiễm COVID-19 tiếp theo.
- **Tài chính (Finance):** Dự đoán các biến động vi mô (microstructure) trong các phiên giao dịch tần suất cao (High-frequency trading).
- **HỆ THỐNG GỢI Ý (Recommendation Systems):** Đây chính là trọng tâm của dự án chúng ta.

### TPP trong Hệ thống Gợi ý (RecSys)
Theo các trích dẫn trong bài báo (Ví dụ: `mei2017neural`, `wang2021sequential`, `meng2024interpretable`), TPP được sử dụng để giải quyết bài toán:
**"Dự đoán thời gian mua sắm tương lai của người dùng và loại sản phẩm họ sẽ mua, dựa trên lịch sử mua hàng trong quá khứ."**

Khác với các hệ thống Sequential Recommendation (SR) thông thường chỉ dự đoán `Next-Item` (Mua áo xong mua gì?), TPP dự đoán đồng thời `Next-Time` và `Next-Item`. Điều này mở ra cánh cửa cho các chiến lược **Promotional (Khuyến mãi)** và **Active Recommendation (Gợi ý chủ động)**:
- Thay vì gửi email quảng cáo lúc nửa đêm khiến khách khó chịu (Spam).
- Hệ thống dùng TPP tính toán hàm Cường độ $\lambda^*(t)$. Khi hàm này đạt đỉnh vào lúc 8h sáng thứ Bảy, hệ thống tự động bắn Notification chứa đúng món hàng khách đang quan tâm.

---

## 2. Ứng dụng 2: Causal Discovery (Khám phá Nhân quả)

Nếu "Event Prediction" là đoán Tương lai, thì "Causal Discovery" là nhìn về Quá khứ để tìm **Nguyên nhân cốt lõi**. Trong Multivariate Hawkes Process (Hawkes Process đa biến), khám phá nhân quả nhằm mục đích khôi phục lại "Cấu trúc nhân quả" (Causal structure) giữa các loại sự kiện khác nhau.

Các lĩnh vực ứng dụng:
- **Khoa học thần kinh (Neuroscience):** Giải mã tín hiệu não. Nếu nơ-ron A chớp (spike), nơ-ron B có chớp theo không?
- **Trí tuệ nhân tạo Vận hành (AI Operations):** Khi hệ thống server sập hàng loạt, lỗi nào là nguyên nhân gốc rễ (Root cause), lỗi nào chỉ là hiệu ứng dây chuyền (Secondary effects)?

### Causal Discovery trong Hành vi Người dùng (Finance & Social)
- **Tài chính:** Giữa hàng triệu lệnh Mua (Bids) và Bán (Asks), lệnh nào thao túng lệnh nào?
- **Mạng xã hội / Bán hàng:** Phân tích cấu trúc Lan truyền. Hành động "Share" của một KOL (Người có sức ảnh hưởng) thực sự tạo ra bao nhiêu hành động "Mua hàng" từ người theo dõi?

**Cách TPP khám phá nhân quả (Granger Causality):**
Trong phương trình Cường độ của quá trình Hawkes:
$\lambda_i^*(t) = \mu_i + \sum_{j=1}^{d}\sum_{t_n < t} \phi_{j,i}(t - t_n)$

Ma trận hàm kích thích $\phi_{j,i}$ chính là chìa khóa. Nếu $\phi_{j,i}(\cdot) = 0$, điều đó chứng tỏ sự kiện $j$ **KHÔNG** phải là nguyên nhân gây ra sự kiện $i$ (Process $j$ does not Granger-cause process $i$). Việc học các ma trận này giúp các kỹ sư vẽ ra được một sơ đồ nhân quả chính xác tuyệt đối từ hàng triệu bản ghi click lộn xộn.

---
**Tóm lại:**
Hai phần này của bài Survey đã củng cố một nền tảng vững chắc: **Dự đoán thời điểm (Next-time prediction)** không chỉ là một ý tưởng bột phát, mà nó là một nhánh nghiên cứu lâu đời và cực kỳ quan trọng trong Khoa học máy tính, được ứng dụng từ y tế, chứng khoán cho đến thương mại điện tử. 

Chúng ta đã hoàn thành việc mổ xẻ bài Survey lớn nhất về TPP. Bạn có muốn mình chuyển sang bài báo **PASRec** (của đối tác bạn) để xem họ dùng **Diffusion Model** kết hợp với dự đoán thời gian như thế nào không?