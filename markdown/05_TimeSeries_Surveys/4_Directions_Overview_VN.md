# 4 Hướng Đi Đột Phá Trong Dự Đoán Chuỗi Thời Gian (Time Series Forecasting - TSF) 2024-2026

Dựa trên các bài khảo sát (Survey) toàn diện nhất từ arXiv trong giai đoạn 2024-2026, lĩnh vực Dự đoán Chuỗi Thời gian (TSF) đang chứng kiến sự dịch chuyển mạnh mẽ. Nếu như trước đây các mô hình thống kê (ARIMA) hay Machine Learning truyền thống thống trị, thì nay Deep Learning và Generative AI đang định hình lại toàn bộ sân chơi.

Dưới đây là phân tích chi tiết về **4 hướng nghiên cứu cốt lõi**:

---

## Hướng 1: Sự Đa Dạng Kiến Trúc Học Sâu (Deep Learning Architectures)

Đây là hướng đi nền tảng, tập trung vào việc thiết kế cấu trúc mạng Neural sao cho bắt được các đặc trưng dài hạn và ngắn hạn của chuỗi thời gian.

**Các bài báo tham chiếu:**
- *Deep Learning for Time Series Forecasting: A Survey* (arXiv:2503.10198)
- *A Comprehensive Survey of Deep Learning for Time Series Forecasting: Architectural Diversity...* (arXiv:2411.05793)

**Những điểm nhấn kỹ thuật:**
1. **Sự thoái trào của RNN/LSTM:** Dù từng là "tiêu chuẩn vàng" cho dữ liệu chuỗi, RNN/LSTM gặp vấn đề lớn về gradient vanishing và khả năng tính toán song song, khiến chúng dần bị thay thế.
2. **Kỷ nguyên của Transformer:** Transformer (với cơ chế Attention) đã giải quyết bài toán phụ thuộc dài hạn (long-term dependencies). Các mô hình như Informer, Autoformer, PatchTST đã thống trị các bảng xếp hạng.
3. **Sự phục hưng của MLP (Mạng nơ-ron truyền thẳng):** Đáng kinh ngạc là các nghiên cứu gần đây (như DLinear) chứng minh rằng: Đối với TSF, đôi khi một kiến trúc tuyến tính đơn giản (Simple Linear Layers) lại có thể đánh bại cả Transformer phức tạp về độ chính xác và tốc độ, do nó bảo toàn được trật tự thời gian tốt hơn.
4. **"Tân binh" Mamba (State Space Models):** Kiến trúc Mamba đang nổi lên như một thế lực mới, hứa hẹn thay thế Transformer bằng cách duy trì bộ nhớ trạng thái với độ phức tạp tuyến tính $O(N)$ thay vì $O(N^2)$, cực kỳ phù hợp cho chuỗi thời gian siêu dài.

---

## Hướng 2: Mô Hình Nền Tảng & Trí Tuệ Nhân Tạo Tạo Sinh (Foundation Models & LLMs)

Việc đào tạo một mô hình từ đầu cho mỗi bộ dữ liệu rất tốn kém và thiếu tính tổng quát. Hướng đi này vay mượn sự thành công của LLMs (như GPT-4, LLaMA) sang thế giới chuỗi thời gian.

**Bài báo tham chiếu:**
- *A Survey of Deep Learning and Foundation Models for Time Series Forecasting* (arXiv:2401.13912)

**Những điểm nhấn kỹ thuật:**
1. **Zero-shot Forecasting:** Các Foundation Models (như TimeGPT, Lag-Llama, Chronos) được huấn luyện trước (Pre-trained) trên hàng tỷ điểm dữ liệu từ nhiều nguồn khác nhau (chứng khoán, thời tiết, điện năng). Nhờ đó, chúng có thể dự đoán một bộ dữ liệu hoàn toàn mới (Zero-shot) mà không cần huấn luyện lại.
2. **Cross-Modality (Bơm tri thức văn bản vào số liệu):** Nhiều nghiên cứu hiện tại tìm cách đưa các kiến thức từ Knowledge Graphs hoặc LLMs (ví dụ: các sự kiện tin tức, quy luật khoa học) vào mô hình dự đoán. Ví dụ: Dự đoán thị trường chứng khoán không chỉ dựa vào giá trị số trong quá khứ, mà còn đọc hiểu cả các báo cáo tài chính.
3. **Prompting cho Time Series:** Giống như NLP, người ta bắt đầu đóng gói dữ liệu chuỗi thời gian thành các "Token" và sử dụng kỹ thuật Prompt để điều khiển mô hình sinh ra dự đoán.

---

## Hướng 3: Mô Hình Khuếch Tán (Diffusion Models)

Nếu Diffusion làm mưa làm gió trong việc tạo ra những bức ảnh siêu thực (Midjourney), thì trong TSF, nó đang tạo ra một cách tiếp cận hoàn toàn mới về dự đoán theo xác suất (Probabilistic Forecasting).

**Các bài báo tham chiếu:**
- *Diffusion Models for Time Series Forecasting: A Survey* (arXiv:2507.14507)
- *The Rise of Diffusion Models in Time-Series Forecasting* (arXiv:2401.03006)

**Những điểm nhấn kỹ thuật:**
1. **Dự đoán không chỉ là 1 con số:** Các mô hình TSF thông thường đưa ra một giá trị điểm (Point forecast). Trong khi đó, Diffusion Models đưa ra một **phân phối xác suất (Probabilistic forecast)**. Nó cho phép đánh giá độ rủi ro và khoảng tin cậy của dự đoán.
2. **Cơ chế Noise - Denoise:** Mô hình học cách thêm nhiễu (noise) vào chuỗi thời gian thực, sau đó học quá trình khử nhiễu (denoise) để khôi phục lại dữ liệu. Quá trình này giúp mô hình nắm bắt được các phân phối dữ liệu cực kỳ phức tạp mà các mô hình truyền thống bó tay.
3. **Conditional Diffusion:** Dự đoán có điều kiện. Mô hình Diffusion được "dẫn đường" (conditioned) bởi các dữ liệu lịch sử hoặc các biến ngoại sinh (như sự kiện thời tiết) để sinh ra tương lai một cách chính xác nhất. Tiêu biểu là các mô hình như TimeGrad, CSDI.

---

## Hướng 4: Chuỗi Đa Biến & AI Lấy Dữ Liệu Làm Trung Tâm (Multivariate & Data-Centric AI)

Dữ liệu trong thực tế hiếm khi đi đơn lẻ. Ví dụ: Dự đoán giao thông cần tốc độ, mật độ, thời tiết; Dự đoán năng lượng cần nhiệt độ, độ ẩm, sức gió.

**Các bài báo tham chiếu:**
- *A Comprehensive Survey of Deep Learning for Multivariate Time Series Forecasting: A Channel Strategy Perspective* (arXiv:2502.10721)
- *Survey and Taxonomy: The Role of Data-Centric AI in Transformer-Based Time Series Forecasting* (arXiv:2407.19784)

**Những điểm nhấn kỹ thuật:**
1. **Chiến lược Kênh (Channel Strategy):** Trong chuỗi thời gian đa biến (Multivariate Time Series - MTS), có hai trường phái:
   - *Channel-mixing (Trộn kênh):* Mô hình hóa sự tương quan giữa các kênh ngay từ đầu.
   - *Channel-independence (Kênh độc lập):* Xử lý từng biến một cách độc lập (như PatchTST), và ngạc nhiên thay, cách này lại thường mang lại kết quả tốt hơn và chống lại việc các kênh "làm nhiễu" lẫn nhau.
2. **Data-Centric AI (AI lấy dữ liệu làm trung tâm):** Sự công nhận rằng "Dữ liệu tốt hơn Mô hình phức tạp". Thay vì liên tục nghĩ ra các kiến trúc mạng mới, hướng đi này tập trung vào:
   - Các kỹ thuật Data Augmentation (Tăng cường dữ liệu).
   - Xử lý Missing Values (Dữ liệu khuyết thiếu), Imputation.
   - Giải quyết Distribution Shift (Sự thay đổi phân phối dữ liệu theo thời gian - một vấn đề cực kỳ nhức nhối trong TSF).

---

## Tóm Lại: Bạn Nên Chọn Hướng Nào?

- Nếu bạn thích kiến trúc nhẹ nhàng, chạy nhanh: Đi theo **Hướng 1 (Mamba hoặc Linear MLPs)**.
- Nếu bạn có tài nguyên tính toán mạnh, muốn làm mô hình tổng quát cho công ty: Đi theo **Hướng 2 (Foundation Models/LLMs)**.
- Nếu bài toán của bạn cần tính toán rủi ro, dự đoán khoảng tin cậy (ví dụ: chứng khoán, y tế): **Hướng 3 (Diffusion Models)** là lựa chọn hàng đầu.
- Nếu bạn làm việc với dữ liệu cảm biến IoT, nhiều thiết bị: Bắt buộc phải nghiên cứu sâu **Hướng 4 (Multivariate / Channel Independence)**.