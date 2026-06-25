# Sự Giao Thoa Giữa Chuỗi Thời Gian và Hệ Thống Gợi Ý: Sequential & Session-based Recommendation (2024-2026)

Khi các kỹ thuật dự đoán chuỗi thời gian (Time Series Forecasting) được áp dụng vào Hệ thống gợi ý (Recommender Systems), bài toán này có những tên gọi đặc thù là: **Sequential Recommendation (Gợi ý tuần tự)**, **Session-based Recommendation (Gợi ý theo phiên)**, hoặc **Temporal/Time-aware Recommendation (Gợi ý theo thời gian)**.

Thay vì dự đoán "nhiệt độ" hay "giá cổ phiếu", mô hình dự đoán "Sản phẩm tiếp theo (Next-item)" mà người dùng sẽ tương tác dựa trên chuỗi hành vi trong quá khứ.

Dưới đây là các hướng nghiên cứu lớn và các bài Survey mới nhất (2024-2026) thống trị sự giao thoa này:

---

## 1. Gợi ý Tuần tự Cơ bản & Các Biến thể Mới (Core Sequential Recommendation)

**Bài báo tiêu biểu:** *A Survey on Sequential Recommendation* (Liwei Pan et al., Dec 2024 - arXiv:2412.12770)

Đây là bài Survey toàn diện nhất hiện nay về Sequential Recommendation (SR). Nó định nghĩa SR không chỉ là chuỗi ID sản phẩm, mà đi sâu vào việc xây dựng "đặc tính" của sản phẩm qua thời gian.

**Các hướng đột phá:**
- **Ultra-long SR (Gợi ý trên chuỗi siêu dài):** Xử lý lịch sử người dùng kéo dài nhiều năm (hàng nghìn tương tác). Đây là nơi các kiến trúc Time Series tiên tiến như **Mamba** (State Space Models) bắt đầu đánh bại Transformer vì Transformer bị giới hạn bởi độ phức tạp $O(N^2)$.
- **Continuous SR (Gợi ý liên tục):** Mô hình hóa khoảng cách thời gian thực tế (Time intervals) giữa các lần click, thay vì coi chúng là các bước nhảy rời rạc (step 1, step 2).
- **Data-augmented SR:** Giải quyết vấn đề dữ liệu thưa thớt bằng cách dùng các mô hình tạo sinh để "sinh ra" các chuỗi tương tác ảo, hoặc dùng Contrastive Learning (Học đối chiếu) để làm phong phú dữ liệu huấn luyện (Được củng cố bởi Survey: *Data Augmentation for Sequential Recommendation*, Sep 2024).

---

## 2. Gợi ý Theo Phiên Kết hợp Dữ liệu Phụ trợ (Side Information-Driven Session-based Recommendation)

**Bài báo tiêu biểu:** *A Survey on Side Information-driven Session-based Recommendation: From a Data-centric Perspective* (Xiaokun Zhang et al., May 2025 - arXiv:2505.12279)

Session-based Recommendation (SBR) khó hơn SR ở chỗ: Nó không biết người dùng là ai (Anonymous user). Nó chỉ có đúng một chuỗi click ngắn ngủi trong một "phiên" lướt web (VD: 15 phút vào Shopee).

**Điểm nhấn công nghệ:**
- Dữ liệu một phiên là quá ngắn để dự đoán. Các mô hình Time Series truyền thống sẽ thất bại. Giải pháp là **Data-centric AI** - tiêm thêm "Side Information" (Dữ liệu phụ trợ) vào chuỗi thời gian:
  - **Tri thức đa phương thức (Multimodal):** Hình ảnh, giá cả, màu sắc của sản phẩm được ghép vào chuỗi click.
  - **Review-Distilled Representations:** Trích xuất các thuộc tính cảm giác (mùi hương, màu sắc) từ Review bằng LLM, sau đó nhúng chúng vào mô hình chuỗi thời gian (như bài *Sensory-Aware Sequential Recommendation*, Mar 2026).

---

## 3. Mạng Nơ-ron Đồ thị (GNN) kết hợp Mô hình Chuỗi

**Bài báo tiêu biểu:** *Graph and Sequential Neural Networks in Session-based Recommendation: A Survey* (Zihao Li et al., Aug 2024 - arXiv:2408.14851)

Thay vì coi chuỗi thời gian chỉ là một đường thẳng $A \rightarrow B \rightarrow C$, người ta biến chuỗi click thành một **Đồ thị chuỗi (Sequential Graph)**.

**Lý do:**
- Trong một phiên lướt web, người dùng hay có thói quen click qua lại (Ví dụ: Xem Áo A \rightarrow Xem Áo B \rightarrow Quay lại xem Áo A).
- GNN kết hợp với RNN/Transformer giúp mô hình hóa các "bước nhảy cóc" này tốt hơn nhiều so với các mô hình Time Series tuyến tính đơn thuần.
- Bài Survey này chia các phương pháp thành hai nhóm lõi: Dựa trên Mạng chuỗi thuần túy (Sequential NN) và Dựa trên Mạng đồ thị (GNN-based).

---

## 4. LLM và "Nhận thức Thời gian" (Temporal Awareness) trong Gợi ý

**Bài báo tiêu biểu:** *Improve Temporal Awareness of LLMs for Sequential Recommendation* (Zhendong Chu et al., May 2024 - arXiv:2405.02778)

Đây là điểm giao thoa nóng nhất giữa LLM-KG-RecSys và Time Series.

**Nỗi đau hiện tại:**
- LLM rất giỏi ngôn ngữ nhưng lại bị "mù thời gian". Nó không hiểu rõ khái niệm "1 tháng trước" khác với "1 ngày trước" như thế nào khi nhìn vào lịch sử mua hàng. Nó đối xử với mọi item trong lịch sử bằng sự chú ý (attention) gần như nhau.

**Giải pháp:**
- Cần có các Framework (như Prompting đặc biệt) để mô phỏng quá trình nhận thức thời gian của con người.
- Kết hợp với **Đồ thị tri thức theo thời gian (Temporal Knowledge Graph)**: Đồ thị không còn tĩnh nữa. Mối quan hệ giữa các thực thể mang theo nhãn thời gian (Timestamps), giúp LLM bắt được xu hướng (Trend) thay đổi sở thích của người dùng.

---

## 5. Federated Learning trong Gợi ý Tuần tự (Federated Sequential Recommendation)

**Bài báo tiêu biểu:** *A Systematic Survey on Federated Sequential Recommendation* (Yichen Li et al., Feb 2025 - arXiv:2504.05313)

- Lịch sử tương tác theo thời gian của người dùng là dữ liệu cực kỳ nhạy cảm (Privacy).
- Hướng đi này cho phép mô hình hóa Time Series Recommendation trực tiếp trên thiết bị của người dùng (Edge device) mà không cần gửi chuỗi dữ liệu gốc lên máy chủ trung tâm.

---

### Tóm Lại: Bức Tranh Toàn Cảnh (LLM + KG + Time Series + RecSys)

Nếu bạn muốn tạo ra một câu chuyện (hoặc một Module dự án) đỉnh cao bao gồm tất cả các yếu tố này, đây là công thức:

1. **Vấn đề cốt lõi:** Lịch sử người dùng là một **Chuỗi thời gian** (Sequential/Temporal Data).
2. **Điểm nghẽn 1:** Các mô hình Time Series (RNN, Transformer) không hiểu được thuộc tính sản phẩm. $\rightarrow$ Giải quyết bằng **Knowledge Graph (KG)**.
3. **Điểm nghẽn 2:** KG tĩnh không bắt được sự thay đổi thời gian. LLM thì lại bị "mù thời gian" và "ảo giác".
4. **Giải pháp Đột phá:** Xây dựng một mô hình "K-RAGRec nhưng có Nhận thức Thời gian (Temporal-aware K-RAGRec)":
   - Dùng mô hình Time Series (như Mamba) để mã hóa lịch sử click.
   - Dùng KG để truy xuất Đồ thị con theo thời gian thực.
   - Dùng LLM làm trung tâm để tổng hợp cả Thời gian + Tri thức Đồ thị $\rightarrow$ Đưa ra kết quả cuối cùng không bị ảo giác.