# Learnable Item Tokenization for Generative Recommendation

**Tên mô hình:** LETTER (LEarnable Tokenizer for generaTivE Recommendation)
**Tác giả:** Wenjie Wang, Honghui Bao, Xinyu Lin, et al. (Đại học Quốc gia Singapore - NUS, Đại học Khoa học và Công nghệ Trung Hoa)
**Nơi công bố/Năm:** CIKM 2024
**Lĩnh vực:** Generative Recommendation (Gợi ý Tạo sinh), Learnable Semantic IDs (Định danh Ngữ nghĩa có thể học được), Collaborative Filtering

---

## 1. Bối cảnh & Phản biện mô hình TIGER (The Problem)

Mặc dù mô hình **TIGER** (sử dụng RQ-VAE để tạo Semantic IDs từ văn bản) đã mở ra kỷ nguyên mới cho Generative RecSys, nhóm tác giả của LETTER đã chỉ ra **2 lỗ hổng tử huyệt** của việc chỉ dùng văn bản (text) để tạo ID:

1. **Sự lệch pha với Tín hiệu Hành vi (Misalignment with Collaborative Signals):** Hai sản phẩm có thể có nội dung mô tả rất giống nhau (ví dụ: *Đàn Guitar Acoustic* và *Đàn Guitar Điện*). RQ-VAE của TIGER sẽ ép chúng vào chung một cụm và cấp cho chúng chuỗi Semantic ID gần giống hệt nhau. Tuy nhiên, hành vi mua sắm thực tế của người dùng lại hoàn toàn khác nhau (người chơi Acoustic hiếm khi mua đồ của Electric). Khi ép một Semantic ID tĩnh vào LLM, mô hình sẽ bị bối rối và gợi ý sai.
2. **Thiên kiến Gán mã & Thiên kiến Tạo sinh (Code Assignment Bias):** RQ-VAE thường bị rơi vào trạng thái mất cân bằng, nơi một số ít token (mã) được gán cho một lượng khổng lồ sản phẩm. Điều này dẫn đến việc LLM khi sinh ra mã (generate) sẽ có xu hướng chỉ sinh ra các mã phổ biến này, bỏ qua các sản phẩm ngách (Item Generation Bias).

---

## 2. Giải pháp Đột phá: Mô hình LETTER

Để giải quyết, nhóm nghiên cứu đề xuất **LETTER** – một Tokenizer (bộ tạo mã) "có thể học được" thay vì bị đóng băng (frozen) như TIGER. LETTER ép các Semantic IDs phải thỏa mãn cả 3 điều kiện thông qua 3 hàm loss (hàm suy hao) khác nhau:

### 2.1. Cụm 1: Semantic Regularization (Định chuẩn Ngữ nghĩa)

- Kế thừa nền tảng của TIGER, LETTER vẫn dùng **RQ-VAE** để nén Vector Văn bản của sản phẩm thành một chuỗi mã phân cấp từ thô đến tinh (Hierarchical Semantics). Đảm bảo tính năng Cold-Start vẫn được giữ nguyên.

### 2.2. Cụm 2: Collaborative Regularization (Định chuẩn Hành vi người dùng - Đột phá cốt lõi)

- Tác giả sử dụng một mô hình truyền thống (như SASRec hoặc LightGCN) để trích xuất ra **Vector Hành vi (CF Embedding)** của các sản phẩm.
- Sử dụng **Học đối chiếu (Contrastive Learning):** Ép Vector Lượng tử hóa (Quantized Embedding) sinh ra từ RQ-VAE phải "khớp" với Vector Hành vi.
- **Kết quả:** Nếu 2 món đồ (dù khác nhau về text) nhưng hay được mua cùng nhau, chúng sẽ bị ép phải có Semantic ID giống nhau. Ngược lại, 2 món đồ mô tả giống nhau nhưng tệp khách hàng khác nhau sẽ bị tách mã ID ra.

### 2.3. Cụm 3: Diversity Regularization (Định chuẩn Đa dạng)

- Sử dụng thuật toán gom cụm (K-Means) trên các vector mã (Code Embeddings). Áp dụng hàm suy hao để đẩy các vector ở các cụm khác nhau ra xa, và kéo các vector cùng cụm lại gần.
- **Kết quả:** Ép RQ-VAE phải sử dụng đồng đều tất cả các token trong Bảng mã (Codebook), giải quyết triệt để tình trạng một mã bị gán cho quá nhiều sản phẩm.

### 2.4. Ranking-Guided Generation Loss (Tối ưu hóa lúc fine-tune LLM)

- Khi tinh chỉnh (fine-tune) LLM, thay vì dùng hàm Loss sinh từ thông thường (Negative Log-Likelihood), tác giả điều chỉnh tham số **Nhiệt độ $\tau$ (Temperature)**. Việc này đánh mạnh hình phạt vào các sản phẩm "Hard-negative" (những sản phẩm bị mô hình đoán sai nhưng lại có điểm số rất cao), giúp cải thiện khả năng Xếp hạng (Ranking) tổng thể của LLM.

---

## 3. Kết quả Thí nghiệm (Results)

Thử nghiệm trên 3 bộ dữ liệu: Instruments, Beauty, Yelp. Mô hình LETTER được gắn vào làm Tokenizer cho 2 mô hình LLM là TIGER và LC-Rec.

1. **Đè bẹp các Baselines:** LETTER-TIGER và LETTER-LC-Rec vượt qua tất cả các mô hình ID cũ (P5-CID, SASRec, LightGCN) và TIGER gốc (không có LETTER).
2. **Cải thiện độ tận dụng Codebook:** Tỉ lệ sử dụng token trong bảng mã tăng vọt (từ việc chỉ dùng 76 mã lên dùng 150 mã trong bảng 256 mã), làm cho các cụm gợi ý đa dạng hơn rất nhiều.
3. **Thống nhất Text và Behavior:** Bài test độ tương đồng cho thấy LETTER tạo ra các mã Semantic ID phản ánh chuẩn xác cả "Sản phẩm đó là gì" và "Ai là người mua nó".

---

## 4. Ví dụ Trực quan (Dễ hình dung)

Hãy tưởng tượng bạn đang xây dựng AI cho một cửa hàng nhạc cụ:

- **Vấn đề của TIGER (Chỉ dùng Text):**

  - Khách hàng đang tìm mua đồ cho nhóm **Nhạc kịch Acoustic**.
  - Sản phẩm A: *Đàn Guitar Acoustic 6 dây* (Semantic ID: `10, 55, 3`).
  - Sản phẩm B: *Đàn Guitar Điện 6 dây* (Do mô tả Text quá giống nhau, TIGER gán ID: `10, 55, 4`).
  - Khi AI sinh ra gợi ý tiếp theo, nó dễ dàng bị nhầm lẫn và gợi ý Guitar Điện cho một người chỉ chơi Acoustic.
- **Giải pháp của LETTER (Kết hợp Hành vi):**

  - LETTER nhìn vào dữ liệu lịch sử mua hàng (Collaborative Signals) và phát hiện ra: "Ê, những người mua Guitar Acoustic toàn mua thêm Capo và Dây đàn gỗ, trong khi hội mua Guitar Điện lại toàn mua Amply và Phơ (Effect pedal). Tệp khách hàng này không liên quan gì nhau!".
  - Ngay lập tức, hàm **Collaborative Regularization** nhảy vào can thiệp. Nó buộc RQ-VAE phải đổi mã ID.
  - Đàn Guitar Acoustic giữ mã: `10, 55, 3`.
  - Đàn Guitar Điện bị đổi thành một mã nhánh khác: `10, 88, 1`.
  - $\rightarrow$ Kết quả: Khi LLM sinh gợi ý, nhờ sự phân tách mã rõ ràng dựa trên hành vi, nó sẽ KHÔNG BAO GIỜ sinh nhầm Guitar Điện cho khách hàng Acoustic nữa! Độ chính xác (Precision/NDCG) tăng vọt!
