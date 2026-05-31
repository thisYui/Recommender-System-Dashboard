# Tóm tắt Chi tiết (Phân tích Chuyên sâu): COMPASS - Knowledge Graph-Augmented LLMs for Explainable Conversational Recommendations

**Tên mô hình:** COMPASS (Compact Preference Analyzer and Summarization System)  
**Tác giả:** Zhangchi Qiu, Linhao Luo, Shirui Pan, Alan Wee-Chung Liew (Griffith University, Monash University)  
**Nơi công bố/Năm:** Đầu năm 2025 (Từ kho lưu trữ Awesome-LLM-KG-RecSys)  
**Lĩnh vực:** Hệ thống gợi ý đàm thoại (Conversational Recommender Systems - CRSs), Gợi ý có tính giải thích (Explainable Recommendations), Hội tụ LLM và KG.

---

## 1. Bối cảnh & Điểm nghẽn (The Problem)

Các Hệ thống gợi ý qua hội thoại (CRSs) có nhiệm vụ trò chuyện với người dùng để tìm ra sở thích và gợi ý sản phẩm. Tuy nhiên, chúng vướng phải một nghịch lý:
- Các CRSs truyền thống lưu trữ "Sở thích người dùng" dưới dạng các **Vector ẩn (Hidden Embeddings)**. Máy tính hiểu rất rõ vector này, nhưng con người thì không. Nó tạo ra các "Hộp đen" (Black-box) không thể giải thích.
- Gần đây, người ta thử dùng Large Language Models (LLMs) để sinh ra các đoạn văn bản giải thích sở thích (Ví dụ: "Người dùng này thích phim hành động"). Nhưng LLM nguyên bản lại thiếu kiến thức chuyên ngành sâu sắc, dẫn đến hiện tượng **Ảo giác (Hallucination)** - đoán sai lệch hoặc bịa ra diễn viên/đạo diễn không liên quan.
- Khi cố gắng ghép Đồ thị tri thức (KG) vào LLM để trị bệnh ảo giác, các nhà khoa học vấp phải **Khoảng cách Phương thức (Modality Gap)**: Cấu trúc toán học của KG và cấu trúc ngôn ngữ tự nhiên của LLM không "nói chung một ngôn ngữ", khiến LLM không thể đọc hiểu KG một cách hiệu quả.

---

## 2. Ý tưởng Đột phá: Mô hình COMPASS

COMPASS được thiết kế như một **mô-đun cắm-và-chạy (Plug-and-play)**. Bất kỳ hệ thống CRS nào cũng có thể cắm COMPASS vào để tăng sức mạnh và khả năng giải thích. COMPASS giải quyết bài toán "Khoảng cách phương thức" thông qua quy trình huấn luyện **2 Giai đoạn (Two-stage Training)**:

### Giai đoạn 1: Graph Entity Captioning Pre-training (Dạy LLM đọc hiểu Đồ thị)
- Đầu tiên, hệ thống dùng Mạng Đồ thị (R-GCN) để mã hóa cấu trúc KG thành các Vector Đồ thị.
- Sau đó, thông qua một lớp Adapter, nó ép LLM phải làm một bài tập: **Dịch Vector Đồ thị thành Văn bản tự nhiên**. (Ví dụ: Nhìn vào vector tọa độ, LLM phải viết được câu *"Đây là diễn viên Robert Downey Jr. đóng vai Iron Man"*).
- Giai đoạn này tạo ra một "Từ điển dịch thuật" hoàn hảo, giúp LLM hoàn toàn thấu hiểu ngôn ngữ toán học của Đồ thị tri thức.

### Giai đoạn 2: Knowledge-Aware Instruction Tuning (Tinh chỉnh theo Lệnh nhận thức Tri thức)
- Sau khi LLM đã "biết đọc" KG, nó được đưa vào thực chiến.
- Đầu vào: Lịch sử trò chuyện của người dùng + Vector Đồ thị của các sản phẩm được nhắc đến trong đoạn chat.
- Nhiệm vụ: LLM phải suy luận và sinh ra một **"Bản tóm tắt sở thích người dùng" (User Preference Summary)** bằng ngôn ngữ tự nhiên. 
- Nhờ có KG làm điểm tựa (Grounding) và khả năng hiểu ngôn ngữ sâu sắc, bản tóm tắt này cực kỳ chính xác, logic và không có tính ảo tưởng.

### Tích hợp vào Hệ thống (Fusion)
Sau khi LLM viết ra Bản tóm tắt sở thích bằng chữ, bản tóm tắt này lại được chuyển thành Vector. COMPASS dùng một **Cơ chế Cổng (Gating mechanism)** để trộn Vector sở thích mới này với Vector sở thích gốc của hệ thống CRS, rồi đưa ra quyết định gợi ý cuối cùng.

---

## 3. Thiết lập Thí nghiệm & Kết quả (Results)

- **Datasets:** Đánh giá trên 2 bộ dữ liệu hội thoại nổi tiếng: **ReDial** (Phim ảnh) và **INSPIRED**.
- **Base CRSs:** Lắp thử COMPASS vào 4 hệ thống CRS truyền thống: KBRD, KGSF, RevCore, và $C^2$-CRS.
- **LLM:** Tác giả sử dụng bộ tóm tắt tinh chỉnh dựa trên nền tảng của Llama. Đánh giá tự động được thực hiện bởi GPT-4o-mini.
- **Kết quả:**
  - **Tăng vọt Hiệu suất Gợi ý:** Khi cắm COMPASS vào các mô hình cũ, hiệu suất (HR@50, NDCG@50) tăng từ **10% đến 47%**. Việc có thêm Bản tóm tắt ngôn ngữ tự nhiên thực sự giúp các mô hình tìm ra sản phẩm chính xác hơn hẳn.
  - **Khả năng Lý luận Siêu việt:** Bản tóm tắt do COMPASS viết ra đạt điểm cao chót vót về **Tính nhất quán Sự thật (Factual Consistency)** và **Năng lực Lý luận (Reasoning Proficiency)** khi so sánh với GPT-4o gốc (không có KG). COMPASS hiểu rõ được ẩn ý của người dùng, biết chính xác đạo diễn, diễn viên và thể loại mà người dùng đang nhắm tới dựa trên những gợi ý mập mờ trong đoạn chat.

---

## 4. Ví dụ Trực quan (Dễ hình dung)

**Đoạn Chat thực tế của người dùng:**
*"Chào bạn, tôi đang muốn tìm một bộ phim nói về tội phạm. Tôi rất thích xem phim The Professional (Sát thủ Leon)."*

- **Hệ thống cũ (Llama-Summary không có Đồ thị):** Bị ảo giác. Nó đoán: *"Người dùng thích phim Làn sóng mới của Pháp (French New Wave), đạo diễn Francois Truffaut"*. (Sai hoàn toàn sự thật về phim Leon).
- **Hệ thống GPT-4o (Không có Đồ thị):** Đoán chung chung: *"Người dùng thích phim tội phạm, hành động, giật gân"*. (Đúng nhưng quá an toàn và nông cạn).
- **Hệ thống COMPASS (Có Đồ thị Tri thức chống lưng):** 
  Nhờ tra cứu chéo trên Đồ thị, nó tìm ra bộ phim *The Professional*, biết được đạo diễn là *Luc Besson*, diễn viên là *Jean Reno* và *Natalie Portman*. Nó suy luận sắc bén và đưa ra bản tóm tắt:
  *"Người dùng đang tìm kiếm phim thể loại **Hành động, Tội phạm, Giật gân**. Họ đặc biệt thích phong cách phim về **Sát thủ chuyên nghiệp, bạo lực ngầm**, giống như các tác phẩm do **Luc Besson** đạo diễn hoặc có mặt **Jean Reno**".*
  
Nhờ bản tóm tắt sắc lẹm và chính xác đến từng centimet này, hệ thống gợi ý ở phía sau dễ dàng bám vào từ khóa "Luc Besson" hoặc "Jean Reno" để gợi ý ngay lập tức các siêu phẩm hành động tương tự, khiến người dùng hoàn toàn bị thuyết phục!