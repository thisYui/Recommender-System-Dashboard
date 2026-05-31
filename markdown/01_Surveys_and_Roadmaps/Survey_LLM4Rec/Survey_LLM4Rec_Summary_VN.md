# A Survey on Large Language Models for Recommendation

**Tên bài báo:** A Survey on Large Language Models for Recommendation
**Tác giả:** Likang Wu, Zhi Zheng, Zhaopeng Qiu, Hao Wang, et al.
**Nơi công bố/Năm:** Cập nhật mới nhất trên arXiv năm 2024 (Bản gốc 2023)
**Lĩnh vực:** Tổng quan về Large Language Models (LLM) trong Hệ thống Gợi ý (RecSys)

---

## 1. Bối cảnh

Sự bùng nổ của Large Language Models (LLMs) như BERT, GPT, LLaMA đã mang lại hai vũ khí tối thượng cho Hệ thống Gợi ý: **Khả năng hiểu ngữ nghĩa sâu sắc** (từ các đoạn text review dài) và **Tri thức thế giới** (biết rằng sách Harry Potter liên quan đến phép thuật dù không ai dạy nó).

Có **2 trường phái lớn** dựa trên bản chất của mô hình ngôn ngữ:

1. **Discriminative LLM for Recommendation (DLLM4Rec):** Sử dụng các mô hình phân biệt (như BERT, RoBERTa). Nhiệm vụ chính của chúng là *Mã hóa* (Encoding) text thành vector nhúng.
2. **Generative LLM for Recommendation (GLLM4Rec):** Sử dụng các mô hình tạo sinh (như GPT, T5, LLaMA). Nhiệm vụ chính của chúng là *Sinh ra* (Generating) câu trả lời hoặc ID sản phẩm.

Dựa trên cách LLM được lắp ghép vào hệ thống RecSys, tác giả chia thành **3 Mô hình Hoạt động (Modeling Paradigms)**:

- **(1) LLM Embeddings + RS:** LLM đóng vai trò làm "Máy vắt nước cam" (Feature Extractor). Đưa text vào, LLM vắt ra Vector. Vector này được ném vào một mô hình RecSys truyền thống (như LightGCN) để chạy tiếp.
- **(2) LLM Tokens + RS:** LLM sinh ra các Tokens (như Semantic IDs trong bài TIGER/LETTER) dựa trên sở thích người dùng. Tokens này được tích hợp vào RS.
- **(3) LLM as RS:** Ép LLM làm luôn Hệ thống Gợi ý. Input là một đoạn văn (Prompt) mô tả người dùng. Output là câu trả lời trực tiếp của LLM (Ví dụ: "Bạn nên mua cuốn sách A").

---

## 2. GLLM4Rec: Kỷ nguyên của Gợi ý Tạo sinh

Vì xu hướng đang dịch chuyển mạnh sang Generative AI, bài báo dành sự tập trung đặc biệt cho **GLLM4Rec**. Tác giả chia GLLM4Rec thành 2 nhóm chiến lược dựa trên việc *có can thiệp vào tệp trọng số của LLM hay không*.

### 2.1. Non-tuning Paradigm (Không Fine-tune - Tiết kiệm chi phí)

Sử dụng LLM nguyên bản (Zero-shot/Few-shot) thông qua Prompt Engineering.

- **Prompting:** Đưa thẳng thông tin vào ChatGPT. Prompt thường gồm 3 phần: Mô tả tác vụ (Task description) + Tiêm thông tin hành vi (Behavior injection) + Ép khuôn định dạng (Format indicator).
  - *Ứng dụng:* Dùng LLM làm **Agent** (Tác tử) điều phối toàn bộ hệ thống gợi ý đa lượt (ChatREC). Hoặc dùng LLM làm **Data Augmentor** (Bộ tăng cường dữ liệu) - yêu cầu ChatGPT đọc tiêu đề nghèo nàn của sản phẩm rồi tự động viết ra một đoạn review chi tiết để làm giàu dữ liệu cho RS truyền thống.
- **In-context Learning (ICL):** Cung cấp thêm một vài ví dụ (Demonstration examples) ngay trong Prompt để LLM tự học nhanh cách trả lời mà không cần train. Vd: "User thích Phim A -> Gợi ý Phim B. Vậy User thích Phim C -> Gợi ý phim gì?".

### 2.2. Tuning Paradigm (Có Fine-tune - Tối ưu hiệu suất)

LLM nguyên bản (GPT-3.5) thường kém hơn các mô hình RS chuyên biệt khi chấm điểm (Scoring). Do đó, người ta phải "độ" lại LLM.

- **Fine-tuning:** Train lại trọng số của LLM để nó phục vụ riêng cho RecSys (Ví dụ: TALLRec dùng dữ liệu Alpaca để fine-tune). Rất tốn kém tài nguyên.
- **Prompt Tuning (Soft Prompt):** Đóng băng LLM, chỉ thêm vài vector nhúng "có thể học được" (Learnable Prompts) vào đầu vào. Prompts này đóng vai trò "dạy" LLM cách tập trung vào các đặc trưng quan trọng cho việc gợi ý.
- **Instruction Tuning:** Dạy LLM bằng hàng loạt các chỉ thị (Instructions) đa dạng (ví dụ: "Dự đoán rating", "Viết review", "Gợi ý tiếp theo"). Việc này giúp LLM làm tốt nhiều tác vụ RecSys cùng lúc và hiểu ý định con người tốt hơn.

---

## 3. Những Điểm nghẽn và Thách thức (Challenges & Findings)

Áp dụng LLM cho RecSys không phải là "viên đạn bạc". Bài báo chỉ ra hàng loạt "tác dụng phụ" cần giải quyết trong tương lai:

### 3.1. Model Bias (Thiên kiến của Mô hình)

- **Position Bias (Thiên kiến vị trí):** LLM mắc bệnh "lười đọc". Nếu bạn đưa cho nó một danh sách 20 sản phẩm ứng viên, nó có xu hướng chỉ chọn những sản phẩm nằm ở đầu danh sách (Top order), mặc kệ sản phẩm cuối hợp lý hơn.
- **Popularity Bias (Thiên kiến độ phổ biến):** LLM được train trên Internet. Nó sẽ ưu tiên gợi ý những sản phẩm nổi tiếng (vì gặp nhiều trong quá trình pre-train), và ngó lơ các sản phẩm ngách (Long-tail items).
- **Personalization Bias:** LLM rất giỏi hiểu Text, nhưng lại... rất dốt trong việc hiểu ID người dùng (User ID) và ID sản phẩm (Item ID). Nó hiểu "iPhone" tốt hơn nhiều so với việc hiểu `Item_492` là gì, làm giảm đi tính cá nhân hóa đặc thù của RecSys truyền thống.

### 3.2. Vấn đề Ảo giác và Kiểm soát (Hallucination & Controlled Generation)

Khi dùng LLM làm RS, nó có thể gợi ý ra một bộ phim "nghe có vẻ rất hay" nhưng... **không hề tồn tại** trên đời, hoặc không có sẵn trong giỏ hàng của doanh nghiệp (Out-of-corpus generation). Việc buộc LLM phải sinh ra các gợi ý "bám chặt" vào cơ sở dữ liệu thật (Grounding) thông qua Knowledge Graph hay Semantic IDs (như TIGER/LETTER) là nhiệm vụ sống còn.

---

## 4. Tổng Kết

Bài Survey này là kim chỉ nam cho bất cứ ai bước chân vào mảng LLM4Rec:

- Nếu bạn có ít tiền và muốn làm nhanh $\rightarrow$ Dùng **Non-tuning (Prompting)**, tận dụng API của GPT-4 để lấy tri thức thế giới bổ trợ cho mô hình.
- Nếu bạn có GPU mạnh và muốn đấu SOTA $\rightarrow$ Đi theo hướng **Generative LLM (Tuning)**, biến bài toán gợi ý thành bài toán sinh chuỗi Token.
- Trận chiến khốc liệt nhất hiện nay không phải là làm sao để LLM hiểu Text tốt hơn, mà là làm sao để LLM hiểu được các **ID định danh phi ngôn ngữ** và khắc phục được căn bệnh **Thiên kiến vị trí (Position Bias)**.
