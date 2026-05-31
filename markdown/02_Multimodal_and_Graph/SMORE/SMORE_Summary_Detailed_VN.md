# Spectrum-based Modality Representation Fusion Graph Convolutional Network for Multimodal Recommendation

**Tên mô hình:** SMORE (Spectrum-based MOdality REpresentation fusion)
**Tác giả:** Rongqing Kenneth Ong và Andy W. H. Khong (Nanyang Technological University, Singapore)
**Hội nghị/Năm:** WSDM '25 (Sắp diễn ra vào tháng 3/2025, bản preprint tháng 12/2024)
**Mã nguồn:** [https://github.com/kennethorq/SMORE](https://github.com/kennethorq/SMORE)
**Lĩnh vực:** Hệ thống Gợi ý Đa phương thức (Multimodal Recommender Systems)

---

## 1. Bối cảnh & Phản biện (The Problem)

Các hệ thống gợi ý hiện tại thường xuyên "pha trộn" (fuse) các đặc trưng đa phương thức (như Hình ảnh và Văn bản) bằng cách nối (concatenation) hoặc cộng (summation) rồi đưa lên đồ thị.

Tuy nhiên, bài báo này chỉ ra một điểm yếu chí mạng của các mô hình hiện tại (kể cả MGCN hay FREEDOM): **Sự khuếch đại nhiễu chéo phương thức (Amplification of cross-modality noise).**

- **Vấn đề:** Trong mỗi phương thức luôn tồn tại "nhiễu" rất đặc thù (Ví dụ: Một bức ảnh bị mờ nét, hoặc một đoạn mô tả văn bản chứa từ khóa rác không liên quan). Nếu chúng ta trộn thẳng Hình ảnh và Văn bản ở Không gian Đặc trưng (Spatial domain) như cách các mô hình cũ vẫn làm, phần nhiễu của Hình ảnh sẽ lây sang Văn bản và ngược lại, tạo ra một khối dữ liệu khổng lồ toàn rác, làm suy giảm nghiêm trọng độ chính xác của biểu diễn sản phẩm.
- **Minh chứng:** Tác giả đưa ra ví dụ trên tập Amazon: Một chiếc búa đồ chơi và một bộ áo liền quần lẽ ra không liên quan gì nhau, nhưng vì người bán đặt một từ khóa rác "Hammer" vào mô tả cái áo, khiến mô hình cũ đánh giá 2 món này giống nhau tới 69.35%. Ngược lại, 2 chiếc túi du lịch giống hệt nhau lại bị đánh giá độ tương đồng chỉ 11.27% vì một trong hai bức ảnh chụp bị nhòe (nhiễu hình ảnh).

---

## 2. Giải pháp Đột phá: SMORE (Đi vào Miền Tần số)

Làm sao để lọc rác khi rác và thông tin hữu ích bị dính chặt vào nhau? Tác giả đã mượn ý tưởng từ ngành **Xử lý Tín hiệu số (Signal Processing)**: Đưa mọi thứ từ *Không gian Đặc trưng* sang *Miền Tần số (Frequency Domain)*!

Mô hình SMORE gồm 3 thành phần chính:

### 2.1. Dung hợp và Lọc nhiễu trên Miền Tần số (Spectrum Modality Fusion)

Thay vì chật vật tìm cách gọt giũa nhiễu ở không gian gốc, SMORE làm như sau:

- **Biến đổi Fourier (Fast Fourier Transform - FFT):** Chuyển đổi các Vector đặc trưng của Hình ảnh và Văn bản sang Miền Tần số. Ở miền này, thông tin cốt lõi (bản chất sản phẩm) thường nằm ở dải tần số thấp, còn các nhiễu rác (như chữ thừa, độ mờ ảnh, nét đứt gãy) sẽ văng ra dải tần số cao.
- **Bộ lọc Động (Dynamic Filter):** Tại miền tần số, SMORE thiết kế một bộ lọc có khả năng tự động học cách "bóp nghẹt" (attenuate) các tần số cao chứa nhiễu, và chỉ cho phép các dải tần số thấp chứa thông tin hữu ích đi qua.
- **Dung hợp (Fusion) bằng Tích chập (Point-wise product):** Việc nhân các dải tần số của Hình ảnh và Văn bản với nhau ở miền Tần số tương đương với phép Tích chập vòng (Circular Convolution) ở không gian gốc. Cách này giúp trộn 2 phương thức cực kỳ mượt mà, giữ được mối tương quan cốt lõi mà không làm khuếch đại nhiễu rác.
- **Đảo ngược (Inverse FFT - IDFT):** Cuối cùng, tín hiệu "sạch" và đã "trộn" được biến đổi Fourier ngược (IDFT) để trở về làm Vector đặc trưng thông thường.

### 2.2. Học Đồ thị Đa phương thức (Multi-modal Graph Learning - MMGL)

Giống như các mô hình GCN mạnh khác, sau khi có được các Vector siêu sạch từ Bước 2.1, SMORE tiếp tục đưa chúng lên 2 loại Đồ thị:

- **Đồ thị Item-Item (Góc nhìn Đa phương thức):** Dùng độ tương đồng Cosine để nối các món đồ giống nhau.
- **Đồ thị User-Item (Góc nhìn Hành vi):** Lan truyền thông tin mua sắm trên đồ thị nhị phân để bắt tín hiệu cộng tác (Collaborative signals).

### 2.3. Cụm Sở thích Nhận thức Phương thức (Modality-Aware Preference Module)

- Nhận ra rằng người dùng có những sở thích rất phức tạp (có người chỉ tin vào hình ảnh, có người lại thích đọc text). SMORE dùng thêm cơ chế Attention để tự động cân bằng giữa "Sở thích Hình ảnh", "Sở thích Văn bản" và "Sở thích Tổng hợp" của từng người dùng, sau đó kết hợp với Contrastive Learning (InfoNCE) để hoàn thiện mô hình.

---

## 3. Thiết lập Thí nghiệm & Kết quả (Experiments & Results)

- **Datasets:** Ba bộ dữ liệu Amazon siêu thưa (Baby, Sports, Clothing).
- **Đặc trưng đa phương thức:** Dùng chuẩn VGG16 (4096-dim) cho ảnh và Sentence-Transformers (384-dim) cho chữ.
- **Baselines so sánh:** BPR, LightGCN, VBPR, MMGCN, GRCN, SLMRec, BM3, **MGCN** (SOTA bài số 3), và **FREEDOM** (SOTA bài số 4).
- **Kết quả:**
  - **SMORE thiết lập SOTA mới:** Nó đánh bại toàn bộ các mô hình (bao gồm cả MGCN và FREEDOM) trên tất cả các tập dữ liệu, đạt ý nghĩa thống kê cao ($p < 0.01$).
  - Đóng góp lớn nhất đến từ module Tần số (MMGL), việc gỡ bỏ module này khiến hiệu năng mô hình rơi tự do, chứng minh sức mạnh của miền Tần số trong việc lọc nhiễu.
  - Phân tích độ phức tạp cũng cho thấy việc dùng Fast Fourier Transform (FFT) giúp giảm độ phức tạp thuật toán xuống mức Logarit ($O(N \log N)$), thay vì $O(N^2)$ như các mô hình Attention cũ, giúp chạy cực nhanh.

---

## 4. Ví dụ Trực quan (Dễ hình dung)

Hãy tưởng tượng đặc trưng của mỗi bức ảnh hay đoạn văn bản giống như một **bản thu âm bài hát**:

- Khi ca sĩ hát, sẽ có tiếng hát (Thông tin cốt lõi).
- Nhưng bản thu bị lẫn tiếng gió rít, tiếng xe cộ ngoài đường, tiếng ho của khán giả (Nhiễu - Noise).

**Cách các mô hình cũ (MGCN, FREEDOM) làm:**

- Họ cố gắng dùng "Kéo" để cắt đi những đoạn âm thanh bị xước, hoặc dùng Hành vi (ID) để làm màng lọc. Nhưng vì tiếng hát và tiếng ồn trộn lẫn vào nhau theo thời gian, họ cắt đi tiếng ồn thì cũng cắt lẹm luôn cả tiếng hát.
- Sau đó họ đem trộn bản thu của Hình ảnh và bản thu của Văn bản lại. Kết quả là tiếng ồn của cả 2 bản thu dội lên nhau, trở thành một mớ âm thanh chói tai (Amplification of cross-modality noise).

**Cách SMORE làm:**

- Đưa âm thanh qua một chiếc **Amply / Equalizer (Biến đổi Fourier - FFT)**.
- Chiếc Amply này chia bản thu âm ra thành các cột sóng (Tần số). Nó nhận ra: "À, tiếng hát của ca sĩ thì trầm ấm và ổn định (Low frequency), còn tiếng rít của gió và tiếng xe cộ thì rất chói tai (High frequency)".
- **Bộ lọc động (Dynamic Filter):** SMORE kéo các cần gạt của dải tần số cao xuống mức 0 (dập tắt tiếng ồn), và giữ nguyên các cần gạt của dải tần số thấp (giữ lại tiếng hát).
- Cuối cùng, nó gộp bản thu "đã lọc nhiễu" của Hình ảnh và Văn bản lại rồi phát ra loa (Inverse FFT).
  $\rightarrow$ Kết quả: Shopee có một bộ hồ sơ sản phẩm cực kỳ "Trong trẻo", chỉ toàn thông tin hữu ích để gợi ý cho khách hàng mà không bị rác dữ liệu làm phiền!
