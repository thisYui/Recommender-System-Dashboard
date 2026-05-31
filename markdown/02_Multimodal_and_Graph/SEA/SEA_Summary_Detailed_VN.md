# It is Never Too Late to Mend - Separate Learning for Multimedia Recommendation

**Tên mô hình:** SEA (SEparate leArning for multimedia recommendation)
**Tác giả:** Zhuangzhuang He, Zihan Wang, Yonghui Yang, Haoyue Bai, và Le Wu (Hefei University of Technology & Arizona State University)
**Năm công bố:** 2024
**Lĩnh vực:** Hệ thống Gợi ý Đa phương thức (Multimodal/Multimedia Recommender Systems)

---

## 1. Bối cảnh & Phản biện (The Problem & Motivation)

Hầu hết các mô hình SOTA từ trước đến nay (kể cả MMGCN, LATTICE hay FREEDOM) đều cố gắng **Đồng bộ hóa (Alignment)** mọi thứ: Ép vector Hình ảnh và vector Văn bản vào chung một không gian, sau đó trộn chúng lại với nhau (Fusion) để gợi ý.

Tuy nhiên, bài báo này chỉ ra một điểm yếu chết người: **Việc đồng bộ hóa hoàn toàn (Full Alignment) sẽ phá hủy các đặc tính độc nhất (Unique attributes) của từng phương thức.**

- Ví dụ: Cả Hình ảnh và Văn bản đều miêu tả chiếc áo "màu đỏ". Nhưng Hình ảnh cho bạn thấy *độ bóng của vải (texture)*, còn Văn bản thì cho bạn biết *chất liệu là 100% cotton*. Nếu cố ép chúng giống hệt nhau, những chi tiết "độc quyền" này sẽ biến mất.

### Phản biện "Orthogonal Learning" (Học Trực giao)

Một số mô hình gần đây đã nhận ra điều này và cố gắng trích xuất **Phần độc nhất (Modal-unique)** bằng cách:

1. Dùng phép **Trừ (Subtraction):** (Phần độc nhất = Toàn bộ Vector - Phần chung).
2. Dùng phép **Trực giao (Orthogonal Constraint):** Ép phần độc nhất của Hình ảnh phải vuông góc (trực giao) với phần độc nhất của Văn bản.

**Tác giả đập tan phương pháp này bằng Toán học (Theorem 1 & 2):**

- Trong không gian nhiều chiều (High-dimensional space $\ge 32$ chiều), hai vector ngẫu nhiên **bất kỳ** vốn dĩ đã có xu hướng vuông góc với nhau (Góc $\approx 90^\circ$ hay $\pi/2$).
- Việc ép chúng "trực giao" là vô nghĩa (ineffective) và không có tác dụng thực sự trong việc tách biệt thông tin. Hơn nữa, phép trừ tuyến tính không thể hiện đúng mối quan hệ phi tuyến phức tạp của dữ liệu.

---

## 2. Kiến trúc Mô hình SEA (Methodology)

Để thay thế cho các phương pháp cũ kỹ, tác giả đề xuất mô hình **SEA (Separate Learning)** dựa trên **Lý thuyết Thông tin (Mutual Information - MI)**. Thay vì dùng hình học (trực giao/phép trừ), SEA dùng xác suất phân phối để ép mô hình học đúng.

Quy trình hoạt động:

### 2.1. Tách đôi Biểu diễn (Splitting Modal Representation)

Mỗi vector phương thức (VD: Vector Hình ảnh) được chẻ ra làm 2 phần:

- **Generic Part ($E_g$):** Phần kiến thức chung (Màu sắc, hình dáng chung).
- **Unique Part ($E_q$):** Phần kiến thức độc quyền (Độ nhăn của vải, độ bóng).

### 2.2. Chiến lược 1: Đẩy xa nhau (Distancing) - Để học phần Độc nhất

- **Mục tiêu:** Làm sao để $E_q$ thực sự khác biệt với $E_g$ trong cùng một bức ảnh?
- **Cách làm:** Mô hình hóa bằng việc **Giảm thiểu Giới hạn trên của Thông tin Tương hỗ (Minimizing the Upper Bound of Mutual Information)**.
- **Ý nghĩa:** Nó ép phần Unique không được chứa bất kỳ thông tin nào trùng lặp với phần Generic. Phép toán này phức tạp nhưng đảm bảo sự tách biệt (Distancing) tuyệt đối về mặt ý nghĩa, chứ không phải chỉ ép 2 góc vuông nhau một cách máy móc như mô hình cũ.

### 2.3. Chiến lược 2: Kéo lại gần (Alignment) - Để học phần Chung

- **Mục tiêu:** Làm sao để $E_g$ của Hình ảnh và $E_g$ của Văn bản có chung một ngôn ngữ?
- **Cách làm:** Sử dụng hàm mất mát mới tên là **SoloSimLoss** để **Tối đa hóa Giới hạn dưới của Thông tin Tương hỗ (Maximizing the Lower Bound of MI)**.
- **Ưu điểm:** Khác với Contrastive Learning (như InfoNCE) bắt buộc phải tạo ra các "Mẫu âm" (Negative sampling) rất tốn thời gian, SoloSimLoss không cần Negative Sampling, chạy cực nhanh nhưng vẫn ép được phần Generic của Hình ảnh và Văn bản tiệm cận vào nhau.

### 2.4. Kết hợp và Gợi ý (Fusion & Optimization)

Sau khi có được các "nguyên liệu tinh khiết" ($E_q$ và $E_g$ của từng phương thức), hệ thống nối (concatenate) chúng lại cùng với Vector Hành vi, truyền qua các lớp Đồ thị (GNN) và tính điểm BPR Loss để gợi ý cho người dùng.

---

## 3. Thiết lập Thí nghiệm & Kết quả (Experiments & Results)

- **Datasets:** Sử dụng lại 3 bộ dữ liệu Amazon (Baby, Sports, Clothing).
- **Baselines so sánh (Rất đồ sộ):** BPR, LightGCN, VBPR, MMGCN, DualGNN, LATTICE, MICRO, SLMRec, BM3, MMSSL, **FREEDOM** (SOTA từ bài báo trước), **MGCN** (SOTA bài trước nữa), và LGMRec.
- **Kết quả:**
  - **SEA đánh bại tất cả:** Vượt mặt cả FREEDOM, MGCN và LGMRec trên tất cả các tập dữ liệu. Trên tập Sports, Recall@10 tăng vọt.
  - **Chứng minh độ linh hoạt (Plug-and-play):** Tác giả đã lấy bộ mã nguồn "Tách đôi/Đẩy xa/Kéo gần" của SEA và cấy nó vào trong mô hình FREEDOM. Kết quả: FREEDOM (vốn đã rất mạnh) lại tiếp tục mạnh hơn nữa (+1.44% Recall), chứng minh rằng việc bảo tồn tính "Độc nhất" là điều mà ngay cả FREEDOM cũng đang bỏ sót.

---

## 4. Ví dụ Trực quan (Dễ hình dung)

Hãy tưởng tượng bạn đang kinh doanh một cửa hàng bán **Bánh kem (Item)**:

- **Hình ảnh** chiếc bánh cho khách hàng thấy: *Hình dáng tròn, có dâu tây, lớp kem bóng bẩy.*
- **Văn bản (Mô tả)** cho khách hàng biết: *Bánh vị Vani, ít ngọt, nguyên liệu hữu cơ, làm trong ngày.*

**Cách các mô hình cũ (MMGCN, LATTICE, FREEDOM) làm:**

- Bắt nhân viên đem bức ảnh và tờ giấy mô tả bỏ vào máy xay sinh tố, xay nhuyễn lại (Full Alignment). Kết quả ra một cốc sinh tố gọi là "Bánh kem ngon". Khách hàng uống xong chỉ thấy ngon, nhưng không còn phân biệt được độ bóng của lớp kem hay việc nó ít ngọt nữa. (Mất đi Modal-Unique).

**Cách SEA (Separate Learning) làm:**

- Nhận ra rằng Hình ảnh và Văn bản có **Điểm chung (Generic)**: "Đều nói về Bánh dâu tây".
- Nhận ra rằng Hình ảnh có **Điểm độc nhất (Unique)**: "Lớp kem rất bóng mịn".
- Nhận ra rằng Văn bản có **Điểm độc nhất (Unique)**: "Vị Vani ít ngọt".
- Thay vì xay sinh tố, SEA khéo léo dùng khay đựng riêng: Để "Vị dâu tây" vào giữa, "Độ bóng" sang một bên, "Ít ngọt" sang một bên.
- Nhờ việc bảo tồn các điểm độc nhất này, khi một khách hàng nữ bước vào tìm kiếm một chiếc bánh "Không quá ngọt để ăn kiêng nhưng phải chụp ảnh đẹp", hệ thống SEA ngay lập tức kết nối được đúng nhu cầu của khách với đúng thuộc tính nguyên bản của sản phẩm, đưa ra lời gợi ý hoàn hảo mà các hệ thống khác đã xay nát từ lâu!
