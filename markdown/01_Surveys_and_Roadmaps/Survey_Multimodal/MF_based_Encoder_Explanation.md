# Giải thích chi tiết: Bộ mã hóa dựa trên Phân tích ma trận (MF-based Encoder)

Dựa trên nội dung bài khảo sát về Hệ thống gợi ý Đa phương thức (Multimodal Recommender System - MRS), phần **Bộ mã hóa dựa trên Phân tích ma trận (MF-based Encoder)** được diễn giải chi tiết các khái niệm và công thức toán học dưới chuẩn LaTeX như sau.

## 1. Ý tưởng cốt lõi
Phân tích ma trận (Matrix Factorization - MF) là một kỹ thuật kinh điển trong lĩnh vực Hệ thống gợi ý. Mục tiêu chính là **phân rã ma trận tương tác người dùng - mục** (ví dụ: ma trận điểm đánh giá, lịch sử nhấp chuột) ký hiệu là $R$, thành hai ma trận nhúng ẩn (hidden embeddings) hạng thấp:
*   **$U$**: Ma trận nhúng ẩn đại diện cho đặc trưng của Người dùng.
*   **$I$**: Ma trận nhúng ẩn đại diện cho đặc trưng của Mục (Sản phẩm, video, bài báo...).

---

## 2. Các phép toán cơ bản và Ý nghĩa của chúng

### 2.1. Phép xấp xỉ ma trận (Dự đoán điểm số)

Để dự đoán sự tương tác hoặc mức độ quan tâm của một người dùng đối với một mục chưa từng tương tác, hệ thống sử dụng phép xấp xỉ sau:

$$ R \approx \hat{R} = U I^T \quad (1) $$

*   **Giải thích toán học:** $\hat{R}$ là ma trận dự đoán. $I^T$ là ma trận chuyển vị của $I$. Phép nhân $U I^T$ bản chất là việc tính **tích vô hướng (dot product)** giữa vector đặc trưng ẩn của một người dùng cụ thể và vector đặc trưng ẩn của một mục cụ thể.
*   **Ví dụ cụ thể:** 
    Giả sử các yếu tố ẩn đại diện cho không gian thể loại phim `[Hành động, Hài hước]`. 
    * User A có sở thích thiên về phim Hành động, biểu diễn bởi vector $u = [0.9, 0.1]$.
    * Phim B là phim thuần Hành động, biểu diễn bởi vector $i = [0.8, 0.2]$.
    * Điểm dự đoán (mức độ phù hợp) hệ thống tính được sẽ là: 
      $$ \hat{r} = u \cdot i^T = (0.9 \times 0.8) + (0.1 \times 0.2) = 0.72 + 0.02 = 0.74 $$
    Điểm số $0.74$ cho thấy mức độ phù hợp cao, do đó hệ thống sẽ ưu tiên gợi ý Phim B cho User A.

### 2.2. Hàm mất mát (Loss Function)

Để mô hình "học" được ma trận $U$ và $I$ chuẩn xác nhất, quá trình huấn luyện sử dụng hàm mất mát để tối thiểu hóa sự sai lệch:

$$ \min_{U,I} \|R - \hat{R}\|_F^2 + \lambda(\|E\|_F^2) \quad (2) $$

*   **Giải thích toán học:**
    *   $\|R - \hat{R}\|_F^2$: Là bình phương của **chuẩn Frobenius (Frobenius norm)**. Chuẩn Frobenius của một ma trận là căn bậc hai của tổng bình phương tất cả các phần tử. Phép toán này tính tổng bình phương sai số giữa dữ liệu tương tác thực tế ($R$) và dữ liệu mô hình dự đoán ($\hat{R}$).
    *   $\lambda(\|E\|_F^2)$: Là thành phần **điều chuẩn (regularization)**. $\|E\|_F^2$ đại diện cho tổng bình phương của các tham số nhúng đang được học. $\lambda$ là một siêu tham số kiểm soát phạt. 
*   **Ý nghĩa:** Cụm $\|R - \hat{R}\|_F^2$ giúp mô hình dự đoán khớp với dữ liệu tương tác trong quá khứ. Tuy nhiên, nếu chỉ tối ưu mỗi cụm này, mô hình rất dễ bị "học vẹt" (overfitting). Việc cộng thêm thành phần phạt $\lambda(\|E\|_F^2)$ nhằm ép các trọng số nhúng không mang các giá trị quá lớn hay quá cực đoan, giúp mô hình duy trì khả năng tổng quát hóa tốt cho những dữ liệu chưa nhìn thấy.

---

## 3. Ứng dụng cụ thể trong Hệ thống Gợi ý Đa phương thức (MRS)

Trong Hệ thống gợi ý Đa phương thức (Multimodal Recommender System), một "mục" không chỉ đơn thuần là một ID. Nó chứa rất nhiều thông tin đến từ các **phương thức (modalities)** khác nhau như văn bản (text), hình ảnh (visual), âm thanh (audio), đồ thị tri thức,... Gọi tập hợp các phương thức là $M$.

Tùy thuộc vào thời điểm và cách thức tiến hành "dung hợp" (fusion) các phương thức này, kiến trúc MF-based Encoder được chia thành 2 chiến lược:

### Chiến lược 1: Bộ mã hóa MF hợp nhất (Unified MF-based encoder) - *Dung hợp đặc trưng*

$$ I = \text{Aggr}(I_m) \quad (3) $$
$$ R \approx \hat{R} = U I^T \quad $$

*   **Giải thích toán học:** Hàm $\text{Aggr}(\cdot)$ (Aggregation) thực hiện việc gộp các ma trận biểu diễn từ từng phương thức $I_m$ (với $m \in M$) thành một ma trận đặc trưng mục hợp nhất $I$. Hàm gộp này có thể là phép nối vector (concatenation), cộng element-wise (sum), hoặc thông qua một cơ chế attention phức tạp. Sau khi có được vector $I$ duy nhất, mô hình tiếp tục thực hiện phép nhân vô hướng với $U$.
*   **Ví dụ trong MRS:** Khi gợi ý mua một chiếc áo. Hệ thống nhúng hình ảnh chiếc áo ra một vector ($I_{visual}$) và văn bản bình luận ra một vector ($I_{text}$). Hàm $\text{Aggr}$ nối hai vector này lại thành một vector dài duy nhất đại diện cho "chiếc áo". Sau đó lấy vector này nhân với vector sở thích của bạn ($U$) để tính điểm.

### Chiến lược 2: Nhiều bộ mã hóa MF (Multiple MF-based encoders) - *Dung hợp dự đoán*

$$ R \approx \hat{R} = \text{Aggr}(U I_m^T) \quad (4) $$

*   **Giải thích toán học:** Mô hình tiến hành tính toán một cách tách biệt. Tích vô hướng $U I_m^T$ được thực hiện cho *từng phương thức lẻ* để tạo ra các điểm dự đoán độc lập (điểm hình ảnh riêng, điểm văn bản riêng). Sau đó, hàm $\text{Aggr}(\cdot)$ mới được gọi để tổng hợp các "điểm số" này lại thành quyết định gợi ý cuối cùng $\hat{R}$.
*   **Ví dụ trong MRS:** 
    *   Điểm tương thích về mặt thị giác: $r_{visual} = U \cdot I_{visual}^T = 0.8$ (Người dùng thích kiểu dáng này).
    *   Điểm tương thích về mặt mô tả: $r_{text} = U \cdot I_{text}^T = 0.4$ (Người dùng không ưng ý về chất liệu viết trong văn bản).
    *   Hệ thống dùng hàm $\text{Aggr}$ (ví dụ tính trung bình hoặc dùng ML để quyết định trọng số): $\hat{r} = (0.8 + 0.4)/2 = 0.6$.

### Cập nhật hàm mất mát cho môi trường Đa phương thức

Bởi vì một mục giờ đây có nhiều không gian nhúng tương ứng với các phương thức ($E_m$), hàm mất mát toàn cục được thiết kế lại thành:

$$ \min_{U,I} \|R - \hat{R}\|_F^2 + \lambda \left( \sum_{m \in M} \|E_m\|_F^2 \right) \quad (5) $$

*   **Ý nghĩa:** Cụm điều chuẩn $\sum_{m \in M} \|E_m\|_F^2$ đảm bảo hình phạt tránh học vẹt (regularization) được áp dụng đồng đều lên toàn bộ tập các tham số học của tất cả các phương thức khác nhau. Điều này giữ cho mạng lưới ổn định và không bị phụ thuộc thái quá vào một phương thức đặc thù nào.

---

## Phụ lục 1: Giải thích về Ma trận Nhúng ẩn (Hidden Embeddings)

Khái niệm **"nhúng ẩn" (hidden embeddings / latent embeddings)** là chìa khóa quan trọng để hiểu cách hệ thống MF và các mô hình học sâu hoạt động.

### 1. "Nhúng" (Embedding) là gì?
Trong đời thực, chúng ta nhận diện người dùng qua "ID" (User_01, User_02) và các mục bằng "Tên" (Phim_A, Phim_B). Máy tính không hiểu ID hay Tên, nó cần các con số để tính toán.
"Nhúng" đơn giản là quá trình biến một đối tượng rời rạc (một người dùng, một sản phẩm, một từ ngữ) thành một **vector số học (một mảng gồm nhiều con số)**. Vector này giống như một "tọa độ" trong không gian nhiều chiều. Những đối tượng nào có tính chất giống nhau (ví dụ: 2 người dùng có cùng sở thích, 2 bộ phim cùng thể loại) thì tọa độ (vector nhúng) của chúng sẽ nằm gần nhau trong không gian này.

### 2. Tại sao gọi là "Ẩn" (Hidden/Latent)?
Gọi là "ẩn" vì các con số trong vector này **không do con người định nghĩa trước**.
*   Trong một hệ thống gợi ý truyền thống dựa trên nội dung (Content-based), con người có thể tạo ra các vector rõ ràng (explicit), ví dụ mảng `[Hành động, Hài hước, Tình cảm]`. Một bộ phim hành động sẽ có vector là `[1.0, 0.0, 0.0]`.
*   Tuy nhiên, trong quá trình Phân tích ma trận (Matrix Factorization) hay mạng Neural, hệ thống sẽ tự động học và tự động gán các con số này dựa trên dữ liệu. Chúng ta chỉ định nghĩa *độ dài* của vector (gọi là $d$ - số chiều ẩn, dimension). Chúng ta **không biết chính xác ý nghĩa ngữ nghĩa của từng chiều trong vector đó là gì**. Có thể chiều số 1 đại diện cho "thích phim hành động có yếu tố cháy nổ", chiều số 2 là "ghét diễn viên nam chính", chiều số 3 là một đặc điểm trừu tượng nào đó mà con người không thể gọi tên được. Do đó, các đặc trưng này được gọi là "đặc trưng ẩn" (latent features).

### 3. Ví dụ trực quan
Giả sử chúng ta cần phân rã một ma trận đánh giá $R$ của 1 triệu người dùng và 100 ngàn bộ phim. Ta thiết lập tham số: số chiều ẩn $d = 3$.
Mô hình MF sau khi huấn luyện sẽ tạo ra 2 ma trận nhúng:
*   **Ma trận $U$ (User Embeddings):** Có kích thước là `[1,000,000 x 3]`. Nghĩa là mỗi người dùng trong số 1 triệu người sẽ được biểu diễn bằng một vector có 3 con số. Chẳng hạn, User_01 là `[0.5, -0.2, 0.8]`.
*   **Ma trận $I$ (Item Embeddings):** Có kích thước là `[100,000 x 3]`. Mỗi bộ phim sẽ là một vector 3 con số. Chẳng hạn, Phim_A là `[0.6, -0.1, 0.9]`.

Khi tính điểm dự đoán $\hat{R} = U \times I^T$, hệ thống lấy vector của User_01 nhân vô hướng với vector của Phim_A. Vì hai vector này có các giá trị tương đồng nhau (`0.5` gần `0.6`, `-0.2` gần `-0.1`, `0.8` gần `0.9`), tích vô hướng của chúng sẽ tạo ra một con số lớn, đồng nghĩa với việc mô hình dự đoán người dùng này sẽ đánh giá cao bộ phim đó.

### 4. Tóm lại
**Ma trận nhúng ẩn** đơn giản là các bảng tính chứa các vector số học, đóng vai trò đại diện cho các đặc trưng, tính cách, hay tính chất ẩn sâu của người dùng và sản phẩm. Chúng được máy tính **tự động "học" và điều chỉnh** thông qua dữ liệu lịch sử tương tác nhằm tối ưu hóa khả năng dự đoán tương lai, thay vì được con người phân loại và gán nhãn thủ công.

---

## Phụ lục 2: Chuẩn Frobenius (Frobenius Norm) là gì?

Trong toán học, đặc biệt là trong Đại số tuyến tính, **Chuẩn Frobenius (Frobenius Norm)** là một cách để đo lường "độ lớn" hoặc "độ dài" của một ma trận. Nó có thể được xem như sự mở rộng trực tiếp của khái niệm khoảng cách Euclid (định lý Pytago) từ vector (1 chiều) sang ma trận (2 chiều).

### 1. Công thức toán học
Đối với một ma trận $A$ có kích thước $m \times n$ (gồm $m$ hàng và $n$ cột), chuẩn Frobenius của $A$, được ký hiệu là $\|A\|_F$, được định nghĩa là **căn bậc hai của tổng bình phương tất cả các phần tử trong ma trận đó**.

Công thức:
$$ \|A\|_F = \sqrt{\sum_{i=1}^{m} \sum_{j=1}^{n} |a_{ij}|^2} $$

Trong đó $a_{ij}$ là phần tử nằm ở hàng $i$, cột $j$ của ma trận $A$.

### 2. Ý nghĩa trong Machine Learning và Hệ thống gợi ý
Khi bạn thấy hàm mất mát có chứa cụm $\|R - \hat{R}\|_F^2$, nó mang ý nghĩa rất cụ thể:

*   $R$: Là ma trận chứa các điểm đánh giá thực tế của người dùng.
*   $\hat{R}$: Là ma trận chứa các điểm đánh giá do mô hình dự đoán.
*   $R - \hat{R}$: Là ma trận **Sai số** (từng phần tử là độ chênh lệch giữa thực tế và dự đoán).

Do đó, **bình phương của chuẩn Frobenius** của ma trận sai số sẽ triệt tiêu dấu căn, còn lại:
$$ \|R - \hat{R}\|_F^2 = \sum_{i} \sum_{j} (R_{ij} - \hat{R}_{ij})^2 $$

Đây chính xác là **Tổng bình phương sai số (Sum of Squared Errors - SSE)** cực kì quen thuộc trong thống kê và học máy. 
Mục tiêu của mô hình (thể hiện qua chữ $\min$) là làm cho cái tổng bình phương sai số này càng nhỏ càng tốt, tức là các điểm dự đoán càng bám sát với điểm thực tế càng tốt.

### 3. Tại sao lại dùng vào điều chuẩn (Regularization)
Trong hàm mất mát ở công thức (2), còn có cụm $\lambda(\|E\|_F^2)$. 
*   $\|E\|_F^2$ là tổng bình phương tất cả các trọng số (các con số trong ma trận nhúng $U$ và $I$).
*   Nếu mô hình chỉ cố gắng làm cho $\|R - \hat{R}\|_F^2$ bằng $0$, nó có thể ép các trọng số nhúng lên những giá trị rất khổng lồ và cực đoan (ví dụ $1000, 5000$) để cố tình khớp hoàn toàn với cả những dữ liệu nhiễu trong tập huấn luyện (hiện tượng **overfitting**).
*   Bằng cách cộng thêm $\lambda\|E\|_F^2$ vào hàm mất mát, chúng ta đang "phạt" mô hình nếu nó dùng các trọng số quá lớn. Mô hình buộc phải tìm cách thỏa hiệp: vừa phải dự đoán đúng (giảm sai số), vừa phải giữ cho các giá trị trong ma trận nhúng ở mức độ nhỏ và mượt mà. Kết quả là mô hình sẽ tổng quát hóa tốt hơn khi dự đoán các tương tác trong tương lai.
