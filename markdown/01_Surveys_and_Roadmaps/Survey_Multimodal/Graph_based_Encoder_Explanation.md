# Giải thích chi tiết: Bộ mã hóa dựa trên Đồ thị (Graph-based Encoder)

Dựa trên nội dung bài khảo sát về Hệ thống gợi ý Đa phương thức (Multimodal Recommender System - MRS), phần **Bộ mã hóa dựa trên Đồ thị (Graph-based Encoder)** được diễn giải chi tiết các khái niệm và công thức toán học dưới chuẩn LaTeX như sau.

## 1. Ý tưởng cốt lõi của Graph-based Encoder
Trong khi MF-based Encoder chỉ nhìn vào các cặp tương tác (Ví dụ: User A thích Item B) một cách độc lập, **Bộ mã hóa dựa trên Đồ thị (Graph-based Encoder)** xem xét toàn bộ hệ thống như một mạng lưới khổng lồ (đồ thị).
*   **Đỉnh/Nút (Nodes):** Là Người dùng (User) và Mục (Item).
*   **Cạnh (Edges):** Là sự tương tác giữa họ (ví dụ: click, mua, đánh giá).

Ý tưởng cốt lõi là **lan truyền thông tin (Message Passing)**: Một người dùng sẽ được biểu diễn không chỉ bởi sở thích cá nhân, mà còn bởi đặc điểm của những mục họ đã tương tác, và thậm chí là sở thích của những người dùng khác có chung hành vi.

Mô hình phổ biến nhất được sử dụng là **Mạng Tích chập Đồ thị (Graph Convolutional Network - GCN)**.

---

## 2. Công thức GCN truyền thống

Đối với một đồ thị $G$ và ma trận kề (adjacency matrix) $A$ biểu diễn các liên kết, sự lan truyền thông tin qua một lớp (layer) của GCN được định nghĩa như sau:

$$ E^{(l)} = \sigma \left( \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} E^{(l-1)} W^{(l-1)} \right) \quad (6) $$

**Giải thích các thành phần:**
*   $E^{(l)}$: Ma trận nhúng (biểu diễn) của tất cả các nút tại lớp thứ $l$. ($E^{(0)}$ chính là các vector đặc trưng ban đầu).
*   $\tilde{A} = A + I$: Ma trận kề có cộng thêm ma trận đơn vị $I$ (Self-connection). Điều này có nghĩa là khi cập nhật thông tin, một nút không chỉ nhận thông tin từ hàng xóm mà còn giữ lại thông tin của chính nó.
*   $\tilde{D}$: Ma trận bậc (Degree matrix) của $\tilde{A}$, với phần tử đường chéo $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$. Nó đếm xem một nút có bao nhiêu liên kết.
*   $\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}$: Đây là bước **chuẩn hóa đối xứng (symmetric normalization)**. Nó đảm bảo các nút "siêu sao" (có quá nhiều liên kết) không lấn át các nút ít liên kết. Mức độ ảnh hưởng được chia đều dựa trên số lượng liên kết của cả nút gửi và nút nhận.
*   $W^{(l-1)}$: Ma trận trọng số có thể học được của lớp $l-1$, dùng để biến đổi không gian đặc trưng.
*   $\sigma(\cdot)$: Hàm kích hoạt phi tuyến tính (ví dụ: ReLU, Sigmoid).

---

## 3. LightGCN - Sự tối giản cho Hệ thống Gợi ý

Trong lĩnh vực gợi ý (Recommendation Systems), người ta phát hiện ra rằng công thức GCN truyền thống mang theo những thành phần không cần thiết. Mô hình **LightGCN** đã chứng minh rằng: **Ma trận trọng số biến đổi ($W$) và Hàm kích hoạt phi tuyến tính ($\sigma$) là không hữu ích**, thậm chí còn làm tăng độ khó khi huấn luyện.

Do đó, GCN được đơn giản hóa thành:

$$ E^{(l)} = \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} E^{(l-1)} \quad (7) $$

*   **Ý nghĩa:** LightGCN chỉ thực hiện duy nhất việc **lan truyền và tổng hợp trung bình có trọng số** (dựa trên cấu trúc đồ thị) các nhúng từ các nút láng giềng. Quá trình này giúp đồ thị học được tính chất "collaborative" (cộng tác) một cách mượt mà và hiệu quả hơn rất nhiều.

Sau khi đi qua $L$ lớp (xếp chồng nhiều lớp lan truyền), biểu diễn cuối cùng của mạng là tập hợp (stack) của tất cả các lớp: $ \bar{E} = \text{Stack}_{l \in L}(E^{(l)}) $.
Quá trình này được gọi gọn là: $ \bar{E} = \text{GCN}(U, I) $. Ma trận $\bar{E}$ sau đó có thể tách lại thành nhúng Người dùng $\bar{U}$ và nhúng Mục $\bar{I}$.

---

## 4. Tăng cường với Đồ thị Đồng nhất (Homogeneous Graphs)

Trong MRS, để khai thác sâu hơn mối quan hệ, người ta còn xây dựng thêm các **đồ thị đồng nhất**: Đồ thị chỉ chứa (User-User) hoặc chỉ chứa (Item-Item).
Hai đồ thị này được tạo ra bằng cách giữ lại top-k những láng giềng có độ tương đồng cao nhất (dựa trên đặc trưng đa phương thức hoặc hành vi).

Ma trận kề cho Item-Item ($S^I$) và User-User ($S^U$) được định nghĩa:

$$ S^I_{i,i'} = \begin{cases} 1, & \text{nếu } S^I_{i,i'} \in \text{top-k}(S^I_{i,i'}) \\ 0, & \text{khác} \end{cases} \quad (8) $$

$$ S^U_{u,u'} = \begin{cases} 1, & \text{nếu } S^U_{u,u'} \in \text{top-k}(S^U_{u,u'}) \\ 0, & \text{khác} \end{cases} \quad (9) $$

**Chuẩn hóa ma trận láng giềng:** Tương tự đồ thị User-Item, đồ thị đồng nhất cũng cần chuẩn hóa (ký hiệu là $\hat{S}_U$ và $\hat{S}_I$). Việc chuẩn hóa (chia cho ma trận bậc $D$) giúp "điều chỉnh mức độ ảnh hưởng của mỗi đỉnh dựa trên khả năng kết nối của nó, ngăn chặn các đỉnh có bậc cao chi phối không cân xứng".

Sự lan truyền trên đồ thị đồng nhất được thực hiện:
$$ \hat{U} = (\hat{S}_U)^{L_u} \bar{U} \quad (10) $$
$$ \hat{I} = (\hat{S}_I)^{L_i} \bar{I} \quad (11) $$
(Với $L_u, L_i$ là số lớp lan truyền trên đồ thị User-User và Item-Item).

Cuối cùng, biểu diễn của User và Item được **tăng cường (Enhancement)** bằng cách cộng gộp thông tin từ đồ thị User-Item ($\bar{U}, \bar{I}$) và đồ thị đồng nhất ($\hat{U}, \hat{I}$):
*   $ \tilde{U} = \hat{U} + \bar{U} $
*   $ \tilde{I} = \hat{I} + \bar{I} $

Để đơn giản hóa toàn bộ quá trình phức tạp này (GCN trên đồ thị dị nhất U-I + GCN trên đồ thị đồng nhất), toàn bộ khối đồ thị hợp nhất được ký hiệu là:
$$ \tilde{U}, \tilde{I} = \text{C-GCN}(U, I) $$

---

## 5. Áp dụng Graph-based vào MRS (Hệ thống gợi ý đa phương thức)

Giống như MF-based, Graph-based Encoder trong môi trường đa phương thức ($M$: hình ảnh, văn bản, âm thanh) cũng có hai cách tiếp cận chính xoay quanh hàm dung hợp $\text{Aggr}(\cdot)$:

### Chiến lược 1: Dung hợp sớm (Early Fusion) / Graph-based hợp nhất
Tất cả các đặc trưng đa phương thức ($I_m$) của một Item được tổng hợp (Nối, Cộng...) thành một vector duy nhất trước khi đưa vào hệ thống đồ thị.

$$ \tilde{U}, \tilde{I} = \text{C-GCN}(U, \text{Aggr}(I_m)) \quad (13) $$
$$ R \approx \hat{R} = \tilde{U} \tilde{I}^T $$

### Chiến lược 2: Dung hợp muộn (Late Fusion) / Nhiều bộ Graph-based
Hệ thống xây dựng và chạy lan truyền trên nhiều đồ thị độc lập cho từng phương thức. Nhận được các dự đoán độc lập rồi mới dung hợp lại ở bước cuối cùng để ra điểm số.

$$ \tilde{U}, \tilde{I}_m = \text{C-GCN}(U, I_m) \quad (14) $$
$$ R \approx \hat{R} = \text{Aggr}(\tilde{U} \tilde{I}_m^T) $$

*Hàm mất mát cho các kiến trúc này tương tự như trong MF-based Encoder (Sử dụng chuẩn Frobenius và Regularization).*

---

## 6. Ví dụ thực tế dễ hiểu: Hệ thống Gợi ý TikTok/Reels

Để hiểu rõ cách Graph-based Encoder khác biệt và mạnh mẽ như thế nào, hãy tưởng tượng một mạng lưới xã hội gồm:

**Bối cảnh:**
*   **Người dùng:** An, Bình, Châu.
*   **Video:** 
    *   **Video 1:** Nhảy múa, nhạc xập xình (Dance, EDM).
    *   **Video 2:** Nấu ăn, nhạc nhẹ nhàng (Cooking, Chill).
    *   **Video 3:** Cover điệu nhảy của Video 1, nhạc xập xình (Dance, EDM).

**Lịch sử xem (Đồ thị tương tác):**
*   An ❤️ Video 1 và Video 3.
*   Bình ❤️ Video 1 và Video 2.
*   Châu ❤️ Video 2.

*Câu hỏi đặt ra:* Hệ thống có nên gợi ý Video 1 cho Châu hay Video 3 cho Bình không?

---

### Graph-based Encoder giải quyết bài toán này như thế nào?

Thay vì chỉ so sánh từng cặp Người - Video một cách khô khan (như MF-based), Graph-based Encoder coi tất cả như một **"mạng lưới truyền miệng"**.

#### Bước 1: Thu thập "Hành trang" ban đầu (Khởi tạo đặc trưng Đa phương thức)
Mỗi Video tự gom tất cả thông tin hình ảnh, âm thanh, chữ viết của mình lại thành một "Gói thông tin" (Vector):
*   **Video 1 đóng gói:** [Hình ảnh người nhảy] + [Âm thanh EDM] + [Caption "#xuhuong"].
*   **Video 2 đóng gói:** [Hình ảnh món ăn] + [Âm thanh Acoustic] + [Caption "#nauan"].

#### Bước 2: Quá trình "Truyền miệng" trên mạng lưới (Message Passing - Lan truyền thông điệp)

Đây là sức mạnh cốt lõi của Đồ thị. Thông tin bắt đầu "chảy" qua lại giữa Người dùng và Video thông qua các tương tác lịch sử.

*   **Vòng 1 (Video "lây" sang Người):** 
    An thích Video 1 và 3. Vậy "hồ sơ" của An sẽ tự động được cộng thêm Gói thông tin của cả hai video này. Sau vòng 1, máy tính tự động đúc kết: *"À, An là một người có đặc trưng thích xem nhảy và nghe nhạc EDM"*.
*   **Vòng 2 (Người "lây" ngược lại cho Video):**
    Video 1 được cả An và Bình thích. Nó sẽ hút lấy một phần tính cách từ hồ sơ của An và Bình. Lúc này, Video 1 không chỉ là "một video nhảy múa" đơn thuần, mà nó được máy tính dán thêm một nhãn ẩn: *"Nội dung này rất phù hợp với nhóm người có gu đa dạng như Bình và An"*.

*   **Hiệu ứng dây chuyền (High-order connectivity - Điểm ăn tiền của Đồ thị):**
    Hãy nhìn vào Châu. Châu mới chỉ thích Video 2. 
    Nhưng khoan đã, Bình cũng thích Video 2. Điều này có nghĩa là Châu và Bình có chung "gu" nấu ăn. 
    Mà Bình lại rất thích Video 1. 
    Dù Châu chưa từng xem Video 1, thông qua "cầu nối" là Video 2 và Bình, thông tin về sự hấp dẫn của Video 1 sẽ "truyền miệng" gián tiếp đến Châu. Máy tính sẽ đánh hơi thấy Châu có tiềm năng thích Video 1!

#### Bước 3: Đi đường tắt nhờ Đồ thị Đồng nhất (Homogeneous Graph)

Đôi khi, dựa vào lịch sử xem là chưa đủ (ví dụ Video 3 là video mới ra lò, ít người xem).
Hệ thống sẽ dùng AI quét nội dung (Hình ảnh, Âm thanh) và nhận ra: *"Chà, Video 1 và Video 3 quá giống nhau (đều là nhảy và EDM)"*.
Lập tức, hệ thống tự vẽ một **"đường tắt"** nối trực tiếp giữa Video 1 và Video 3. 
Nhờ đường tắt này, hễ Video 1 có tương tác tốt, nó sẽ tự động san sẻ thông tin tốt đó sang cho Video 3. Từ đó, hệ thống sẽ tự tin gợi ý Video 3 cho Bình (vì Bình thích Video 1).

#### Bước 4: Chốt hạ (Dự đoán)

Sau vài vòng "truyền miệng" và "chia sẻ thông tin" như trên, mỗi Người dùng và mỗi Video giờ đây đều sở hữu một **"Hồ sơ siêu cấp"** (chính là ma trận nhúng $\tilde{U}$ và $\tilde{I}$). Hồ sơ này không chỉ chứa thông tin gốc, mà còn chứa đựng cả sở thích lây lan từ cộng đồng.

Cuối cùng, hệ thống chỉ việc mang Hồ sơ siêu cấp của Châu ra so khớp với Hồ sơ siêu cấp của Video 1. Vì thông tin đã được lan truyền qua lại liên kết họ với nhau, độ khớp nhau (tích vô hướng) sẽ rất cao, và hệ thống sẽ quyết định đưa Video 1 lên màn hình của Châu!

---

## 7. Phụ lục: Giải mã Toán học của GCN (Cho người mới bắt đầu)

Để hiểu rõ công thức toán học trông có vẻ đáng sợ của LightGCN: $E^{(l)} = \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} E^{(l-1)}$, chúng ta hãy "bóc tách" nó ra thành từng phần nhỏ bằng một ví dụ bằng số cực kỳ đơn giản.

Hãy tưởng tượng một vũ trụ thu nhỏ chỉ có 2 Người dùng ($U_1, U_2$) và 2 Video ($V_1, V_2$).
Lịch sử xem:
*   $U_1$ xem $V_1$.
*   $U_2$ xem $V_1$ và $V_2$.

### Bước 1: Ma trận kề $A$ (Adjacency Matrix) là gì?
Đây đơn giản là bảng ghi chép "ai quen ai". Kích thước của bảng này là (Tổng số Nút $\times$ Tổng số Nút) = $4 \times 4$.
Hàng và Cột lần lượt là: $U_1, U_2, V_1, V_2$. Nếu có tương tác thì ghi số `1`, không thì số `0`.

$$
A = \begin{bmatrix}
0 & 0 & 1 & 0 \\
0 & 0 & 1 & 1 \\
1 & 1 & 0 & 0 \\
0 & 1 & 0 & 0 
\end{bmatrix}
$$
*(Ví dụ: Dòng 2 ($U_2$) có số `1` ở cột 3 ($V_1$) và cột 4 ($V_2$))*

### Bước 2: $\tilde{A} = A + I$ (Thêm vòng lặp tự thân)
Ma trận đơn vị $I$ là đường chéo toàn số `1`. Khi lấy $A + I$, nghĩa là ta thêm số `1` vào đường chéo.
**Ý nghĩa thực tế:** Khi tôi đi thu thập thông tin từ bạn bè, tôi không được quên giữ lại thông tin của chính bản thân tôi!

$$
\tilde{A} = \begin{bmatrix}
1 & 0 & 1 & 0 \\
0 & 1 & 1 & 1 \\
1 & 1 & 1 & 0 \\
0 & 1 & 0 & 1 
\end{bmatrix}
$$

### Bước 3: Ma trận Bậc $\tilde{D}$ (Degree Matrix) là gì?
"Bậc" đơn giản là đếm xem mỗi nút có bao nhiêu liên kết (bao gồm cả chính nó).
*   $U_1$ nối với $V_1$ và chính nó => Bậc = 2.
*   $U_2$ nối với $V_1, V_2$ và chính nó => Bậc = 3.
*   $V_1$ nối với $U_1, U_2$ và chính nó => Bậc = 3.
*   $V_2$ nối với $U_2$ và chính nó => Bậc = 2.

Ma trận $\tilde{D}$ chỉ điền các con số này lên đường chéo:
$$
\tilde{D} = \begin{bmatrix}
2 & 0 & 0 & 0 \\
0 & 3 & 0 & 0 \\
0 & 0 & 3 & 0 \\
0 & 0 & 0 & 2 
\end{bmatrix}
$$

### Bước 4: Sự kỳ diệu của phép nhân $\tilde{A} \times E$
Giả sử $E$ là ma trận nhúng hiện tại (ví dụ vector $U_1$ là hàng 1 của $E$).
Phép tính $\tilde{A} \times E$ trong đại số tuyến tính bản chất chính là **phép cộng các vector**.
Khi lấy hàng của $\tilde{A}$ nhân với $E$, con số `1` ở vị trí nào sẽ bốc vector ở vị trí đó đem cộng lại.
Ví dụ cập nhật cho $U_1$ (hàng 1 của $\tilde{A}$ là `[1, 0, 1, 0]`): Nó sẽ lấy vector của chính nó ($U_1$) cộng với vector của ($V_1$).
=> *Đây chính là hành động "truyền miệng", thu gom thông tin từ hàng xóm!*

### Bước 5: Tại sao lại phải kẹp giữa hai cái $\tilde{D}^{-\frac{1}{2}}$ ?
Ký hiệu rườm rà $\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}$ được gọi là **Chuẩn hóa đối xứng (Symmetric Normalization)**.

**Vấn đề:** Nếu một video ($V_{hot}$) được 1 triệu người xem. Khi 1 triệu người này cập nhật thông tin (như Bước 4), họ sẽ cộng vector của họ với vector của $V_{hot}$. Con số sẽ phình to ra khổng lồ và làm sai lệch mọi thứ.

**Giải pháp:** Ta cần phải **chia (giảm bớt)** lượng thông tin lan truyền đi dựa trên độ "hot" của nó.
Toán học quy định $\tilde{D}^{-\frac{1}{2}} = \frac{1}{\sqrt{\text{Bậc}}}$.
Khi kẹp $\tilde{A}$ ở giữa 2 ma trận này, lượng thông tin truyền từ nút $i$ sang nút $j$ sẽ bị chia cho: $\sqrt{\text{Bậc của } i} \times \sqrt{\text{Bậc của } j}$.

**Ý nghĩa siêu dễ hiểu:** 
*   Lượng ảnh hưởng của MỘT cái bắt tay phụ thuộc vào độ nổi tiếng của cả hai người.
*   Nếu $U_1$ (ít xem) xem $V_2$ (ít nổi) => Đây là sở thích cá nhân cực kỳ đặc biệt. Trọng số truyền thông tin sẽ rất **cao** ($\frac{1}{\sqrt{2 \times 2}}$).
*   Nếu $U_2$ (xem tạp nham) xem $V_{hot}$ (triệu người xem) => Ai cũng xem cái này, nó chả nói lên gu cá nhân đặc sắc nào cả. Trọng số truyền tin sẽ bị ép xuống rất **thấp** ($\frac{1}{\sqrt{3 \times 1000000}}$).

### Kết luận công thức LightGCN
$$ E^{(l)} = \underbrace{\left( \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} \right)}_{\text{Bảng phân bổ trọng số}} \times \underbrace{E^{(l-1)}}_{\text{Vector cũ}} $$

**Dịch ra tiếng Việt:** Để lấy vector mới ($E^{(l)}$), hãy nhặt vector của tất cả láng giềng cộng vào vector của mình ($\tilde{A} \times E^{(l-1)}$), nhưng nhớ phải dìm bớt sức ảnh hưởng của những anh láng giềng nào có quá nhiều bạn bè ($\tilde{D}^{-\frac{1}{2}}$). Quá trình này được lặp đi lặp lại qua các lớp (layer).
