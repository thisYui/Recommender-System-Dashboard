# Personalized Temporal Encoding

---

## 1. Tóm tắt

Hướng nghiên cứu về personalized temporal encoding cho sequential recommendation đã không đi đến kết quả như kỳ vọng.

Ý tưởng ban đầu: mở rộng RoTE (SIGIR'26) bằng cách cho mỗi user một frequency spectrum riêng — giả thuyết là người mua hằng ngày và người mua hằng năm nên "cảm nhận" thời gian khác nhau.

**Kết luận: giả thuyết này sai, và  đã chạy kiểm chứng để chắc chắn**

---

## 2. Lý do 1 — Giả thuyết không đúng

### Thiết kế thí nghiệm

Bốn nhánh, chỉ khác nhau ở **nguồn của frequency spectrum**, và **giống hệt nhau về mặt số học tại thời điểm khởi tạo**:

| Nhánh      | Spectrum                                                               | Vai trò               |
| ----------- | ---------------------------------------------------------------------- | ---------------------- |
| `fixed`   | cố định, không học                                                | mốc tham chiếu       |
| `global`  | học chung cho mọi user                                               | **baseline**     |
| `user`    | hypernetwork sinh theo feature của user                               | **giả thuyết** |
| `shuffle` | như`user`, nhưng mỗi user nhận feature của **user khác** | **đối chứng** |

### Kết quả (ML-1M, 5 seeds, full-catalog ranking)

| Nhánh      | N@10                       | R@10                       |
| ----------- | -------------------------- | -------------------------- |
| `fixed`   | **0,1083** ± 0,0017 | **0,2059** ± 0,0018 |
| `global`  | 0,1059 ± 0,0019           | 0,1999 ± 0,0034           |
| `user`    | 0,1059 ± 0,0014           | 0,2003 ± 0,0025           |
| `shuffle` | 0,1062 ± 0,0011           | 0,2007 ± 0,0017           |

**`user` vs `global`: +0,04%, p = 0,93.**

### Tại sao kết quả âm tính này là đáng tin

- Độ phân tán spectrum giữa các user của nhánh `user` là **1,15**, so với 0,0005 ở các nhánh không điều kiện hóa. Nghĩa là mô hình **thực sự** học ra spectrum khác nhau cho từng user — và điều đó không tạo ra khác biệt đáng kể nào về độ chính xác.
- **Nhánh đối chứng xác nhận.** `shuffle` (feature bị hoán đổi giữa các user) cho kết quả tương đương `user`. Phần chênh lệch nhỏ còn lại không đến từ cá nhân hóa.

### Một quan sát phụ

Nhánh `fixed` (spectrum đóng băng) **thắng tất cả các nhánh có học**, khoảng 3% tương đối trên R@10 — gấp nhiều lần nhiễu giữa các seed. Tức là không chỉ cá nhân hóa không giúp ích, mà **việc học spectrum nói chung cũng có thể không giúp ích**.

---

## 3. Lý do 2 — Hướng này đã chật

Từ 10/2025 đến 7/2026 đã có ít nhất năm công trình về learned/adaptive time-frequency spectrum:

| Công trình | Thời điểm | Nội dung trùng                                                                                            |
| ------------ | ------------ | ----------------------------------------------------------------------------------------------------------- |
| TO-RoPE      | 10/2025      | kết hợp index và timestamp trong rotary                                                                  |
| SIREN-RoPE   | 04/2026      | thay lịch tần số cố định bằng mạng học ra góc quay                                                |
| AdaRoPE      | 06/2026      | frequency học được theo từng head, từng chiều                                                        |
| ClockRoPE    | 07/2026      | chứng minh mọi hàm điều biến khả tách dương đều xấp xỉ được bằng random Fourier rotations |
| CARoPE       | —           | frequency phụ thuộc nội dung token                                                                       |

Ngoài ra còn MUFFIN (CIKM'25) đã làm cá nhân hóa frequency theo user — tuy ở miền sequence-index chứ không phải miền thời gian liên tục.

---

## 4. Lý do 3 — Tiền đề ban đầu vốn đã mỏng

Khi đọc kỹ lại RoTE (Mục 2.2, Eq. 3), phân rã Y/M/D của họ là **số năm, số tháng, số ngày tích lũy kể từ 1/1/1970**. Ví dụ 1/1/1971 → (1, 12, 365).

Nghĩa là ba giá trị này **không phải ba khía cạnh thời gian độc lập** ->chúng là **cùng một đại lượng đo bằng ba đơn vị khác nhau**, với tỉ lệ cố định ~1 : 12 : 365.

Tính lại bằng chính hằng số công bố của RoTE (base 10⁶/10⁴/10², trọng số 1,5/1,0/0,5):

- **Thành phần day áp đảo ở cả 32/32 chiều**
- **Year + month chỉ đóng góp tối đa 6,87% vào góc quay tổng**

Tức là mô hình "coarse-to-fine multi-level" trên thực tế gần như chỉ là mô hình một tầng day. Cái mà em định cá nhân hóa vốn đã không có nhiều cấu trúc để cá nhân hóa.
