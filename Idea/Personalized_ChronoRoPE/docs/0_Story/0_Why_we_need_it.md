Để hiểu được RoPE, ta cần phải biết được tại sao RoPE được sinh ra

Transformer bị "mù" về thứ tự bởi vì attention và feed forward block không có cách tự nhiên nào để phân biệt thứ tự của dữ liệu đầu vào.

### Các hướng giaỉ quyết cũ

Absolute positional embeddings

Vậy làm sao mà các mô hình như BERT và GPT-2 biết được vị trí của các từ ở đâu ?

Đơn gỉan ta chỉ cộng 2 vector, vector embedding của từ đó và vector embedding cho vị trí đó

Token Vector (mang thông tin ngữ nghĩa) + Positional vector (mang thông tin vị trí) -> input ![1782402968049](image/0_Why_we_need_it/1782402968049.png)

Đơn giản nhưng lại qúa ngây thơ -> Đặc tính của ngôn ngữ là tính tương đối nhưng ta lại đi mã hoá vị trí một cách tuyệt đối

![1782403502692](image/0_Why_we_need_it/1782403502692.png)

vector biểu diễn khác nhau cho cùng 1 từ -> Mô hình xem như 2 từ khác nhau hoàn toàn -> Mô hình cần phải học lại ngữ nghĩa cho từng vị trí -> tốn kém

-> Chúng ta cần một cách biểu diễn linh hoạt hơn dựa trên khoảng cách tương đối

![1782403774319](image/0_Why_we_need_it/1782403774319.png)

Mục tiêu là attention score chỉ nên dựa vào content (vector query và key) và vị trí tương đối (m-n) -> đây là lúc mà chúng ta đưa RoPE vào

Làm sao chúng ta có thể đưa thông tin vị trí vào mà không phá hỏng thông tin ngữ nghĩa ban đầu ?  -> Không đi cộng nữa, chúng ta đi xoay, nhưng xoay cái gì và xoay ở đâu ?

![1782404086560](image/0_Why_we_need_it/1782404086560.png)

Chúng ta sẽ xoay 2 vector Q và K ở mọi layer -> trước bước attention
