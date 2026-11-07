# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **SEED:** 6011.
- **Nguồn số liệu:** mình dùng notebook và JSON đính kèm để tổng hợp kết quả.
- **Phạm vi thực hiện:** mình đã tổng hợp các thí nghiệm đã chạy và viết code thử thách JSON cùng phản ánh cuối bài. Notebook còn lỗi API ở thí nghiệm 4.A; các ô stress test, kiểm tra tính tay, thử thách và báo cáo cuối chưa có output. JSON hiện tại là bản xuất trước các phần này.

## 1. Bằng chứng

| Tệp | Trạng thái |
| --- | --- |
| [day01_lab.ipynb](day01_lab.ipynb) | Mình đã lưu output các thí nghiệm chính và điền ô code thử thách, phản ánh, AI log |
| [lab1_report.json](lab1_report.json) | Có SEED, 10 dự đoán, 10 kiểm tra đạt và kết quả 7 phương án; chưa có kiểm tra tính tay hoặc thử thách |
| [hand_calc.jpg](hand_calc.jpg) | Có ảnh tính tay đính kèm |

## 2. Kết quả chính

| Phương án | F1 test | F1 SUB 40 | p95 (ms/tin) | USD/1.000 tin |
| --- | ---: | ---: | ---: | ---: |
| BoW + LogReg | 0.8914 | 0.8235 | 0.1588 | 0 |
| BoW + MLP | 0.9290 | 0.9474 | 2.4542 | 0 |
| MiniLM + kNN | 0.9005 | 0.9474 | 64.2349 | 0 |
| MiniLM + LogReg | 0.9641 | 0.9744 | 38.6474 | 0 |
| MiniLM + MLP head | 0.9794 | 0.9744 | 36.4371 | 0 |
| LLM nhỏ zero-shot | — | 0.9231 | 987.5298 | 0.025500 |
| LLM nhỏ few-shot | — | 0.9500 | 853.1963 | 0.057583 |

LLM chỉ được đánh giá trên SUB 40 tin, không có kết quả toàn bộ test. Chi phí bằng 0 ở các mô hình cục bộ là chi phí API; chưa bao gồm hạ tầng. Chưa có kết quả model LLM lớn để so sánh.

## 3. Dự đoán và thực tế

Ba điểm khác biệt có bằng chứng trong kết quả đã lưu:

1. **`ml_f1`:** dự đoán F1 khoảng 0.85; thực tế 0.8914, cao hơn 0.0414. Nhận định precision cao hơn recall đúng: precision = 1.0, recall = 0.8041. BoW nhận diện tốt nhiều từ đặc trưng nhưng vẫn bỏ sót spam viết như thông báo thông thường.
2. **`dl_relu`:** dự đoán bỏ ReLU làm F1 giảm; output thực tế vẫn là 0.929, bằng kết quả MLP khi làm tròn ba chữ số. Hai lớp tuyến tính không có ReLU vẫn biểu diễn một phép biến đổi tuyến tính; kết quả cho thấy phi tuyến không tạo chênh lệch rõ trong lần chạy này.
3. **`tradeoff`:** dự đoán BoW + LogReg và MiniLM + MLP head thuộc Pareto; output còn có BoW + MLP. Mô hình này có p95 2.4542 ms, F1 SUB 0.9474, tạo thêm một lựa chọn trung gian giữa tốc độ và chất lượng.

## 4. Phân tích 50 case khó

Mình chưa có output của stress test trong notebook đã lưu, nên chưa kết luận mô hình tụt nhiều nhất hoặc prompt phòng thủ cải thiện bao nhiêu. Mình phân biệt dự đoán `hard_eval` với kết quả đo thực tế.

Hai ví dụ có kết quả thực tế ở thí nghiệm 3.B (tách biệt với bộ 50 case):

- “Exclusive: get a brand new phone at no cost…”: nhãn spam, BoW đoán ham, Encoder + LogReg đoán spam. Cách nói “at no cost” có thể né tín hiệu từ khóa mà BoW đã học.
- “Last chance! Your account rewards expire tonight…”: nhãn spam, BoW đoán ham, Encoder + LogReg đoán spam. Ngữ nghĩa phần thưởng và hạn chót giúp biểu diễn embedding nhận diện nội dung đáng ngờ.

Trong sáu tin của thí nghiệm này, BoW đúng 4/6, Encoder + LogReg đúng 6/6. Mình chưa dùng kết quả sáu tin này để kết luận cho bộ 50 case.

## 5. Thử thách cá nhân

SEED 6011 cho chỉ số `6011 % 6 = 5`, tương ứng **đề #6**: sửa prompt trả JSON có `label` và `reason`, viết bộ parse JSON, chạy 10 tin SUB và so sánh token/chi phí với bản một từ.

Mình đã viết prompt JSON, dùng `json.loads` để đọc và kiểm tra `label`, `reason`, rồi so sánh với nhãn một từ trên cùng 10 tin SUB. Code báo F1, lỗi định dạng, token và chi phí ước tính; kết quả được lưu vào `_LOG["challenge_result"]`. Parser đã được kiểm tra, nhưng mình chưa có số đo API thực tế để kết luận chênh lệch. Mình chọn nhãn một từ khi chỉ cần phân loại, JSON khi cần lý do để đọc lỗi. Token được ước tính bằng số từ chia 0,75, chưa tính token suy luận. Phần code có AI hỗ trợ như mình ghi trong AI log.

## 6. Phương án cho sản phẩm thật

Dựa vào số đo của mình trong notebook, mình chọn **MiniLM + MLP head**: F1 test 0.9794, F1 SUB 0.9744, p95 36.4371 ms, đạt yêu cầu p95 < 50 ms/tin. BoW + LogReg, BoW + MLP và MiniLM + LogReg cũng đạt ngưỡng độ trễ; kNN và hai phương án LLM vượt ngưỡng.

MiniLM + MLP head đo được khoảng 32.685 tin/giây khi suy luận cả lô, thấp hơn tải trung bình 10 triệu SMS/ngày (115.74 tin/giây). Ước lượng tuyến tính cần ít nhất 4 đơn vị xử lý tương đương; đây chưa phải kết quả kiểm thử tải. Cần đo thêm hàng đợi, tải đỉnh và độ trễ khi triển khai song song để xác nhận SLA.

Nếu gửi toàn bộ 10 triệu tin/ngày qua LLM, ngoại suy đơn giá đã đo cho 30 ngày: zero-shot khoảng **7.650 USD/tháng**, few-shot khoảng **17.275 USD/tháng**, chưa gồm hạ tầng. Có thể dùng BoW cho tin chắc chắn và chuyển tin lưỡng lự sang embedding; LLM chỉ xử lý một phần cần phân tích thêm. Hệ thống lai cần được đánh giá lại về chất lượng và độ trễ.

## 7. AI log và điều mình đã học

| Phần | Mình hỏi AI gì | AI hỗ trợ gì | Kiểm chứng |
| --- | --- | --- | --- |
| Chuẩn bị bài nộp | Hoàn thiện report/individual theo SUBMISSION.md | Đối chiếu notebook/JSON và tổng hợp báo cáo | Số liệu được đối chiếu với output đã lưu |
| Hoàn thiện các ô cuối Phần 5 | Điền phần còn thiếu và sửa giọng văn báo cáo | Viết code thử thách JSON, kiểm tra parser và soạn phản ánh | Parser được kiểm tra độc lập; chưa có phép đo API mới |
| Rà soát bài nộp | Kiểm tra thư mục report theo SUBMISSION.md | Bổ sung trả lời leakage, sửa phép đếm tham số MLP, kiểm tra cấu trúc và tên repository | Giữ nguyên số đo và lịch sử chạy; ghi rõ các ô chưa có output |

Mình có dùng AI hỗ trợ code thử thách mặc dù đề ghi không dùng AI. Mình ghi rõ điều này để phân biệt phần được hỗ trợ với phần tự thực hiện; mình không khai nhận phần code đó là tự làm độc lập.

Mình hiểu rõ hơn sự khác nhau giữa BoW dựa trên từ trong từ điển và embedding biểu diễn ngữ nghĩa. Mình dự đoán bỏ ReLU sẽ làm F1 giảm, nhưng kết quả vẫn khoảng 0,929; độ phức tạp của mô hình không đảm bảo cải thiện trên mọi bộ dữ liệu. Validation loss thấp nhất ở epoch 5 cũng cho mình thấy cần theo dõi overfitting. Khi chọn mô hình, mình xem đồng thời F1, p95, throughput, chi phí và phạm vi tập đánh giá.
