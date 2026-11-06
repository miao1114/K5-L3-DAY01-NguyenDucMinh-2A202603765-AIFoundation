# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **SEED:** 6011.
- **Nguồn số liệu:** notebook và JSON đính kèm; giữ nguyên kết quả đã lưu.
- **Trạng thái:** đã tổng hợp kết quả hiện có; stress test, thử thách cá nhân và ô báo cáo cuối chưa có output trong notebook. Cần hoàn thành các phần này trước khi xem bài là hoàn chỉnh.

## 1. Bằng chứng

| Tệp | Trạng thái |
| --- | --- |
| [day01_lab.ipynb](day01_lab.ipynb) | Có output các thí nghiệm chính; chưa chạy hết các ô cuối |
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

Ô chạy stress test và ô đọc lỗi chưa có output. Chưa đủ bằng chứng xác định mô hình tụt nhiều nhất, độ giảm F1 theo nhóm hoặc hiệu quả prompt phòng thủ. Không dùng dự đoán `hard_eval` thay cho số đo thực tế.

Hai ví dụ có kết quả thực tế ở thí nghiệm 3.B (tách biệt với bộ 50 case):

- “Exclusive: get a brand new phone at no cost…”: nhãn spam, BoW đoán ham, Encoder + LogReg đoán spam. Cách nói “at no cost” có thể né tín hiệu từ khóa mà BoW đã học.
- “Last chance! Your account rewards expire tonight…”: nhãn spam, BoW đoán ham, Encoder + LogReg đoán spam. Ngữ nghĩa phần thưởng và hạn chót giúp biểu diễn embedding nhận diện nội dung đáng ngờ.

Trong sáu tin của thí nghiệm này, BoW đúng 4/6, Encoder + LogReg đúng 6/6. Cần chạy riêng bộ 50 case để hoàn thành yêu cầu stress test.

## 5. Thử thách cá nhân

SEED 6011 cho chỉ số `6011 % 6 = 5`, tương ứng **đề #6**: sửa prompt trả JSON có `label` và `reason`, viết bộ parse JSON, chạy 10 tin SUB và so sánh token/chi phí với bản một từ.

Ô code thử thách đang trống, ô nhận đề chưa có output. Chưa có kết quả hoặc chi phí để báo cáo. Học viên cần tự thực hiện phần thử thách theo quy định của lab.

## 6. Phương án cho sản phẩm thật

Theo số đo đã lưu và lựa chọn đã viết trong notebook, chọn **MiniLM + MLP head**: F1 test 0.9794, F1 SUB 0.9744, p95 36.4371 ms, đạt yêu cầu p95 < 50 ms/tin. BoW + LogReg, BoW + MLP và MiniLM + LogReg cũng đạt ngưỡng độ trễ; kNN và hai phương án LLM vượt ngưỡng.

MiniLM + MLP head đo được khoảng 32.685 tin/giây khi suy luận cả lô, thấp hơn tải trung bình 10 triệu SMS/ngày (115.74 tin/giây). Ước lượng tuyến tính cần ít nhất 4 đơn vị xử lý tương đương; đây chưa phải kết quả kiểm thử tải. Cần đo thêm hàng đợi, tải đỉnh và độ trễ khi triển khai song song để xác nhận SLA.

Nếu gửi toàn bộ 10 triệu tin/ngày qua LLM, ngoại suy đơn giá đã đo cho 30 ngày: zero-shot khoảng **7.650 USD/tháng**, few-shot khoảng **17.275 USD/tháng**, chưa gồm hạ tầng. Có thể dùng BoW cho tin chắc chắn và chuyển tin lưỡng lự sang embedding; LLM chỉ xử lý một phần cần phân tích thêm. Hệ thống lai cần được đánh giá lại về chất lượng và độ trễ.

## 7. AI log và điều đã học

| Phần | Yêu cầu gửi AI | AI thực hiện | Kiểm chứng / cần làm tiếp |
| --- | --- | --- | --- |
| Chuẩn bị bài nộp | Đẩy toàn bộ lên Git, hoàn thiện report/individual theo SUBMISSION.md | Đọc yêu cầu, đối chiếu notebook/JSON, chuẩn hóa tên file và tổng hợp báo cáo từ bằng chứng hiện có | Đối chiếu số liệu với JSON/output; ghi rõ các phần chưa chạy. Học viên cần đọc lại nội dung trước khi nộp |

Các lần sử dụng AI trước phiên chuẩn bị bài nộp này chưa được ghi trong notebook; học viên cần bổ sung lịch sử thực tế nếu có.

Các nhận xét đã thể hiện trong câu trả lời notebook: BoW phụ thuộc vào từ xuất hiện trong từ điển, còn embedding có thể nhận ra nội dung tương tự dù đổi cách dùng từ. MLP có thể overfit: validation loss thấp nhất ở epoch 5 trong lần chạy này. Khi chọn mô hình cần xét F1 cùng độ trễ, tài nguyên và phạm vi tập đánh giá.
