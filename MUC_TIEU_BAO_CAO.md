# Phát biểu Mục tiêu (Dán vào đầu Notebook / Báo cáo)

## Vấn đề

Giám sát người xâm nhập vùng cấm bằng Webcam dựa trên khung xương.

## Giả thuyết

> "Chúng tôi dự đoán việc lọc Gaussian (Ch. 2) sẽ giúp giảm 15% tỷ lệ nhận diện sai khung xương do nhiễu môi trường."

## Tiêu chí thành công

Hệ thống đạt độ chính xác (Accuracy) trên **90%** trong điều kiện ánh sáng phòng.

## Phương pháp

Pipeline minh bạch theo chương trình học:

1. **Ch. 2** — Toán tử điểm (tương phản/độ sáng), lọc Gaussian, biến đổi hình học (ROI)
2. **Ch. 3** — Phát hiện biên Canny, phát hiện đường thẳng Hough
3. **Ch. 4** — Phân đoạn vùng chuyển động bằng Contours + trừ nền MOG2
4. **Ch. 5** — MediaPipe Pose trích xuất khung xương; cảnh báo khi khung xương nằm trong ROI

## Dữ liệu thực nghiệm

Sử dụng **webcam thực tế** và **bối cảnh phòng** của sinh viên — không dùng ảnh mẫu trên mạng.
