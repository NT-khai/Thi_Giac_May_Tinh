# Hệ thống Giám sát Xâm nhập Vùng Cấm — Pipeline Thị giác Máy tính

**Trường Đại học Giao thông Vận tải TP. Hồ Chí Minh**
Học phần: Xử lý hình ảnh và thị giác máy tính (Mã HP: 121036)
Giảng viên hướng dẫn: Võ Thượng Anh
Nhóm thực hiện: Nhóm A

---

## 1. Giới thiệu Đề tài

Dự án tập trung xây dựng một hệ thống giám sát an ninh thời gian thực giúp phát hiện người xâm nhập vào vùng quan tâm (ROI) xác định qua webcam. Hệ thống không chỉ là một "hộp đen" nhận diện mà triển khai một pipeline minh bạch, kết hợp các kỹ thuật xử lý ảnh truyền thống làm tiền đề cho mô hình học sâu MediaPipe Pose.

### Pipeline 4 giai đoạn xử lý

| Giai đoạn  | Chương | Kỹ thuật                                                                |
| ---------- | ------ | ----------------------------------------------------------------------- |
| Tiền xử lý | Ch.2   | Toán tử điểm, Lọc Gaussian, Cắt vùng ROI                                |
| Đặc trưng  | Ch.3   | Canny Edge Detection và Hough Lines                                     |
| Phân đoạn  | Ch.4   | Trừ nền MOG2, Phép toán hình thái học và Contours                       |
| Nhận dạng  | Ch.5   | Trích xuất và vẽ khung xương người hình cây với 33 điểm mốc (landmarks) |

---

## 2. Hướng dẫn Cài đặt

### Bước 1: Clone Project và chuẩn bị môi trường

```bash
# Clone dự án từ repository
git clone https://github.com/NT-khai/Thi_Giac_May_Tinh.git
cd Thi_Giac_May_Tinh

# Tạo môi trường ảo (khuyến nghị)
python -m venv venv
venv/bin/activate

# Hũy chạy máy ảo
deactivateg
```

### Bước 2: Cài đặt thư viện

Dự án sử dụng các thư viện CV tiêu chuẩn:

```bash
pip install opencv-python numpy mediapipe matplotlib scikit-image
```

### Bước 3: Tải Model nhận dạng

Trước khi khởi động lần đầu, hãy chạy script sau để tải model `pose_landmarker_lite.task` từ Google:

```bash
python download_model.py
```

---

## 3. Hướng dẫn Sử dụng

### 3.1. Chạy giám sát thời gian thực

Chạy pipeline chính để bắt đầu giám sát qua webcam:

```bash
python main.py --camera 0 --roi 160 80 480 360 --gaussian-kernel 5 --confidence 0.5

# Chạy mặc định
python main.py
```

**Giai đoạn Học nền (Calibration):** Hệ thống sẽ đếm 90 khung hình đầu tiên (~3 giây) để thuật toán MOG2 học nền tĩnh. Lưu ý: đứng yên và không đi vào vùng giám sát trong giai đoạn này.

**Điều khiển và Tương tác:**

| Phím      | Chức năng                                                                                       |
| --------- | ----------------------------------------------------------------------------------------------- |
| `q`       | Thoát chương trình                                                                              |
| `s`       | Lưu 4 ảnh minh họa trung gian (Lọc nhiễu, Canny, Phân đoạn, Kết quả cuối) vào thư mục `output/` |
| `r`       | Thực hiện học lại nền (Reset Calibration)                                                       |
| `1` – `4` | Chọn tham số để chỉnh (1: Gaussian, 2: Canny Low, 3: Canny High, 4: Confidence)                 |
| `+` / `-` | Tăng hoặc giảm giá trị tham số đang chọn trực tiếp                                              |

### 3.2. Công cụ Khảo sát tham số (Parameter Sweep)

Sử dụng công cụ này để tạo bảng so sánh định lượng phục vụ báo cáo thực nghiệm:

```bash
# Khảo sát 3 mức ngưỡng Canny (Ch.3)
python parameter_sweep.py --type canny --values 30 50 80

# Khảo sát 3 mức độ tin cậy MediaPipe (Ch.5)
python parameter_sweep.py --type confidence --values 0.3 0.5 0.7

# Chạy mặc định
python parameter_sweep.py
```

Khi chạy, nhấn `SPACE` để chụp khung hình chuẩn từ webcam làm dữ liệu thực nghiệm. Kết quả sẽ được lưu tại thư mục `sweep_results/`.

---

## 4. Kết quả và Giải thích hiển thị

Hệ thống hiển thị 4 cửa sổ song song để minh chứng quá trình biến đổi của pipeline:

1. **Ảnh sau lọc nhiễu (Ch.2):** Kết quả của toán tử điểm và lọc Gaussian giúp mịn ảnh.
2. **Canny + Hough (Ch.3):** Biên Canny (xanh lá) và các đoạn thẳng Hough Lines (đỏ).
3. **Mặt nạ phân đoạn (Ch.4):** Vùng chuyển động màu trắng và contour bao quanh màu xanh.
4. **Kết quả cuối (Ch.5):** Khung xương MediaPipe màu cam/vàng. Khung ROI sẽ chuyển từ **Xanh (An toàn)** sang **Đỏ (Cảnh báo)** nếu có ≥ 3 điểm mốc xâm nhập.

---

## 5. Cấu trúc Project

```
.
├── main.py              # Pipeline giám sát và logic cảnh báo chính
├── parameter_sweep.py   # Script khảo sát tham số định lượng
├── download_model.py    # Script quản lý và tải model Pose Landmarker
├── output/               # Thư mục chứa các ảnh snapshot lưu trong quá trình chạy
└── sweep_results/        # Thư mục chứa kết quả so sánh tham số
```

---

_Báo cáo và mã nguồn được thực hiện bởi nhóm sinh viên lớp Xử lý hình ảnh và thị giác máy tính - UTH._
