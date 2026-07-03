# Hệ thống Giám sát Xâm nhập Vùng Cấm — Khung xương MediaPipe

Bài tập lớn **Thị giác máy tính** — Phát hiện người xâm nhập vùng cấm (ROI) qua Webcam, vẽ khung xương giống `87_human_action_reg.png`.

> Pipeline **minh bạch**: các kỹ thuật Ch.2–4 chạy **trước** model MediaPipe Pose (Ch.5), thể hiện hiểu biết bản chất xử lý ảnh — không chỉ "gọi API".

---

## Phát biểu Mục tiêu (dán vào báo cáo)

Xem file [`MUC_TIEU_BAO_CAO.md`](MUC_TIEU_BAO_CAO.md).

---

## Pipeline đầy đủ

| Giai đoạn      | Chương   | Kỹ thuật                                    | Cửa sổ hiển thị      |
| -------------- | -------- | ------------------------------------------- | -------------------- |
| Tiền xử lý     | **Ch.2** | Toán tử điểm (α, β), Gaussian Blur, cắt ROI | (1) Sau lọc nhiễu    |
| Phát hiện biên | **Ch.3** | Canny Edge + Hough Lines                    | (2) Canny + Hough    |
| Phân đoạn      | **Ch.4** | MOG2 + Contours + Morphology                | (3) Mặt nạ phân đoạn |
| Nhận dạng      | **Ch.5** | MediaPipe Pose → khung xương                | (4) Kết quả cuối     |

**Logic cảnh báo:** Nếu ≥ 3 điểm khớp khung xương (landmark) nằm trong ROI → khung **Xanh → Đỏ** + hiển thị cảnh báo.

---

## Cài đặt

```bash
cd Thi_Giac_May_Tinh
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### Thoát env

```bash
deactivate
```

Thư viện: `opencv-python`, `numpy`, `scikit-image`, `matplotlib`, `mediapipe`.

---

## Cách chạy

### Giám sát realtime (4 cửa sổ)

```bash
python main.py
```

**Giai đoạn khởi động:** Camera tự **học nền 90 khung hình (~3 giây)** trước khi hiện ROI. Hãy đứng yên, không bước vào vùng giám sát.

| Phím | Chức năng                                    |
| ---- | -------------------------------------------- |
| `q`  | Thoát                                        |
| `s`  | Lưu 4 ảnh minh họa vào `output/`             |
| `r`  | Học lại nền (chạy lại giai đoạn calibration) |

**Tham số tùy chỉnh:**

```bash
python main.py --roi 160 80 480 360 --canny-low 50 --canny-high 150 --confidence 0.5
```

| Tham số                | Mặc định         | Chương | Mô tả                               |
| ---------------------- | ---------------- | ------ | ----------------------------------- |
| `--roi X Y W H`        | `160 80 480 360` | Ch.2   | Vùng cấm                            |
| `--gaussian-kernel`    | `5`              | Ch.2   | Kernel Gaussian                     |
| `--brightness`         | `10`             | Ch.2   | Độ sáng (β)                         |
| `--contrast`           | `1.2`            | Ch.2   | Tương phản (α)                      |
| `--canny-low`          | `50`             | Ch.3   | Ngưỡng Canny thấp                   |
| `--canny-high`         | `150`            | Ch.3   | Ngưỡng Canny cao                    |
| `--threshold`          | `127`            | Ch.4   | Ngưỡng mặt nạ                       |
| `--confidence`         | `0.5`            | Ch.5   | Tin cậy MediaPipe                   |
| `--calibration-frames` | `90`             | —      | Số khung học nền trước khi hiện ROI |

### Khảo sát tham số (Parameter Sweep)

```bash
# 3 ngưỡng Canny (Ch.3)
python parameter_sweep.py --type canny --values 30 50 80

# 3 ngưỡng tin cậy MediaPipe (Ch.5)
python parameter_sweep.py --type confidence --values 0.3 0.5 0.7
```

Nhấn **SPACE** để chụp khung hình từ webcam (dữ liệu thực tế). Kết quả lưu tại `sweep_results/`.

---

## Cấu trúc project

```
CUOIKI/
├── main.py                 # Pipeline Ch.2-5 + webcam
├── download_model.py       # Tải model Pose Landmarker (lần đầu)
├── parameter_sweep.py      # Khảo sát tham số Canny / Confidence
├── models/                 # Model .task (tự tạo khi chạy lần đầu)
├── requirements.txt
├── README.md
├── MUC_TIEU_BAO_CAO.md     # Phát biểu mục tiêu (dán vào báo cáo)
├── output/                 # Ảnh lưu khi nhấn 's'
└── sweep_results/          # Kết quả parameter sweep
```

---

## Hướng dẫn thực nghiệm

1. **Dữ liệu thực tế:** Dùng webcam và bối cảnh phòng của bạn — không dùng ảnh mẫu mạng.
2. **Học nền:** Chương trình tự đếm 90 khung hình học nền — **đứng yên**, ROI chưa hiện trong giai đoạn này.
3. **ROI:** Chỉnh `--roi` sao cho vùng cấm khớp không gian cần giám sát (cửa, lối đi...).
4. **Khung xương:** Đứng đủ thân trong tầm nhìn camera; MediaPipe cần thấy ≥ nửa cơ thể.
5. **Báo cáo:** Nhấn `s` lưu 4 ảnh trung gian; chạy `parameter_sweep.py` để có bảng so sánh.

---

## Giải thích hiển thị

- **Cửa sổ 1:** Ảnh sau toán tử điểm + Gaussian (Ch.2)
- **Cửa sổ 2:** Biên Canny (xanh lá) + đường Hough (đỏ) — mép cửa, vạch sàn (Ch.3)
- **Cửa sổ 3:** Mặt nạ chuyển động + contour xanh (Ch.4)
- **Cửa sổ 4:** Khung xương MediaPipe (cam/vàng) + ROI xanh/đỏ + cảnh báo (Ch.5)

---

## Xử lý lỗi

| Vấn đề                                 | Cách khắc phục                                     |
| -------------------------------------- | -------------------------------------------------- |
| Không mở webcam                        | `--camera 1`, kiểm tra quyền camera                |
| `mediapipe has no attribute solutions` | Đã sửa — dùng Tasks API; chạy lại `python main.py` |
| Không thấy khung xương                 | Đứng xa hơn, đủ ánh sáng, hạ `--confidence 0.3`    |
| Lỗi tải model                          | Chạy `python download_model.py` thủ công           |
| Canny quá nhiễu                        | Tăng `--canny-low`, tăng `--gaussian-kernel`       |
| Cảnh báo sai                           | Tăng `--confidence`, chỉnh ROI nhỏ hơn             |

---

## Tác giả

Bài tập lớn Đại học — Học phần Thị giác máy tính.
