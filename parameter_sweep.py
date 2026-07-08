"""
Khảo sát tham số (Parameter Sweep) — Phục vụ báo cáo thực nghiệm (Mục 4)
==========================================================================
Mặc định khảo sát 3 giá trị của Gaussian Kernel (Ch.2): 21 - 11 - 5
(có thể đổi sang khảo sát Canny hoặc Confidence qua --type)

Với MỖI giá trị tham số, script xuất ra ĐẦY ĐỦ 5 giai đoạn của pipeline:
  0. Ảnh gốc                      (Ch.2 — trước xử lý)
  1. Ảnh sau lọc nhiễu (Gaussian) (Ch.2)
  2. Canny + Hough                (Ch.3)
  3. Mặt nạ phân đoạn             (Ch.4)
  4. Kết quả cuối (khung xương)   (Ch.5)

Kết quả: 1 ảnh so sánh dạng lưới (hàng = giá trị tham số, cột = 5 giai đoạn)
         + ảnh riêng lẻ từng giai đoạn/tham số + file metrics
         Lưu tại sweep_results/
"""

import argparse
import os
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np

from main import (
    DEFAULT_ROI,
    IntrusionPoseDetector,
    draw_final_result,
)

# 5 giai đoạn hiển thị cho mỗi tham số
STAGE_NAMES = [
    "0. Anh goc",
    "1. Sau loc nhieu (Ch.2)",
    "2. Canny + Hough (Ch.3)",
    "3. Mat na phan doan (Ch.4)",
    "4. Ket qua cuoi (Ch.5)",
]
STAGE_FILE_TAGS = ["00_goc", "01_loc_nhieu", "02_canny_hough", "03_phan_doan", "04_ket_qua"]

# Giá trị mặc định cho từng loại khảo sát
DEFAULT_VALUES = {
    "gaussian": [21, 11, 5],
    "canny": [30, 50, 80],
    "confidence": [0.3, 0.5, 0.8],
}


def capture_frame_interactive(camera, roi):
    """Chụp khung hình từ webcam (nhấn SPACE), dùng dữ liệu thực tế."""
    print("Dang mo webcam... Dung trong / di chuyen vao vung cam.")
    print("Nhan SPACE de chup | ESC de huy.")
    cap = cv2.VideoCapture(camera)
    frame = None
    while cap.isOpened():
        ret, f = cap.read()
        if not ret:
            break
        preview = f.copy()
        x, y, w, h = roi
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Nhan SPACE de chup | ESC de huy", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            raise RuntimeError("Da huy chup anh")
        if key == 32:
            frame = f.copy()
            break
    cap.release()
    cv2.destroyAllWindows()
    return frame


def build_param_overrides(sweep_type, val):
    """Tạo dict tham số truyền vào IntrusionPoseDetector + nhãn hiển thị."""
    if sweep_type == "gaussian":
        val = int(val) | 1  # Gaussian kernel phai le
        return {"gaussian_kernel": val}, f"Gaussian kernel={val}"
    elif sweep_type == "canny":
        low = int(val)
        high = low * 3
        return {"canny_low": low, "canny_high": high}, f"Canny low={low}, high={high}"
    elif sweep_type == "confidence":
        return {"min_confidence": val}, f"Confidence={val}"
    else:
        raise ValueError(f"sweep_type khong hop le: {sweep_type}")


def run_single_experiment(frame, roi, param_overrides, output_dir, run_label):
    """Chạy pipeline với một bộ tham số, trả về 5 panel (goc..ket qua) + metrics."""
    detector = IntrusionPoseDetector(roi=roi, **param_overrides)

    # Cho MOG2 học nền trên chính khung hình khảo sát
    result = None
    for _ in range(40):
        result = detector.process_frame(frame)

    final = draw_final_result(result, roi, frame.shape[1], frame.shape[0])

    x, y, w, h = roi

    # 0. Ảnh gốc
    panel_original = result["original"].copy()
    cv2.rectangle(panel_original, (x, y), (x + w, y + h), (255, 255, 0), 2)

    # 1. Sau lọc nhiễu (Gaussian)
    panel_denoised = result["denoised"].copy()
    cv2.rectangle(panel_denoised, (x, y), (x + w, y + h), (255, 255, 0), 2)

    # 2. Canny + Hough
    panel_canny = result["canny_vis"].copy()
    cv2.rectangle(panel_canny, (x, y), (x + w, y + h), (255, 255, 0), 2)

    # 3. Mặt nạ phân đoạn
    mask_bgr = cv2.cvtColor(result["segmentation_mask"], cv2.COLOR_GRAY2BGR)
    if result["contour"] is not None:
        cv2.drawContours(mask_bgr, [result["contour"]], -1, (0, 255, 0), 2)
    panel_mask = result["original"].copy()
    panel_mask[y : y + h, x : x + w] = mask_bgr

    # 4. Kết quả cuối
    panel_final = final.copy()

    panels = [panel_original, panel_denoised, panel_canny, panel_mask, panel_final]

    detector.close()

    # Lưu từng ảnh riêng lẻ
    sub_dir = os.path.join(output_dir, run_label)
    os.makedirs(sub_dir, exist_ok=True)
    for panel, tag in zip(panels, STAGE_FILE_TAGS):
        cv2.imwrite(os.path.join(sub_dir, f"{tag}.png"), panel)

    metrics = {
        "params": param_overrides,
        "intrusion": result["intrusion"],
        "landmarks_in_roi": result["num_landmarks_in_roi"],
        "contour_area": int(result["contour_area"]),
        "has_motion": result["has_motion"],
        "pose_detected": len(result["pose_landmarks"]) > 0,
    }
    return panels, metrics


def run_parameter_sweep(
    sweep_type="gaussian",
    values=None,
    camera=0,
    roi=None,
    output_dir="sweep_results",
    image_path=None,
):
    """
    Khảo sát tham số và tạo ảnh so sánh dạng lưới:
      hang = gia tri tham so, cot = 5 giai doan pipeline.

    sweep_type:
      'gaussian'   — Gaussian kernel (Ch.2), mac dinh 21 - 11 - 5
      'canny'      — canny_low (canny_high = 3 * canny_low)          (Ch.3)
      'confidence' — nguong tin cay MediaPipe                        (Ch.5)
    """
    if roi is None:
        roi = DEFAULT_ROI

    if values is None:
        values = DEFAULT_VALUES[sweep_type]

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if image_path:
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Khong doc duoc anh: {image_path}")
    else:
        frame = capture_frame_interactive(camera, roi)

    all_metrics = []
    all_panels = []  # list cua list-5-panel, 1 phan tu / gia tri

    print(f"\n{'=' * 65}")
    print(f"  PARAMETER SWEEP: {sweep_type}")
    print(f"  Gia tri thu: {values}")
    print(f"{'=' * 65}")

    for val in values:
        overrides, label = build_param_overrides(sweep_type, val)
        run_label = "_".join(f"{k}={v}" for k, v in overrides.items())

        print(f"\n--- {label} ---")
        panels, metrics = run_single_experiment(frame, roi, overrides, output_dir, run_label)
        all_metrics.append(metrics)
        all_panels.append(panels)

        print(f"  Xam nhap: {metrics['intrusion']}")
        print(f"  Diem khop trong ROI: {metrics['landmarks_in_roi']}")
        print(f"  Pose detected: {metrics['pose_detected']}")

    # ------------------------------------------------------------------
    # Ảnh so sánh dạng lưới: hàng = gia tri tham so, cot = 5 giai doan
    # ------------------------------------------------------------------
    n_rows = len(values)
    n_cols = len(STAGE_NAMES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])

    for row, (val, panels, m) in enumerate(zip(values, all_panels, all_metrics)):
        _, row_label = build_param_overrides(sweep_type, val)
        for col, (panel, stage_name) in enumerate(zip(panels, STAGE_NAMES)):
            ax = axes[row][col]
            rgb = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
            ax.imshow(rgb)
            ax.axis("off")
            if row == 0:
                ax.set_title(stage_name, fontsize=11)
            if col == 0:
                ax.set_ylabel(row_label, fontsize=10)
                ax.axis("on")
                ax.set_xticks([])
                ax.set_yticks([])
        # Ghi chu metrics ben canh anh cuoi cung (cot cuoi) cua hang
        info = (
            f"{row_label}\n"
            f"Xam nhap: {m['intrusion']} | Landmarks: {m['landmarks_in_roi']}"
        )
        axes[row][n_cols - 1].set_xlabel(info, fontsize=9)

    plt.suptitle(f"Parameter Sweep — {sweep_type} (5 giai doan pipeline)", fontsize=14)
    plt.tight_layout()

    compare_path = os.path.join(output_dir, f"{timestamp}_compare_{sweep_type}.png")
    plt.savefig(compare_path, dpi=150)
    plt.close()

    # ------------------------------------------------------------------
    # File metrics
    # ------------------------------------------------------------------
    report_path = os.path.join(output_dir, f"{timestamp}_metrics_{sweep_type}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Parameter Sweep: {sweep_type}\n")
        f.write(f"Thoi gian: {timestamp}\n")
        f.write("-" * 55 + "\n")
        for val, m in zip(values, all_metrics):
            f.write(f"\nGia tri = {val}\n")
            for k, v in m.items():
                f.write(f"  {k}: {v}\n")

    print(f"\n{'=' * 65}")
    print("HOAN TAT PARAMETER SWEEP")
    print(f"  Anh so sanh (luoi 5 giai doan) : {compare_path}")
    print(f"  Bao cao                        : {report_path}")
    print(f"  Anh rieng le tung giai doan    : {output_dir}/<tham_so>/")
    print(f"{'=' * 65}")

    return all_metrics, compare_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Khao sat tham so (Gaussian / Canny / Confidence) — xuat 5 giai doan pipeline"
    )
    parser.add_argument(
        "--type", choices=["gaussian", "canny", "confidence"], default="gaussian",
        help="Loai tham so khao sat (mac dinh: gaussian, gia tri 21-11-5)",
    )
    parser.add_argument(
        "--values", type=float, nargs="+", default=None,
        help="3 gia tri tham so (vd: --values 21 11 5)",
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--roi", type=int, nargs=4, default=list(DEFAULT_ROI),
        metavar=("X", "Y", "W", "H"),
    )
    parser.add_argument("--output-dir", type=str, default="sweep_results")
    parser.add_argument("--image", type=str, default=None,
                        help="Anh tinh (khuyen nghi dung webcam thuc te)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Chuyen values ve int neu sweep gaussian/canny
    values = args.values
    if values and args.type in ("gaussian", "canny"):
        values = [int(v) for v in values]

    run_parameter_sweep(
        sweep_type=args.type,
        values=values,
        camera=args.camera,
        roi=tuple(args.roi),
        output_dir=args.output_dir,
        image_path=args.image,
    )