"""
Khảo sát tham số (Parameter Sweep) — Phục vụ báo cáo thực nghiệm (Mục 4)
==========================================================================
Thử nghiệm ít nhất 3 giá trị của:
  - Ngưỡng Canny (canny_low / canny_high)  — Ch.3
  - Ngưỡng tin cậy MediaPipe (confidence)  — Ch.5

Kết quả: ảnh so sánh 3 cột + file metrics lưu tại sweep_results/
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
    make_display_panels,
)


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


def run_single_experiment(frame, roi, param_overrides, output_dir):
    """Chạy pipeline với một bộ tham số, trả về panel kết quả và metrics."""
    detector = IntrusionPoseDetector(roi=roi, **param_overrides)

    # Cho MOG2 học nền
    result = None
    for _ in range(40):
        result = detector.process_frame(frame)

    final = draw_final_result(result, roi, frame.shape[1], frame.shape[0])
    panels = make_display_panels(result, final, roi)
    detector.close()

    param_label = "_".join(f"{k}={v}" for k, v in param_overrides.items())
    sub_dir = os.path.join(output_dir, param_label)
    os.makedirs(sub_dir, exist_ok=True)

    cv2.imwrite(os.path.join(sub_dir, "canny.png"), panels[1])
    cv2.imwrite(os.path.join(sub_dir, "mask.png"), panels[2])
    cv2.imwrite(os.path.join(sub_dir, "final.png"), panels[3])

    metrics = {
        "params": param_overrides,
        "intrusion": result["intrusion"],
        "landmarks_in_roi": result["num_landmarks_in_roi"],
        "contour_area": int(result["contour_area"]),
        "has_motion": result["has_motion"],
        "pose_detected": result["pose_landmarks"] is not None,
    }
    return panels[3], metrics


def run_parameter_sweep(
    sweep_type="canny",
    values=None,
    camera=0,
    roi=None,
    output_dir="sweep_results",
    image_path=None,
):
    """
    Khảo sát tham số và tạo ảnh so sánh.

    sweep_type:
      'canny'      — thử 3 cặp canny_low (canny_high = 3 * canny_low)
      'confidence' — thử 3 ngưỡng tin cậy MediaPipe
    """
    if roi is None:
        roi = DEFAULT_ROI

    if values is None:
        values = [30, 50, 80] if sweep_type == "canny" else [0.3, 0.5, 0.7]

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if image_path:
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Khong doc duoc anh: {image_path}")
    else:
        frame = capture_frame_interactive(camera, roi)

    all_metrics = []
    final_images = []

    print(f"\n{'=' * 65}")
    print(f"  PARAMETER SWEEP: {sweep_type}")
    print(f"  Gia tri thu: {values}")
    print(f"{'=' * 65}")

    for val in values:
        if sweep_type == "canny":
            overrides = {"canny_low": val, "canny_high": val * 3}
            label = f"Canny low={val}, high={val * 3}"
        else:
            overrides = {"min_confidence": val}
            label = f"Confidence={val}"

        print(f"\n--- {label} ---")
        final_panel, metrics = run_single_experiment(frame, roi, overrides, output_dir)
        all_metrics.append(metrics)
        final_images.append(final_panel)

        print(f"  Xam nhap: {metrics['intrusion']}")
        print(f"  Diem khop trong ROI: {metrics['landmarks_in_roi']}")
        print(f"  Pose detected: {metrics['pose_detected']}")

    # Ảnh so sánh 3 cột
    fig, axes = plt.subplots(1, len(values), figsize=(5 * len(values), 5))
    if len(values) == 1:
        axes = [axes]

    for ax, val, img, m in zip(axes, values, final_images, all_metrics):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        title = (
            f"{sweep_type}={val}\n"
            f"Intrusion: {m['intrusion']} | "
            f"Landmarks: {m['landmarks_in_roi']}"
        )
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.suptitle(f"Parameter Sweep — {sweep_type}", fontsize=13)
    plt.tight_layout()

    compare_path = os.path.join(output_dir, f"{timestamp}_compare_{sweep_type}.png")
    plt.savefig(compare_path, dpi=150)
    plt.close()

    # File metrics
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
    print(f"  Anh so sanh : {compare_path}")
    print(f"  Bao cao     : {report_path}")
    print(f"{'=' * 65}")

    return all_metrics, compare_path


def parse_args():
    parser = argparse.ArgumentParser(description="Khao sat tham so Canny / Confidence")
    parser.add_argument(
        "--type", choices=["canny", "confidence"], default="canny",
        help="Loai tham so khao sat",
    )
    parser.add_argument(
        "--values", type=float, nargs="+", default=None,
        help="3 gia tri tham so (vd: --values 30 50 80)",
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
    # Chuyển values về int nếu sweep canny
    values = args.values
    if values and args.type == "canny":
        values = [int(v) for v in values]

    run_parameter_sweep(
        sweep_type=args.type,
        values=values,
        camera=args.camera,
        roi=tuple(args.roi),
        output_dir=args.output_dir,
        image_path=args.image,
    )
