"""
Hệ thống Giám sát Xâm nhập Vùng Cấm — Pipeline Thị giác Máy tính
=================================================================
Pipeline đầy đủ theo Chương 2 → 3 → 4 → 5:

  Ch.2  Toán tử điểm | Gaussian Blur | Biến đổi hình học (ROI)
  Ch.3  Canny Edge Detection | Hough Lines
  Ch.4  Phân đoạn vùng chuyển động (MOG2 + Contours)
  Ch.5  MediaPipe Pose Landmarker — trích xuất & vẽ khung xương

Logic cảnh báo: Nếu tọa độ khung xương nằm trong ROI → khung Xanh → Đỏ.

Giai đoạn khởi động: Camera HỌC NỀN (đếm khung hình) trước khi hiện ROI.
"""

import argparse
import os
import time
from datetime import datetime

import cv2
import numpy as np

from download_model import ensure_pose_model

# MediaPipe Tasks API (>= 0.10.14 — không còn mp.solutions)
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python.core import base_options as base_options_lib
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarksConnections
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_lib


# =============================================================================
# Cấu hình mặc định
# =============================================================================
DEFAULT_ROI = (160, 80, 480, 360)
DEFAULT_GAUSSIAN_KERNEL = 3 # 21 - 5 - 3
DEFAULT_BRIGHTNESS = 10 
DEFAULT_CONTRAST = 1.2
DEFAULT_CANNY_LOW = 80 # 10-30 50-150 80-240
DEFAULT_CANNY_HIGH = 240
DEFAULT_BINARY_THRESHOLD = 127
DEFAULT_MORPH_KERNEL = 5
DEFAULT_MIN_CONFIDENCE = 0.8 # 0.1 - 0.5 - 0.8
DEFAULT_CALIBRATION_FRAMES = 90       # ~3 giây @ 30fps — học nền trước khi hiện ROI
MIN_CONTOUR_AREA = 600
MIN_LANDMARKS_IN_ROI = 3

POSE_CONNECTIONS = PoseLandmarksConnections.POSE_LANDMARKS


def _landmark_score(lm):
    """Lấy điểm tin cậy của landmark (visibility hoặc presence)."""
    if lm.visibility is not None:
        return lm.visibility
    if lm.presence is not None:
        return lm.presence
    return 1.0


def draw_pose_skeleton(frame, all_pose_landmarks, frame_w, frame_h):
    """
    Ch.5 — Vẽ khung xương MediaPipe (giống 87_human_action_reg.png).
    """
    output = frame.copy()
    if not all_pose_landmarks:
        return output

    for pose_landmarks in all_pose_landmarks:
        pts = []
        for lm in pose_landmarks:
            px = int(lm.x * frame_w)
            py = int(lm.y * frame_h)
            pts.append((px, py))

        # Vẽ các đoạn nối
        for conn in POSE_CONNECTIONS:
            i, j = conn.start, conn.end
            if i < len(pts) and j < len(pts):
                cv2.line(output, pts[i], pts[j], (255, 100, 0), 3)

        # Vẽ điểm khớp
        for px, py in pts:
            cv2.circle(output, (px, py), 4, (0, 255, 255), -1)

    return output


class IntrusionPoseDetector:
    """Pipeline phát hiện xâm nhập vùng cấm qua khung xương MediaPipe Pose."""

    def __init__(
        self,
        roi=DEFAULT_ROI,
        gaussian_kernel=DEFAULT_GAUSSIAN_KERNEL,
        brightness=DEFAULT_BRIGHTNESS,
        contrast=DEFAULT_CONTRAST,
        canny_low=DEFAULT_CANNY_LOW,
        canny_high=DEFAULT_CANNY_HIGH,
        binary_threshold=DEFAULT_BINARY_THRESHOLD,
        morph_kernel=DEFAULT_MORPH_KERNEL,
        min_confidence=DEFAULT_MIN_CONFIDENCE,
    ):
        self.roi = roi
        self.gaussian_kernel = gaussian_kernel | 1
        self.brightness = brightness
        self.contrast = contrast
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.binary_threshold = binary_threshold
        self.morph_kernel = morph_kernel | 1
        self.min_confidence = min_confidence

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )

        # Ch.5: MediaPipe Tasks API — Pose Landmarker
        model_path = ensure_pose_model()
        options = PoseLandmarkerOptions(
            base_options=base_options_lib.BaseOptions(model_asset_path=model_path),
            running_mode=running_mode_lib.VisionTaskRunningMode.VIDEO,
            num_poses=5,  # Cho phep nhan dien toi da 5 nguoi cung luc
            min_pose_detection_confidence=min_confidence,
            min_pose_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self.pose_landmarker = PoseLandmarker.create_from_options(options)
        self._video_timestamp_ms = 0

    # =========================================================================
    # CHƯƠNG 2 — Tiền xử lý
    # =========================================================================
    def apply_point_operations(self, frame):
        """Ch.2 — Toán tử điểm: pixel_moi = alpha * pixel_cu + beta."""
        return cv2.convertScaleAbs(frame, alpha=self.contrast, beta=self.brightness)

    def apply_gaussian_blur(self, frame):
        """Ch.2 — Lọc Gaussian khử nhiễu."""
        return cv2.GaussianBlur(
            frame, (self.gaussian_kernel, self.gaussian_kernel), sigmaX=0
        )

    def extract_roi(self, frame):
        """Ch.2 — Biến đổi hình học: cắt vùng ROI."""
        x, y, w, h = self.roi
        return frame[y : y + h, x : x + w].copy()

    def preprocess(self, frame):
        """Ch.2 — Pipeline tiền xử lý."""
        return self.apply_gaussian_blur(self.apply_point_operations(frame))

    # =========================================================================
    # Giai đoạn HỌC NỀN (trước khi hiện ROI)
    # =========================================================================
    def learn_background(self, frame):
        """
        Cho MOG2 học nền tĩnh — chỉ gọi trong giai đoạn calibration.
        Không hiển thị ROI, không chạy pipeline đầy đủ.
        """
        denoised = self.preprocess(frame)
        roi_frame = self.extract_roi(denoised)
        self.bg_subtractor.apply(roi_frame)
        return denoised

    # =========================================================================
    # CHƯƠNG 3 — Phát hiện biên & Đường thẳng
    # =========================================================================
    def detect_canny_edges(self, frame):
        """Ch.3 — Canny Edge Detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, self.canny_low, self.canny_high)

    def detect_hough_lines(self, edges):
        """Ch.3 — Hough Lines từ ảnh biên."""
        return cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180,
            threshold=80, minLineLength=40, maxLineGap=10,
        )

    def visualize_edges_and_lines(self, frame, edges, lines):
        """Ch.3 — Overlay Canny (xanh) + Hough (đỏ)."""
        vis = frame.copy()
        vis[edges > 0] = [0, 255, 0]
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.reshape(-1)
                cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return vis

    # =========================================================================
    # CHƯƠNG 4 — Phân đoạn ảnh
    # =========================================================================
    def segment_motion(self, roi_frame):
        """Ch.4 — MOG2 + ngưỡng nhị phân."""
        fg_mask = self.bg_subtractor.apply(roi_frame)
        _, binary = cv2.threshold(fg_mask, self.binary_threshold, 255, cv2.THRESH_BINARY)
        return binary

    def clean_segmentation_mask(self, binary_mask):
        """Ch.4 — Morphology Opening/Closing."""
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.morph_kernel, self.morph_kernel)
        )
        opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    def extract_contours(self, mask):
        """Ch.4 — Trích xuất contour lớn nhất."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest, max_area = None, 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area, largest = area, cnt
        return largest, max_area, max_area >= MIN_CONTOUR_AREA

    # =========================================================================
    # CHƯƠNG 5 — Nhận dạng (MediaPipe Tasks API)
    # =========================================================================
    def detect_pose(self, frame):
        """Ch.5 — Pose Landmarker (Tasks API)."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        self._video_timestamp_ms += 33  # ~30 fps
        return self.pose_landmarker.detect_for_video(mp_image, self._video_timestamp_ms)

    def get_all_poses(self, pose_result):
        """Lấy danh sách landmark của tất cả các pose."""
        if pose_result and pose_result.pose_landmarks:
            return pose_result.pose_landmarks
        return []

    def count_landmarks_in_roi(self, all_pose_landmarks, frame_w, frame_h):
        """Ch.5 — Đếm điểm khớp tin cậy cao nằm trong ROI."""
        if not all_pose_landmarks:
            return 0, []

        x, y, w, h = self.roi
        in_roi = []
        for pose_landmarks in all_pose_landmarks:
            for lm in pose_landmarks:
                if _landmark_score(lm) < self.min_confidence:
                    continue
                px, py = int(lm.x * frame_w), int(lm.y * frame_h)
                if x <= px <= x + w and y <= py <= y + h:
                    in_roi.append((px, py))
        return len(in_roi), in_roi

    # =========================================================================
    # Pipeline tổng hợp (sau khi học nền xong)
    # =========================================================================
    def process_frame(self, frame):
        """Xử lý đầy đủ Ch.2 → Ch.5."""
        h, w = frame.shape[:2]
        original = frame.copy()

        denoised = self.preprocess(frame)
        roi_frame = self.extract_roi(denoised)

        canny_edges = self.detect_canny_edges(denoised)
        hough_lines = self.detect_hough_lines(canny_edges)
        canny_vis = self.visualize_edges_and_lines(denoised, canny_edges, hough_lines)

        motion_mask = self.segment_motion(roi_frame)
        clean_mask = self.clean_segmentation_mask(motion_mask)
        contour, contour_area, has_motion = self.extract_contours(clean_mask)

        pose_result = self.detect_pose(denoised)
        all_pose_landmarks = self.get_all_poses(pose_result)
        num_in_roi, roi_landmarks = self.count_landmarks_in_roi(all_pose_landmarks, w, h)

        intrusion = num_in_roi >= MIN_LANDMARKS_IN_ROI

        return {
            "original": original,
            "denoised": denoised,
            "canny_vis": canny_vis,
            "segmentation_mask": clean_mask,
            "contour": contour,
            "contour_area": contour_area,
            "has_motion": has_motion,
            "pose_landmarks": all_pose_landmarks,
            "num_landmarks_in_roi": num_in_roi,
            "roi_landmarks": roi_landmarks,
            "intrusion": intrusion,
        }

    def set_confidence(self, new_conf):
        self.min_confidence = new_conf
        self.pose_landmarker.close()
        
        model_path = ensure_pose_model()
        options = PoseLandmarkerOptions(
            base_options=base_options_lib.BaseOptions(model_asset_path=model_path),
            running_mode=running_mode_lib.VisionTaskRunningMode.VIDEO,
            num_poses=5,
            min_pose_detection_confidence=self.min_confidence,
            min_pose_presence_confidence=self.min_confidence,
            min_tracking_confidence=self.min_confidence,
        )
        self.pose_landmarker = PoseLandmarker.create_from_options(options)

    def reset_background(self):
        """Reset MOG2 và timestamp video."""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )
        self._video_timestamp_ms = 0

    def close(self):
        self.pose_landmarker.close()


# =============================================================================
# Hiển thị
# =============================================================================
def draw_calibration_screen(frame, current, total):
    """
    Màn hình HỌC NỀN — không hiển thị ROI.
    Camera đếm khung hình cho MOG2 học background.
    """
    vis = frame.copy()
    h, w = vis.shape[:2]
    progress = current / total

    # Thanh tiến trình
    bar_x, bar_y, bar_w, bar_h = 50, h - 60, w - 100, 30
    cv2.rectangle(vis, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
    cv2.rectangle(
        vis, (bar_x, bar_y),
        (bar_x + int(bar_w * progress), bar_y + bar_h),
        (0, 200, 0), -1,
    )

    cv2.putText(
        vis, "GIAI DOAN HOC NEN — VUI LONG DUNG YEN",
        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
    )
    cv2.putText(
        vis, f"Dang hoc: {current}/{total} khung hinh",
        (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
    )
    cv2.putText(
        vis, "ROI se xuat hien sau khi hoc nen xong",
        (30, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1,
    )
    cv2.putText(
        vis, "KHONG di chuyen vao vung giam sat",
        (30, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 180, 255), 2,
    )
    return vis


def draw_final_result(result, roi, frame_w, frame_h):
    """Kết quả cuối: khung xương + ROI xanh/đỏ + cảnh báo."""
    x, y, rw, rh = roi
    output = draw_pose_skeleton(result["original"], result["pose_landmarks"], frame_w, frame_h)

    roi_color = (0, 0, 255) if result["intrusion"] else (0, 255, 0)
    cv2.rectangle(output, (x, y), (x + rw, y + rh), roi_color, 3)

    if result["intrusion"]:
        status, color = "CANH BAO: NGUOI XAM NHAP VUNG CAM!", (0, 0, 255)
    else:
        status, color = "An toan - Vung cam trong", (0, 255, 0)

    cv2.putText(output, status, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if result["pose_landmarks"]:
        cv2.putText(
            output, f"Diem khop trong ROI: {result['num_landmarks_in_roi']}",
            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )

    for pt in result["roi_landmarks"]:
        cv2.circle(output, pt, 8, (0, 0, 255), 2)

    return output


def make_display_panels(result, final_output, roi):
    """4 cửa sổ trung gian cho báo cáo."""
    x, y, w, h = roi

    panel_denoised = result["denoised"].copy()
    cv2.rectangle(panel_denoised, (x, y), (x + w, y + h), (255, 255, 0), 2)
    cv2.putText(panel_denoised, "1. Sau loc nhieu (Ch.2)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    panel_canny = result["canny_vis"].copy()
    cv2.rectangle(panel_canny, (x, y), (x + w, y + h), (255, 255, 0), 2)
    cv2.putText(panel_canny, "2. Canny + Hough (Ch.3)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    mask_bgr = cv2.cvtColor(result["segmentation_mask"], cv2.COLOR_GRAY2BGR)
    if result["contour"] is not None:
        cv2.drawContours(mask_bgr, [result["contour"]], -1, (0, 255, 0), 2)
    panel_mask = result["original"].copy()
    panel_mask[y : y + h, x : x + w] = mask_bgr
    cv2.putText(panel_mask, "3. Mat na phan doan (Ch.4)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    panel_final = final_output.copy()
    cv2.putText(panel_final, "4. Khung xuong + Canh bao (Ch.5)", (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    return panel_denoised, panel_canny, panel_mask, panel_final


def save_snapshot(panels, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    names = ["01_loc_nhieu", "02_canny", "03_phan_doan", "04_ket_qua"]
    paths = []
    for panel, name in zip(panels, names):
        path = os.path.join(output_dir, f"{timestamp}_{name}.png")
        cv2.imwrite(path, panel)
        paths.append(path)
    return paths


def run_calibration(cap, detector, total_frames):
    """
    Giai đoạn 1: Camera học nền, đếm khung hình, KHÔNG hiện ROI.
    Trả về True nếu hoàn tất, False nếu người dùng nhấn 'q'.
    """
    print(f"\n>>> GIAI DOAN HOC NEN: {total_frames} khung hinh (~{total_frames // 30}s)")
    print(">>> Hay dung yen, KHONG di vao vung giam sat.\n")

    count = 0
    while count < total_frames:
        ret, frame = cap.read()
        if not ret:
            print("Khong doc duoc khung hinh.")
            return False

        detector.learn_background(frame)
        count += 1
        screen = draw_calibration_screen(frame, count, total_frames)
        cv2.imshow("Hoc nen - Calibration", screen)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return False

    # Thông báo hoàn tất (1 giây)
    for _ in range(30):
        ret, frame = cap.read()
        if ret:
            done = frame.copy()
            cv2.putText(done, "HOC NEN HOAN TAT!", (80, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(done, "Bat dau hien thi ROI...", (80, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Hoc nen - Calibration", done)
            cv2.waitKey(33)

    cv2.destroyWindow("Hoc nen - Calibration")
    print(">>> Hoc nen hoan tat! Bat dau giam sat voi ROI.\n")
    return True


def run_webcam(args):
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Khong mo duoc webcam (index={args.camera})")

    roi = tuple(args.roi)
    detector = IntrusionPoseDetector(
        roi=roi,
        gaussian_kernel=args.gaussian_kernel,
        brightness=args.brightness,
        contrast=args.contrast,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        binary_threshold=args.threshold,
        morph_kernel=args.morph_kernel,
        min_confidence=args.confidence,
    )

    print("=" * 65)
    print("  HE THONG GIAM SAT XAM NHAP VUNG CAM")
    print("=" * 65)

    # --- Giai đoạn học nền (không hiện ROI) ---
    if not run_calibration(cap, detector, args.calibration_frames):
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        return

    print("Phim tat: q=Thoat | s=Luu anh | r=Hoc lai nen")
    print("Phim tat: 1=Kernel, 2=CannyLow, 3=CannyHigh, 4=Confidence | +/-=Dieu chinh")
    print(f"ROI: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
    print("=" * 65)

    active_param = 1

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            result = detector.process_frame(frame)
            final = draw_final_result(result, roi, w, h)

            info_text = [
                f"1. Gaussian: {detector.gaussian_kernel}" + (" <--" if active_param == 1 else ""),
                f"2. Canny Low: {detector.canny_low}" + (" <--" if active_param == 2 else ""),
                f"3. Canny High: {detector.canny_high}" + (" <--" if active_param == 3 else ""),
                f"4. Confidence: {detector.min_confidence:.2f}" + (" <--" if active_param == 4 else "")
            ]
            for i, text in enumerate(info_text):
                color = (0, 255, 255) if (i + 1) == active_param else (255, 255, 255)
                cv2.putText(final, text, (10, 150 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            panels = make_display_panels(result, final, roi)

            cv2.imshow("0. Anh goc", frame)
            cv2.imshow("1. Sau loc nhieu (Ch.2)", panels[0])
            cv2.imshow("2. Canny + Hough (Ch.3)", panels[1])
            cv2.imshow("3. Mat na phan doan (Ch.4)", panels[2])
            cv2.imshow("4. Ket qua cuoi (Ch.5)", panels[3])

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("1"): active_param = 1
            elif key == ord("2"): active_param = 2
            elif key == ord("3"): active_param = 3
            elif key == ord("4"): active_param = 4
            elif key == ord("+") or key == ord("="):
                if active_param == 1:
                    detector.gaussian_kernel += 2
                elif active_param == 2:
                    detector.canny_low = min(255, detector.canny_low + 10)
                elif active_param == 3:
                    detector.canny_high = min(255, detector.canny_high + 10)
                elif active_param == 4:
                    new_conf = min(1.0, detector.min_confidence + 0.1)
                    if abs(new_conf - detector.min_confidence) > 0.01:
                        detector.set_confidence(new_conf)
            elif key == ord("-") or key == ord("_"):
                if active_param == 1:
                    detector.gaussian_kernel = max(1, detector.gaussian_kernel - 2)
                elif active_param == 2:
                    detector.canny_low = max(0, detector.canny_low - 10)
                elif active_param == 3:
                    detector.canny_high = max(0, detector.canny_high - 10)
                elif active_param == 4:
                    new_conf = max(0.0, detector.min_confidence - 0.1)
                    if abs(new_conf - detector.min_confidence) > 0.01:
                        detector.set_confidence(new_conf)
            elif key == ord("s"):
                paths = save_snapshot(panels, args.output_dir)
                print("Da luu:")
                for p in paths:
                    print(f"  - {p}")
            elif key == ord("r"):
                detector.reset_background()
                if not run_calibration(cap, detector, args.calibration_frames):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Giam sat xam nhap — Pipeline Ch.2-5 + MediaPipe Tasks API"
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--roi", type=int, nargs=4, default=list(DEFAULT_ROI),
                        metavar=("X", "Y", "W", "H"))
    parser.add_argument("--gaussian-kernel", type=int, default=DEFAULT_GAUSSIAN_KERNEL)
    parser.add_argument("--brightness", type=int, default=DEFAULT_BRIGHTNESS)
    parser.add_argument("--contrast", type=float, default=DEFAULT_CONTRAST)
    parser.add_argument("--canny-low", type=int, default=DEFAULT_CANNY_LOW)
    parser.add_argument("--canny-high", type=int, default=DEFAULT_CANNY_HIGH)
    parser.add_argument("--threshold", type=int, default=DEFAULT_BINARY_THRESHOLD)
    parser.add_argument("--morph-kernel", type=int, default=DEFAULT_MORPH_KERNEL)
    parser.add_argument("--confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument(
        "--calibration-frames", type=int, default=DEFAULT_CALIBRATION_FRAMES,
        help="So khung hinh hoc nen truoc khi hien ROI (mac dinh: 90 ~ 3 giay)",
    )
    parser.add_argument("--output-dir", type=str, default="output")
    return parser.parse_args()


if __name__ == "__main__":
    run_webcam(parse_args())
