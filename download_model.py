"""
Tải model Pose Landmarker cho MediaPipe Tasks API (>= 0.10.14).
Chạy tự động lần đầu nếu chưa có file model.
"""

import os
import urllib.request

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "pose_landmarker_lite.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)


def ensure_pose_model():
    """Tải model nếu chưa tồn tại, trả về đường dẫn file .task."""
    if os.path.isfile(MODEL_PATH):
        return MODEL_PATH

    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Dang tai model Pose Landmarker lan dau...")
    print(f"  URL: {MODEL_URL}")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"  Da luu: {MODEL_PATH}")
    return MODEL_PATH
