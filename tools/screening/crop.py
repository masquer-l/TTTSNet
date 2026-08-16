"""Adaptive crop to remove black background outside the endoscopic circular view.

Parameters are estimated per video and stored in the videos table.
Crop is applied on-the-fly when serving images and during export.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


def _safe_int(v: float) -> int:
    return int(round(v))


def detect_circle_in_frame(
    image: np.ndarray,
    threshold: int = 15,
    min_radius_ratio: float = 0.25,
    max_radius_ratio: float = 0.65,
) -> Optional[Tuple[float, float, float]]:
    """Detect the largest bright circle in a frame.

    Returns (center_x, center_y, radius) or None if detection fails.
    """
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    h, w = gray.shape[:2]
    min_dim = min(w, h)

    # Threshold to separate dark background from bright circular region.
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    # Close small holes inside the circle.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Pick the contour whose enclosing circle has the largest radius and is centered.
    best = None
    best_score = 0.0
    for cnt in contours:
        if len(cnt) < 5:
            continue
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if radius < min_dim * min_radius_ratio or radius > min_dim * max_radius_ratio:
            continue
        area = cv2.contourArea(cnt)
        # Favor large, circular contours near the image center.
        dist_from_center = np.hypot(x - w / 2, y - h / 2)
        score = area - dist_from_center * 0.5
        if score > best_score:
            best_score = score
            best = (float(x), float(y), float(radius))
    return best


def estimate_video_crop(
    video_path: str,
    sample_count: int = 12,
    margin: float = 1.0,
) -> Optional[Dict[str, float]]:
    """Estimate a square crop region for a video by sampling frames.

    Returns dict with keys: center_x, center_y, crop_size.
    Returns None if estimation fails.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if total <= 0 or fps <= 0:
        cap.release()
        return None

    # Read one frame to know native resolution.
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, sample = cap.read()
    if not ret or sample is None:
        cap.release()
        return None
    native_h, native_w = sample.shape[:2]

    sample_indices = [
        int(round(i * total / (sample_count + 1))) for i in range(1, sample_count + 1)
    ]
    detections = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        det = detect_circle_in_frame(frame)
        if det:
            detections.append(det)
    cap.release()

    if len(detections) < 3:
        return None

    centers = np.array([d[:2] for d in detections])
    radii = np.array([d[2] for d in detections])
    center_x = float(np.median(centers[:, 0]))
    center_y = float(np.median(centers[:, 1]))
    radius = float(np.median(radii))
    crop_size = 2 * radius * margin

    # Clamp crop size to image bounds while keeping the crop square.
    max_size = min(native_w, native_h)
    crop_size = min(crop_size, max_size)
    crop_size = int(round(crop_size))
    if crop_size % 2 == 1:
        crop_size += 1

    return {
        "center_x": center_x,
        "center_y": center_y,
        "crop_size": float(crop_size),
    }


def get_crop_box(crop_params: Dict[str, float], image_shape: Tuple[int, ...]) -> Tuple[int, int, int, int]:
    """Compute integer crop box (x1, y1, x2, y2) from params and image shape."""
    h, w = image_shape[:2]
    cx = crop_params["center_x"]
    cy = crop_params["center_y"]
    size = crop_params["crop_size"]
    half = size / 2.0
    x1 = _safe_int(cx - half)
    y1 = _safe_int(cy - half)
    x2 = _safe_int(cx + half)
    y2 = _safe_int(cy + half)

    # Clamp to image bounds, keeping square if possible.
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > w:
        x1 = max(0, x1 - (x2 - w))
        x2 = w
    if y2 > h:
        y1 = max(0, y1 - (y2 - h))
        y2 = h
    return (x1, y1, x2, y2)


def crop_image(image: np.ndarray, crop_params: Dict[str, float]) -> np.ndarray:
    """Crop a color image using crop_params."""
    x1, y1, x2, y2 = get_crop_box(crop_params, image.shape)
    return image[y1:y2, x1:x2]


def crop_mask(mask: np.ndarray, crop_params: Dict[str, float]) -> np.ndarray:
    """Crop a single-channel mask using the same params."""
    x1, y1, x2, y2 = get_crop_box(crop_params, mask.shape)
    return mask[y1:y2, x1:x2]


def scale_crop_params(
    crop_params: Dict[str, float], src_size: Tuple[int, int], dst_size: Tuple[int, int]
) -> Dict[str, float]:
    """Scale crop params from src resolution to dst resolution."""
    src_h, src_w = src_size
    dst_h, dst_w = dst_size
    scale_x = dst_w / src_w
    scale_y = dst_h / src_h
    return {
        "center_x": crop_params["center_x"] * scale_x,
        "center_y": crop_params["center_y"] * scale_y,
        "crop_size": crop_params["crop_size"] * scale_x,
    }
