#!/usr/bin/env python3
"""Extract full-resolution frames for a segment."""

import argparse
import sqlite3
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tools.screening.config import DB_PATH, SEGMENT_FRAMES_DIR, ensure_dirs
from tools.screening.db import init_db, set_frame_paths


def _case_and_stem(segment_id: str) -> tuple:
    parts = segment_id.split("_")
    return parts[0], "_".join(parts[1:-2])


def extract_segment_frames(segment_id: str, quality: int = 95) -> int:
    ensure_dirs()
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        seg = conn.execute(
            "SELECT * FROM segments WHERE segment_id = ?", (segment_id,)
        ).fetchone()
    if seg is None:
        return 0
    video_path = seg["video_path"]
    start = seg["start_frame"]
    end = seg["end_frame"]
    case, video_stem = _case_and_stem(segment_id)
    out_dir = SEGMENT_FRAMES_DIR / case / video_stem / segment_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    count = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for frame_idx in tqdm(range(start, end + 1), desc=f"Extract {segment_id}", leave=False):
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        filename = f"{case}_{video_stem}_{frame_idx:08d}.jpg"
        out_path = out_dir / filename
        rel_path = str(out_path.relative_to(SEGMENT_FRAMES_DIR))
        if not out_path.exists():
            cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        set_frame_paths(segment_id, frame_idx, fullres_path=rel_path)
        count += 1
    cap.release()
    return count


def main():
    parser = argparse.ArgumentParser(description="Extract full-res frames for segment(s).")
    parser.add_argument("--segment-id", required=True)
    args = parser.parse_args()
    n = extract_segment_frames(args.segment_id)
    print(f"Extracted {n} frames for {args.segment_id}")


if __name__ == "__main__":
    main()
