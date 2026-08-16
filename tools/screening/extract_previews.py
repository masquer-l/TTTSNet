#!/usr/bin/env python3
"""Extract low-resolution preview frames from all readable SFY videos."""

import argparse
import shutil
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tools.screening.config import (
    DB_PATH,
    PREVIEW_DIR,
    PREVIEW_INTERVAL_SEC,
    PREVIEW_MAX_SIZE,
    PREVIEW_QUALITY,
    ensure_dirs,
)
from tools.screening.db import init_db, set_frame_paths_batch


def _resize_max_edge(image: np.ndarray, max_size: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(max_size / max(h, w), 1.0)
    if scale == 1.0:
        return image
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _check_disk(min_gb: int = 20) -> None:
    usage = shutil.disk_usage(PREVIEW_DIR)
    free_gb = usage.free / (1024**3)
    if free_gb < min_gb:
        raise RuntimeError(
            f"Disk space too low: {free_gb:.1f} GB free, need at least {min_gb} GB."
        )


def _video_preview_tasks(case: Optional[str] = None, video_name: Optional[str] = None):
    """Return mapping video_path -> list of (segment_id, frame_idx)."""
    init_db()
    tasks: dict = {}
    where = ["f.is_preview = 1 AND f.preview_path IS NULL"]
    params: list = []
    if case:
        where.append("s.`case` = ?")
        params.append(case)
    if video_name:
        where.append("s.video_name = ?")
        params.append(video_name)
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""
            SELECT f.segment_id, f.frame_idx, s.video_path
            FROM frames f
            JOIN segments s ON f.segment_id = s.segment_id
            WHERE {' AND '.join(where)}
            ORDER BY s.video_path, f.frame_idx
            """,
            params,
        ).fetchall()
    for r in rows:
        vp = r["video_path"]
        tasks.setdefault(vp, []).append((r["segment_id"], r["frame_idx"]))
    return tasks


def _video_stem_from_segment_id(segment_id: str) -> str:
    parts = segment_id.split("_")
    return "_".join(parts[1:-2])


def _case_from_segment_id(segment_id: str) -> str:
    return segment_id.split("_")[0]


def _extract_video_previews(video_path: str, tasks: list, force: bool = False) -> tuple:
    video_path = Path(video_path)
    if not video_path.exists():
        return 0, 0, f"missing video: {video_path}"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, 0, f"cannot open: {video_path}"

    target_indices = {frame_idx for _, frame_idx in tasks}
    index_to_segments = {}
    for segment_id, frame_idx in tasks:
        index_to_segments.setdefault(frame_idx, []).append(segment_id)

    saved = 0
    skipped = 0
    updates = []
    frame_idx = 0
    pbar = tqdm(total=len(tasks), desc=f"Extracting {video_path.name}", leave=False)
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if frame_idx in target_indices:
            for segment_id in index_to_segments[frame_idx]:
                case = _case_from_segment_id(segment_id)
                video_stem = _video_stem_from_segment_id(segment_id)
                rel_dir = Path(case) / video_stem
                filename = f"{case}_{video_stem}_{frame_idx:08d}.jpg"
                out_path = PREVIEW_DIR / rel_dir / filename
                rel_path = str(rel_dir / filename)

                if out_path.exists() and not force:
                    updates.append((rel_path, None, None, None, None, None, segment_id, frame_idx))
                    skipped += 1
                else:
                    resized = _resize_max_edge(frame, PREVIEW_MAX_SIZE)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(
                        str(out_path),
                        resized,
                        [int(cv2.IMWRITE_JPEG_QUALITY), PREVIEW_QUALITY],
                    )
                    updates.append((rel_path, None, None, None, None, None, segment_id, frame_idx))
                    saved += 1
            pbar.update(len(index_to_segments[frame_idx]))
        frame_idx += 1
    pbar.close()
    cap.release()

    if updates:
        set_frame_paths_batch(updates)
    return saved, skipped, ""


def main():
    parser = argparse.ArgumentParser(description="Extract 2-second preview frames.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel videos")
    parser.add_argument("--force", action="store_true", help="Overwrite existing previews")
    parser.add_argument("--case", help="Only process a specific case")
    parser.add_argument("--video-name", help="Only process a specific video (requires --case)")
    args = parser.parse_args()

    ensure_dirs()
    _check_disk()
    init_db()

    tasks_by_video = _video_preview_tasks(case=args.case, video_name=args.video_name)
    total = sum(len(v) for v in tasks_by_video.values())
    print(f"Videos to process: {len(tasks_by_video)}, preview frames: {total}")

    if not total:
        print("No pending preview frames.")
        return

    saved_total = 0
    skipped_total = 0
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_extract_video_previews, vp, tasks, args.force): vp
            for vp, tasks in tasks_by_video.items()
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting previews"):
            vp = futures[future]
            try:
                saved, skipped, err = future.result()
                saved_total += saved
                skipped_total += skipped
                if err:
                    errors.append(f"{vp}: {err}")
            except Exception as e:
                errors.append(f"{vp}: {e}")

    print(f"Saved: {saved_total}, skipped: {skipped_total}, errors: {len(errors)}")
    for err in errors[:10]:
        print("  ERR:", err)


if __name__ == "__main__":
    main()
