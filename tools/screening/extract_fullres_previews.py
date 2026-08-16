#!/usr/bin/env python3
"""Extract full-resolution frames at the 2-second preview interval.

This populates frames.fullres_path so that coarse-screening zoom and downstream
annotation can use original-resolution frames instead of downscaled previews.
"""

import argparse
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import cv2
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tools.screening.config import (
    DB_PATH,
    PREVIEW_INTERVAL_SEC,
    SEGMENT_FRAMES_DIR,
    ensure_dirs,
)
from tools.screening.db import init_db, set_frame_paths_batch


def _case_from_segment_id(segment_id: str) -> str:
    return segment_id.split("_")[0]


def _video_stem_from_segment_id(segment_id: str) -> str:
    parts = segment_id.split("_")
    return "_".join(parts[1:-2])


def _pending_preview_tasks(case: Optional[str] = None, video_name: Optional[str] = None):
    """Return mapping video_path -> list of (segment_id, frame_idx)."""
    init_db()
    where = ["f.is_preview = 1 AND f.fullres_path IS NULL AND s.status != 'invalid'"]
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
    tasks: dict = {}
    for r in rows:
        tasks.setdefault(r["video_path"], []).append((r["segment_id"], r["frame_idx"]))
    return tasks


def _extract_video_fullres(video_path: str, tasks: list) -> tuple:
    video_path = Path(video_path)
    if not video_path.exists():
        return 0, 0, f"missing video: {video_path}"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, 0, f"cannot open: {video_path}"

    target_indices = {frame_idx for _, frame_idx in tasks}
    index_to_segments: dict = {}
    for segment_id, frame_idx in tasks:
        index_to_segments.setdefault(frame_idx, []).append(segment_id)

    saved = 0
    skipped = 0
    updates = []
    frame_idx = 0
    pbar = tqdm(total=len(tasks), desc=f"Fullres {video_path.name}", leave=False)
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if frame_idx in target_indices:
            for segment_id in index_to_segments[frame_idx]:
                case = _case_from_segment_id(segment_id)
                video_stem = _video_stem_from_segment_id(segment_id)
                out_dir = SEGMENT_FRAMES_DIR / case / video_stem / segment_id
                filename = f"{case}_{video_stem}_{frame_idx:08d}.jpg"
                out_path = out_dir / filename
                rel_path = str(Path(case) / video_stem / segment_id / filename)

                if not out_path.exists():
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(
                        str(out_path),
                        frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95],
                    )
                    saved += 1
                else:
                    skipped += 1
                updates.append((None, None, rel_path, None, None, None, segment_id, frame_idx))
            pbar.update(len(index_to_segments[frame_idx]))
        frame_idx += 1
    pbar.close()
    cap.release()

    if updates:
        set_frame_paths_batch(updates)
    return saved, skipped, ""


def main():
    parser = argparse.ArgumentParser(description="Extract full-resolution 2-second preview frames.")
    parser.add_argument("--workers", type=int, default=2, help="Parallel videos")
    parser.add_argument("--case", help="Specific case to process")
    parser.add_argument("--video-name", help="Specific video to process (requires --case)")
    args = parser.parse_args()

    ensure_dirs()
    init_db()

    tasks_by_video = _pending_preview_tasks(case=args.case, video_name=args.video_name)
    total = sum(len(v) for v in tasks_by_video.values())
    print(f"Videos to process: {len(tasks_by_video)}, fullres preview frames: {total}")

    if not total:
        print("No pending fullres preview frames.")
        return

    saved_total = 0
    skipped_total = 0
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_extract_video_fullres, vp, tasks): vp
            for vp, tasks in tasks_by_video.items()
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fullres previews"):
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
