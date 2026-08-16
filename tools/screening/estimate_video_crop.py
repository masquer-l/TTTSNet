#!/usr/bin/env python3
"""Estimate adaptive crop parameters for videos and store them in the DB."""

import argparse
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tools.screening.config import DB_PATH
from tools.screening.crop import estimate_video_crop
from tools.screening.db import get_video_crop, init_db, update_video_crop


def _videos_missing_crop():
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT `case`, video_name, unified_path
            FROM videos
            WHERE status = 'ok' AND crop_size IS NULL
            ORDER BY `case`, video_name
            """
        ).fetchall()
    return [dict(r) for r in rows]


def estimate_and_store(case: str, video_name: str, video_path: str) -> bool:
    existing = get_video_crop(case, video_name)
    if existing:
        print(f"Crop already exists for {case}/{video_name}, skipping.")
        return True
    params = estimate_video_crop(video_path)
    if params is None:
        print(f"Failed to estimate crop for {case}/{video_name}")
        return False
    update_video_crop(case, video_name, params["center_x"], params["center_y"], params["crop_size"])
    print(
        f"Stored crop for {case}/{video_name}: "
        f"center=({params['center_x']:.1f}, {params['center_y']:.1f}), size={params['crop_size']:.1f}"
    )
    return True


def main():
    parser = argparse.ArgumentParser(description="Estimate adaptive crop parameters for SFY videos.")
    parser.add_argument("--case", help="Specific case to process")
    parser.add_argument("--video-name", help="Specific video to process (requires --case)")
    parser.add_argument("--force", action="store_true", help="Re-estimate even if params exist")
    args = parser.parse_args()

    init_db()

    if args.case and args.video_name:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT unified_path FROM videos WHERE `case` = ? AND video_name = ?",
                (args.case, args.video_name),
            ).fetchone()
        if not row:
            print(f"Video not found: {args.case}/{args.video_name}")
            return
        estimate_and_store(args.case, args.video_name, row["unified_path"])
    elif args.case:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT video_name, unified_path FROM videos WHERE `case` = ? AND status = 'ok'",
                (args.case,),
            ).fetchall()
        for r in rows:
            estimate_and_store(args.case, r["video_name"], r["unified_path"])
    else:
        videos = _videos_missing_crop()
        for v in videos:
            estimate_and_store(v["case"], v["video_name"], v["unified_path"])


if __name__ == "__main__":
    main()
