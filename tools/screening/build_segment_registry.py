#!/usr/bin/env python3
"""Build segment/frame registry from video metadata."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tools.screening.config import METADATA_CSV
from tools.screening.db import build_registry, init_db, load_videos_from_metadata


def main():
    parser = argparse.ArgumentParser(description="Build SFY segment/frame registry.")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=METADATA_CSV,
        help="Path to sfy_video_metadata.csv",
    )
    parser.add_argument(
        "--segment-sec",
        type=float,
        default=None,
        help="Segment length in seconds (default from config)",
    )
    args = parser.parse_args()

    init_db()
    n_videos = load_videos_from_metadata(args.metadata)
    print(f"Loaded {n_videos} videos from metadata.")
    n_processed = build_registry(segment_sec=args.segment_sec)
    print(f"Built registry for {n_processed} OK videos.")


if __name__ == "__main__":
    main()
