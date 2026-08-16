#!/usr/bin/env python3
"""Render reviewed LabelMe polygons to binary masks and update review.db.

Scans batch_001/labels/ for JSON files with flags.reviewed == true,
renders the polygon shapes to PNG masks in batch_001/masks_reviewed/,
and updates frames.annotation_mask_path in review.db.
"""

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tools.screening.db import connect_db


def render_polygon_mask(data: dict, image_width: int, image_height: int) -> np.ndarray:
    """Render polygon shapes from LabelMe JSON data to a binary mask."""
    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "polygon":
            continue
        points = shape.get("points", [])
        if len(points) < 3:
            continue
        pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 255)

    return mask


def update_reviewed_masks(batch_dir: Path, reviewer: str = "default"):
    batch_dir = Path(batch_dir)
    csv_path = batch_dir / "annotation_batch.csv"
    labels_dir = batch_dir / "labels"
    masks_dir = batch_dir / "masks_reviewed"
    masks_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return
    if not labels_dir.exists():
        print(f"Labels directory not found: {labels_dir}")
        return

    # Load CSV to map image_path -> segment_id, frame_idx
    csv_rows = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_path = Path(row["image_path"]).name
            csv_rows[image_path] = row

    reviewed_items = []
    for json_path in sorted(labels_dir.glob("*.json")):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        flags = data.get("flags", {})
        if not flags.get("reviewed"):
            continue

        image_path = Path(data.get("imagePath", json_path.name.replace(".json", ".jpg"))).name
        if image_path not in csv_rows:
            print(f"Warning: {image_path} not found in CSV, skipping")
            continue

        csv_row = csv_rows[image_path]
        segment_id = csv_row["segment_id"]
        frame_idx = int(csv_row["frame_idx"])

        image_height = data.get("imageHeight")
        image_width = data.get("imageWidth")
        if image_height is None or image_width is None:
            print(f"Warning: missing image size in {json_path.name}, skipping")
            continue

        mask = render_polygon_mask(data, image_width, image_height)
        mask_path = masks_dir / f"{json_path.stem}.png"
        cv2.imwrite(str(mask_path), mask)

        reviewed_items.append((segment_id, frame_idx, mask_path, json_path))

    print(f"Rendered {len(reviewed_items)} reviewed masks to {masks_dir}")

    # Update review.db
    updated = 0
    with connect_db() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        for segment_id, frame_idx, mask_path, json_path in reviewed_items:
            rel_mask_path = str(mask_path.relative_to(Path("/mnt/d/torch_project/dataset/sfy_screening")))
            rel_json_path = str(json_path.relative_to(Path("/mnt/d/torch_project/dataset/sfy_screening")))

            cursor.execute(
                """
                UPDATE frames
                SET status = 'valid',
                    annotation_mask_path = ?,
                    tags = 'reviewed',
                    label_source = 'manual'
                WHERE segment_id = ? AND frame_idx = ?
                """,
                (rel_mask_path, segment_id, frame_idx),
            )

            cursor.execute(
                """
                INSERT INTO frame_labels (segment_id, frame_idx, label, tags, notes, reviewer,
                                          annotation_json_path, annotation_mask_path, label_source, timestamp)
                VALUES (?, ?, 'valid', 'reviewed', 'reviewed mask rendered', ?, ?, ?, 'manual',
                        datetime('now'))
                ON CONFLICT DO UPDATE SET
                    label = excluded.label,
                    tags = excluded.tags,
                    notes = excluded.notes,
                    reviewer = excluded.reviewer,
                    annotation_json_path = excluded.annotation_json_path,
                    annotation_mask_path = excluded.annotation_mask_path,
                    label_source = excluded.label_source
                """,
                (segment_id, frame_idx, reviewer, rel_json_path, rel_mask_path),
            )
            updated += 1

        conn.commit()

    print(f"Updated {updated} frames in review.db")


def main():
    parser = argparse.ArgumentParser(description="Render reviewed masks and update review.db.")
    parser.add_argument("--batch-dir", default="/mnt/d/torch_project/dataset/sfy_screening/annotations/batch_001")
    parser.add_argument("--reviewer", default="default")
    args = parser.parse_args()

    update_reviewed_masks(Path(args.batch_dir), reviewer=args.reviewer)


if __name__ == "__main__":
    main()
