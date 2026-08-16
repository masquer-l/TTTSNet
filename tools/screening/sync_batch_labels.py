#!/usr/bin/env python3
"""Sync X-AnyLabeling LabelMe JSON flags back to annotation_batch.csv and review.db.

Scans labels/ for flags:
  - "reviewed": frame has been manually refined and is ready as GT
  - "unreviewable": frame cannot be annotated (e.g., no vessel, out of focus)

Updates CSV annotation_status and writes invalid frames back to review.db.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tools.screening.db import connect_db, update_frame_label


def parse_json_flags(json_path: Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("flags", {})


def sync_batch(
    batch_dir: Path,
    labels_subdir: str = "labels",
    reviewer: str = "default",
    write_db: bool = True,
) -> dict:
    batch_dir = Path(batch_dir)
    csv_path = batch_dir / "annotation_batch.csv"
    labels_dir = batch_dir / labels_subdir

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    # Ensure required columns exist
    for col in ["annotation_status", "invalid_reason", "reviewer"]:
        if col not in fieldnames:
            fieldnames.append(col)

    stats = {"reviewed": 0, "unreviewable": 0, "vit_pseudo": 0, "other": 0}

    for row in rows:
        image_path = Path(row["image_path"])
        label_json_path = labels_dir / f"{image_path.stem}.json"

        if not label_json_path.exists():
            # Keep existing status if JSON is missing
            status = row.get("annotation_status", "vit_pseudo")
            stats[status] = stats.get(status, 0) + 1
            continue

        flags = parse_json_flags(label_json_path)

        if flags.get("unreviewable"):
            row["annotation_status"] = "unreviewable"
            row["invalid_reason"] = flags.get("unreviewable_reason", "")
            row["reviewer"] = reviewer
            stats["unreviewable"] += 1
        elif flags.get("reviewed"):
            row["annotation_status"] = "reviewed"
            row["invalid_reason"] = ""
            row["reviewer"] = reviewer
            stats["reviewed"] += 1
        else:
            # Keep original vit_pseudo or existing status
            if row.get("annotation_status") not in ("reviewed", "unreviewable"):
                row["annotation_status"] = "vit_pseudo"
            stats[row["annotation_status"]] = stats.get(row["annotation_status"], 0) + 1

    # Write CSV
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Write back to review.db for unreviewable frames
    if write_db:
        for row in rows:
            json_path = batch_dir / "labels" / f"{Path(row['image_path']).stem}.json"
            if row["annotation_status"] == "unreviewable":
                update_frame_label(
                    segment_id=row["segment_id"],
                    frame_idx=int(row["frame_idx"]),
                    label="invalid",
                    invalid_reason=row.get("invalid_reason") or "unreviewable",
                    reviewer=reviewer,
                    annotation_json_path=str(json_path),
                    label_source="manual",
                )
            elif row["annotation_status"] == "reviewed":
                update_frame_label(
                    segment_id=row["segment_id"],
                    frame_idx=int(row["frame_idx"]),
                    label="valid",
                    tags="reviewed",
                    notes="reviewed via X-AnyLabeling",
                    reviewer=reviewer,
                    annotation_json_path=str(json_path),
                    label_source="manual",
                )

    print(f"Synced {len(rows)} frames in {batch_dir}")
    print(f"  reviewed:     {stats['reviewed']}")
    print(f"  unreviewable: {stats['unreviewable']}")
    print(f"  vit_pseudo:   {stats['vit_pseudo']}")
    print(f"  other:        {stats['other']}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Sync X-AnyLabeling flags back to CSV and DB.")
    parser.add_argument("--batch-dir", default="/mnt/d/torch_project/dataset/sfy_screening/annotations/batch_001")
    parser.add_argument("--labels-subdir", default="labels")
    parser.add_argument("--reviewer", default="default")
    parser.add_argument("--no-db", action="store_true", help="Skip writing back to review.db")
    args = parser.parse_args()

    sync_batch(
        batch_dir=Path(args.batch_dir),
        labels_subdir=args.labels_subdir,
        reviewer=args.reviewer,
        write_db=not args.no_db,
    )


if __name__ == "__main__":
    main()
