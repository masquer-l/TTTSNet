#!/usr/bin/env python3
"""Create a working subset of a batch that only contains unfinished frames.

This lets the annotator reopen X-AnyLabeling without loading already-reviewed
or unreviewable frames.

Usage:
  1. Run after some frames are marked reviewed/unreviewable in X-AnyLabeling
     and synced via sync_batch_labels.py.
  2. Open working/images/ in X-AnyLabeling.
  3. After further annotation, run sync_batch_labels.py again with
     --labels-subdir working/labels to sync changes back.
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


FINISHED_STATUSES = {"reviewed", "unreviewable"}


def filter_working_set(
    batch_dir: Path,
    output_subdir: str = "working",
    status_csv_col: str = "annotation_status",
    copy_images: bool = False,
) -> Path:
    batch_dir = Path(batch_dir)
    csv_path = batch_dir / "annotation_batch.csv"
    src_images_dir = batch_dir / "images"
    src_labels_dir = batch_dir / "labels"

    out_dir = batch_dir / output_subdir
    out_images_dir = out_dir / "images"
    out_labels_dir = out_dir / "labels"

    # Clean old working set to avoid stale symlinks for finished frames.
    if out_dir.exists():
        for p in list(out_images_dir.glob("*")) + list(out_labels_dir.glob("*")):
            p.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    unfinished = [r for r in rows if r.get(status_csv_col) not in FINISHED_STATUSES]
    finished = [r for r in rows if r.get(status_csv_col) in FINISHED_STATUSES]

    for row in unfinished:
        image_path = Path(row["image_path"])
        label_json_path = src_labels_dir / f"{image_path.stem}.json"

        dst_image = out_images_dir / image_path.name
        if copy_images:
            shutil.copy2(str(image_path), str(dst_image))
        else:
            if dst_image.exists() or dst_image.is_symlink():
                dst_image.unlink()
            dst_image.symlink_to(image_path.resolve())

        dst_label = out_labels_dir / f"{image_path.stem}.json"
        if label_json_path.exists():
            if dst_label.exists() or dst_label.is_symlink():
                dst_label.unlink()
            dst_label.symlink_to(label_json_path.resolve())
        else:
            # Create empty placeholder if source label missing
            empty_labelme = {
                "version": "2.4.0",
                "flags": {},
                "shapes": [],
                "imagePath": image_path.name,
                "imageData": None,
                "imageHeight": row.get("crop_y2", 1080),
                "imageWidth": row.get("crop_x2", 1080),
            }
            with open(dst_label, "w", encoding="utf-8") as f:
                json.dump(empty_labelme, f, indent=2, ensure_ascii=False)

    # Write a small report
    report = {
        "total": len(rows),
        "unfinished": len(unfinished),
        "finished": len(finished),
        "finished_breakdown": {
            status: sum(1 for r in finished if r.get(status_csv_col) == status)
            for status in FINISHED_STATUSES
        },
    }
    with open(out_dir / "working_set_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Working set created at {out_dir}")
    print(f"  unfinished: {len(unfinished)}")
    print(f"  finished:   {len(finished)} (reviewed={report['finished_breakdown']['reviewed']}, unreviewable={report['finished_breakdown']['unreviewable']})")
    print(f"\nNext step:")
    print(f"  cd /mnt/d/torch_project/X-AnyLabeling")
    print(f"  python anylabeling/app.py {out_images_dir} --output {out_labels_dir} --labels vessel --autosave")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Create a working subset of unfinished frames.")
    parser.add_argument("--batch-dir", default="/mnt/d/torch_project/dataset/sfy_screening/annotations/batch_001")
    parser.add_argument("--output-subdir", default="working")
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of symlinking")
    args = parser.parse_args()

    filter_working_set(
        batch_dir=Path(args.batch_dir),
        output_subdir=args.output_subdir,
        copy_images=args.copy_images,
    )


if __name__ == "__main__":
    main()
