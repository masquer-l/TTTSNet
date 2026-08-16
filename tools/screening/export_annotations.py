#!/usr/bin/env python3
"""Export valid frames to a training-ready dataset."""

import argparse
import csv
import json
import random
import shutil
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tools.screening.config import (
    ANNOTATIONS_DIR,
    EXPORTS_DIR,
    OVERLAY_DIR,
    PREVIEW_DIR,
    SEGMENT_FRAMES_DIR,
    ensure_dirs,
)
from tools.screening.db import connect_db, get_video_crop


def _case_and_stem(segment_id: str) -> tuple:
    parts = segment_id.split("_")
    return parts[0], "_".join(parts[1:-2])


def export_annotations(
    name: str,
    copy_images: bool = False,
    split_ratio: float = 0.8,
    allow_cnn_fallback: bool = False,
) -> Path:
    ensure_dirs()
    export_dir = EXPORTS_DIR / f"{name}_{_now_timestamp()}"
    export_dir.mkdir(parents=True, exist_ok=True)
    images_dir = export_dir / "images"
    labels_dir = export_dir / "labels"
    json_dir = export_dir / "json"
    if copy_images:
        images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    # Build the WHERE clause. By default we require a manual pixel mask to
    # prevent silent export of CNN predictions as ground truth.
    if allow_cnn_fallback:
        mask_clause = "(f.annotation_mask_path IS NOT NULL OR f.overlay_path IS NOT NULL)"
    else:
        mask_clause = "f.annotation_mask_path IS NOT NULL"

    with connect_db() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""
            SELECT f.segment_id, f.frame_idx, f.preview_path, f.overlay_path,
                   f.fullres_path, f.annotation_mask_path, f.label_source, s.`case`, s.video_name,
                   s.video_path, s.start_frame
            FROM frames f
            JOIN segments s ON f.segment_id = s.segment_id
            JOIN videos v ON s.`case` = v.`case` AND s.video_name = v.video_name
            WHERE f.status = 'valid' AND s.status = 'valid' AND {mask_clause}
              AND v.review_status NOT IN ('invalid', 'not_ttts')
            ORDER BY s.`case`, s.video_name, f.frame_idx
            """
        ).fetchall()

    if not rows:
        print("No valid frames to export.")
        if not allow_cnn_fallback:
            print(
                "Hint: use --allow-cnn-fallback to export frames without manual masks "
                "(labels will be CNN predictions, not ground truth)."
            )
        return export_dir

    # Split by case
    cases = sorted({r["case"] for r in rows})
    random.shuffle(cases)
    split_at = int(len(cases) * split_ratio)
    train_cases = set(cases[:split_at])
    val_cases = set(cases[split_at:])

    # Cache crop parameters per video (original resolution).
    crop_cache: dict = {}

    def _get_crop(case: str, video_name: str):
        key = (case, video_name)
        if key not in crop_cache:
            crop_cache[key] = get_video_crop(case, video_name)
        return crop_cache[key]

    manifest = []
    missing = 0
    for r in rows:
        case = r["case"]
        video_name = r["video_name"]
        segment_id = r["segment_id"]
        frame_idx = r["frame_idx"]
        _, video_stem = _case_and_stem(segment_id)
        base_name = f"{case}_{video_stem}_{frame_idx:08d}"

        # Source image priority: fullres > preview
        src_image = None
        if r["fullres_path"]:
            src_image = SEGMENT_FRAMES_DIR / r["fullres_path"]
        elif r["preview_path"]:
            src_image = PREVIEW_DIR / r["preview_path"]

        # Label priority: manual mask only unless explicit fallback is enabled.
        src_label = None
        label_source = r["label_source"] or "screening"
        if r["annotation_mask_path"]:
            src_label = ANNOTATIONS_DIR / r["annotation_mask_path"]
            label_source = "manual"
        elif allow_cnn_fallback and r["overlay_path"]:
            src_label = OVERLAY_DIR / r["overlay_path"]
            label_source = "cnn_fallback"

        if src_image is None or src_label is None:
            missing += 1
            continue

        image_bgr = cv2.imread(str(src_image))
        if image_bgr is None:
            missing += 1
            continue
        image_shape = image_bgr.shape

        crop_params = _get_crop(case, video_name)

        if copy_images:
            dst_image = images_dir / f"{base_name}.jpg"
            if not dst_image.exists():
                shutil.copy2(str(src_image), str(dst_image))
            image_path = str(dst_image.relative_to(export_dir))
        else:
            image_path = str(src_image)

        dst_label = labels_dir / f"{base_name}.png"
        if not dst_label.exists():
            label_img = cv2.imread(str(src_label), cv2.IMREAD_GRAYSCALE)
            if label_img is None:
                missing += 1
                continue
            if label_img.shape[:2] != image_shape[:2]:
                label_img = cv2.resize(
                    label_img, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST
                )
            cv2.imwrite(str(dst_label), label_img)
        label_path = str(dst_label.relative_to(export_dir))

        # LabelMe v3 JSON with empty shapes (placeholder for desktop refinement)
        labelme_json = {
            "version": "3.0.0",
            "flags": {},
            "shapes": [],
            "imagePath": f"{base_name}.jpg",
            "imageData": None,
            "imageHeight": int(image_shape[0]),
            "imageWidth": int(image_shape[1]),
        }
        json_path = json_dir / f"{base_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(labelme_json, f, indent=2, ensure_ascii=False)

        split = "train" if case in train_cases else "val"
        entry = {
            "case": case,
            "video_name": video_name,
            "segment_id": segment_id,
            "frame_idx": frame_idx,
            "video_path": r["video_path"],
            "image_path": image_path,
            "label_path": label_path,
            "label_source": label_source,
            "split": split,
        }
        if crop_params is not None:
            entry["crop_center_x"] = crop_params["center_x"]
            entry["crop_center_y"] = crop_params["center_y"]
            entry["crop_size"] = crop_params["crop_size"]
        manifest.append(entry)

    manifest_path = export_dir / "frame_manifest.csv"
    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "video_name",
                "segment_id",
                "frame_idx",
                "video_path",
                "image_path",
                "label_path",
                "label_source",
                "split",
                "crop_center_x",
                "crop_center_y",
                "crop_size",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest)

    dataset_config = {
        "img_size": 448,
        "binary": True,
        "frame_manifest": str(manifest_path),
    }
    with open(export_dir / "dataset_config.json", "w", encoding="utf-8") as f:
        json.dump(dataset_config, f, indent=2, ensure_ascii=False)

    print(f"Exported {len(manifest)} frames to {export_dir} (missing {missing}).")
    print(f"  train cases: {len(train_cases)}, val cases: {len(val_cases)}")
    return export_dir


def _now_timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def main():
    parser = argparse.ArgumentParser(description="Export valid frames to training set.")
    parser.add_argument("--name", default="sfy_manual")
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--split-ratio", type=float, default=0.8)
    parser.add_argument(
        "--allow-cnn-fallback",
        action="store_true",
        help="Allow CNN overlay to be used as label when no manual mask exists. "
             "Exported label_source will be 'cnn_fallback', not ground truth.",
    )
    args = parser.parse_args()
    export_annotations(
        args.name,
        copy_images=args.copy_images,
        split_ratio=args.split_ratio,
        allow_cnn_fallback=getattr(args, "allow_cnn_fallback"),
    )


if __name__ == "__main__":
    main()
