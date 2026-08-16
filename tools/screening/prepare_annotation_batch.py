#!/usr/bin/env python3
"""Prepare a batch of candidate frames for pixel-level annotation.

Outputs a directory tree compatible with X-AnyLabeling:
    <output_dir>/
      images/              # symlinks to preview/fullres frames
      labels/              # empty LabelMe JSON files for X-AnyLabeling
      mask_grayscale_map.json
      annotation_batch.csv
      README.md
"""

import argparse
import csv
import cv2
import json
import random
import sqlite3
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tools.screening.config import PREVIEW_DIR, SEGMENT_FRAMES_DIR
from tools.screening.crop import crop_image, get_crop_box
from tools.screening.db import get_video_crop


DEFAULT_CASES = ["71", "29", "27", "33", "30", "26", "12", "28", "74", "31"]
FRAMES_PER_CASE = 20


def fetch_valid_frames(conn: sqlite3.Connection, cases: list[str]) -> list[dict]:
    placeholders = ",".join("?" * len(cases))
    rows = conn.execute(
        f"""
        SELECT f.segment_id, f.frame_idx, f.frame_time_sec, f.preview_path, f.fullres_path,
               f.overlay_path, f.annotation_mask_path, f.label_source, f.tags,
               s.`case`, s.video_name, s.start_frame, s.end_frame
        FROM frames f
        JOIN segments s ON f.segment_id = s.segment_id
        WHERE f.status = 'valid' AND s.`case` IN ({placeholders})
        ORDER BY s.`case`, s.video_name, f.frame_idx
        """,
        cases,
    ).fetchall()
    return [dict(r) for r in rows]


def sample_dispersed(frames: list[dict], case: str, n: int, rng: random.Random) -> list[dict]:
    """Sample n frames for a case, preferring different segments and time dispersion."""
    case_frames = [f for f in frames if f["case"] == case]
    if not case_frames:
        return []

    # Group by segment
    by_segment: dict[str, list[dict]] = {}
    for f in case_frames:
        by_segment.setdefault(f["segment_id"], []).append(f)

    # Sort segments by their earliest frame time
    segments = sorted(by_segment.items(), key=lambda x: min(f["frame_time_sec"] for f in x[1]))
    n_segments = len(segments)

    selected: list[dict] = []
    selected_keys: set[tuple[str, int]] = set()

    if n_segments >= n:
        # Pick n segments uniformly across time, take the middle frame of each
        indices = np.linspace(0, n_segments - 1, n, dtype=int)
        for i in indices:
            seg_frames = segments[i][1]
            seg_frames_sorted = sorted(seg_frames, key=lambda f: f["frame_time_sec"])
            frame = seg_frames_sorted[len(seg_frames_sorted) // 2]
            selected.append(frame)
            selected_keys.add((frame["segment_id"], frame["frame_idx"]))
    else:
        # Take middle frame from every segment first
        for _, seg_frames in segments:
            seg_frames_sorted = sorted(seg_frames, key=lambda f: f["frame_time_sec"])
            frame = seg_frames_sorted[len(seg_frames_sorted) // 2]
            selected.append(frame)
            selected_keys.add((frame["segment_id"], frame["frame_idx"]))

        # Fill the rest by time-quantile sampling from remaining frames
        remaining = [f for f in case_frames if (f["segment_id"], f["frame_idx"]) not in selected_keys]
        remaining_sorted = sorted(remaining, key=lambda f: f["frame_time_sec"])
        needed = n - len(selected)
        if remaining_sorted and needed > 0:
            quantiles = np.linspace(0, 1, needed + 2)[1:-1]
            for q in quantiles:
                idx = int(round(q * (len(remaining_sorted) - 1)))
                frame = remaining_sorted[idx]
                key = (frame["segment_id"], frame["frame_idx"])
                if key not in selected_keys:
                    selected.append(frame)
                    selected_keys.add(key)

    return sorted(selected, key=lambda f: (f["video_name"], f["frame_idx"]))


def build_labelme_json(image_path: Path, image_width: int, image_height: int) -> dict:
    return {
        "version": "2.4.0",
        "flags": {},
        "shapes": [],
        "imagePath": str(image_path.name),
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width,
    }


def prepare_batch(
    output_dir: Path,
    cases: list[str] = None,
    frames_per_case: int = FRAMES_PER_CASE,
    seed: int = 42,
    use_fullres: bool = True,
    apply_crop: bool = True,
) -> Path:
    if cases is None:
        cases = DEFAULT_CASES

    rng = random.Random(seed)
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    db_path = Path("/mnt/d/torch_project/dataset/sfy_screening/db/review.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    frames = fetch_valid_frames(conn, cases)

    sampled: list[dict] = []
    for case in cases:
        sampled.extend(sample_dispersed(frames, case, frames_per_case, rng))

    # Write mask grayscale map for X-AnyLabeling binary semantic segmentation export
    mask_map = {"type": "grayscale", "colors": {"vessel": 255}}
    with open(output_dir / "mask_grayscale_map.json", "w", encoding="utf-8") as f:
        json.dump(mask_map, f, indent=2, ensure_ascii=False)

    csv_rows = []
    for item in sampled:
        case = item["case"]
        video_name = item["video_name"]
        segment_id = item["segment_id"]
        frame_idx = item["frame_idx"]

        # Determine source image path
        if use_fullres and item.get("fullres_path"):
            src_image = SEGMENT_FRAMES_DIR / item["fullres_path"]
            resolution_note = "fullres"
        elif item.get("preview_path"):
            src_image = PREVIEW_DIR / item["preview_path"]
            resolution_note = "preview"
        else:
            src_image = None
            resolution_note = "missing"

        base_name = f"{case}_{video_name}_{frame_idx:08d}"
        dst_image = images_dir / f"{base_name}.jpg"

        crop_params = None
        crop_box = None
        if apply_crop:
            crop_params = get_video_crop(case, video_name)

        image_bgr = None
        if src_image and src_image.exists():
            image_bgr = cv2.imread(str(src_image))

        if image_bgr is not None:
            if apply_crop and crop_params is not None:
                image_bgr = crop_image(image_bgr, crop_params)
                crop_box = get_crop_box(crop_params, image_bgr.shape)
            cv2.imwrite(str(dst_image), image_bgr)
            image_ready = True
        else:
            image_ready = False

        # Create empty LabelMe JSON for X-AnyLabeling
        label_json_path = labels_dir / f"{base_name}.json"
        if image_ready:
            h, w = image_bgr.shape[:2]
        else:
            h, w = 1080, 1920

        labelme = build_labelme_json(dst_image, w, h)
        with open(label_json_path, "w", encoding="utf-8") as f:
            json.dump(labelme, f, indent=2, ensure_ascii=False)

        csv_rows.append({
            "batch_id": output_dir.name,
            "case_id": case,
            "video_id": video_name,
            "segment_id": segment_id,
            "frame_idx": frame_idx,
            "time_sec": item["frame_time_sec"],
            "quality_level": "",
            "difficulty_tags": item.get("tags", "") or "",
            "selection_method": "dispersed_segment_sampling",
            "selection_reason": f"{resolution_note}; segment coverage",
            "image_path": str(dst_image),
            "label_json_path": str(label_json_path),
            "annotation_status": "pending",
            "reviewer": "",
            "crop_center_x": crop_params["center_x"] if crop_params else "",
            "crop_center_y": crop_params["center_y"] if crop_params else "",
            "crop_size": crop_params["crop_size"] if crop_params else "",
            "crop_x1": crop_box[0] if crop_box else "",
            "crop_y1": crop_box[1] if crop_box else "",
            "crop_x2": crop_box[2] if crop_box else "",
            "crop_y2": crop_box[3] if crop_box else "",
        })

    csv_path = output_dir / "annotation_batch.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()) if csv_rows else [])
        writer.writeheader()
        writer.writerows(csv_rows)

    # Summary
    summary = {
        "total_frames": len(sampled),
        "cases": cases,
        "frames_per_case": frames_per_case,
        "unique_segments": len({(r["case_id"], r["segment_id"]) for r in csv_rows}),
        "images_ready": sum(1 for r in csv_rows if Path(r["image_path"]).exists()),
    }
    with open(output_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(f"# Annotation Batch: {output_dir.name}\n\n")
        f.write(f"```json\n{json.dumps(summary, indent=2)}\n```\n\n")
        f.write("Open `images/` in X-AnyLabeling. Save annotations to `labels/` as LabelMe JSON.\n")
        f.write("Export masks using `mask_grayscale_map.json` with `vessel` mapped to 255.\n")

    print(f"Prepared {len(sampled)} frames in {output_dir}")
    print(f"  unique segments covered: {summary['unique_segments']}")
    print(f"  images ready: {summary['images_ready']}/{summary['total_frames']}")
    print(f"  CSV: {csv_path}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Prepare a pixel annotation batch for X-AnyLabeling.")
    parser.add_argument("--output-dir", default="/mnt/d/torch_project/dataset/sfy_screening/annotations/batch_001")
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES, help="Case IDs to include")
    parser.add_argument("--frames-per-case", type=int, default=FRAMES_PER_CASE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview", action="store_true", help="Use preview images instead of fullres")
    parser.add_argument("--no-crop", dest="apply_crop", action="store_false", help="Disable per-video circular-view crop")
    args = parser.parse_args()

    prepare_batch(
        output_dir=Path(args.output_dir),
        cases=args.cases,
        frames_per_case=args.frames_per_case,
        seed=args.seed,
        use_fullres=not args.preview,
        apply_crop=args.apply_crop,
    )


if __name__ == "__main__":
    main()
