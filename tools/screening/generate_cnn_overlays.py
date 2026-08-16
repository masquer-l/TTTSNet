#!/usr/bin/env python3
"""Generate CNN baseline overlay masks for preview frames."""

import argparse
import sqlite3
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tools.screening.config import (
    CNN_BATCH_SIZE,
    CNN_CHECKPOINT,
    CNN_CONFIG,
    DB_PATH,
    OVERLAY_DIR,
    PREVIEW_DIR,
    ensure_dirs,
)
from tools.screening.db import init_db, set_frame_paths_batch
from tools.eval_sfy import create_model, load_checkpoint, load_config


def _case_and_stem_from_segment_id(segment_id: str) -> tuple:
    parts = segment_id.split("_")
    return parts[0], "_".join(parts[1:-2])


def _update_batch_with_retry(updates: list, max_retries: int = 5) -> None:
    """Write a batch of frame path updates, retrying on SQLite lock."""
    for attempt in range(max_retries):
        try:
            set_frame_paths_batch(updates)
            return
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() and attempt < max_retries - 1:
                wait = 0.5 * (2 ** attempt)
                time.sleep(wait)
            else:
                raise


class PreviewDataset(Dataset):
    def __init__(self, rows, img_size: int, preview_root: Path):
        self.rows = rows
        self.img_size = img_size
        self.preview_root = preview_root
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        preview_path = self.preview_root / r["preview_path"]
        image = cv2.imread(str(preview_path))
        if image is None:
            # Return a blank image; will be skipped in post-processing
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        return {
            "tensor": tensor,
            "orig_size": (orig_h, orig_w),
            "segment_id": r["segment_id"],
            "frame_idx": r["frame_idx"],
            "preview_path": r["preview_path"],
        }


def _collate(batch):
    tensors = torch.stack([b["tensor"] for b in batch])
    return tensors, batch


def generate_overlays(batch_size: int, cpu: bool = False, force: bool = False) -> None:
    ensure_dirs()
    init_db()

    cfg = load_config(str(CNN_CONFIG))
    img_size = cfg["dataset_config"].get("img_size", 448)
    device = torch.device("cpu" if cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = create_model(cfg, device)
    load_checkpoint(model, str(CNN_CHECKPOINT), device)
    model.eval()

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        if force:
            rows = conn.execute(
                """
                SELECT f.segment_id, f.frame_idx, f.preview_path
                FROM frames f
                WHERE f.is_preview = 1 AND f.preview_path IS NOT NULL
                ORDER BY f.segment_id, f.frame_idx
                """
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT f.segment_id, f.frame_idx, f.preview_path
                FROM frames f
                WHERE f.is_preview = 1 AND f.preview_path IS NOT NULL AND f.overlay_path IS NULL
                ORDER BY f.segment_id, f.frame_idx
                """
            ).fetchall()

    if not rows:
        print("No pending overlay frames.")
        return

    dataset = PreviewDataset([dict(r) for r in rows], img_size, PREVIEW_DIR)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate,
    )

    saved = 0
    errors = 0
    batch_updates = []
    for tensors, batch in tqdm(loader, desc="Generating overlays"):
        tensors = tensors.to(device)
        with torch.no_grad():
            logits = model(tensors)
            probs = torch.softmax(logits, dim=1)[:, 1:2]  # [B,1,H,W]

        for i, item in enumerate(batch):
            segment_id = item["segment_id"]
            frame_idx = item["frame_idx"]
            preview_rel = item["preview_path"]
            orig_size = item["orig_size"]

            if orig_size[0] == 0 or orig_size[1] == 0:
                errors += 1
                continue

            prob = probs[i : i + 1]
            prob_up = F.interpolate(prob, size=orig_size, mode="bilinear", align_corners=False)
            prob_np = prob_up.squeeze().cpu().numpy()
            pred = (prob_np > 0.5).astype(np.uint8)

            case, video_stem = _case_and_stem_from_segment_id(segment_id)
            rel_dir = Path(case) / video_stem
            filename = f"{case}_{video_stem}_{frame_idx:08d}.png"
            out_path = OVERLAY_DIR / rel_dir / filename
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), pred * 255)

            area_ratio = float(pred.mean())
            max_conf = float(prob_np.max())
            mean_conf = float(prob_np.mean())
            rel_overlay = str(rel_dir / filename)
            batch_updates.append(
                (
                    preview_rel,
                    rel_overlay,
                    None,
                    max_conf,
                    mean_conf,
                    area_ratio,
                    segment_id,
                    frame_idx,
                )
            )
            saved += 1

        if batch_updates:
            _update_batch_with_retry(batch_updates)
            batch_updates = []

    print(f"Overlays saved: {saved}, errors: {errors}")


def main():
    parser = argparse.ArgumentParser(description="Generate CNN overlays for previews.")
    parser.add_argument("--batch-size", type=int, default=CNN_BATCH_SIZE)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    generate_overlays(args.batch_size, cpu=args.cpu, force=args.force)


if __name__ == "__main__":
    main()
