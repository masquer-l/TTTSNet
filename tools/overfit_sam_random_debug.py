#!/usr/bin/env python3
"""Overfit and visualize the SAM random-point data path.

This script freezes a tiny augmented subset in memory, overfits SAM on it, and
exports visual panels for checking raw input, augmentation, random prompts,
prediction, and GT alignment.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

TTTS_SAM_ROOT = Path("/root/autodl-fs/masquer.li/code/TTTS_SAM")
sys.path.insert(0, str(TTTS_SAM_ROOT))
from segment_anything import sam_model_registry

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset_sam_random import SAMRandomPointDataset, collate_fn
from tools.train_sam_random import compute_loss, load_config, set_seed


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=-1)
    return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)


def draw_points(image_rgb: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    canvas = image_rgb.copy()
    for x, y in points_xy.astype(int):
        cv2.circle(canvas, (int(x), int(y)), 5, (255, 255, 0), thickness=-1)
        cv2.circle(canvas, (int(x), int(y)), 7, (0, 0, 0), thickness=1)
    return canvas


def mask_to_rgb(mask: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask > 0] = color
    return out


def overlay_masks(image_rgb: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred_rgb = mask_to_rgb(pred, (255, 0, 0))
    gt_rgb = mask_to_rgb(gt, (0, 255, 0))
    overlay = image_rgb.copy()
    overlay = cv2.addWeighted(overlay, 1.0, gt_rgb, 0.35, 0)
    overlay = cv2.addWeighted(overlay, 1.0, pred_rgb, 0.35, 0)
    return overlay


def make_dataset(cfg: Dict[str, Any], sample_count: int, seed: int) -> List[Dict[str, Any]]:
    ds_cfg = cfg["dataset_config"]
    prompt_cfg = cfg.get("prompt_config", {})
    aug_cfg = cfg.get("augmentation_config", {})
    train_cfg = cfg.get("training_config", {})

    dataset = SAMRandomPointDataset(
        data_path=ds_cfg["train_paths"][0],
        mode="train",
        img_size=ds_cfg.get("img_size", 1024),
        prompt_points_num=prompt_cfg.get("prompt_points_num", 20),
        disable_augmentation=train_cfg.get("disable_augmentation", False),
        custom_defects_p=aug_cfg.get("custom_defects_p", 0.5),
        deterministic_points=True,
        points_seed=seed,
    )

    samples: List[Dict[str, Any]] = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if sample is None:
            continue
        samples.append(sample)
        if len(samples) >= sample_count:
            break

    if not samples:
        raise RuntimeError("No valid positive samples found for overfit debug.")
    return samples


def create_model(cfg: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    model_cfg = cfg["model_config"]
    model = sam_model_registry[model_cfg["model_type"]](
        checkpoint=model_cfg.get("checkpoint"),
        device=str(device),
    )
    return model.to(device)


@torch.no_grad()
def predict_batch(
    model: torch.nn.Module,
    batch: Dict[str, Any],
    device: torch.device,
    use_box_prompt: bool,
) -> List[np.ndarray]:
    model.eval()
    images = batch["images"].to(device)
    points = batch["points"].to(device)
    point_labels = batch["point_labels"].to(device)
    boxes = batch["boxes"].to(device) if use_box_prompt else None

    image_embeddings = model.image_encoder(images)
    sparse_embeddings, dense_embeddings = model.prompt_encoder(
        points=(points, point_labels),
        boxes=boxes,
        masks=None,
    )
    low_res_masks, _ = model.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    preds: List[np.ndarray] = []
    for i in range(low_res_masks.shape[0]):
        masks = model.postprocess_masks(
            low_res_masks[i].unsqueeze(0),
            input_size=batch["size_before_pad"][i],
            original_size=batch["image_sizes"][i],
        )
        pred = (torch.sigmoid(masks.squeeze()) > 0.5).cpu().numpy().astype(np.uint8)
        preds.append(pred)
    return preds


def save_visualization(
    model: torch.nn.Module,
    batch: Dict[str, Any],
    device: torch.device,
    output_dir: Path,
    step: int,
    use_box_prompt: bool,
) -> None:
    preds = predict_batch(model, batch, device, use_box_prompt)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, pred in enumerate(preds):
        raw_rgb = bgr_to_rgb(batch["images_raw"][i])
        aug_rgb = bgr_to_rgb(batch["images_original"][i])
        gt = batch["labels"][i].squeeze().cpu().numpy().astype(np.uint8)
        points = batch["points_original"][i].cpu().numpy()

        sam_input = batch["images"][i].detach().cpu().numpy().transpose(1, 2, 0)
        sam_input = np.clip(sam_input, 0, 255).astype(np.uint8)
        sam_input_rgb = bgr_to_rgb(sam_input)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        panels = [
            ("raw image", raw_rgb),
            ("augmented + random points", draw_points(aug_rgb, points)),
            ("SAM resized input", sam_input_rgb),
            ("prediction", pred * 255),
            ("GT", gt * 255),
            ("overlay: GT green, pred red", overlay_masks(aug_rgb, pred, gt)),
        ]

        for ax, (title, image) in zip(axes.flat, panels):
            ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
            ax.set_title(title)
            ax.axis("off")

        fig.tight_layout()
        name = batch["image_names"][i].replace("/", "_")
        fig.savefig(output_dir / f"step_{step:04d}_{i}_{name}.png", dpi=150)
        plt.close(fig)


def train_step(
    model: torch.nn.Module,
    batch: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
    iou_weight: float,
    use_box_prompt: bool,
) -> Tuple[float, float, float]:
    model.train()
    images = batch["images"].to(device)
    points = batch["points"].to(device)
    point_labels = batch["point_labels"].to(device)
    boxes = batch["boxes"].to(device) if use_box_prompt else None

    with autocast(enabled=use_amp):
        image_embeddings = model.image_encoder(images)
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=(points, point_labels),
            boxes=boxes,
            masks=None,
        )
        low_res_masks, iou_predictions = model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        loss, miou, dice, valid = compute_loss(
            model, low_res_masks, iou_predictions, batch, criterion, device, iou_weight
        )

    if valid <= 0:
        return float("nan"), float("nan"), float("nan")

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    return loss.item(), miou, dice


def main() -> None:
    parser = argparse.ArgumentParser(description="Overfit SAM random-point pipeline and export visual checks.")
    parser.add_argument("--config", type=str, default="configs/config_sam_random.json")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--sample_count", type=int, default=2)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--visualize_every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"experiments/overfit_sam_random_debug_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    samples = make_dataset(cfg, args.sample_count, args.seed)
    loader = DataLoader(samples, batch_size=len(samples), shuffle=False, num_workers=0, collate_fn=collate_fn)
    batch = next(iter(loader))

    model = create_model(cfg, device)
    train_cfg = cfg.get("training_config", {})
    prompt_cfg = cfg.get("prompt_config", {})
    lr = args.lr if args.lr is not None else train_cfg.get("lr", 1e-4)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=train_cfg.get("weight_decay", 0.01))

    import monai
    criterion = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean").to(device)
    use_amp = bool(train_cfg.get("use_amp", 1))
    scaler = GradScaler(enabled=use_amp)
    iou_weight = train_cfg.get("iou_loss_weight", 0.1)
    use_box_prompt = bool(prompt_cfg.get("box_prompt", False))

    save_visualization(model, batch, device, output_dir, step=0, use_box_prompt=use_box_prompt)

    log_rows = []
    for step in range(1, args.steps + 1):
        loss, miou, dice = train_step(
            model,
            batch,
            optimizer,
            criterion,
            scaler,
            device,
            use_amp,
            iou_weight,
            use_box_prompt,
        )
        log_rows.append({"step": step, "loss": loss, "miou": miou, "dice": dice})
        print(f"step {step}/{args.steps}: loss={loss:.4f}, miou={miou:.4f}, dice={dice:.4f}")

        if step % args.visualize_every == 0 or step == args.steps:
            save_visualization(model, batch, device, output_dir, step=step, use_box_prompt=use_box_prompt)

    with open(output_dir / "overfit_history.csv", "w", encoding="utf-8") as f:
        f.write("step,loss,miou,dice\n")
        for row in log_rows:
            f.write(f"{row['step']},{row['loss']},{row['miou']},{row['dice']}\n")

    torch.save({"model_state_dict": model.state_dict(), "history": log_rows}, output_dir / "overfit_model.pth")
    print(f"Overfit debug outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
