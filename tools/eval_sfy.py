#!/usr/bin/env python3
"""Evaluate TTTSNet-style models on the SFY 0923 external dataset."""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.TTTSNet import TTTSNet
from models.tttsnet_vit import TTTSNetViT
from models.tttsnet_transformer_decoder import TTTSNetTransformerDecoder, TTTSNetViTTransformerDecoder
from utils.losses import soft_skeletonize


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_model(cfg: Dict, device: torch.device) -> torch.nn.Module:
    model_cfg = cfg["model_config"]
    model_name = model_cfg.get("name", "TTTSNet")
    if model_name == "TTTSNetViT":
        model = TTTSNetViT(
            classes=model_cfg.get("classes", 2),
            block_1=model_cfg.get("block_1", 3),
            block_2=model_cfg.get("block_2", 8),
            num_features=model_cfg.get("num_features", 64),
            sam_checkpoint=model_cfg.get("sam_checkpoint", "/autodl-fs/data/masquer.li/model/sam_vit_b_01ec64.pth"),
            freeze_vit=model_cfg.get("freeze_vit", False),
        )
    elif model_name == "TTTSNetTransformerDecoder":
        model = TTTSNetTransformerDecoder(
            classes=model_cfg.get("classes", 2),
            block_1=model_cfg.get("block_1", 3),
            block_2=model_cfg.get("block_2", 8),
            num_features=model_cfg.get("num_features", 64),
            transformer_dim=model_cfg.get("transformer_dim", 128),
            transformer_heads=model_cfg.get("transformer_heads", 4),
            transformer_pooled_size=model_cfg.get("transformer_pooled_size", 28),
            transformer_num_layers=model_cfg.get("transformer_num_layers", 1),
            transformer_use_pos_embed=model_cfg.get("transformer_use_pos_embed", True),
        )
    elif model_name == "TTTSNetViTTransformerDecoder":
        model = TTTSNetViTTransformerDecoder(
            classes=model_cfg.get("classes", 2),
            block_1=model_cfg.get("block_1", 3),
            block_2=model_cfg.get("block_2", 8),
            num_features=model_cfg.get("num_features", 64),
            sam_checkpoint=model_cfg.get("sam_checkpoint", "/autodl-fs/data/masquer.li/model/sam_vit_b_01ec64.pth"),
            freeze_vit=model_cfg.get("freeze_vit", False),
            transformer_dim=model_cfg.get("transformer_dim", 128),
            transformer_heads=model_cfg.get("transformer_heads", 4),
            transformer_pooled_size=model_cfg.get("transformer_pooled_size", 28),
            transformer_num_layers=model_cfg.get("transformer_num_layers", 1),
            transformer_use_pos_embed=model_cfg.get("transformer_use_pos_embed", True),
        )
    else:
        model = TTTSNet(
            classes=model_cfg.get("classes", 2),
            block_1=model_cfg.get("block_1", 3),
            block_2=model_cfg.get("block_2", 8),
            num_features=model_cfg.get("num_features", 64),
        )
    return model.to(device)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)


def list_pairs(dataset_path: Path) -> List[Tuple[Path, Path]]:
    image_dir = dataset_path / "images"
    label_dir = dataset_path / "labels"
    pairs = []
    for image_path in sorted(image_dir.glob("*.jpg")):
        label_path = label_dir / f"{image_path.stem}.png"
        if label_path.exists():
            pairs.append((image_path, label_path))
    return pairs


def preprocess_image(image_path: Path, img_size: int, device: torch.device) -> Tuple[torch.Tensor, Tuple[int, int]]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_size = image.shape[:2]
    image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    return tensor, orig_size


def load_mask(label_path: Path) -> np.ndarray:
    mask = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read label: {label_path}")
    return (mask > 0).astype(np.uint8)


def binary_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    pred_sum = pred.sum()
    gt_sum = gt.sum()
    eps = 1e-7
    return {
        "miou": float((inter + eps) / (union + eps)),
        "dice": float((2 * inter + eps) / (pred_sum + gt_sum + eps)),
        "pixel_acc": float((pred == gt).mean()),
    }


def cldice_metric(pred: np.ndarray, gt: np.ndarray, iterations: int = 10) -> float:
    pred_t = torch.from_numpy(pred[None, None].astype(np.float32))
    gt_t = torch.from_numpy(gt[None, None].astype(np.float32))
    pred_skel = soft_skeletonize(pred_t, iterations=iterations)
    gt_skel = soft_skeletonize(gt_t, iterations=iterations)
    smooth = 1.0
    tprec = ((pred_skel * gt_t).sum() + smooth) / (pred_skel.sum() + smooth)
    tsens = ((gt_skel * pred_t).sum() + smooth) / (gt_skel.sum() + smooth)
    return float((2.0 * tprec * tsens + smooth) / (tprec + tsens + smooth))


def surface_dice(pred: np.ndarray, gt: np.ndarray, tolerance: int = 2) -> float:
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    pred_boundary = pred - cv2.erode(pred, kernel)
    gt_boundary = gt - cv2.erode(gt, kernel)
    if pred_boundary.sum() == 0 and gt_boundary.sum() == 0:
        return 1.0
    if pred_boundary.sum() == 0 or gt_boundary.sum() == 0:
        return 0.0

    dist_to_gt = cv2.distanceTransform(1 - gt_boundary, cv2.DIST_L2, 3)
    dist_to_pred = cv2.distanceTransform(1 - pred_boundary, cv2.DIST_L2, 3)
    pred_match = (dist_to_gt[pred_boundary > 0] <= tolerance).sum()
    gt_match = (dist_to_pred[gt_boundary > 0] <= tolerance).sum()
    return float((pred_match + gt_match) / (pred_boundary.sum() + gt_boundary.sum()))


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    img_size = cfg["dataset_config"].get("img_size", 448)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    model = create_model(cfg, device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    pairs = list_pairs(Path(args.dataset_path))
    rows = []
    for image_path, label_path in tqdm(pairs, desc="Evaluating SFY"):
        image, orig_size = preprocess_image(image_path, img_size, device)
        gt = load_mask(label_path)
        logits = model(image)
        prob = torch.softmax(logits, dim=1)[:, 1:2]
        prob = F.interpolate(prob, size=orig_size, mode="bilinear", align_corners=False)
        pred = (prob.squeeze().cpu().numpy() > args.threshold).astype(np.uint8)

        metrics = binary_metrics(pred, gt)
        metrics["cldice"] = cldice_metric(pred, gt, iterations=args.cldice_iterations)
        metrics["surface_dice_2px"] = surface_dice(pred, gt, tolerance=2)
        metrics["surface_dice_5px"] = surface_dice(pred, gt, tolerance=5)
        rows.append({"image": image_path.name, **metrics})

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image", "miou", "dice", "pixel_acc", "cldice", "surface_dice_2px", "surface_dice_5px"]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {key: float(np.mean([row[key] for row in rows])) for key in fieldnames if key != "image"}
    summary["n"] = len(rows)
    with open(output_csv.with_suffix(".summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved per-image results to: {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TTTSNet models on SFY 0923.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_path", default="/autodl-fs/data/masquer.li/SFY_Training_Dataset/0923")
    parser.add_argument("--output_csv", default="experiments/sfy_generalization_results.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cldice_iterations", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
