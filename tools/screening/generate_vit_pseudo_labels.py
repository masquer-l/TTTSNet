#!/usr/bin/env python3
"""Generate ViT TTTSNet pseudo-labels for an annotation batch.

Writes:
- masks/             binary PNG masks (foreground=255)
- labels/            LabelMe JSON with polygon shapes from mask contours
- annotation_batch.csv updated with mask_path and annotation_status='vit_pseudo'
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.TTTSNet import TTTSNet
from models.tttsnet_vit import TTTSNetViT
from models.tttsnet_transformer_decoder import TTTSNetTransformerDecoder, TTTSNetViTTransformerDecoder


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_model(cfg: dict, device: torch.device) -> torch.nn.Module:
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


def preprocess_image(image_path: Path, img_size: int, device: torch.device) -> torch.Tensor:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    return tensor


def mask_to_polygons(mask: np.ndarray, epsilon_factor: float = 0.0025) -> list[list[list[float]]]:
    """Convert a binary mask to a list of polygon point lists."""
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
        epsilon = epsilon_factor * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) < 3:
            approx = cnt
        pts = approx.reshape(-1, 2).astype(np.float32).tolist()
        polygons.append(pts)
    return polygons


def build_labelme_json(image_path: Path, image_width: int, image_height: int, polygons: list) -> dict:
    shapes = []
    for pts in polygons:
        shapes.append({
            "label": "vessel",
            "score": None,
            "points": pts,
            "group_id": None,
            "description": "",
            "difficult": False,
            "shape_type": "polygon",
            "flags": {},
            "attributes": {},
            "kie_linking": [],
            "visible": True,
        })
    return {
        "version": "2.4.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": str(image_path.name),
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width,
    }


@torch.no_grad()
def generate_pseudo_labels(
    batch_dir: Path,
    config_path: Path,
    checkpoint_path: Path,
    threshold: float = 0.5,
    device_str: str = "auto",
):
    batch_dir = Path(batch_dir)
    images_dir = batch_dir / "images"
    labels_dir = batch_dir / "labels"
    masks_dir = batch_dir / "masks"
    labels_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(str(config_path))
    img_size = cfg["dataset_config"].get("img_size", 448)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Using device: {device}")

    model = create_model(cfg, device)
    load_checkpoint(model, str(checkpoint_path), device)
    model.eval()

    image_paths = sorted(images_dir.glob("*.jpg"))
    if not image_paths:
        print(f"No images found in {images_dir}")
        return

    for image_path in tqdm(image_paths, desc="Generating ViT pseudo-labels"):
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"Warning: cannot read {image_path}")
            continue
        orig_h, orig_w = image_bgr.shape[:2]

        tensor = preprocess_image(image_path, img_size, device)
        logits = model(tensor)
        prob = torch.softmax(logits, dim=1)[:, 1:2]
        prob = F.interpolate(prob, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        pred = (prob.squeeze().cpu().numpy() > threshold).astype(np.uint8)

        mask_u8 = (pred * 255).astype(np.uint8)
        mask_path = masks_dir / f"{image_path.stem}.png"
        cv2.imwrite(str(mask_path), mask_u8)

        polygons = mask_to_polygons(pred)
        labelme = build_labelme_json(image_path, orig_w, orig_h, polygons)
        label_path = labels_dir / f"{image_path.stem}.json"
        with open(label_path, "w", encoding="utf-8") as f:
            json.dump(labelme, f, indent=2, ensure_ascii=False)

    # Update CSV
    csv_path = batch_dir / "annotation_batch.csv"
    if csv_path.exists():
        rows = []
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                row["annotation_status"] = "vit_pseudo"
                row["mask_path"] = str(masks_dir / f"{Path(row['image_path']).stem}.png")
                rows.append(row)
        if "mask_path" not in fieldnames:
            fieldnames.append("mask_path")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"Generated {len(image_paths)} pseudo-labels in {batch_dir}")
    print(f"  masks: {masks_dir}")
    print(f"  labels: {labels_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate ViT pseudo-labels for an annotation batch.")
    parser.add_argument("--batch-dir", default="/mnt/d/torch_project/dataset/sfy_screening/annotations/batch_001")
    parser.add_argument("--config", default="/mnt/d/torch_project/TTTSNet/experiments/tttsnet_vit_layerwise_lr_20260627_083345/config.json")
    parser.add_argument("--checkpoint", default="/mnt/d/torch_project/TTTSNet/experiments/tttsnet_vit_layerwise_lr_20260627_083345/checkpoints/best_model.pth")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    generate_pseudo_labels(
        batch_dir=Path(args.batch_dir),
        config_path=Path(args.config),
        checkpoint_path=Path(args.checkpoint),
        threshold=args.threshold,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
