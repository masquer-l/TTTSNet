#!/usr/bin/env python3
"""
为无标注视频生成伪标签
使用训练好的 TTTSNet 模型对无标注帧进行推理，保存置信度高于阈值的二值伪标签
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.TTTSNet import TTTSNet


class UnlabeledVideoDataset(Dataset):
    """无标注视频帧数据集"""

    def __init__(self, data_path: str, img_size: int = 448):
        self.data_path = data_path
        self.img_size = img_size
        self.image_paths = sorted(self._find_images(data_path))

    def _find_images(self, data_path: str):
        """支持 .png 和 .jpg"""
        paths = []
        for ext in ["png", "jpg", "jpeg"]:
            paths.extend(glob.glob(os.path.join(data_path, f"**/*.{ext}"), recursive=True))
        return paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法读取图像: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
        return img_tensor, img_path


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(checkpoint_path: str, cfg: Dict[str, Any], device: torch.device):
    model_cfg = cfg["model_config"]
    model = TTTSNet(
        classes=model_cfg.get("classes", 2),
        block_1=model_cfg.get("block_1", 3),
        block_2=model_cfg.get("block_2", 8),
        num_features=model_cfg.get("num_features", 64),
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def normalize_image(img: torch.Tensor) -> torch.Tensor:
    """ImageNet normalization"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img.device)
    return (img - mean) / std


@torch.no_grad()
def generate_pseudo_labels(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: str,
    confidence_threshold: float = 0.9,
):
    """生成伪标签并保存"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    total_count = 0
    high_conf_count = 0

    for images, img_paths in tqdm(loader, desc="Generating pseudo labels"):
        images = images.to(device, non_blocking=True)
        images = normalize_image(images)

        outputs = model(images)  # [B, 2, H, W]
        probs = F.softmax(outputs, dim=1)[:, 1, :, :]  # [B, H, W]

        for i in range(len(img_paths)):
            total_count += 1
            prob = probs[i]
            max_conf = prob.max().item()

            if max_conf >= confidence_threshold:
                high_conf_count += 1
                pred_mask = (prob > 0.5).cpu().numpy().astype(np.uint8) * 255

                # 保存到与原路径对应的输出目录
                rel_path = os.path.relpath(img_paths[i], "/autodl-fs/data/masquer.li/temperal_data/sfy_data_v1_20251019/")
                out_path = output_dir / rel_path.replace("/images/", "/pseudo_labels/").replace(".jpg", ".png")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_path), pred_mask)
                saved_count += 1

    print(f"\nPseudo-label generation completed:")
    print(f"  Total frames: {total_count}")
    print(f"  High confidence (>={confidence_threshold}): {high_conf_count} ({high_conf_count/total_count*100:.1f}%)")
    print(f"  Saved pseudo labels: {saved_count}")
    print(f"  Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate pseudo labels for unlabeled video frames")
    parser.add_argument("--config", type=str, default="configs/config.json", help="Path to config JSON")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--unlabeled_path", type=str,
                        default="/autodl-fs/data/masquer.li/temperal_data/sfy_data_v1_20251019/",
                        help="Path to unlabeled video data")
    parser.add_argument("--output_dir", type=str, default="pseudo_labels/sfy_data_v1",
                        help="Output directory for pseudo labels")
    parser.add_argument("--confidence_threshold", type=float, default=0.9,
                        help="Confidence threshold for saving pseudo labels")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, cfg, device)

    dataset = UnlabeledVideoDataset(args.unlabeled_path, img_size=cfg["dataset_config"].get("img_size", 448))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    generate_pseudo_labels(
        model, loader, device,
        output_dir=args.output_dir,
        confidence_threshold=args.confidence_threshold,
    )


if __name__ == "__main__":
    import glob
    main()
