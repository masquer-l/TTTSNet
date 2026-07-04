#!/usr/bin/env python3
"""Compute efficiency statistics for TTTSNet-style models."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
        )
    else:
        model = TTTSNet(
            classes=model_cfg.get("classes", 2),
            block_1=model_cfg.get("block_1", 3),
            block_2=model_cfg.get("block_2", 8),
            num_features=model_cfg.get("num_features", 64),
        )
    return model.to(device)


def count_flops(model: torch.nn.Module, input_tensor: torch.Tensor) -> int:
    """Estimate FLOPs using torchprofile or fvcore if available, else return -1."""
    try:
        from thop import profile
        flops, _ = profile(model, inputs=(input_tensor,), verbose=False)
        return int(flops)
    except Exception:
        pass
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, input_tensor)
        return int(flops.total())
    except Exception:
        return -1


def measure_latency(model: torch.nn.Module, input_tensor: torch.Tensor, warmup: int = 10, repeats: int = 50) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            _ = model(input_tensor)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    return float(np.median(times)) * 1000.0  # ms


def evaluate_one(config_path: str, img_size: int, device: torch.device) -> dict:
    cfg = load_config(config_path)
    model = create_model(cfg, device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
    flops = count_flops(model, dummy_input)
    latency = measure_latency(model, dummy_input)

    return {
        "model": cfg["model_config"].get("name", "TTTSNet"),
        "config": Path(config_path).name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "flops": flops,
        "latency_ms": latency,
        "img_size": img_size,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute efficiency stats for TTTSNet models")
    parser.add_argument("--output_csv", default="experiments/efficiency_statistics.csv")
    parser.add_argument("--img_size", type=int, default=448)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    configs = [
        "configs/config.json",
        "configs/config_vit_backbone.json",
        "configs/config_temporal_v3.json",
        "configs/config_transformer_decoder.json",
        "configs/config_vit_transformer_cldice.json",
        "configs/config_semi_v2.json",
    ]

    rows = []
    for config_path in configs:
        try:
            row = evaluate_one(config_path, args.img_size, device)
            rows.append(row)
            print(f"{row['model']:30s} params={row['total_params']:,} flops={row['flops']:,} latency={row['latency_ms']:.2f}ms")
        except Exception as e:
            print(f"Error processing {config_path}: {e}")
            rows.append({
                "model": Path(config_path).stem,
                "config": Path(config_path).name,
                "total_params": -1,
                "trainable_params": -1,
                "flops": -1,
                "latency_ms": -1,
                "img_size": args.img_size,
            })

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "config", "total_params", "trainable_params", "flops", "latency_ms", "img_size"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved efficiency statistics to {output_path}")


if __name__ == "__main__":
    main()
