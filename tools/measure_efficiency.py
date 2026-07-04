#!/usr/bin/env python3
"""Measure parameter count, optional FLOPs, and inference speed."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.TTTSNet import TTTSNet
from models.tttsnet_vit import TTTSNetViT
from models.tttsnet_transformer_decoder import TTTSNetTransformerDecoder, TTTSNetViTTransformerDecoder


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
    return model.to(device).eval()


def measure_latency(model: torch.nn.Module, x: torch.Tensor, warmup: int, iterations: int) -> Dict[str, float]:
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if x.is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            _ = model(x)
        if x.is_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    ms = elapsed * 1000.0 / iterations
    return {"latency_ms": ms, "fps": 1000.0 / ms}


def try_flops(model: torch.nn.Module, x: torch.Tensor):
    try:
        from thop import profile
    except ImportError:
        return None
    flops, _ = profile(model, inputs=(x,), verbose=False)
    return flops


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure TTTSNet model efficiency.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    img_size = cfg["dataset_config"].get("img_size", 448)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = create_model(cfg, device)
    x = torch.randn(args.batch_size, 3, img_size, img_size, device=device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    latency = measure_latency(model, x, args.warmup, args.iterations)
    flops = try_flops(model, x)

    result = {
        "config": args.config,
        "device": str(device),
        "batch_size": args.batch_size,
        "img_size": img_size,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "flops": flops,
        **latency,
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
