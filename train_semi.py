#!/usr/bin/env python3
"""
TTTSNet 半监督训练脚本
使用有标注数据（FetReg）+ 伪标签数据（无标注视频）训练
伪标签由训练好的 baseline 模型生成
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.TTTSNet import TTTSNet
from dataset_tttsnet import TTTSNetDataset
from dataset_semi import TTTSNetSemiDataset
from utils.losses import DiceLoss
from utils.metrics_binary import calc_miou_and_dice, calc_pixel_accuracy
from utils.tracker import ExperimentTracker


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def semi_collate_fn(batch):
    """batch 元素: (image, mask, weight)"""
    images = torch.stack([item[0] for item in batch], dim=0)
    masks = torch.stack([item[1] for item in batch], dim=0)
    weights = torch.tensor([item[2] for item in batch], dtype=torch.float32)
    return images, masks, weights


def create_dataloaders(cfg: Dict[str, Any]):
    ds_cfg = cfg["dataset_config"]
    train_cfg = cfg["training_config"]

    labeled_paths = ds_cfg.get("train_paths", [])
    pseudo_path = ds_cfg.get("pseudo_data_path", "")
    val_paths = ds_cfg.get("val_paths", [])

    train_dataset = TTTSNetSemiDataset(
        labeled_data_paths=labeled_paths,
        pseudo_data_path=pseudo_path,
        mode="train",
        img_size=ds_cfg.get("img_size", 448),
        binary=ds_cfg.get("binary", True),
    )

    if len(val_paths) == 1:
        val_dataset = TTTSNetDataset(
            data_path=val_paths[0],
            mode="valid",
            img_size=ds_cfg.get("img_size", 448),
            binary=ds_cfg.get("binary", True),
        )
    else:
        from torch.utils.data import ConcatDataset
        val_dataset = ConcatDataset([
            TTTSNetDataset(p, mode="valid", img_size=ds_cfg.get("img_size", 448),
                           binary=ds_cfg.get("binary", True))
            for p in val_paths
        ])

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
        collate_fn=semi_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, train_dataset.get_stats()


def create_model(cfg: Dict[str, Any], device: torch.device):
    model_cfg = cfg["model_config"]
    model = TTTSNet(
        classes=model_cfg.get("classes", 2),
        block_1=model_cfg.get("block_1", 3),
        block_2=model_cfg.get("block_2", 8),
        num_features=model_cfg.get("num_features", 64),
    )
    model = model.to(device)
    return model


def create_optimizer(model: nn.Module, cfg: Dict[str, Any]):
    train_cfg = cfg["training_config"]
    return AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg.get("weight_decay", 0.01))


def create_scheduler(optimizer: torch.optim.Optimizer, cfg: Dict[str, Any]):
    train_cfg = cfg["training_config"]
    scheduler_type = train_cfg.get("scheduler", "cosine")
    if scheduler_type == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=train_cfg.get("scheduler_t_max", train_cfg["num_epochs"]),
            eta_min=train_cfg.get("eta_min", 1e-7),
        )
    else:
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        return ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)


def create_losses(cfg: Dict[str, Any], device: torch.device):
    train_cfg = cfg["training_config"]
    dice_weight = train_cfg.get("loss_dice_weight", 1.0)
    bce_weight = train_cfg.get("loss_bce_weight", 1.0)

    dice_loss = DiceLoss(mode="binary", from_logits=True).to(device)
    bce_loss = nn.BCEWithLogitsLoss(reduction="none").to(device)
    ce_loss = nn.CrossEntropyLoss(reduction="none").to(device)

    return dice_loss, bce_loss, ce_loss, dice_weight, bce_weight


def compute_loss(pred: torch.Tensor, target: torch.Tensor, sample_weights: torch.Tensor,
                 dice_loss, bce_loss, ce_loss, dice_weight: float, bce_weight: float):
    """sample_weights: [B] per-sample weight for labeled/pseudo"""
    if target.dim() == 4:
        target = target.squeeze(1)

    B = pred.size(0)
    vessel_logit = pred[:, 1:2, :, :]

    loss_dice = dice_loss(vessel_logit, target.unsqueeze(1).float())

    # BCE with per-sample weighting
    bce_per_pixel = bce_loss(vessel_logit, target.unsqueeze(1).float())  # [B, 1, H, W]
    bce_per_sample = bce_per_pixel.view(B, -1).mean(dim=1)  # [B]
    loss_bce = (bce_per_sample * sample_weights).mean()

    # CE with per-sample weighting
    ce_per_pixel = ce_loss(pred, target.long())  # [B, H, W]
    ce_per_sample = ce_per_pixel.view(B, -1).mean(dim=1)  # [B]
    loss_ce = (ce_per_sample * sample_weights).mean()

    total_loss = dice_weight * loss_dice + bce_weight * loss_bce + 0.5 * loss_ce

    return total_loss, {
        "loss_dice": loss_dice.item(),
        "loss_bce": loss_bce.item(),
        "loss_ce": loss_ce.item(),
    }


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    dice_loss, bce_loss, ce_loss, dice_weight: float, bce_weight: float,
                    device: torch.device, scaler: GradScaler, epoch: int, tracker: ExperimentTracker,
                    grad_accum: int = 1, use_amp: bool = True):
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    total_bce = 0.0
    total_ce = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc=f"Train Semi Epoch {epoch}")
    for batch_idx, (images, masks, weights) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            outputs = model(images)
            loss, loss_dict = compute_loss(
                outputs, masks, weights, dice_loss, bce_loss, ce_loss, dice_weight, bce_weight
            )
            loss = loss / grad_accum

        scaler.scale(loss).backward()

        if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum
        total_dice += loss_dict["loss_dice"]
        total_bce += loss_dict["loss_bce"]
        total_ce += loss_dict["loss_ce"]
        num_batches += 1

        global_step = epoch * len(loader) + batch_idx
        tracker.log_step(global_step, epoch, {
            "train/loss": loss.item() * grad_accum,
            "train/loss_dice": loss_dict["loss_dice"],
            "train/loss_bce": loss_dict["loss_bce"],
            "train/loss_ce": loss_dict["loss_ce"],
        })

        pbar.set_postfix({
            "loss": f"{loss.item() * grad_accum:.4f}",
            "dice": f"{loss_dict['loss_dice']:.4f}",
            "bce": f"{loss_dict['loss_bce']:.4f}",
        })

    return {
        "train/loss": total_loss / max(num_batches, 1),
        "train/loss_dice": total_dice / max(num_batches, 1),
        "train/loss_bce": total_bce / max(num_batches, 1),
        "train/loss_ce": total_ce / max(num_batches, 1),
    }


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device,
             dice_loss, bce_loss, ce_loss, dice_weight: float, bce_weight: float,
             use_amp: bool = True):
    model.eval()
    total_loss = 0.0
    total_miou = 0.0
    total_dice = 0.0
    total_acc = 0.0
    num_batches = 0
    num_valid = 0

    for images, masks in tqdm(loader, desc="Validate"):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            outputs = model(images)
            # validation 时用 weight=1 计算 loss（仅用于监控）
            dummy_weights = torch.ones(outputs.size(0), device=device)
            loss, _ = compute_loss(outputs, masks, dummy_weights, dice_loss, bce_loss, ce_loss, dice_weight, bce_weight)

        vessel_prob = torch.softmax(outputs, dim=1)[:, 1:2, :, :]
        miou, dice = calc_miou_and_dice(vessel_prob, masks)
        acc = calc_pixel_accuracy(vessel_prob, masks)

        if not np.isnan(miou):
            total_miou += miou
            total_dice += dice
            num_valid += 1

        total_loss += loss.item()
        total_acc += acc
        num_batches += 1

    return {
        "val/loss": total_loss / max(num_batches, 1),
        "val/miou": total_miou / max(num_valid, 1) if num_valid > 0 else 0.0,
        "val/dice": total_dice / max(num_valid, 1) if num_valid > 0 else 0.0,
        "val/pixel_acc": total_acc / max(num_batches, 1),
    }


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler._LRScheduler,
                    epoch: int, metrics: Dict[str, float], path: Path, is_best: bool = False):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    if is_best:
        best_path = path.parent / "best_model.pth"
        torch.save(checkpoint, best_path)


def main():
    parser = argparse.ArgumentParser(description="TTTSNet semi-supervised training")
    parser.add_argument("--config", type=str, default="config_semi.json", help="Path to config JSON")
    parser.add_argument("--work_dir", type=str, default=None, help="Experiment output directory")
    parser.add_argument("--num_epochs", type=int, default=None, help="Override num_epochs")
    parser.add_argument("--debug", type=int, default=None, help="Override debug mode")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg["training_config"]
    runtime_cfg = cfg["runtime_config"]

    num_epochs = args.num_epochs if args.num_epochs is not None else train_cfg["num_epochs"]
    debug = args.debug if args.debug is not None else runtime_cfg.get("debug", 0)

    seed = train_cfg.get("seed", 42)
    set_seed(seed)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{runtime_cfg.get('run_name', 'tttsnet_semi')}_{timestamp}"
    if args.work_dir:
        exp_dir = Path(args.work_dir) / run_name
    else:
        exp_dir = Path(runtime_cfg.get("work_dir", "experiments")) / run_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    tracker = ExperimentTracker(
        experiment_dir=str(exp_dir),
        experiment_name=run_name,
        config=cfg,
        use_tensorboard=True,
        use_csv=True,
        use_json=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, data_stats = create_dataloaders(cfg)
    print(f"Data stats: {data_stats}")
    print(f"Val samples: {len(val_loader.dataset)}")

    model = create_model(cfg, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")

    optimizer = create_optimizer(model, cfg)
    scheduler = create_scheduler(optimizer, cfg)
    dice_loss, bce_loss, ce_loss, dice_weight, bce_weight = create_losses(cfg, device)

    use_amp = bool(train_cfg.get("use_amp", 1))
    scaler = GradScaler(enabled=use_amp)

    tracker.log_lr(optimizer, step=0)

    best_miou = 0.0
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer,
            dice_loss, bce_loss, ce_loss, dice_weight, bce_weight,
            device, scaler, epoch, tracker,
            grad_accum=train_cfg.get("gradient_accumulation_steps", 1),
            use_amp=use_amp,
        )

        val_metrics = validate(
            model, val_loader, device,
            dice_loss, bce_loss, ce_loss, dice_weight, bce_weight,
            use_amp=use_amp,
        )

        epoch_time = time.time() - epoch_start

        if isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        else:
            scheduler.step(val_metrics["val/miou"])

        epoch_metrics = {**train_metrics, **val_metrics, "epoch/time_s": epoch_time}
        tracker.log_epoch(epoch, epoch_metrics)
        tracker.log_lr(optimizer, step=epoch)

        print(f"Epoch {epoch}/{num_epochs} | Time: {epoch_time:.1f}s | "
              f"Train Loss: {train_metrics['train/loss']:.4f} | "
              f"Val mIoU: {val_metrics['val/miou']:.4f} | Val Dice: {val_metrics['val/dice']:.4f}")

        if debug == 0:
            save_freq = runtime_cfg.get("save_frequency", 10)
            if epoch % save_freq == 0 or epoch == num_epochs:
                save_checkpoint(
                    model, optimizer, scheduler, epoch, epoch_metrics,
                    exp_dir / "checkpoints" / f"model_epoch_{epoch:03d}.pth"
                )

            if val_metrics["val/miou"] > best_miou:
                best_miou = val_metrics["val/miou"]
                save_checkpoint(
                    model, optimizer, scheduler, epoch, epoch_metrics,
                    exp_dir / "checkpoints" / "best_model.pth",
                    is_best=True
                )
                print(f"  *** New best mIoU: {best_miou:.4f}")

        tracker.flush()

    total_time = time.time() - start_time
    final_metrics = {
        "best_val_miou": best_miou,
        "total_time_h": total_time / 3600,
    }
    tracker.summarize(final_metrics)
    tracker.close()

    print(f"\nTraining completed. Best val mIoU: {best_miou:.4f}")
    print(f"Experiment directory: {exp_dir}")


if __name__ == "__main__":
    main()
