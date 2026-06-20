#!/usr/bin/env python3
"""
TTTSNet 时序一致性训练脚本
训练：3 帧片段输入，共享模型权重，中间帧 GT 监督 + 相邻帧时序一致性约束
推理：单帧输入（与 baseline 相同）
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.TTTSNet import TTTSNet
from dataset_tttsnet import TTTSNetDataset
from dataset_temporal import TTTSNetTemporalDataset
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


def create_dataloaders(cfg: Dict[str, Any]):
    """创建时序训练集和单帧验证集"""
    ds_cfg = cfg["dataset_config"]
    aug_cfg = cfg.get("augmentation_config", {})
    train_cfg = cfg["training_config"]

    train_paths = ds_cfg.get("train_paths", [])
    val_paths = ds_cfg.get("val_paths", [])
    use_strong_aug = aug_cfg.get("use_strong_aug", False)

    if len(train_paths) == 1:
        train_dataset = TTTSNetTemporalDataset(
            data_path=train_paths[0],
            mode="train",
            img_size=ds_cfg.get("img_size", 448),
            binary=ds_cfg.get("binary", True),
            use_strong_aug=use_strong_aug,
        )
    else:
        from torch.utils.data import ConcatDataset
        train_dataset = ConcatDataset([
            TTTSNetTemporalDataset(p, mode="train", img_size=ds_cfg.get("img_size", 448),
                                   binary=ds_cfg.get("binary", True),
                                   use_strong_aug=use_strong_aug)
            for p in train_paths
        ])

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
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


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
    temporal_weight = train_cfg.get("loss_temporal_weight", 0.1)

    dice_loss = DiceLoss(mode="binary", from_logits=True).to(device)
    bce_loss = nn.BCEWithLogitsLoss().to(device)
    ce_loss = nn.CrossEntropyLoss().to(device)

    return dice_loss, bce_loss, ce_loss, dice_weight, bce_weight, temporal_weight


def temporal_consistency_loss(pred_probs: torch.Tensor, lambda_temp: float = 0.1):
    """
    pred_probs: [B, 3, 1, H, W] vessel probabilities for 3 frames
    """
    p0 = pred_probs[:, 0]
    p1 = pred_probs[:, 1]
    p2 = pred_probs[:, 2]

    # uncertainty-based weight: high near 0.5, low near 0/1
    conf0 = p0 * (1 - p0) * 4
    conf1 = p1 * (1 - p1) * 4
    conf2 = p2 * (1 - p2) * 4

    w01 = torch.minimum(conf0, conf1)
    w12 = torch.minimum(conf1, conf2)

    diff01 = (p0 - p1).abs()
    diff12 = (p1 - p2).abs()

    loss = (w01 * diff01).mean() + (w12 * diff12).mean()
    return lambda_temp * loss


def compute_supervision_loss(pred: torch.Tensor, target: torch.Tensor,
                             dice_loss, bce_loss, ce_loss,
                             dice_weight: float, bce_weight: float):
    """单帧监督 loss（与 train.py 相同）"""
    if target.dim() == 4:
        target = target.squeeze(1)

    vessel_logit = pred[:, 1:2, :, :]
    loss_dice = dice_loss(vessel_logit, target.unsqueeze(1).float())
    loss_bce = bce_loss(vessel_logit, target.unsqueeze(1).float())
    loss_ce = ce_loss(pred, target.long())

    total = dice_weight * loss_dice + bce_weight * loss_bce + 0.5 * loss_ce
    return total, {
        "loss_dice": loss_dice.item(),
        "loss_bce": loss_bce.item(),
        "loss_ce": loss_ce.item(),
    }


def forward_temporal(model: nn.Module, clip: torch.Tensor):
    """
    clip: [B, 3, 3, H, W]
    return: [B, 3, 2, H, W] logits
    """
    B, T, C, H, W = clip.shape
    clip_flat = clip.view(B * T, C, H, W)
    logits_flat = model(clip_flat)  # [B*T, 2, H, W]
    logits = logits_flat.view(B, T, 2, H, W)
    return logits


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    dice_loss, bce_loss, ce_loss, dice_weight: float, bce_weight: float,
                    temporal_weight: float, device: torch.device, scaler: GradScaler,
                    epoch: int, tracker: ExperimentTracker, grad_accum: int = 1,
                    use_amp: bool = True):
    model.train()
    total_loss = 0.0
    total_sup = 0.0
    total_temp = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc=f"Train Temporal Epoch {epoch}")
    for batch_idx, (clips, masks) in enumerate(pbar):
        clips = clips.to(device, non_blocking=True)  # [B, 3, 3, H, W]
        masks = masks.to(device, non_blocking=True)  # [B, 1, H, W]

        with autocast(enabled=use_amp):
            logits = forward_temporal(model, clips)  # [B, 3, 2, H, W]
            mid_logits = logits[:, 1, :, :, :]  # [B, 2, H, W]

            sup_loss, loss_dict = compute_supervision_loss(
                mid_logits, masks, dice_loss, bce_loss, ce_loss, dice_weight, bce_weight
            )

            # 时序一致性 loss
            probs = torch.softmax(logits, dim=2)[:, :, 1:2, :, :]  # [B, 3, 1, H, W]
            temp_loss = temporal_consistency_loss(probs, temporal_weight)

            loss = sup_loss + temp_loss
            loss = loss / grad_accum

        scaler.scale(loss).backward()

        if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum
        total_sup += sup_loss.item()
        total_temp += temp_loss.item()
        num_batches += 1

        global_step = epoch * len(loader) + batch_idx
        tracker.log_step(global_step, epoch, {
            "train/loss": loss.item() * grad_accum,
            "train/loss_sup": sup_loss.item(),
            "train/loss_temporal": temp_loss.item(),
            "train/loss_dice": loss_dict["loss_dice"],
            "train/loss_bce": loss_dict["loss_bce"],
            "train/loss_ce": loss_dict["loss_ce"],
        })

        pbar.set_postfix({
            "loss": f"{loss.item() * grad_accum:.4f}",
            "sup": f"{sup_loss.item():.4f}",
            "temp": f"{temp_loss.item():.4f}",
        })

    return {
        "train/loss": total_loss / max(num_batches, 1),
        "train/loss_sup": total_sup / max(num_batches, 1),
        "train/loss_temporal": total_temp / max(num_batches, 1),
    }


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device,
             dice_loss, bce_loss, ce_loss, dice_weight: float, bce_weight: float,
             use_amp: bool = True):
    """单帧验证，与 baseline 可比"""
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
            loss, _ = compute_supervision_loss(outputs, masks, dice_loss, bce_loss, ce_loss, dice_weight, bce_weight)

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
    parser = argparse.ArgumentParser(description="TTTSNet temporal consistency training")
    parser.add_argument("--config", type=str, default="configs/config_temporal.json", help="Path to config JSON")
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
    run_name = f"{runtime_cfg.get('run_name', 'tttsnet_temporal')}_{timestamp}"
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

    train_loader, val_loader = create_dataloaders(cfg)
    print(f"Train clips: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    model = create_model(cfg, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")

    optimizer = create_optimizer(model, cfg)
    scheduler = create_scheduler(optimizer, cfg)
    dice_loss, bce_loss, ce_loss, dice_weight, bce_weight, temporal_weight = create_losses(cfg, device)

    use_amp = bool(train_cfg.get("use_amp", 1))
    scaler = GradScaler(enabled=use_amp)

    tracker.log_lr(optimizer, step=0)

    best_miou = 0.0
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer,
            dice_loss, bce_loss, ce_loss, dice_weight, bce_weight, temporal_weight,
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
