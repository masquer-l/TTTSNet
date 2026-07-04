#!/usr/bin/env python3
"""
TTTSNet 单帧全监督基线训练脚本
复现原论文配置：custom TTTSNet 模型、448×448 输入、血管二分类、custom augmentations。
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

# 添加项目根目录和 src 到路径（兼容 data_loader.py 的 utils 导入）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.TTTSNet import TTTSNet
from models.tttsnet_vit import TTTSNetViT
from models.tttsnet_transformer_decoder import TTTSNetTransformerDecoder, TTTSNetViTTransformerDecoder
from dataset_tttsnet import TTTSNetDataset
from utils.losses import DiceLoss, AsymmetricUnifiedFocalLoss, SoftCLDiceLoss
from utils.metrics_binary import calc_miou_and_dice, calc_pixel_accuracy
from utils.tracker import ExperimentTracker


def set_seed(seed: int):
    """设置随机种子以保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TeeLogger:
    """同时将输出写入文件和控制台的日志重定向器。"""

    def __init__(self, filepath: Path, mode: str = "a"):
        self.terminal = sys.stdout
        self.log = open(filepath, mode, buffering=1, encoding="utf-8")

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_dataloaders(cfg: Dict[str, Any], aug_cfg: Dict[str, float]):
    """创建训练/验证 DataLoader"""
    ds_cfg = cfg["dataset_config"]
    train_cfg = cfg["training_config"]

    train_paths = ds_cfg.get("train_paths", [])
    val_paths = ds_cfg.get("val_paths", [])

    if len(train_paths) == 1:
        train_dataset = TTTSNetDataset(
            data_path=train_paths[0],
            mode="train",
            img_size=ds_cfg.get("img_size", 448),
            binary=ds_cfg.get("binary", True),
        )
    else:
        from torch.utils.data import ConcatDataset
        train_dataset = ConcatDataset([
            TTTSNetDataset(p, mode="train", img_size=ds_cfg.get("img_size", 448),
                           binary=ds_cfg.get("binary", True))
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
    """创建 TTTSNet 或 TTTSNet-ViT 模型"""
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
    model = model.to(device)
    return model


def create_optimizer(model: nn.Module, cfg: Dict[str, Any]):
    """创建 AdamW 优化器，支持按参数组设置不同学习率"""
    train_cfg = cfg["training_config"]
    lr = train_cfg["lr"]
    weight_decay = train_cfg.get("weight_decay", 0.01)

    # 支持 backbone 与 decoder 分层学习率，例如 ViT encoder 使用更小 lr
    backbone_lr_multiplier = train_cfg.get("backbone_lr_multiplier", 1.0)
    backbone_param_pattern = train_cfg.get("backbone_param_pattern", "")

    if backbone_lr_multiplier != 1.0 and backbone_param_pattern:
        backbone_params = []
        other_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if backbone_param_pattern in name:
                backbone_params.append(param)
            else:
                other_params.append(param)
        param_groups = [
            {"params": backbone_params, "lr": lr * backbone_lr_multiplier},
            {"params": other_params, "lr": lr},
        ]
        print(
            f"[Optimizer] backbone params ({backbone_param_pattern}): "
            f"{len(backbone_params)} groups with lr={lr * backbone_lr_multiplier}; "
            f"other params: {len(other_params)} groups with lr={lr}"
        )
        optimizer = AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, cfg: Dict[str, Any]):
    """创建学习率调度器"""
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
    """创建损失函数"""
    train_cfg = cfg["training_config"]
    loss_type = train_cfg.get("loss_type", "default")
    dice_weight = train_cfg.get("loss_dice_weight", 1.0)
    bce_weight = train_cfg.get("loss_bce_weight", 1.0)
    cldice_weight = train_cfg.get("loss_cldice_weight", 0.0)

    if loss_type == "aufl":
        # AsymmetricUnifiedFocalLoss，适合类别不平衡的细血管分割
        auft_loss = AsymmetricUnifiedFocalLoss(
            weight=train_cfg.get("aufl_weight", 0.5),
            delta=train_cfg.get("aufl_delta", 0.6),
            gamma=train_cfg.get("aufl_gamma", 0.2),
            n_classes=2,
        ).to(device)
        cldice_loss = SoftCLDiceLoss(
            iterations=train_cfg.get("cldice_iterations", 10)
        ).to(device) if cldice_weight > 0 else None
        return auft_loss, None, None, cldice_loss, dice_weight, bce_weight, cldice_weight

    dice_loss = DiceLoss(mode="binary", from_logits=True).to(device)
    bce_loss = nn.BCEWithLogitsLoss().to(device)
    ce_loss = nn.CrossEntropyLoss().to(device)

    cldice_loss = SoftCLDiceLoss(
        iterations=train_cfg.get("cldice_iterations", 10)
    ).to(device) if cldice_weight > 0 else None

    return dice_loss, bce_loss, ce_loss, cldice_loss, dice_weight, bce_weight, cldice_weight


def compute_loss(pred: torch.Tensor, target: torch.Tensor,
                 dice_loss, bce_loss, ce_loss, cldice_loss,
                 dice_weight: float, bce_weight: float, cldice_weight: float,
                 loss_type: str = "default"):
    """
    pred: [B, 2, H, W] logits (binary class: background + vessel)
    target: [B, 1, H, W] or [B, H, W] with values 0/1
    """
    if target.dim() == 4:
        target = target.squeeze(1)

    if loss_type == "aufl" and dice_loss is not None:
        # AUFL 内部自己做激活和 one-hot
        loss = dice_loss(pred, target.long())
        loss_cldice = torch.tensor(0.0, device=pred.device)
        if cldice_loss is not None and cldice_weight > 0:
            vessel_prob = torch.softmax(pred, dim=1)[:, 1:2, :, :]
            loss_cldice = cldice_loss(vessel_prob, target.unsqueeze(1).float())
            loss = loss + cldice_weight * loss_cldice
        return loss, {
            "loss_dice": 0.0,
            "loss_bce": 0.0,
            "loss_ce": loss.item(),
            "loss_cldice": loss_cldice.item(),
        }

    # 提取 vessel logit [B, 1, H, W]
    vessel_logit = pred[:, 1:2, :, :]

    # Dice loss (binary, on vessel channel)
    loss_dice = dice_loss(vessel_logit, target.unsqueeze(1).float())

    # BCE loss (on vessel channel)
    loss_bce = bce_loss(vessel_logit, target.unsqueeze(1).float())

    # 可选：CrossEntropy loss 作为正则
    loss_ce = ce_loss(pred, target.long())
    loss_cldice = torch.tensor(0.0, device=pred.device)
    if cldice_loss is not None and cldice_weight > 0:
        vessel_prob = torch.sigmoid(vessel_logit)
        loss_cldice = cldice_loss(vessel_prob, target.unsqueeze(1).float())

    total_loss = (
        dice_weight * loss_dice
        + bce_weight * loss_bce
        + 0.5 * loss_ce
        + cldice_weight * loss_cldice
    )

    return total_loss, {
        "loss_dice": loss_dice.item(),
        "loss_bce": loss_bce.item(),
        "loss_ce": loss_ce.item(),
        "loss_cldice": loss_cldice.item(),
    }


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    dice_loss, bce_loss, ce_loss, cldice_loss,
                    dice_weight: float, bce_weight: float, cldice_weight: float,
                    device: torch.device, scaler: GradScaler, epoch: int, tracker: ExperimentTracker,
                    grad_accum: int = 1, use_amp: bool = True, loss_type: str = "default"):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    total_bce = 0.0
    total_ce = 0.0
    total_cldice = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}")
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            outputs = model(images)
            loss, loss_dict = compute_loss(
                outputs, masks, dice_loss, bce_loss, ce_loss, cldice_loss,
                dice_weight, bce_weight, cldice_weight, loss_type=loss_type
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
        total_cldice += loss_dict["loss_cldice"]
        num_batches += 1

        # 记录每步 loss
        global_step = epoch * len(loader) + batch_idx
        tracker.log_step(global_step, epoch, {
            "train/loss": loss.item() * grad_accum,
            "train/loss_dice": loss_dict["loss_dice"],
            "train/loss_bce": loss_dict["loss_bce"],
            "train/loss_ce": loss_dict["loss_ce"],
            "train/loss_cldice": loss_dict["loss_cldice"],
        })

        pbar.set_postfix({
            "loss": f"{loss.item() * grad_accum:.4f}",
            "dice": f"{loss_dict['loss_dice']:.4f}",
            "bce": f"{loss_dict['loss_bce']:.4f}",
            "cldice": f"{loss_dict['loss_cldice']:.4f}",
        })

    avg_loss = total_loss / max(num_batches, 1)
    avg_dice = total_dice / max(num_batches, 1)
    avg_bce = total_bce / max(num_batches, 1)
    avg_ce = total_ce / max(num_batches, 1)
    avg_cldice = total_cldice / max(num_batches, 1)

    return {
        "train/loss": avg_loss,
        "train/loss_dice": avg_dice,
        "train/loss_bce": avg_bce,
        "train/loss_ce": avg_ce,
        "train/loss_cldice": avg_cldice,
    }


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device,
             dice_loss, bce_loss, ce_loss, cldice_loss,
             dice_weight: float, bce_weight: float, cldice_weight: float,
             use_amp: bool = True, loss_type: str = "default"):
    """验证一个 epoch"""
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
            loss, _ = compute_loss(
                outputs, masks, dice_loss, bce_loss, ce_loss, cldice_loss,
                dice_weight, bce_weight, cldice_weight, loss_type=loss_type
            )

        # 计算指标：取 vessel 概率
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

    avg_loss = total_loss / max(num_batches, 1)
    avg_miou = total_miou / max(num_valid, 1) if num_valid > 0 else 0.0
    avg_dice = total_dice / max(num_valid, 1) if num_valid > 0 else 0.0
    avg_acc = total_acc / max(num_batches, 1)

    return {
        "val/loss": avg_loss,
        "val/miou": avg_miou,
        "val/dice": avg_dice,
        "val/pixel_acc": avg_acc,
    }


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler._LRScheduler,
                    epoch: int, metrics: Dict[str, float], path: Path, is_best: bool = False):
    """保存 checkpoint"""
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
    parser = argparse.ArgumentParser(description="TTTSNet single-frame baseline training")
    parser.add_argument("--config", type=str, default="configs/config.json", help="Path to config JSON")
    parser.add_argument("--work_dir", type=str, default=None, help="Experiment output directory")
    parser.add_argument("--num_epochs", type=int, default=None, help="Override num_epochs")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--debug", type=int, default=None, help="Override debug mode (1=debug)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg["training_config"]
    runtime_cfg = cfg["runtime_config"]

    # 命令行覆盖
    num_epochs = args.num_epochs if args.num_epochs is not None else train_cfg["num_epochs"]
    seed = args.seed if args.seed is not None else train_cfg.get("seed", 42)
    debug = args.debug if args.debug is not None else runtime_cfg.get("debug", 0)

    # 设置随机种子
    set_seed(seed)

    # 创建实验目录
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{runtime_cfg.get('run_name', 'tttsnet')}_{timestamp}"
    if args.work_dir:
        exp_dir = Path(args.work_dir) / run_name
    else:
        exp_dir = Path(runtime_cfg.get("work_dir", "experiments")) / run_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 将 stdout/stderr 同时重定向到实验目录的 training.log
    tee = TeeLogger(exp_dir / "training.log", mode="a")
    sys.stdout = tee
    sys.stderr = tee

    # 保存配置副本
    with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    # 保存版本号文件（如配置中提供）
    version = runtime_cfg.get("version")
    exp_id = runtime_cfg.get("exp_id")
    if version or exp_id:
        version_file = exp_dir / "VERSION"
        with open(version_file, "w", encoding="utf-8") as f:
            if exp_id:
                f.write(f"EXP-ID: {exp_id}\n")
            if version:
                f.write(f"Model-Version: {version}\n")
            f.write(f"Seed: {seed}\n")
            f.write("Status: in progress\n")
            f.write(f"Description: {run_name}\n")

    # 初始化实验追踪
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

    # 数据
    aug_cfg = cfg.get("augmentation_config", {})
    train_loader, val_loader = create_dataloaders(cfg, aug_cfg)
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    # 模型
    model = create_model(cfg, device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")

    # 优化器、调度器、损失
    optimizer = create_optimizer(model, cfg)
    scheduler = create_scheduler(optimizer, cfg)
    dice_loss, bce_loss, ce_loss, cldice_loss, dice_weight, bce_weight, cldice_weight = create_losses(cfg, device)
    loss_type = train_cfg.get("loss_type", "default")

    # AMP
    use_amp = bool(train_cfg.get("use_amp", 1))
    scaler = GradScaler(enabled=use_amp)

    # 记录学习率
    tracker.log_lr(optimizer, step=0)

    # 训练循环
    best_miou = 0.0
    best_epoch = 0
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer,
            dice_loss, bce_loss, ce_loss, cldice_loss, dice_weight, bce_weight, cldice_weight,
            device, scaler, epoch, tracker,
            grad_accum=train_cfg.get("gradient_accumulation_steps", 1),
            use_amp=use_amp, loss_type=loss_type,
        )

        val_metrics = validate(
            model, val_loader, device,
            dice_loss, bce_loss, ce_loss, cldice_loss, dice_weight, bce_weight, cldice_weight,
            use_amp=use_amp, loss_type=loss_type,
        )

        epoch_time = time.time() - epoch_start

        # 学习率调度
        if isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        else:
            scheduler.step(val_metrics["val/miou"])

        # 记录 epoch 指标
        epoch_metrics = {**train_metrics, **val_metrics, "epoch/time_s": epoch_time}
        tracker.log_epoch(epoch, epoch_metrics)
        tracker.log_lr(optimizer, step=epoch)

        print(f"Epoch {epoch}/{num_epochs} | Time: {epoch_time:.1f}s | "
              f"Train Loss: {train_metrics['train/loss']:.4f} | "
              f"Val mIoU: {val_metrics['val/miou']:.4f} | Val Dice: {val_metrics['val/dice']:.4f}")

        # 保存 checkpoint
        if debug == 0:
            save_freq = runtime_cfg.get("save_frequency", 10)
            checkpoint_dir = exp_dir / "checkpoints"
            if epoch % save_freq == 0 or epoch == num_epochs:
                save_checkpoint(
                    model, optimizer, scheduler, epoch, epoch_metrics,
                    checkpoint_dir / f"model_epoch_{epoch:03d}.pth"
                )
                # 只保留最新的一个 epoch checkpoint，其余视为无效中间状态并删除以释放磁盘
                keep_last_n = runtime_cfg.get("keep_last_n_checkpoints", 1)
                epoch_ckpts = sorted(checkpoint_dir.glob("model_epoch_*.pth"))
                for old_ckpt in epoch_ckpts[:-keep_last_n]:
                    old_ckpt.unlink()

            if val_metrics["val/miou"] > best_miou:
                best_miou = val_metrics["val/miou"]
                best_epoch = epoch
                save_checkpoint(
                    model, optimizer, scheduler, epoch, epoch_metrics,
                    checkpoint_dir / "best_model.pth",
                    is_best=True
                )
                print(f"  *** New best mIoU: {best_miou:.4f} at epoch {best_epoch}")

        tracker.flush()

    # 训练结束
    total_time = time.time() - start_time
    final_metrics = {
        "best_val_miou": best_miou,
        "best_epoch": best_epoch,
        "total_time_h": total_time / 3600,
    }
    tracker.summarize(final_metrics)
    tracker.close()

    print(f"\nTraining completed. Best val mIoU: {best_miou:.4f} at epoch {best_epoch}")
    print(f"Experiment directory: {exp_dir}")

    # 恢复 stdout/stderr
    if 'tee' in locals():
        sys.stdout = tee.terminal
        sys.stderr = tee.terminal
        tee.close()


if __name__ == "__main__":
    main()
