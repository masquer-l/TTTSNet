#!/usr/bin/env python3
"""SAM + 随机点提示训练脚本（从 TTTS_SAM 拷贝到 TTTSNet）

保持与 TTTS_SAM 随机点实验一致：
- model_type: vit_b
- 输入 1024×1024
- 随机点提示（整张图内随机，label 全 1）
- Loss: DiceCELoss + 0.1 * IoU MSE loss

使用 image_encoder + prompt_encoder + mask_decoder 的直接调用方式，
与 TTTS_SAM/src/training/inference.py 的 forward_with_batch_info 保持一致。
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# 引入 TTTS_SAM 的 segment_anything
TTTS_SAM_ROOT = Path("/root/autodl-fs/masquer.li/code/TTTS_SAM")
sys.path.insert(0, str(TTTS_SAM_ROOT))
from segment_anything import sam_model_registry

# TTTSNet 内部模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset_sam_random import SAMRandomPointDataset, collate_fn
from utils.metrics_binary import calc_miou_and_dice
from utils.tracker import ExperimentTracker


def set_seed(seed: int):
    if seed is None:
        print("Seed is null; using non-deterministic randomness to match TTTS_SAM config.")
        return
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


def create_dataloaders(cfg: Dict[str, Any]):
    ds_cfg = cfg["dataset_config"]
    train_cfg = cfg["training_config"]
    prompt_cfg = cfg.get("prompt_config", {})
    aug_cfg = cfg.get("augmentation_config", {})

    points_sample_mode = prompt_cfg.get("points_sample_mode", "RANDOM")
    prompt_with_gt = bool(prompt_cfg.get("prompt_with_gt", 1))

    train_dataset = SAMRandomPointDataset(
        data_path=ds_cfg["train_paths"][0],
        mode="train",
        img_size=ds_cfg.get("img_size", 1024),
        prompt_points_num=prompt_cfg.get("prompt_points_num", 20),
        disable_augmentation=train_cfg.get("disable_augmentation", False),
        custom_defects_p=aug_cfg.get("custom_defects_p", 0.5),
        points_sample_mode=points_sample_mode,
        prompt_with_gt=prompt_with_gt,
    )
    val_dataset = SAMRandomPointDataset(
        data_path=ds_cfg["val_paths"][0],
        mode="valid",
        img_size=ds_cfg.get("img_size", 1024),
        prompt_points_num=prompt_cfg.get("prompt_points_num", 20),
        disable_augmentation=True,
        points_sample_mode=points_sample_mode,
        prompt_with_gt=prompt_with_gt,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


def create_model(cfg: Dict[str, Any], device: torch.device):
    model_cfg = cfg["model_config"]
    sam = sam_model_registry[model_cfg["model_type"]](
        checkpoint=model_cfg.get("checkpoint"),
        device=str(device),
    )
    sam = sam.to(device)

    if model_cfg.get("freeze_fv", False):
        for p in sam.image_encoder.parameters():
            p.requires_grad = False
        print("Image encoder frozen")

    return sam


def create_optimizer(model: torch.nn.Module, cfg: Dict[str, Any]):
    train_cfg = cfg["training_config"]
    return AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )


def create_scheduler(optimizer: torch.optim.Optimizer, cfg: Dict[str, Any]):
    train_cfg = cfg["training_config"]
    return CosineAnnealingLR(
        optimizer,
        T_max=train_cfg.get("scheduler_t_max", train_cfg["num_epochs"]),
        eta_min=train_cfg.get("eta_min", 1e-6),
    )


def true_iou(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """计算 batch 内每张图的 IoU。"""
    p = (torch.sigmoid(pred) > 0.5).float()
    g = (gt > 0.5).float()
    inter = (p * g).sum(dim=(2, 3))
    union = p.sum(dim=(2, 3)) + g.sum(dim=(2, 3)) - inter
    return inter / (union + 1e-8)


def compute_loss(
    sam_model: torch.nn.Module,
    low_res_masks: torch.Tensor,
    iou_predictions: torch.Tensor,
    batch: Dict[str, Any],
    criterion: torch.nn.Module,
    device: torch.device,
    iou_weight: float = 0.1,
):
    """复现 TTTS_SAM 的 loss：DiceCE + IoU MSE。"""
    B = low_res_masks.shape[0]
    total_loss = torch.tensor(0.0, device=device)
    total_miou = 0.0
    total_dice = 0.0
    valid = 0

    for i in range(B):
        if i >= len(batch["labels"]) or batch["labels"][i] is None:
            continue

        gt = batch["labels"][i].to(device).float()
        if gt.dim() == 3:
            gt = gt.unsqueeze(0)  # [1, 1, H, W]
        masks = sam_model.postprocess_masks(
            torch.clamp(low_res_masks[i], -20.0, 20.0).unsqueeze(0),
            input_size=batch["size_before_pad"][i],
            original_size=batch["image_sizes"][i],
        )
        pred = masks.squeeze(1).unsqueeze(0).to(device).float()
        if pred.shape != gt.shape:
            continue

        mask_loss = criterion(pred, gt)

        iou_loss = torch.tensor(0.0, device=device)
        if iou_predictions is not None:
            pred_iou = iou_predictions[i].squeeze()
            if pred_iou.dim() > 0:
                pred_iou = pred_iou[0]
            true_iou_val = true_iou(pred, gt).squeeze()
            iou_loss = F.mse_loss(pred_iou, true_iou_val)

        total_loss = total_loss + mask_loss + iou_weight * iou_loss

        m, d = calc_miou_and_dice(pred.sigmoid(), gt)
        if not np.isnan(m) and not np.isnan(d):
            total_miou += m
            total_dice += d
            valid += 1

    return total_loss, total_miou / max(valid, 1), total_dice / max(valid, 1), valid


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    scaler: GradScaler,
    epoch: int,
    tracker: ExperimentTracker,
    use_amp: bool = True,
    iou_weight: float = 0.1,
    use_box_prompt: bool = False,
):
    model.train()
    total_loss = 0.0
    total_miou = 0.0
    total_dice = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        if batch is None:
            continue

        images = batch["images"].to(device, non_blocking=True)
        points = batch["points"].to(device, non_blocking=True)
        point_labels = batch["point_labels"].to(device, non_blocking=True)
        boxes = batch["boxes"].to(device, non_blocking=True) if use_box_prompt else None

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

        if valid > 0:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            total_loss += loss.item()
            total_miou += miou
            total_dice += dice
            num_batches += 1

            global_step = epoch * len(loader) + batch_idx
            tracker.log_step(global_step, epoch, {
                "train/loss": loss.item(),
                "train/miou": miou,
                "train/dice": dice,
            })

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "miou": f"{miou:.4f}",
                "dice": f"{dice:.4f}",
            })

    return {
        "train/loss": total_loss / max(num_batches, 1),
        "train/miou": total_miou / max(num_batches, 1),
        "train/dice": total_dice / max(num_batches, 1),
    }


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    use_amp: bool = True,
    iou_weight: float = 0.1,
    use_box_prompt: bool = False,
):
    model.eval()
    total_loss = 0.0
    total_miou = 0.0
    total_dice = 0.0
    num_batches = 0

    for batch in tqdm(loader, desc="Validate"):
        if batch is None:
            continue

        images = batch["images"].to(device, non_blocking=True)
        points = batch["points"].to(device, non_blocking=True)
        point_labels = batch["point_labels"].to(device, non_blocking=True)
        boxes = batch["boxes"].to(device, non_blocking=True) if use_box_prompt else None

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

        if valid > 0:
            total_loss += loss.item()
            total_miou += miou
            total_dice += dice
            num_batches += 1

    return {
        "val/loss": total_loss / max(num_batches, 1),
        "val/miou": total_miou / max(num_batches, 1),
        "val/dice": total_dice / max(num_batches, 1),
    }


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    metrics: Dict[str, float],
    path: Path,
    is_best: bool = False,
):
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
        torch.save(checkpoint, path.parent / "best_model.pth")


def main():
    parser = argparse.ArgumentParser(description="SAM random points training")
    parser.add_argument("--config", type=str, default="configs/config_sam_random.json")
    parser.add_argument("--work_dir", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--debug", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg["training_config"]
    runtime_cfg = cfg["runtime_config"]
    prompt_cfg = cfg.get("prompt_config", {})

    num_epochs = args.num_epochs if args.num_epochs is not None else train_cfg["num_epochs"]
    debug = args.debug if args.debug is not None else runtime_cfg.get("debug", 0)

    seed = train_cfg.get("seed", runtime_cfg.get("seed", None))
    set_seed(seed)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{runtime_cfg.get('run_name', 'sam_random')}_{timestamp}"
    exp_dir = Path(args.work_dir) / run_name if args.work_dir else Path(runtime_cfg.get("work_dir", "experiments")) / run_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 将 stdout/stderr 同时重定向到实验目录的 training.log
    tee = TeeLogger(exp_dir / "training.log", mode="a")
    sys.stdout = tee
    sys.stderr = tee

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
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
    print(f"Prompt config: prompt_with_gt={prompt_cfg.get('prompt_with_gt', 1)}, "
          f"points_sample_mode={prompt_cfg.get('points_sample_mode', 'RANDOM')}, "
          f"box_prompt={prompt_cfg.get('box_prompt', False)}")

    model = create_model(cfg, device)
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = create_optimizer(model, cfg)
    scheduler = create_scheduler(optimizer, cfg)

    import monai
    criterion = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean").to(device)

    use_amp = bool(train_cfg.get("use_amp", 1))
    scaler = GradScaler(enabled=use_amp)
    iou_weight = train_cfg.get("iou_loss_weight", 0.1)
    use_box_prompt = bool(prompt_cfg.get("box_prompt", False))
    validation_frequency = int(runtime_cfg.get("validation_frequency", 1))

    tracker.log_lr(optimizer, step=0)

    best_miou = 0.0
    best_epoch = 0
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, epoch, tracker,
            use_amp=use_amp, iou_weight=iou_weight, use_box_prompt=use_box_prompt,
        )
        # 对齐 TTTS_SAM：验证 epoch 为 1、validation_frequency 的倍数、
        # (epoch+1) 为 validation_frequency 的倍数，以及最后 3 个 epoch
        should_validate = (
            epoch == 1
            or epoch % validation_frequency == 0
            or (epoch + 1) % validation_frequency == 0
            or epoch >= num_epochs - 2
        )
        if should_validate:
            val_metrics = validate(
                model,
                val_loader,
                criterion,
                device,
                use_amp=use_amp,
                iou_weight=iou_weight,
                use_box_prompt=use_box_prompt,
            )
        else:
            val_metrics = {"val/loss": None, "val/miou": None, "val/dice": None}
        epoch_time = time.time() - epoch_start

        # 对齐 TTTS_SAM：cosine scheduler 只在验证 epoch step
        if should_validate:
            scheduler.step()

        epoch_metrics = {**train_metrics, **val_metrics, "epoch/time_s": epoch_time}
        tracker.log_epoch(epoch, epoch_metrics)
        tracker.log_lr(optimizer, step=epoch)

        if should_validate:
            print(f"Epoch {epoch}/{num_epochs} | Time: {epoch_time:.1f}s | "
                  f"Train Loss: {train_metrics['train/loss']:.4f} | "
                  f"Val mIoU: {val_metrics['val/miou']:.4f} | Val Dice: {val_metrics['val/dice']:.4f}")
        else:
            print(f"Epoch {epoch}/{num_epochs} | Time: {epoch_time:.1f}s | "
                  f"Train Loss: {train_metrics['train/loss']:.4f} | "
                  f"Val: skipped (frequency={validation_frequency})")

        if debug == 0:
            save_freq = runtime_cfg.get("save_frequency", 10)
            if epoch % save_freq == 0 or epoch == num_epochs:
                save_checkpoint(
                    model, optimizer, scheduler, epoch, epoch_metrics,
                    exp_dir / "checkpoints" / f"model_epoch_{epoch:03d}.pth"
                )
                # 只保留最新的 N 个 epoch checkpoint，释放磁盘
                keep_last_n = runtime_cfg.get("keep_last_n_checkpoints", 1)
                checkpoint_dir = exp_dir / "checkpoints"
                epoch_ckpts = sorted(checkpoint_dir.glob("model_epoch_*.pth"))
                for old_ckpt in epoch_ckpts[:-keep_last_n]:
                    old_ckpt.unlink()
            if should_validate and val_metrics["val/miou"] > best_miou:
                best_miou = val_metrics["val/miou"]
                best_epoch = epoch
                save_checkpoint(
                    model, optimizer, scheduler, epoch, epoch_metrics,
                    exp_dir / "checkpoints" / "best_model.pth",
                    is_best=True,
                )
                print(f"  *** New best mIoU: {best_miou:.4f} at epoch {best_epoch}")

        tracker.flush()

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
