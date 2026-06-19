#!/usr/bin/env python3
"""二分类分割指标（mIoU / Dice）"""

import torch
import numpy as np


def calc_miou_and_dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-7):
    """
    计算二分类的 mIoU 和 Dice。

    Args:
        pred: 模型输出概率，形状 [B, 1, H, W] 或 [B, H, W]
        target: GT mask，形状 [B, 1, H, W] 或 [B, H, W]，值为 0 或 1
        threshold: 二值化阈值
        eps: 防止除零

    Returns:
        miou: 平均 IoU（只考虑有前景的样本）
        dice: 平均 Dice（只考虑有前景的样本）
    """
    if pred.dim() == 4 and pred.size(1) == 1:
        pred = pred.squeeze(1)
    if target.dim() == 4 and target.size(1) == 1:
        target = target.squeeze(1)

    pred_bin = (pred > threshold).float()
    target = target.float()

    intersection = (pred_bin * target).sum(dim=(1, 2))
    union = ((pred_bin + target) > 0).float().sum(dim=(1, 2))
    dice_den = pred_bin.sum(dim=(1, 2)) + target.sum(dim=(1, 2))

    iou_per_sample = (intersection + eps) / (union + eps)
    dice_per_sample = (2 * intersection + eps) / (dice_den + eps)

    # 只统计 target 中有前景的样本
    valid = target.sum(dim=(1, 2)) > 0
    if valid.sum() == 0:
        return float('nan'), float('nan')

    miou = iou_per_sample[valid].mean().item()
    dice = dice_per_sample[valid].mean().item()
    return miou, dice


def calc_pixel_accuracy(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
    """计算像素级准确率"""
    if pred.dim() == 4 and pred.size(1) == 1:
        pred = pred.squeeze(1)
    if target.dim() == 4 and target.size(1) == 1:
        target = target.squeeze(1)

    pred_bin = (pred > threshold).float()
    correct = (pred_bin == target.float()).float().sum()
    total = target.numel()
    return (correct / total).item()
