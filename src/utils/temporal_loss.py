#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temporal consistency losses for unlabeled consecutive frames.

当前实现：Minimal Temporal-v1
- 只实现基于概率图的一阶 L1 一致性（S1: 相邻帧，Δt=1）；
- 权重使用 Bidirectional / agreement 风格：
  conf = |p - 0.5|, w = min(conf_t, conf_{t+1})；
- 设计为纯函数，方便后续扩展到 L2 / feature-level / Teacher 加权。
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor


def _validate_shapes(p_t: Tensor, p_tp1: Tensor) -> None:
    """简单的形状检查，确保时序 pair 可以逐点对齐。"""
    if p_t.shape != p_tp1.shape:
        raise ValueError(
            f"temporal_consistency_l1: 预测张量形状不匹配: "
            f"p_t.shape={p_t.shape}, p_tp1.shape={p_tp1.shape}"
        )


def temporal_consistency_l1(
    p_t: Tensor,
    p_tp1: Tensor,
    eps: float = 1e-6,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> Tensor:
    """概率图级的一阶时序一致性损失（L1 + 置信度加权）。

    Args:
        p_t: t 帧预测概率，范围约在 [0, 1]，形状 [B, 1, H, W] 或 [B, H, W] 等。
        p_tp1: t+1 帧预测概率，形状必须与 p_t 一致。
        eps: 防止数值边界问题的小常数。
        reduction: 'mean' | 'sum' | 'none'，同 PyTorch 通用约定。

    Returns:
        标量 loss（mean/sum）或与输入同形状的逐点 loss（none）。
    """
    if not isinstance(p_t, Tensor) or not isinstance(p_tp1, Tensor):
        raise TypeError("temporal_consistency_l1 期望输入为 torch.Tensor")

    _validate_shapes(p_t, p_tp1)

    # 保证在 [eps, 1-eps] 内，避免后续数值问题
    p_t = p_t.clamp(min=eps, max=1.0 - eps)
    p_tp1 = p_tp1.clamp(min=eps, max=1.0 - eps)

    # 置信度：远离 0.5 越远越“自信”
    conf_t = (p_t - 0.5).abs()
    conf_tp1 = (p_tp1 - 0.5).abs()
    weight = torch.minimum(conf_t, conf_tp1)

    # 一阶差异（L1）
    diff = (p_t - p_tp1).abs()
    loss_map = weight * diff

    if reduction == "mean":
        return loss_map.mean()
    if reduction == "sum":
        return loss_map.sum()
    if reduction == "none":
        return loss_map

    raise ValueError(f"temporal_consistency_l1: 不支持的 reduction = {reduction!r}")


__all__ = ["temporal_consistency_l1"]

