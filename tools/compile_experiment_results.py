#!/usr/bin/env python3
"""实验结果汇总脚本

扫描 TTTSNet/experiments/ 下所有实验，读取 summary.json 或 epoch_history.csv，
生成对比表格和曲线图，输出到 project_material/。
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
OUTPUT_DIR = Path("/root/autodl-fs/masquer.li/code/project_material")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_epoch_csv(exp_dir: Path):
    """递归查找 epoch_history.csv"""
    candidates = list(exp_dir.rglob("epoch_history.csv"))
    if not candidates:
        return None
    # 优先选子目录（实际实验输出），若无则取根目录
    candidates.sort(key=lambda p: len(str(p)))
    return candidates[0]


def load_experiment(exp_dir: Path):
    """加载单个实验的元信息和最佳指标。"""
    summary_files = list(exp_dir.rglob("summary.json"))
    csv_file = find_epoch_csv(exp_dir)

    # 使用 epoch_history.csv 所在目录作为实验名（更精确）
    if csv_file:
        name_dir = csv_file.parent
    else:
        name_dir = exp_dir
    name = name_dir.name

    record = {
        "name": name,
        "dir": str(name_dir.relative_to(PROJECT_ROOT)),
        "total_epochs": None,
        "best_val_miou": None,
        "best_epoch": None,
        "best_val_dice": None,
        "total_time_h": None,
        "df": None,
        "csv": None,
    }

    if summary_files:
        with open(summary_files[0], "r") as f:
            summary = json.load(f)
        fm = summary.get("final_metrics", {})
        bm = summary.get("best_metrics", {})
        record["total_epochs"] = summary.get("total_epochs")
        record["best_val_miou"] = fm.get("best_val_miou")
        record["best_epoch"] = fm.get("best_epoch")
        record["best_val_dice"] = bm.get("best_val/dice")
        record["total_time_h"] = fm.get("total_time_h")

    if csv_file and csv_file.exists():
        df = pd.read_csv(csv_file)
        if "val/miou" in df.columns:
            best_idx = df["val/miou"].idxmax()
            record["best_val_miou"] = df.loc[best_idx, "val/miou"]
            record["best_epoch"] = int(df.loc[best_idx, "epoch"])
        if "val/dice" in df.columns:
            best_idx = df["val/dice"].idxmax()
            record["best_val_dice"] = df.loc[best_idx, "val/dice"]
        record["df"] = df
        record["csv"] = str(csv_file.relative_to(PROJECT_ROOT))
        if record["total_epochs"] is None:
            record["total_epochs"] = len(df)

    return record


def main():
    # 扫描所有实验目录
    exp_dirs = [d for d in EXPERIMENTS_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")]
    # 排除已知的非实验目录和 debug 目录
    skip_names = {"deprecated", "pseudo_label_audit_20260620", "comparison_baseline_vs_temporal_v1"}
    exp_dirs = [d for d in exp_dirs if d.name not in skip_names]

    records = []
    for exp_dir in sorted(exp_dirs):
        try:
            rec = load_experiment(exp_dir)
            # 只保留完整实验（≥10 epochs），排除短周期 debug 验证
            if rec["total_epochs"] is not None and rec["total_epochs"] < 10:
                continue
            if rec["df"] is not None or rec["best_val_miou"] is not None:
                records.append(rec)
        except Exception as e:
            print(f"Skip {exp_dir.name}: {e}")

    name_map = {
        "tttsnet_single_baseline_20260619_233056": "Baseline",
        "tttsnet_vit_backbone_20260620_155115": "ViT Backbone",
        "tttsnet_temporal_20260620_013154": "Temporal v1",
        "tttsnet_temporal_v2_20260620_021249": "Temporal v2",
        "tttsnet_temporal_no_loss_20260620_114905": "Temporal no-loss",
        "tttsnet_temporal_v3_20260620_213749": "Temporal v3",
        "tttsnet_aufl_30ep_20260620_103650": "AUFL 30ep",
        "tttsnet_semi_20260620_085338": "Semi-supervised",
        "sam_random_points_20260620_211639": "SAM random points",
    }

    change_description = {
        "tttsnet_single_baseline_20260619_233056": "原版 TTTSNet，448x448，Dice+BCE+CE",
        "tttsnet_vit_backbone_20260620_155115": "TTTSNet + SAM ViT-B backbone 替换 Init_Block",
        "tttsnet_temporal_20260620_013154": "TTTSNet + 3帧时序一致性 + 弱同步增强 (lambda=0.1)",
        "tttsnet_temporal_v2_20260620_021249": "TTTSNet + 3帧时序一致性 + 弱同步增强 (lambda=1.0)",
        "tttsnet_temporal_no_loss_20260620_114905": "TTTSNet + 3帧输入 + temporal loss=0",
        "tttsnet_temporal_v3_20260620_213749": "TTTSNet + 3帧时序一致性 + 强同步增强 (lambda=0.1)",
        "tttsnet_aufl_30ep_20260620_103650": "TTTSNet + Asymmetric Unified Focal Loss",
        "tttsnet_semi_20260620_085338": "TTTSNet + 伪标签半监督 (max_conf>=0.9, 10ep 暂停)",
        "sam_random_points_20260620_211639": "SAM ViT-B + 随机点提示，1024x1024",
    }

    # 生成对比表
    rows = []
    for rec in records:
        rows.append({
            "实验": name_map.get(rec["name"], rec["name"]),
            "改动点": change_description.get(rec["name"], ""),
            "Best val_mIoU": f"{rec['best_val_miou']:.4f}" if rec["best_val_miou"] is not None else "-",
            "Best Epoch": rec["best_epoch"] if rec["best_epoch"] is not None else "-",
            "Best val_Dice": f"{rec['best_val_dice']:.4f}" if rec["best_val_dice"] is not None else "-",
            "Epochs": rec["total_epochs"] if rec["total_epochs"] is not None else "-",
            "Time(h)": f"{rec['total_time_h']:.2f}" if rec["total_time_h"] is not None else "-",
        })
    df_summary = pd.DataFrame(rows)
    df_summary = df_summary.sort_values("Best val_mIoU", ascending=False)

    print("\n=== 实验结果汇总 ===")
    print(df_summary.to_string(index=False))

    # 保存 CSV
    df_summary.to_csv(OUTPUT_DIR / "experiment_summary.csv", index=False)

    # 生成对比图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.tab10.colors
    plotted = 0
    for rec in records:
        df = rec.get("df")
        if df is None or "val/miou" not in df.columns:
            continue
        label = name_map.get(rec["name"], rec["name"])
        color = colors[plotted % len(colors)]
        axes[0].plot(df["epoch"], df["val/miou"], label=label, color=color)
        if rec["best_epoch"] is not None and rec["best_val_miou"] is not None:
            axes[0].scatter(rec["best_epoch"], rec["best_val_miou"], color=color, zorder=5)
        plotted += 1

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Val mIoU")
    axes[0].set_title("Val mIoU over Epochs")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    plotted = 0
    for rec in records:
        df = rec.get("df")
        if df is None or "val/dice" not in df.columns:
            continue
        label = name_map.get(rec["name"], rec["name"])
        color = colors[plotted % len(colors)]
        axes[1].plot(df["epoch"], df["val/dice"], label=label, color=color)
        plotted += 1

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val Dice")
    axes[1].set_title("Val Dice over Epochs")
    axes[1].legend(loc="lower right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "all_experiments_curves.png"
    plt.savefig(fig_path, dpi=150)
    print(f"\nSaved curves to {fig_path}")
    print(f"Saved summary to {OUTPUT_DIR / 'experiment_summary.csv'}")


if __name__ == "__main__":
    main()
