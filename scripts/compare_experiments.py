#!/usr/bin/env python3
"""
实验对比脚本：读取多个实验目录的 epoch_history.csv，生成对比表格和曲线
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_experiment(exp_dir: Path):
    """加载单个实验的指标和配置"""
    exp_dir = Path(exp_dir)
    history_path = list(exp_dir.rglob("epoch_history.csv"))
    if not history_path:
        return None

    df = pd.read_csv(history_path[0])

    summary_path = list(exp_dir.rglob("summary.json"))
    summary = {}
    if summary_path:
        with open(summary_path[0]) as f:
            summary = json.load(f)

    config_path = list(exp_dir.rglob("config.json"))
    config = {}
    if config_path:
        with open(config_path[0]) as f:
            config = json.load(f)

    return {
        "name": exp_dir.name,
        "df": df,
        "summary": summary,
        "config": config,
    }


def generate_comparison(experiments: list, output_dir: Path):
    """生成对比表格和图表"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 对比表格
    rows = []
    for exp in experiments:
        if exp is None:
            continue
        df = exp["df"]
        best_idx = df["val/miou"].idxmax()
        row = {
            "Experiment": exp["name"],
            "Best val_mIoU": f"{df.loc[best_idx, 'val/miou']:.4f}",
            "Best val_Dice": f"{df.loc[best_idx, 'val/dice']:.4f}",
            "Best Epoch": int(df.loc[best_idx, "epoch"]),
            "Final val_mIoU": f"{df['val/miou'].iloc[-1]:.4f}",
            "Total Epochs": len(df),
        }
        rows.append(row)

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(output_dir / "comparison_table.csv", index=False)
    print(comparison_df.to_string(index=False))

    # 对比曲线
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for exp in experiments:
        if exp is None:
            continue
        df = exp["df"]
        axes[0].plot(df["epoch"], df["val/miou"], label=exp["name"], marker="o", markersize=2)
        axes[1].plot(df["epoch"], df["val/dice"], label=exp["name"], marker="o", markersize=2)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("val mIoU")
    axes[0].set_title("Validation mIoU Comparison")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("val Dice")
    axes[1].set_title("Validation Dice Comparison")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "comparison_curves.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved comparison plots to {output_dir / 'comparison_curves.png'}")


def main():
    parser = argparse.ArgumentParser(description="Compare TTTSNet experiments")
    parser.add_argument("--exp_dirs", nargs="+", required=True, help="Experiment directories")
    parser.add_argument("--output_dir", type=str, default="experiments/comparison", help="Output directory")
    args = parser.parse_args()

    experiments = [load_experiment(Path(d)) for d in args.exp_dirs]
    experiments = [e for e in experiments if e is not None]

    if not experiments:
        print("No valid experiments found")
        return

    generate_comparison(experiments, Path(args.output_dir))


if __name__ == "__main__":
    main()
