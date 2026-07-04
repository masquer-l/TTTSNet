#!/usr/bin/env python3
"""清理实验目录中的中间 checkpoint，保留 best_model.pth 和训练记录。

用法:
    python tools/cleanup_checkpoints.py experiments/tttsnet_vit_backbone_20260625_153352
    python tools/cleanup_checkpoints.py experiments/*
    python tools/cleanup_checkpoints.py --delete-failed experiments/tttsnet_vit_backbone_20260625_194538
"""

import argparse
import shutil
import sys
from pathlib import Path


def cleanup_experiment_dir(exp_dir: Path, delete_failed: bool = False) -> dict:
    """清理单个实验目录，返回删除文件数和释放字节数。"""
    exp_dir = Path(exp_dir)
    result = {"deleted_files": 0, "freed_bytes": 0}

    if not exp_dir.exists():
        return result

    checkpoints_dir = exp_dir / "checkpoints"
    if not checkpoints_dir.exists():
        # 没有 checkpoint 目录，视为失败/空实验
        if delete_failed and exp_dir.is_dir():
            size = sum(f.stat().st_size for f in exp_dir.rglob("*") if f.is_file())
            shutil.rmtree(exp_dir)
            result["deleted_files"] += 1
            result["freed_bytes"] += size
            print(f"[DELETE FAILED] {exp_dir} (freed {size / 1024**3:.2f} GB)")
        return result

    # 删除中间 epoch checkpoint，保留 best_model.pth
    for ckpt in checkpoints_dir.glob("model_epoch_*.pth"):
        size = ckpt.stat().st_size
        ckpt.unlink()
        result["deleted_files"] += 1
        result["freed_bytes"] += size

    if result["deleted_files"] > 0:
        print(
            f"[CLEANUP] {exp_dir}: deleted {result['deleted_files']} intermediate checkpoints, "
            f"freed {result['freed_bytes'] / 1024**3:.2f} GB"
        )

    return result


def main():
    parser = argparse.ArgumentParser(description="Cleanup intermediate checkpoints.")
    parser.add_argument("paths", nargs="+", help="Experiment directories to clean")
    parser.add_argument(
        "--delete-failed",
        action="store_true",
        help="Delete entire experiment directory if it has no checkpoints/ looks failed",
    )
    args = parser.parse_args()

    total_deleted = 0
    total_freed = 0

    for pattern in args.paths:
        path = Path(pattern)
        if "*" in pattern or "?" in pattern:
            matches = list(Path(".").glob(pattern))
        else:
            matches = [path]

        for match in matches:
            if match.is_dir():
                res = cleanup_experiment_dir(match, delete_failed=args.delete_failed)
                total_deleted += res["deleted_files"]
                total_freed += res["freed_bytes"]

    print(
        f"\nTotal: deleted {total_deleted} files, freed {total_freed / 1024**3:.2f} GB"
    )


if __name__ == "__main__":
    main()
