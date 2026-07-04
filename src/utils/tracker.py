#!/usr/bin/env python3
"""
轻量级实验追踪器 - 支持 TensorBoard、JSON 和 CSV 日志
改编自 TTTS_SAM/src/experiment/tracker.py，去除 MetricsLogger 依赖。
"""

import os
import json
import csv
import time
import platform
from datetime import datetime
from typing import Dict, Any, Optional, Union
from pathlib import Path

import torch
import numpy as np


class ExperimentTracker:
    """统一实验追踪器"""

    def __init__(
        self,
        experiment_dir: str,
        experiment_name: str = "",
        config: Optional[Any] = None,
        use_tensorboard: bool = True,
        use_csv: bool = True,
        use_json: bool = True,
    ):
        self.experiment_dir = Path(experiment_dir)
        self.experiment_name = experiment_name or self.experiment_dir.name
        self.use_tensorboard = use_tensorboard
        self.use_csv = use_csv
        self.use_json = use_json

        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self._start_time = time.time()
        self._step_data: list = []
        self._epoch_data: list = []
        self._tb_writer = None

        if self.use_tensorboard:
            self._init_tensorboard()
        if self.use_json:
            self._init_json_log()

        if config is not None:
            self.log_config(config)

        print(f"ExperimentTracker initialized: {self.experiment_dir}")
        if self.use_tensorboard:
            print(f"   TensorBoard: tensorboard --logdir {self.experiment_dir / 'tb_logs'}")

    def _init_tensorboard(self):
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_dir = self.experiment_dir / "tb_logs"
            tb_dir.mkdir(parents=True, exist_ok=True)
            self._tb_writer = SummaryWriter(log_dir=str(tb_dir))
        except ImportError:
            print("tensorboard not installed, skipping TensorBoard logs")
            self.use_tensorboard = False

    def log_scalar(self, tag: str, value: float, step: int):
        if self._tb_writer is not None and value is not None and not np.isnan(value):
            self._tb_writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, values: Dict[str, float], step: int):
        for sub_tag, value in values.items():
            if value is not None:
                self.log_scalar(f"{main_tag}/{sub_tag}", value, step)

    def log_image(self, tag: str, image: torch.Tensor, step: int):
        if self._tb_writer is not None:
            if image.dim() == 2:
                image = image.unsqueeze(0)
            self._tb_writer.add_image(tag, image, step)

    def log_lr(self, optimizer: torch.optim.Optimizer, step: int):
        for i, pg in enumerate(optimizer.param_groups):
            name = pg.get("name", f"group_{i}")
            self.log_scalar(f"lr/{name}", pg["lr"], step)

    def _init_json_log(self):
        self._json_path = self.experiment_dir / "experiment_meta.json"
        self._meta = {
            "experiment_name": self.experiment_name,
            "start_time": datetime.now().isoformat(),
            "platform": {
                "python": platform.python_version(),
                "pytorch": torch.__version__,
                "cuda": torch.version.cuda if torch.cuda.is_available() else "N/A",
                "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"),
                "hostname": platform.node(),
            },
            "config": {},
            "timeline": [],
        }

    def log_config(self, config: Union[Dict[str, Any], Any]) -> None:
        if hasattr(config, "model_dump"):
            config_dict = config.model_dump(mode="json", by_alias=True)
        elif hasattr(config, "dict"):
            config_dict = config.dict()
        else:
            config_dict = dict(config)

        self._meta["config"] = _make_serializable(config_dict)
        self._save_json()

    def log_event(self, event: str, data: Optional[Dict] = None):
        entry = {
            "time": datetime.now().isoformat(),
            "elapsed_s": round(time.time() - self._start_time, 1),
            "event": event,
        }
        if data:
            entry["data"] = _make_serializable(data)
        self._meta["timeline"].append(entry)
        self._save_json()

    def _save_json(self):
        if self.use_json and hasattr(self, "_json_path"):
            with open(self._json_path, "w", encoding="utf-8") as f:
                json.dump(self._meta, f, indent=2, ensure_ascii=False)

    def log_step(self, step: int, epoch: int, metrics: Dict[str, float]):
        row = {"step": step, "epoch": epoch, **metrics}
        self._step_data.append(row)

        if self.use_csv and len(self._step_data) % 50 == 0:
            self._flush_csv("step_history.csv", self._step_data)

    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        row = {"epoch": epoch, **metrics}
        self._epoch_data.append(row)

        if self.use_csv:
            self._flush_csv("epoch_history.csv", self._epoch_data)

    def _flush_csv(self, filename: str, data: list):
        if not data:
            return
        csv_path = self.experiment_dir / filename
        fieldnames = list(data[0].keys())
        for row in data[1:]:
            for k in row:
                if k not in fieldnames:
                    fieldnames.append(k)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(data)

    def summarize(self, final_metrics: Optional[Dict[str, float]] = None):
        elapsed = time.time() - self._start_time

        summary = {
            "experiment_name": self.experiment_name,
            "total_time_s": round(elapsed, 1),
            "total_time_h": round(elapsed / 3600, 2),
            "total_epochs": len(self._epoch_data),
            "total_steps": len(self._step_data),
        }
        if final_metrics:
            summary["final_metrics"] = final_metrics
        if self._epoch_data:
            summary["best_metrics"] = self._find_best_metrics()

        self._meta["summary"] = summary
        self._save_json()

        if self.use_csv:
            self._flush_csv("step_history.csv", self._step_data)
            self._flush_csv("epoch_history.csv", self._epoch_data)

        summary_path = self.experiment_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print(f"Experiment summary: {self.experiment_name}")
        print(f"   Total time: {summary['total_time_h']:.2f}h")
        print(f"   Total epochs: {summary['total_epochs']}")
        if final_metrics:
            for k, v in final_metrics.items():
                print(f"   {k}: {v:.4f}")
        print(f"{'='*60}")

        return summary

    def _find_best_metrics(self) -> Dict[str, float]:
        best = {}
        if not self._epoch_data:
            return best
        metric_keys = [
            k
            for k in self._epoch_data[0]
            if k not in ("epoch", "time_s") and isinstance(self._epoch_data[0].get(k), (int, float))
        ]
        for key in metric_keys:
            values = [row.get(key) for row in self._epoch_data if row.get(key) is not None]
            if not values:
                continue
            if "loss" in key.lower():
                best[f"best_{key}"] = min(values)
            else:
                best[f"best_{key}"] = max(values)
        return best

    def flush(self):
        if self._tb_writer is not None:
            self._tb_writer.flush()
        if self.use_csv:
            if self._step_data:
                self._flush_csv("step_history.csv", self._step_data)
            if self._epoch_data:
                self._flush_csv("epoch_history.csv", self._epoch_data)
        self._save_json()

    def close(self):
        self.flush()
        if self._tb_writer is not None:
            self._tb_writer.close()


def _make_serializable(obj):
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if hasattr(obj, "__dict__"):
        return str(obj)
    return obj
