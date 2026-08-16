"""Dataset that loads frames on-the-fly from a frame_manifest.csv.

Use this when disk is tight and you do not want to copy full-resolution images.
"""

import csv
import sys
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tools" / "screening"))
from crop import crop_image, crop_mask, scale_crop_params


class VideoFrameDataset(Dataset):
    """Dataset that reads image/label paths from a manifest.

    The manifest CSV must contain at least:
        image_path, label_path, split
    Optional: video_path, frame_idx (used to extract the frame from the video
    when image_path points to a video file), label_source (manual/cnn_fallback),
    case, crop_center_x/crop_center_y/crop_size.

    For static image files, image_path should be an absolute or relative path to
    the image file.
    """

    def __init__(
        self,
        manifest_path: str,
        mode: str = "train",
        img_size: int = 448,
        binary: bool = True,
        label_source: Optional[str] = None,
        crop_source_size: Optional[Tuple[int, int]] = (1080, 1920),
    ):
        assert mode in ("train", "valid")
        self.mode = mode
        self.img_size = img_size
        self.binary = binary
        self.label_source = label_source
        self.crop_source_size = crop_source_size
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest_dir = self.manifest_path.parent
        self.rows = []
        with open(manifest_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Respect the case-level split written by export_annotations.
                if row.get("split", "train") != mode:
                    continue
                # Allow training on only manual GT or only CNN pseudo labels.
                if label_source is not None and row.get("label_source") != label_source:
                    continue
                self.rows.append(row)

        if not self.rows:
            raise ValueError(
                f"No {mode} rows found in {manifest_path} "
                f"(label_source filter={label_source})."
            )

        # Sanity-check case leakage.
        self._validate_split()
        self._print_summary()

        self.resize_transform = A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(),
            ToTensorV2(),
        ])
        self.train_transform = A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.OneOf([
                A.Blur(blur_limit=(3, 7), p=0.25),
                A.MotionBlur(blur_limit=(3, 7), p=0.45),
            ], p=0.2),
            A.ShiftScaleRotate(
                border_mode=cv2.BORDER_CONSTANT,
                shift_limit=0.025,
                rotate_limit=40,
                scale_limit=0.2,
                p=0.2,
            ),
            A.ColorJitter(saturation=0.2, hue=0.15, p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.15, 0.05), contrast_limit=(-0.1, 0.2), p=0.3
            ),
            A.CLAHE(clip_limit=1.0, tile_grid_size=(16, 16), p=0.15),
            A.Normalize(),
            ToTensorV2(),
        ])

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        return self.manifest_dir / p

    def _read_image(self, row: dict) -> np.ndarray:
        image_path = self._resolve(row["image_path"])
        if image_path.is_file():
            image = cv2.imread(str(image_path))
        else:
            # Fallback: extract from video_path at frame_idx
            video_path = row.get("video_path")
            frame_idx = int(row.get("frame_idx", 0))
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, image = cap.read()
            cap.release()
        if image is None:
            raise ValueError(f"Cannot read image for row: {row}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _read_label(self, row: dict) -> np.ndarray:
        label_path = self._resolve(row["label_path"])
        mask = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Cannot read label for row: {row}")
        if self.binary:
            mask = (mask > 0).astype(np.uint8)
        return mask

    def _read_crop(self, row: dict):
        cx = row.get("crop_center_x")
        cy = row.get("crop_center_y")
        size = row.get("crop_size")
        if cx is None or cy is None or size is None or size == "":
            return None
        return {
            "center_x": float(cx),
            "center_y": float(cy),
            "crop_size": float(size),
        }

    def _validate_split(self) -> None:
        """Ensure no case appears in both train and valid partitions."""
        train_cases = set()
        valid_cases = set()
        with open(self.manifest_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                case = row.get("case")
                if not case:
                    continue
                split = row.get("split", "train")
                if split == "train":
                    train_cases.add(case)
                elif split == "valid":
                    valid_cases.add(case)
        leaked = train_cases & valid_cases
        if leaked:
            raise ValueError(
                f"Case-level data leak detected: cases {sorted(leaked)} appear in both train and valid splits."
            )

    def _print_summary(self) -> None:
        """Print a concise summary of the loaded split."""
        cases = sorted({r.get("case", "unknown") for r in self.rows})
        sources = {}
        for r in self.rows:
            sources[r.get("label_source", "unknown")] = sources.get(r.get("label_source", "unknown"), 0) + 1
        print(
            f"[VideoFrameDataset] mode={self.mode}, samples={len(self.rows)}, "
            f"cases={len(cases)}, label_sources={sources}"
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        image = self._read_image(row)
        label = self._read_label(row)
        crop_params = self._read_crop(row)
        if crop_params is not None and self.crop_source_size is not None:
            h, w = image.shape[:2]
            scaled = scale_crop_params(crop_params, self.crop_source_size, (h, w))
            image = crop_image(image, scaled)
            label = crop_mask(label, scaled)
        if self.mode == "train":
            transformed = self.train_transform(image=image, mask=label)
        else:
            transformed = self.resize_transform(image=image, mask=label)
        return transformed["image"], transformed["mask"]
