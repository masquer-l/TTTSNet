import os
import glob
from typing import Tuple, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.custom_augmentations import (
    AddDustParticles,
    AddLaserPointer,
    AddOpticalFiber,
    AddStructuralDefects,
    CustomDefectsAugmentation,
)


class TTTSNetDataset(Dataset):
    """
    TTTSNet 数据集，基于原论文 FetoscopicDataset 修改：
    - 修复 Albumentations 2.x 下的自定义增强兼容性问题
    - 明确血管二分类标签处理
    - 输出 image [3, H, W] 和 mask [1, H, W]
    """

    def __init__(
        self,
        data_path: str,
        mode: str = "train",
        img_size: int = 448,
        binary: bool = True,
    ):
        assert mode in ["train", "valid"]
        self.data_path = data_path
        self.mode = mode
        self.img_size = img_size
        self.binary = binary

        self.images = sorted(glob.glob(os.path.join(data_path, "*/images/*.png"), recursive=True))
        self.labels = sorted(glob.glob(os.path.join(data_path, "*/labels/*.png"), recursive=True))
        assert len(self.images) == len(self.labels), (
            f"图像和标签数量不匹配: images={len(self.images)}, labels={len(self.labels)}"
        )

        self.train_transform = A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.OneOf([
                A.Blur(blur_limit=(3, 7), p=0.25),
                A.MotionBlur(blur_limit=(3, 7), p=0.45)
            ], p=0.2),
            CustomDefectsAugmentation(p=0.5),
            A.ShiftScaleRotate(
                border_mode=cv2.BORDER_CONSTANT,
                shift_limit=0.025,
                rotate_limit=40,
                scale_limit=0.2,
                p=0.2,
            ),
            A.ColorJitter(saturation=0.2, hue=0.15, p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.15, 0.05), contrast_limit=(-0.1, 0.2), p=0.3),
            A.CLAHE(clip_limit=1.0, tile_grid_size=(16, 16), p=0.15),
            A.Normalize(),
            ToTensorV2(),
        ])

        self.valid_transform = A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(),
            ToTensorV2(),
        ])

    def __len__(self) -> int:
        return len(self.images)

    def _load_label(self, label_path: str) -> np.ndarray:
        """加载并二值化标签"""
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise ValueError(f"无法读取标签: {label_path}")

        if self.binary:
            # 原论文逻辑：label > 1 的像素置 0
            label = np.where(label > 1, 0, label)
            # 确保只有 0 和 1
            label = (label > 0).astype(np.uint8)
        return label

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.images[idx]
        label_path = self.labels[idx]

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = self._load_label(label_path)

        if self.mode == "train":
            transformed = self.train_transform(image=image, mask=label)
        else:
            transformed = self.valid_transform(image=image, mask=label)

        image_tensor = transformed["image"]
        mask_tensor = transformed["mask"].unsqueeze(0).float()

        return image_tensor, mask_tensor

    def get_sample_paths(self) -> List[Tuple[str, str]]:
        """返回 (image_path, label_path) 列表，用于固定数据划分"""
        return list(zip(self.images, self.labels))
