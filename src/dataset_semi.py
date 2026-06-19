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
)


class CustomDefectsAugmentation(A.ImageOnlyTransform):
    """同 dataset_tttsnet.py"""

    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.laser = AddLaserPointer(p=1.0)
        self.dust = AddDustParticles(p=1.0)
        self.structural = AddStructuralDefects(p=1.0)
        self.fiber = AddOpticalFiber(p=1.0)

    def apply(self, img, **params):
        r = np.random.random()
        if r < 0.10:
            img = self.laser(image=img)["image"]
        elif r < 0.25:
            img = self.dust(image=img)["image"]
        elif r < 0.40:
            img = self.structural(image=img)["image"]
        elif r < 0.65:
            img = self.fiber(image=img)["image"]
        elif r < 0.75:
            img = self.dust(image=img)["image"]
            img = self.structural(image=img)["image"]
        elif r < 0.85:
            img = self.laser(image=img)["image"]
            img = self.fiber(image=img)["image"]
        elif r < 0.92:
            img = self.laser(image=img)["image"]
            img = self.dust(image=img)["image"]
            img = self.fiber(image=img)["image"]
        elif r < 0.98:
            img = self.laser(image=img)["image"]
            img = self.structural(image=img)["image"]
            img = self.fiber(image=img)["image"]
        else:
            img = self.dust(image=img)["image"]
            img = self.structural(image=img)["image"]
            img = self.fiber(image=img)["image"]
        return img

    def get_transform_init_args_names(self):
        return ()


class TTTSNetSemiDataset(Dataset):
    """
    半监督数据集：混合有标注数据（FetReg）和伪标签数据（无标注视频）
    """

    def __init__(
        self,
        labeled_data_paths: List[str],
        pseudo_data_path: str,
        mode: str = "train",
        img_size: int = 448,
        binary: bool = True,
    ):
        assert mode in ["train", "valid"]
        self.mode = mode
        self.img_size = img_size
        self.binary = binary

        # 加载有标注数据
        self.labeled_samples = []
        for data_path in labeled_data_paths:
            images = sorted(glob.glob(os.path.join(data_path, "*/images/*.png"), recursive=True))
            labels = sorted(glob.glob(os.path.join(data_path, "*/labels/*.png"), recursive=True))
            assert len(images) == len(labels)
            for img_path, lbl_path in zip(images, labels):
                self.labeled_samples.append((img_path, lbl_path, 1.0))  # weight=1.0 for labeled

        # 加载伪标签数据
        self.pseudo_samples = []
        pseudo_images = sorted(glob.glob(os.path.join(pseudo_data_path, "**/*.jpg"), recursive=True))
        for img_path in pseudo_images:
            # 对应的伪标签路径
            pseudo_lbl_path = img_path.replace("/images/", "/pseudo_labels/").replace(".jpg", ".png")
            if os.path.exists(pseudo_lbl_path):
                self.pseudo_samples.append((img_path, pseudo_lbl_path, 0.5))  # weight=0.5 for pseudo

        self.samples = self.labeled_samples + self.pseudo_samples

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
        return len(self.samples)

    def _load_label(self, label_path: str) -> np.ndarray:
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise ValueError(f"无法读取标签: {label_path}")
        if self.binary:
            label = np.where(label > 1, 0, label)
            label = (label > 0).astype(np.uint8)
        return label

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        img_path, lbl_path, weight = self.samples[idx]

        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"无法读取图像: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = self._load_label(lbl_path)

        if self.mode == "train":
            transformed = self.train_transform(image=image, mask=label)
        else:
            transformed = self.valid_transform(image=image, mask=label)

        image_tensor = transformed["image"]
        mask_tensor = transformed["mask"].unsqueeze(0).float()

        return image_tensor, mask_tensor, weight

    def get_stats(self) -> dict:
        return {
            "labeled": len(self.labeled_samples),
            "pseudo": len(self.pseudo_samples),
            "total": len(self.samples),
        }
