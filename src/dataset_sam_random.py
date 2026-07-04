"""SAM 随机点提示数据集（从 TTTS_SAM 简化拷贝）

与 TTTS_SAM/src/data/dataset.py 的 SAMDataset 保持核心逻辑一致，
但只保留 RANDOM 点提示模式，去除 box prompt、APG、Memory Bank 等无关逻辑。
图像 resize 到 1024×1024，与 SAM 原配置一致。
"""

import glob
import os
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode

from utils.custom_augmentations import (
    AddDustParticles,
    AddLaserPointer,
    AddOpticalFiber,
    AddStructuralDefects,
)


# ---------------------------------------------------------------------------
# Prompt 生成辅助函数（与 TTTS_SAM/src/prompts/prompt_utils.py 保持一致）
# ---------------------------------------------------------------------------
def _find_box_from_mask(mask: np.ndarray) -> List[float]:
    """从 mask 中提取边界框 [x1, y1, x2, y2]。"""
    if mask is None or mask.size == 0:
        return [0.0, 0.0, 1.0, 1.0]
    if mask.ndim > 2:
        mask = mask.squeeze()
    y_coords, x_coords = np.where(mask > 0)
    if len(x_coords) == 0 or len(y_coords) == 0:
        h, w = mask.shape
        return [0.0, 0.0, float(w - 1), float(h - 1)]
    return [float(x_coords.min()), float(y_coords.min()),
            float(x_coords.max()), float(y_coords.max())]


def _generate_points_from_mask(
    mask: np.ndarray,
    num_points: int = 10,
    fg_ratio: float = 0.5,
    sample_mode: str = "random",
) -> Tuple[np.ndarray, np.ndarray]:
    """从 GT mask 中采样前景/背景点（RANDOM / UNIFORM）。

    返回:
        points: (num_points, 2) 坐标数组
        labels: (num_points,)  标签数组 (1=前景, 0=背景)
    """
    if mask is None or mask.size == 0:
        points = np.zeros((num_points, 2), dtype=np.float32)
        labels = np.zeros(num_points, dtype=np.float32)
        return points, labels

    if mask.ndim > 2:
        mask = mask.squeeze()

    h, w = mask.shape
    num_fg = max(1, int(num_points * fg_ratio))
    num_bg = num_points - num_fg

    fg_coords = np.where(mask > 0)
    if len(fg_coords[0]) > 0:
        if sample_mode == "random":
            indices = np.random.choice(
                len(fg_coords[0]), min(num_fg, len(fg_coords[0])), replace=False
            )
        else:
            indices = np.linspace(0, len(fg_coords[0]) - 1, num_fg, dtype=int)
        fg_points = np.column_stack([fg_coords[1][indices], fg_coords[0][indices]])
        if len(fg_points) < num_fg:
            fg_points = np.pad(fg_points, ((0, num_fg - len(fg_points)), (0, 0)), mode="constant")
    else:
        fg_points = np.zeros((num_fg, 2), dtype=np.float32)

    bg_coords = np.where(mask == 0)
    if len(bg_coords[0]) > 0:
        if sample_mode == "random":
            indices = np.random.choice(
                len(bg_coords[0]), min(num_bg, len(bg_coords[0])), replace=False
            )
        else:
            indices = np.linspace(0, len(bg_coords[0]) - 1, num_bg, dtype=int)
        bg_points = np.column_stack([bg_coords[1][indices], bg_coords[0][indices]])
        if len(bg_points) < num_bg:
            bg_points = np.pad(bg_points, ((0, num_bg - len(bg_points)), (0, 0)), mode="constant")
    else:
        bg_points = np.zeros((num_bg, 2), dtype=np.float32)

    points = np.concatenate([fg_points, bg_points], axis=0).astype(np.float32)
    labels = np.concatenate([
        np.ones(len(fg_points), dtype=np.float32),
        np.zeros(len(bg_points), dtype=np.float32),
    ]).astype(np.float32)

    if len(points) < num_points:
        pad = num_points - len(points)
        points = np.pad(points, ((0, pad), (0, 0)), mode="constant")
        labels = np.pad(labels, (0, pad), mode="constant")
    elif len(points) > num_points:
        points = points[:num_points]
        labels = labels[:num_points]

    return points, labels


class SAMRandomPointDataset(Dataset):
    """SAM 随机点提示数据集（对齐 TTTS_SAM A0.2.5）。

    当 prompt_with_gt=True 且 points_sample_mode='RANDOM' 时，
    从 GT mask 中随机采样 `prompt_points_num` 个点（前景+背景混合标签），
    与 TTTS_SAM 的 generate_positive_prompts 行为一致。

    当 prompt_with_gt=False 时，回退到整张图内的全随机/网格点（label 全 1）。
    """

    def __init__(
        self,
        data_path: str,
        mode: str = "train",
        img_size: int = 1024,
        prompt_points_num: int = 20,
        disable_augmentation: bool = False,
        custom_defects_p: float = 0.5,
        deterministic_points: bool = False,
        points_seed: int = 0,
        points_sample_mode: str = "RANDOM",
        prompt_with_gt: bool = True,
    ):
        self.data_path = data_path
        self.mode = mode
        self.img_size = img_size
        self.prompt_points_num = prompt_points_num
        self.disable_augmentation = disable_augmentation
        self.custom_defects_p = custom_defects_p
        self.deterministic_points = deterministic_points
        self.points_seed = points_seed
        self.points_sample_mode = points_sample_mode.upper()
        self.prompt_with_gt = prompt_with_gt

        self.resize_transform = ResizeLongestSide(target_length=img_size)

        self.images, self.labels = self._load_data_paths()

        self.train_geom_transform = self._create_train_geom_transform()
        self.train_image_transform = self._create_train_image_transform()
        self.valid_transform = self._create_valid_transform()

    def _load_data_paths(self) -> Tuple[List[str], List[str]]:
        """加载图像和标签路径（与 FetoscopicDataset 保持一致的目录结构）。"""
        img_patterns = [
            os.path.join(self.data_path, "*/images/*.png"),
            os.path.join(self.data_path, "*/images/*.jpg"),
            os.path.join(self.data_path, "images/*.png"),
            os.path.join(self.data_path, "images/*.jpg"),
        ]
        label_patterns = [
            os.path.join(self.data_path, "*/labels/*.png"),
            os.path.join(self.data_path, "labels/*.png"),
        ]

        images = []
        for pattern in img_patterns:
            images.extend(glob.glob(pattern, recursive=True))
        images = sorted(images)

        labels = []
        for pattern in label_patterns:
            labels.extend(glob.glob(pattern, recursive=True))
        labels = sorted(labels)

        if len(images) != len(labels):
            raise ValueError(f"图像数量({len(images)})与标签数量({len(labels)})不匹配")

        return images, labels

    def _create_train_geom_transform(self) -> A.Compose:
        """几何变换，同时作用于 image 和 mask，与 TTTS_SAM 对齐。"""
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0),
                A.Transpose(p=1.0),
            ], p=0.75),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.25,
                rotate_limit=45,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5,
            ),
            A.OneOf([
                A.ElasticTransform(
                    alpha=20,
                    sigma=5,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=1.0,
                ),
                A.GridDistortion(
                    num_steps=5,
                    distort_limit=0.3,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=1.0,
                ),
                A.OpticalDistortion(
                    distort_limit=0.25,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=1.0,
                ),
            ], p=0.4),
        ])

    def _create_train_image_transform(self) -> A.Compose:
        """只作用于 image 的颜色/噪声增强，与 TTTS_SAM 对齐。"""
        return A.Compose([
            A.OneOf([
                A.Blur(blur_limit=(3, 7), p=0.25),
                A.MotionBlur(blur_limit=(3, 7), p=0.45),
            ], p=0.2),
            A.ColorJitter(saturation=0.2, hue=0.15, p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.15, 0.05),
                contrast_limit=(-0.1, 0.2),
                p=0.3,
            ),
            A.CLAHE(clip_limit=1.0, tile_grid_size=(16, 16), p=0.15),
            A.OneOf([
                A.OneOf([
                    AddLaserPointer(always_apply=False, p=0.99),
                    AddDustParticles(p=0.2),
                    AddStructuralDefects(p=0.2),
                    AddOpticalFiber(p=0.4),
                ], p=0.4),
                A.Sequential([AddDustParticles(), AddStructuralDefects()], p=0.25),
                A.Sequential([AddLaserPointer(), AddOpticalFiber()], p=0.25),
                A.Sequential([AddLaserPointer(), AddDustParticles(), AddOpticalFiber()], p=0.15),
                A.Sequential([AddLaserPointer(), AddStructuralDefects(), AddOpticalFiber()], p=0.15),
                A.Sequential([AddDustParticles(), AddStructuralDefects(), AddOpticalFiber()], p=0.15),
                A.Sequential([
                    AddLaserPointer(),
                    AddDustParticles(),
                    AddStructuralDefects(),
                    AddOpticalFiber(),
                ], p=0.05),
            ], p=self.custom_defects_p),
        ])

    def _create_valid_transform(self) -> A.Compose:
        return A.Compose([])

    def _load_image(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")
        if image.shape[-1] > 3:
            image = image[:, :, :3]
        if len(image.shape) == 2:
            image = np.repeat(image[:, :, None], 3, axis=-1)
        return image

    def _load_gt(self, label_path: str) -> np.ndarray:
        gt = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise ValueError(f"无法加载标签: {label_path}")
        # FetReg 格式：label > 1 置 0，最终 {0, 1}
        gt = ((gt == 1) | (gt == 255)).astype(np.uint8)
        return gt

    def _generate_random_points(
        self,
        image_size: Tuple[int, int],
        index: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """在整张图像内完全随机采样点，label 全为 1。"""
        h, w = image_size
        rng = np.random
        if self.deterministic_points and index is not None:
            rng = np.random.default_rng(self.points_seed + index)
            xs = rng.integers(0, w, self.prompt_points_num)
            ys = rng.integers(0, h, self.prompt_points_num)
        else:
            xs = rng.randint(0, w, self.prompt_points_num)
            ys = rng.randint(0, h, self.prompt_points_num)
        points = np.stack([xs, ys], axis=1).astype(np.float32)
        labels = np.ones(self.prompt_points_num, dtype=np.int64)
        return points, labels

    def _process_image_for_sam(
        self,
        image: np.ndarray,
        gt: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """resize 长边到 1024 并 pad 成方形，与 SAM 原始预处理一致。

        GT 保持原始尺寸，在 loss 计算时通过 SAM 的 postprocess_masks 上采样到 original_size 对齐。
        """
        resized_image = self.resize_transform.apply_image(
            image, interpolation=InterpolationMode.BILINEAR
        )
        resized_image = resized_image.transpose(2, 0, 1)[None, :, :, :].astype(np.float32)

        resized_gt = None
        if gt is not None:
            resized_gt = (gt > 0).astype(np.uint8)

        return resized_image, resized_gt

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Optional[Dict]:
        try:
            image_raw = self._load_image(self.images[index])
            image = image_raw.copy()
            gt = self._load_gt(self.labels[index])
            original_size = image.shape[:2]

            if self.mode == "train" and not self.disable_augmentation:
                geom = self.train_geom_transform(image=image, mask=gt)
                image = geom["image"]
                gt = geom["mask"]
                image = self.train_image_transform(image=image)["image"]

            resized_image, resized_gt = self._process_image_for_sam(image, gt)

            # 过滤负样本（与 TTTS_SAM 保持一致）
            if resized_gt is not None and np.all(resized_gt == 0):
                return None

            size_before_pad = resized_image.shape[-2:]

            # 生成点提示：对齐 TTTS_SAM A0.2.5
            if self.prompt_with_gt and self.mode != "test":
                mask = (gt > 0).astype(np.uint8)
                points_np, point_labels = _generate_points_from_mask(
                    mask,
                    num_points=self.prompt_points_num,
                    fg_ratio=0.5,
                    sample_mode=self.points_sample_mode.lower(),
                )
                box_np = _find_box_from_mask(mask)
            else:
                points_np, point_labels = self._generate_random_points(original_size, index=index)
                h, w = original_size
                box_np = [h * 0.1, w * 0.1, h * 0.9, w * 0.9]

            points_original = torch.from_numpy(points_np.copy()).float()
            points = torch.from_numpy(points_np).float()
            point_labels = torch.from_numpy(point_labels).long()
            points = self.resize_transform.apply_coords_torch(points, original_size)

            box = torch.tensor([box_np], dtype=torch.float32)
            box = self.resize_transform.apply_boxes_torch(box, original_size)

            gt2D = torch.tensor(resized_gt, dtype=torch.long).unsqueeze(0)

            return {
                "img_name": os.path.basename(self.images[index]),
                "resize_img": resized_image,
                "image_raw": image_raw,
                "image_org": image,
                "gt2D": gt2D,
                "box": box,
                "points": points,
                "points_original": points_original,
                "points_label": point_labels,
                "image_ori_size": tuple(original_size),
                "size_before_pad": tuple(size_before_pad),
            }
        except Exception as e:
            print(f"加载数据失败 index={index}: {e}")
            return None


def collate_fn(batched_input: List[Optional[Dict]]) -> Optional[Dict]:
    """SAM 随机点 batch collate。"""
    valid_inputs = [item for item in batched_input if item is not None]
    if not valid_inputs:
        return None

    images = []
    images_raw = []
    images_original = []
    points = []
    points_original = []
    point_labels = []
    boxes = []
    image_sizes = []
    image_names = []
    labels = []
    size_before_pad = []

    for item in valid_inputs:
        image_tensor = torch.from_numpy(item["resize_img"])
        if image_tensor.dim() == 4 and image_tensor.shape[0] == 1:
            image_tensor = image_tensor.squeeze(0)
        images.append(image_tensor)

        images_raw.append(item["image_raw"])
        images_original.append(item["image_org"])

        pts = item["points"]
        pls = item["points_label"]
        if pts.dim() == 2:
            pts = pts.unsqueeze(0)
            pls = pls.unsqueeze(0)
        points.append(pts)
        points_original.append(item["points_original"])
        point_labels.append(pls)

        box = item["box"]
        if box.dim() == 1:
            box = box.unsqueeze(0)
        boxes.append(box)

        size_before_pad.append(item["size_before_pad"])
        image_sizes.append(item["image_ori_size"])
        image_names.append(item["img_name"])
        labels.append(item["gt2D"])

    return {
        "images": torch.stack(images, dim=0),
        "images_raw": images_raw,
        "images_original": images_original,
        "points": torch.cat(points, dim=0),
        "points_original": torch.stack(points_original, dim=0),
        "point_labels": torch.cat(point_labels, dim=0),
        "boxes": torch.cat(boxes, dim=0),
        "image_sizes": image_sizes,
        "image_names": image_names,
        "size_before_pad": size_before_pad,
        "labels": labels,
    }
