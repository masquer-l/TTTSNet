import os
import glob
import re
from typing import Tuple, List, Dict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.custom_augmentations import CustomDefectsAugmentation


class TTTSNetTemporalDataset(Dataset):
    """
    TTTSNet 时序数据集：返回 3 帧连续片段 [t-1, t, t+1] 和中间帧 mask。
    3 帧共享同一个模型，训练时用时序一致性约束相邻帧输出。

    注意：为保证时序 loss 的时空对齐，本数据集对 3 帧应用同一组增强参数
    （确定性 resize + normalize，同步颜色增强，同步水平翻转）。
    可选强增强（use_strong_aug=True）会加入 CustomDefectsAugmentation，
    同样通过 ReplayCompose 在三帧间同步。
    """

    def __init__(
        self,
        data_path: str,
        mode: str = "train",
        img_size: int = 448,
        binary: bool = True,
        use_strong_aug: bool = False,
    ):
        assert mode in ["train", "valid"]
        self.data_path = data_path
        self.mode = mode
        self.img_size = img_size
        self.binary = binary
        self.use_strong_aug = use_strong_aug

        self.video_clips = self._build_video_clips()
        self.total_clips = len(self.video_clips)

        # 基础训练增强（所有 temporal 实验共享）
        base_train_transforms = [
            A.Resize(self.img_size, self.img_size),
            A.ColorJitter(saturation=0.2, hue=0.15, p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.15, 0.05), contrast_limit=(-0.1, 0.2), p=0.3),
            A.CLAHE(clip_limit=1.0, tile_grid_size=(16, 16), p=0.15),
            A.Normalize(),
            ToTensorV2(),
        ]

        if self.use_strong_aug:
            # 在 resize 后、normalize 前插入与 baseline 类似的强增强
            # 注意：水平翻转已在 transform 前通过 numpy 同步完成，此处不再重复
            strong_transforms = [
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
            ]
            # 插入到 Resize 之后、ColorJitter 之前
            base_train_transforms = [
                base_train_transforms[0],
                *strong_transforms,
                *base_train_transforms[1:],
            ]

        # 使用时序同步增强：对 3 帧应用同一组随机参数，保证时序一致性
        self.train_transform = A.ReplayCompose(base_train_transforms)

        self.valid_transform = A.ReplayCompose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(),
            ToTensorV2(),
        ])

    def _extract_video_frame(self, path: str) -> Tuple[str, int]:
        """从路径中提取 video_id 和 frame number"""
        basename = os.path.basename(path)
        match = re.match(r"(Video\d+)_(?:frame)?(\d+)\.png", basename)
        if match:
            return match.group(1), int(match.group(2))
        video_id = os.path.basename(os.path.dirname(os.path.dirname(path)))
        return video_id, 0

    def _build_video_clips(self) -> List[Dict[str, List[str]]]:
        """构建每个 video 的连续帧列表，并生成 3 帧 clip 索引"""
        image_paths = sorted(glob.glob(os.path.join(self.data_path, "*/images/*.png"), recursive=True))
        label_paths = sorted(glob.glob(os.path.join(self.data_path, "*/labels/*.png"), recursive=True))
        assert len(image_paths) == len(label_paths)

        video_frames: Dict[str, List[Tuple[str, str, int]]] = {}
        for img_path, lbl_path in zip(image_paths, label_paths):
            video_id, frame_num = self._extract_video_frame(img_path)
            if video_id not in video_frames:
                video_frames[video_id] = []
            video_frames[video_id].append((img_path, lbl_path, frame_num))

        clips = []
        for video_id in sorted(video_frames.keys()):
            frames = sorted(video_frames[video_id], key=lambda x: x[2])
            for i in range(1, len(frames) - 1):
                clips.append({
                    "video_id": video_id,
                    "image_paths": [frames[i-1][0], frames[i][0], frames[i+1][0]],
                    "label_paths": [frames[i-1][1], frames[i][1], frames[i+1][1]],
                    "frame_nums": [frames[i-1][2], frames[i][2], frames[i+1][2]],
                })

        return clips

    def _load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"无法读取图像: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _load_label(self, path: str) -> np.ndarray:
        label = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise ValueError(f"无法读取标签: {path}")
        if self.binary:
            label = np.where(label > 1, 0, label)
            label = (label > 0).astype(np.uint8)
        return label

    def __len__(self) -> int:
        return self.total_clips

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        clip = self.video_clips[idx]
        image_paths = clip["image_paths"]
        label_paths = clip["label_paths"]

        images = [self._load_image(p) for p in image_paths]
        labels = [self._load_label(p) for p in label_paths]

        # 训练模式下对 3 帧和中间帧 mask 做一致的水平翻转
        if self.mode == "train" and np.random.random() < 0.5:
            images = [np.fliplr(img).copy() for img in images]
            labels = [np.fliplr(lbl).copy() for lbl in labels]

        if self.mode == "train":
            transform = self.train_transform
        else:
            transform = self.valid_transform

        # 中间帧同步增强 image + mask，并记录随机参数
        middle = transform(image=images[1], mask=labels[1])
        replay_params = middle["replay"]

        # 用同一组参数增强相邻帧，保证三帧增强一致
        transformed_images = [
            A.ReplayCompose.replay(replay_params, image=images[0])["image"],
            middle["image"],
            A.ReplayCompose.replay(replay_params, image=images[2])["image"],
        ]
        mask_tensor = middle["mask"].unsqueeze(0).float()

        clip_tensor = torch.stack(transformed_images, dim=0)  # [3, 3, H, W]

        return clip_tensor, mask_tensor

    def get_sample_paths(self, idx: int) -> Dict:
        return self.video_clips[idx]
