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


class TTTSNetTemporalDataset(Dataset):
    """
    TTTSNet 时序数据集：返回 3 帧连续片段 [t-1, t, t+1] 和中间帧 mask。
    3 帧共享同一个模型，训练时用时序一致性约束相邻帧输出。

    注意：为保证时序 loss 的空间对齐，本数据集只做确定性的 resize + normalize，
    不做随机几何变换。后续可加入基于 ReplayCompose 的一致增强。
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

        self.video_clips = self._build_video_clips()
        self.total_clips = len(self.video_clips)

        self.train_transform = A.Compose([
            A.Resize(self.img_size, self.img_size),
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

        if self.mode == "train":
            transformed_images = [self.train_transform(image=img)["image"] for img in images]
            transformed_label = self.train_transform(image=images[1], mask=labels[1])
        else:
            transformed_images = [self.valid_transform(image=img)["image"] for img in images]
            transformed_label = self.valid_transform(image=images[1], mask=labels[1])
        mask_tensor = transformed_label["mask"].unsqueeze(0).float()

        clip_tensor = torch.stack(transformed_images, dim=0)  # [3, 3, H, W]

        return clip_tensor, mask_tensor

    def get_sample_paths(self, idx: int) -> Dict:
        return self.video_clips[idx]
