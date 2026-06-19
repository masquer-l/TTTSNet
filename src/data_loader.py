import os
import numpy as np
import glob
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple
from utils.custom_augmentations import AddDustParticles, AddLaserPointer, AddOpticalFiber, AddStructuralDefects


class FetoscopicDataset(Dataset):
    """
    FetoscopicDataset class for loading and transforming fetoscopic image data.

    Args:
        data_path (str): Path to the data directory.
        mode (str): Mode of the dataset, either 'train', 'valid', or None.
        original_image_output (bool): Whether to output the original image alongside the transformed one.
        img_size (int): Size to resize the images.
        crop_size (int): Size to crop the images.
        binary (bool): Whether to binarize the labels.
    """

    def __init__(self,
                 data_path: str = None,
                 mode: str = None,
                 original_image_output: bool = False,
                 img_size: int = 448,
                 crop_size: int = 256,
                 binary: bool = True
                 ) -> None:
        """
        Initializes the dataset with paths, transformations, and other configurations.

        Args:
            data_path (str): Path to the dataset.
            mode (str): Mode of the dataset, either 'train', 'valid', or None.
            original_image_output (bool): If True, returns original images along with augmented ones.
            img_size (int): Size to resize the images.
            crop_size (int): Size to crop the images.
            binary (bool): If True, labels are binarized.
        """
        self.data_path = data_path
        assert mode in ["train", "valid", None]
        self.mode = mode
        self.org_img_out = original_image_output
        self.img_size = img_size
        self.crop_size = crop_size
        self.binary = binary

        self.video_ID = []
        for video in glob.glob(self.data_path + "/*/images/*.png", recursive=True):
            video_name = os.path.basename(video)
            video_name = video_name.split("_")[0]
            self.video_ID.append(video_name)

        self.images = sorted(
            glob.glob(self.data_path + "/*/images/*.png", recursive=True))
        self.labels = sorted(
            glob.glob(self.data_path + "/*/labels/*.png", recursive=True))
        assert len(self.images) == len(self.labels)

        self.resize_transform = A.Compose([
            A.Resize(self.img_size, self.img_size),
            ToTensorV2()
        ])

        self.train_transform = A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.OneOf([
                A.Blur(blur_limit=(3, 7), p=0.25),
                A.MotionBlur(blur_limit=(3, 7), p=0.45)
            ], p=0.2),
            A.OneOf([
                A.OneOf([
                    AddLaserPointer(p=0.1),
                    AddDustParticles(p=0.2),
                    AddStructuralDefects(p=0.2),
                    AddOpticalFiber(p=0.4),
                ], p=0.4),
                A.Sequential([
                    AddDustParticles(),
                    AddStructuralDefects()
                ], p=0.25),
                A.Sequential([
                    AddLaserPointer(),
                    AddOpticalFiber()
                ], p=0.25),
                A.Sequential([
                    AddLaserPointer(),
                    AddDustParticles(),
                    AddOpticalFiber()
                ], p=0.15),
                A.Sequential([
                    AddLaserPointer(),
                    AddDustParticles(),
                    AddOpticalFiber()
                ], p=0.15),
                A.Sequential([
                    AddLaserPointer(),
                    AddDustParticles(),
                    AddOpticalFiber()
                ], p=0.15),
                A.Sequential([
                    AddLaserPointer(),
                    AddStructuralDefects(),
                    AddOpticalFiber()
                ], p=0.15),
                A.Sequential([
                    AddDustParticles(),
                    AddStructuralDefects(),
                    AddOpticalFiber()
                ], p=0.15),
                A.Sequential([
                    AddLaserPointer(),
                    AddDustParticles(),
                    AddStructuralDefects(),
                    AddOpticalFiber()
                ], p=0.05),
            ], p=0.5),
            A.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT,
                               shift_limit=0.025,
                               rotate_limit=40,
                               scale_limit=0.2,
                               p=0.2),
            A.ColorJitter(saturation=0.2, hue=0.15, p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.15, 0.05), contrast_limit=(-0.1, 0.2), p=0.3),
            A.CLAHE(clip_limit=1.0, tile_grid_size=(16, 16), p=0.15),
            A.Normalize(),
            ToTensorV2()
        ])

        self.valid_transform = A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(),
            ToTensorV2()
        ])

    def __getitem__(self,
                    x: int
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gets the item at the specified index.

        Args:
            x (int): Index of the item.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Transformed image and label tensors.
        """
        image_org = cv2.imread(self.images[x])
        image_org = cv2.cvtColor(image_org, cv2.COLOR_BGR2RGB)
        label_org = cv2.imread(self.labels[x], 0)
        if self.binary:
            label_org = np.where(label_org > 1, 0, label_org)

        if self.mode == "train":
            transformed_img = self.train_transform(image=image_org, mask=label_org)
        elif self.mode == "valid":
            transformed_img = self.valid_transform(image=image_org, mask=label_org)

        image = transformed_img["image"]
        label = transformed_img["mask"]

        if self.org_img_out:
            transformed_img = self.resize_transform(image=image_org, mask=label_org)
            image_org = transformed_img["image"]
            label_org = transformed_img["mask"]
            return image, image_org, label, label_org
        else:
            return image, label

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.images)
