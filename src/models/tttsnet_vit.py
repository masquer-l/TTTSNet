# TTTSNet with SAM ViT-B backbone

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# 使用当前项目内的 segment_anything
from segment_anything import sam_model_registry

# 复用 TTTSNet 的模块
from models.TTTSNet import (
    Conv,
    DownSamplingBlock,
    InputInjection,
    MAD,
    RFFM_A,
    RFFM_B,
    SEM_B_Block,
)

__all__ = ["TTTSNetViT"]


class ViTAdapter(nn.Module):
    """将 SAM ViT-B image encoder 的输出适配到 TTTSNet 后续模块期望的尺寸。

    SAM ViT-B 输出 [B, 256, H/16, W/16]；
    原 TTTSNet Init_Block 输出 [B, num_features, H/2, W/2]。
    因此需要：1x1 conv 降维 + 8x 上采样。
    """

    def __init__(self, in_chans: int = 256, out_chans: int = 64):
        super().__init__()
        self.conv1x1 = Conv(in_chans, out_chans, kSize=1, stride=1, padding=0, bn_acti=True)

    def forward(self, x: torch.Tensor, target_size: torch.Size) -> torch.Tensor:
        x = self.conv1x1(x)
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return x


class TTTSNetViT(nn.Module):
    """TTTSNet 变体：将 Init_Block 替换为 SAM ViT-B image encoder + adapter。

    其余结构（RFFM-A/B、SEM-B blocks、MAD）与原版 TTTSNet 保持一致，
    用于在受控条件下评估 ViT backbone 相对原版 CNN stem 的收益。
    """

    def __init__(
        self,
        classes: int = 2,
        block_1: int = 3,
        block_2: int = 8,
        num_features: int = 64,
        sam_checkpoint: str = "/autodl-fs/data/masquer.li/model/sam_vit_b_01ec64.pth",
        freeze_vit: bool = False,
    ):
        super().__init__()
        self.classes = classes
        self.block_1 = block_1
        self.block_2 = block_2
        self.num_features = num_features

        # SAM ViT-B image encoder
        # 允许 sam_checkpoint 为 None 或文件不存在：仅初始化结构，后续由完整 checkpoint 覆盖权重
        if sam_checkpoint is None:
            sam = sam_model_registry["vit_b"](checkpoint=None)
        elif Path(sam_checkpoint).exists():
            sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
        else:
            print(f"Warning: sam_checkpoint not found: {sam_checkpoint}, initializing from scratch.")
            sam = sam_model_registry["vit_b"](checkpoint=None)
        self.vit_encoder = sam.image_encoder

        if freeze_vit:
            for p in self.vit_encoder.parameters():
                p.requires_grad = False

        # adapter：ViT 输出 256ch -> num_features，上采样 8x
        self.vit_adapter = ViTAdapter(in_chans=256, out_chans=num_features)

        # 原 TTTSNet 后续结构
        self.down_1 = InputInjection(1)
        self.down_2 = InputInjection(2)
        self.down_3 = InputInjection(3)

        self.RFFM_A = RFFM_A(self.num_features + 3)

        self.downsample_1 = DownSamplingBlock(self.num_features + 3, 2 * self.num_features)

        self.SEM_B_Block1 = SEM_B_Block(
            num_channels=2 * self.num_features,
            num_block=self.block_1,
            dilation=[2, 2, 2],
            flag=1,
        )

        self.RFFM_B1 = RFFM_B(ch_in=4 * self.num_features + 3, ch_MEDCAM=2 * self.num_features)

        self.downsample_2 = DownSamplingBlock(4 * self.num_features + 3, 4 * self.num_features)

        self.SEM_B_Block2 = SEM_B_Block(
            num_channels=4 * self.num_features,
            num_block=self.block_2,
            dilation=[4, 4, 8, 8, 16, 16, 32, 32],
            flag=2,
        )

        self.RFFM_B2 = RFFM_B(ch_in=8 * self.num_features + 3, ch_MEDCAM=4 * self.num_features)

        self.MAD = MAD(classes=self.classes, num_features=self.num_features)

        # SAM 预训练使用的 pixel mean/std
        self.register_buffer(
            "pixel_mean",
            torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
            persistent=False,
        )

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """TTTSNet dataset 输出 [0,1]，需要转换到 SAM 的 [mean/std] 空间。"""
        x = x * 255.0
        x = (x - self.pixel_mean) / (self.pixel_std + 1e-8)
        return x

    def _resize_pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        """将预训练的 1024 位置编码插值到当前输入尺寸。"""
        if self.vit_encoder.pos_embed is None:
            return x

        # x: [B, H, W, C] after patch_embed
        B, H, W, C = x.shape
        pos_embed = self.vit_encoder.pos_embed  # [1, Hp, Wp, C]
        if H == pos_embed.shape[1] and W == pos_embed.shape[2]:
            return x + pos_embed

        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [1, C, Hp, Wp]
        pos_embed_resized = F.interpolate(pos_embed, size=(H, W), mode="bilinear", align_corners=False)
        pos_embed_resized = pos_embed_resized.permute(0, 2, 3, 1)  # [1, H, W, C]
        return x + pos_embed_resized

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        B, _, H, W = input.shape
        target_size = (H // 2, W // 2)

        # ViT backbone
        x = self._preprocess(input)
        x = self.vit_encoder.patch_embed(x)  # [B, H/16, W/16, C]
        x = self._resize_pos_embed(x)

        for blk in self.vit_encoder.blocks:
            x = blk(x)

        x = self.vit_encoder.neck(x.permute(0, 3, 1, 2))  # [B, 256, H/16, W/16]
        out_init_block = self.vit_adapter(x, target_size)  # [B, num_features, H/2, W/2]

        # 后续结构与原版 TTTSNet 完全一致
        down_1 = self.down_1(input)
        input_RFFM_a = out_init_block, down_1
        out_RFFM_a = self.RFFM_A(input_RFFM_a)

        out_downsample_1 = self.downsample_1(out_RFFM_a)
        out_sem_block1 = self.SEM_B_Block1(out_downsample_1)

        down_2 = self.down_2(input)
        input_sem1_MEDCAM1 = out_sem_block1, out_downsample_1, down_2
        out_RFFM_b1 = self.RFFM_B1(input_sem1_MEDCAM1)

        out_downsample_2 = self.downsample_2(out_RFFM_b1)
        out_se_block2 = self.SEM_B_Block2(out_downsample_2)

        down_3 = self.down_3(input)
        input_sem2_MEDCAM2 = out_se_block2, out_downsample_2, down_3
        out_RFFM_b2 = self.RFFM_B2(input_sem2_MEDCAM2)

        input_RFFMb1_RFFMb2 = out_RFFM_b1, out_RFFM_b2
        out_mad = self.MAD(input_RFFMb1_RFFMb2)

        return out_mad


if __name__ == "__main__":
    # 兼容直接运行本文件
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    image = torch.rand((1, 3, 448, 448))
    model = TTTSNetViT(classes=2, num_features=64, freeze_vit=True)
    out = model(image)
    print("Output shape:", out.shape)
    print("Trainable params:", count_parameters(model))
