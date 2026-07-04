"""TTTSNet variants with a lightweight transformer-enhanced decoder."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.TTTSNet import (
    Conv,
    DownSamplingBlock,
    Init_Block,
    InputInjection,
    RFFM_A,
    RFFM_B,
    SEM_B_Block,
)
from models.tttsnet_vit import TTTSNetViT, ViTAdapter
from segment_anything import sam_model_registry

__all__ = ["TTTSNetTransformerDecoder", "TTTSNetViTTransformerDecoder"]


class GlobalTransformerBlock(nn.Module):
    """Global context block applied on pooled deep decoder features."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        pooled_size: int = 28,
        num_layers: int = 1,
        use_pos_embed: bool = True,
    ):
        super().__init__()
        self.pooled_size = pooled_size
        self.use_pos_embed = use_pos_embed
        self.in_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Conv2d(embed_dim, in_channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(in_channels)

        if self.use_pos_embed:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, pooled_size * pooled_size, embed_dim)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_size = x.shape[-2:]
        pooled = F.interpolate(
            x,
            size=(self.pooled_size, self.pooled_size),
            mode="bilinear",
            align_corners=False,
        )
        tokens = self.in_proj(pooled).flatten(2).transpose(1, 2)
        if self.use_pos_embed:
            tokens = tokens + self.pos_embed
        tokens = self.encoder(tokens)
        context = tokens.transpose(1, 2).reshape(
            x.shape[0],
            -1,
            self.pooled_size,
            self.pooled_size,
        )
        context = self.out_proj(context)
        context = F.interpolate(context, size=original_size, mode="bilinear", align_corners=False)
        return self.norm(x + context)


class TransformerMAD(nn.Module):
    """MAD decoder with a lightweight global transformer on deep features."""

    def __init__(
        self,
        c1: int = 32,
        c2: int = 64,
        classes: int = 2,
        num_features: int = 64,
        transformer_dim: int = 128,
        transformer_heads: int = 4,
        transformer_pooled_size: int = 28,
        transformer_num_layers: int = 1,
        transformer_use_pos_embed: bool = True,
    ):
        super().__init__()
        self.c1, self.c2 = c1, c2
        self.num_features = num_features
        deep_channels = 8 * self.num_features + 3

        self.mid_layer_1x1 = Conv(4 * self.num_features + 3, c1, 1, 1, padding=0, bn_acti=False)
        self.deep_layer_1x1 = Conv(deep_channels, c2, 1, 1, padding=0, bn_acti=False)
        self.global_context = GlobalTransformerBlock(
            in_channels=deep_channels,
            embed_dim=transformer_dim,
            num_heads=transformer_heads,
            pooled_size=transformer_pooled_size,
            num_layers=transformer_num_layers,
            use_pos_embed=transformer_use_pos_embed,
        )

        self.DwConv1 = Conv(self.c1 + self.c2, self.c1 + self.c2, (3, 3), 1, padding=(1, 1),
                            groups=self.c1 + self.c2, bn_acti=True)
        self.PwConv1 = Conv(self.c1 + self.c2, classes, 1, 1, padding=0, bn_acti=False)
        self.DwConv2 = Conv(deep_channels, deep_channels, (3, 3), 1, padding=(1, 1),
                            groups=deep_channels, bn_acti=True)
        self.PwConv2 = Conv(deep_channels, classes, 1, 1, padding=0, bn_acti=False)

    def forward(self, x):
        x1, x2 = x
        x2 = self.global_context(x2)
        x2_size = x2.size()[2:]

        x1_ = self.mid_layer_1x1(x1)
        x2_ = self.deep_layer_1x1(x2)
        x2_ = F.interpolate(x2_, [x2_size[0] * 2, x2_size[1] * 2], mode="bilinear", align_corners=False)

        att = torch.cat([x1_, x2_], 1)
        att = self.DwConv1(att)
        att = torch.sigmoid(self.PwConv1(att))

        out = self.DwConv2(x2)
        out = self.PwConv2(out)
        out = F.interpolate(out, [x2_size[0] * 2, x2_size[1] * 2], mode="bilinear", align_corners=False)
        out = out * att
        out = F.interpolate(out, [x2_size[0] * 8, x2_size[1] * 8], mode="bilinear", align_corners=False)
        return out


class TTTSNetTransformerDecoder(nn.Module):
    def __init__(
        self,
        classes: int = 2,
        block_1: int = 3,
        block_2: int = 8,
        num_features: int = 64,
        transformer_dim: int = 128,
        transformer_heads: int = 4,
        transformer_pooled_size: int = 28,
        transformer_num_layers: int = 1,
        transformer_use_pos_embed: bool = True,
    ):
        super().__init__()
        self.classes = classes
        self.block_1 = block_1
        self.block_2 = block_2
        self.num_features = num_features
        self.Init_Block = Init_Block(n_features=self.num_features)
        self.down_1 = InputInjection(1)
        self.down_2 = InputInjection(2)
        self.down_3 = InputInjection(3)
        self.RFFM_A = RFFM_A(self.num_features + 3)
        self.downsample_1 = DownSamplingBlock(self.num_features + 3, 2 * self.num_features)
        self.SEM_B_Block1 = SEM_B_Block(num_channels=2 * self.num_features, num_block=self.block_1,
                                        dilation=[2, 2, 2], flag=1)
        self.RFFM_B1 = RFFM_B(ch_in=4 * self.num_features + 3, ch_MEDCAM=2 * self.num_features)
        self.downsample_2 = DownSamplingBlock(4 * self.num_features + 3, 4 * self.num_features)
        self.SEM_B_Block2 = SEM_B_Block(num_channels=4 * self.num_features, num_block=self.block_2,
                                        dilation=[4, 4, 8, 8, 16, 16, 32, 32], flag=2)
        self.RFFM_B2 = RFFM_B(ch_in=8 * self.num_features + 3, ch_MEDCAM=4 * self.num_features)
        self.MAD = TransformerMAD(
            classes=self.classes,
            num_features=self.num_features,
            transformer_dim=transformer_dim,
            transformer_heads=transformer_heads,
            transformer_pooled_size=transformer_pooled_size,
            transformer_num_layers=transformer_num_layers,
            transformer_use_pos_embed=transformer_use_pos_embed,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out_init_block = self.Init_Block(input)
        down_1 = self.down_1(input)
        out_RFFM_a = self.RFFM_A((out_init_block, down_1))
        out_downsample_1 = self.downsample_1(out_RFFM_a)
        out_sem_block1 = self.SEM_B_Block1(out_downsample_1)
        down_2 = self.down_2(input)
        out_RFFM_b1 = self.RFFM_B1((out_sem_block1, out_downsample_1, down_2))
        out_downsample_2 = self.downsample_2(out_RFFM_b1)
        out_se_block2 = self.SEM_B_Block2(out_downsample_2)
        down_3 = self.down_3(input)
        out_RFFM_b2 = self.RFFM_B2((out_se_block2, out_downsample_2, down_3))
        return self.MAD((out_RFFM_b1, out_RFFM_b2))


class TTTSNetViTTransformerDecoder(TTTSNetViT):
    def __init__(
        self,
        classes: int = 2,
        block_1: int = 3,
        block_2: int = 8,
        num_features: int = 64,
        sam_checkpoint: str = "/autodl-fs/data/masquer.li/model/sam_vit_b_01ec64.pth",
        freeze_vit: bool = False,
        transformer_dim: int = 128,
        transformer_heads: int = 4,
        transformer_pooled_size: int = 28,
        transformer_num_layers: int = 1,
        transformer_use_pos_embed: bool = True,
    ):
        nn.Module.__init__(self)
        self.classes = classes
        self.block_1 = block_1
        self.block_2 = block_2
        self.num_features = num_features

        sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
        self.vit_encoder = sam.image_encoder
        if freeze_vit:
            for p in self.vit_encoder.parameters():
                p.requires_grad = False

        self.vit_adapter = ViTAdapter(in_chans=256, out_chans=num_features)
        self.down_1 = InputInjection(1)
        self.down_2 = InputInjection(2)
        self.down_3 = InputInjection(3)
        self.RFFM_A = RFFM_A(self.num_features + 3)
        self.downsample_1 = DownSamplingBlock(self.num_features + 3, 2 * self.num_features)
        self.SEM_B_Block1 = SEM_B_Block(num_channels=2 * self.num_features, num_block=self.block_1,
                                        dilation=[2, 2, 2], flag=1)
        self.RFFM_B1 = RFFM_B(ch_in=4 * self.num_features + 3, ch_MEDCAM=2 * self.num_features)
        self.downsample_2 = DownSamplingBlock(4 * self.num_features + 3, 4 * self.num_features)
        self.SEM_B_Block2 = SEM_B_Block(num_channels=4 * self.num_features, num_block=self.block_2,
                                        dilation=[4, 4, 8, 8, 16, 16, 32, 32], flag=2)
        self.RFFM_B2 = RFFM_B(ch_in=8 * self.num_features + 3, ch_MEDCAM=4 * self.num_features)
        self.MAD = TransformerMAD(
            classes=self.classes,
            num_features=self.num_features,
            transformer_dim=transformer_dim,
            transformer_heads=transformer_heads,
            transformer_pooled_size=transformer_pooled_size,
            transformer_num_layers=transformer_num_layers,
            transformer_use_pos_embed=transformer_use_pos_embed,
        )
        self.register_buffer("pixel_mean", torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1), persistent=False)


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    image = torch.rand((1, 3, 448, 448))
    model = TTTSNetTransformerDecoder(classes=2, num_features=64)
    print(model(image).shape)
