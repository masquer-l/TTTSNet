# TTTSNet - Temporal Test-Time Training Segmentation Network #

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

__all__ = ["TTTSNet"]


def channel_split(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将张量在通道维度上对半分割
    
    Args:
        x (torch.Tensor): 输入张量 (B, C, H, W)
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 分割后的两个张量
    """
    c = int(x.size()[1])
    c1 = round(c * 0.5)
    x1 = x[:, :c1, :, :].contiguous()
    x2 = x[:, c1:, :, :].contiguous()
    return x1, x2


# ============================================================================
# 基础卷积组件 (Basic Convolution Components)
# ============================================================================

class Conv(nn.Module):
    """
    基础卷积模块，可选择性地包含BatchNorm和PReLU激活
    
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int): 卷积核大小
        stride (int): 步长
        padding (int): 填充
        dilation (tuple): 膨胀系数
        groups (int): 分组卷积的组数
        with_bn_activation (bool): 是否包含BN和激活
        bias (bool): 是否使用偏置
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int, 
                 stride: int, 
                 padding: int, 
                 dilation: Tuple[int, int] = (1, 1), 
                 groups: int = 1, 
                 with_bn_activation: bool = False, 
                 bias: bool = False):
        super().__init__()

        self.with_bn_activation = with_bn_activation

        self.conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=kernel_size,
            stride=stride, 
            padding=padding,
            dilation=dilation, 
            groups=groups, 
            bias=bias
        )

        if self.with_bn_activation:
            self.bn_prelu = BNPReLU(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.conv(x)
        
        if self.with_bn_activation:
            x = self.bn_prelu(x)

        return x


class BNPReLU(nn.Module):
    """
    BatchNorm + PReLU 激活组合
    
    Args:
        num_channels (int): 输入通道数
    """
    def __init__(self, num_channels: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_channels, eps=1e-3)
        self.activation = nn.PReLU(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.bn(x)
        x = self.activation(x)
        return x


class Conv3x3Resume(nn.Module):
    """
    3x3卷积后接1x1卷积恢复通道数的组合模块
    
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int): 卷积核大小
        stride (int): 步长
        padding (int): 填充
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int, 
                 stride: int, 
                 padding: int, 
                 dilation: Tuple[int, int] = (1, 1), 
                 groups: int = 1, 
                 with_bn_activation: bool = False, 
                 bias: bool = False):
        super().__init__()
        
        # 3x3 卷积处理一半通道
        self.conv3x3 = Conv(
            in_channels // 2, in_channels // 2, 
            kernel_size, 1, padding=1, 
            with_bn_activation=True
        )
        
        # 1x1 卷积恢复通道数
        self.conv1x1_resume = Conv(
            in_channels // 2, in_channels, 
            1, 1, padding=0, 
            with_bn_activation=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.conv3x3(x)
        x = self.conv1x1_resume(x)
        return x


# ============================================================================
# 网络初始化和下采样模块 (Initialization and Downsampling Modules)
# ============================================================================

class InitBlock(nn.Module):
    """
    网络初始化块，用于提取基础特征
    
    Args:
        num_features (int): 输出特征通道数
    """
    def __init__(self, num_features: int = 32):
        super().__init__()
        self.num_features = num_features
        
        self.stem = nn.Sequential(
            Conv(3, self.num_features, 3, 2, padding=1, with_bn_activation=True), 
            Conv(self.num_features, self.num_features, 3, 1, padding=1, with_bn_activation=True),
            Conv(self.num_features, self.num_features, 3, 1, padding=1, with_bn_activation=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.stem(x)


class DownSamplingBlock(nn.Module):
    """
    下采样块，结合卷积和最大池化
    
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 确定卷积输出通道数
        if self.in_channels < self.out_channels:
            conv_channels = out_channels - in_channels
        else:
            conv_channels = out_channels

        self.conv3x3 = Conv(in_channels, conv_channels, 3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.bn_prelu = BNPReLU(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        conv_out = self.conv3x3(x)

        # 如果输入通道数小于输出通道数，拼接池化结果
        if self.in_channels < self.out_channels:
            pool_out = self.max_pool(x)
            output = torch.cat([conv_out, pool_out], dim=1)
        else:
            output = conv_out

        return self.bn_prelu(output)


class InputInjection(nn.Module):
    """
    输入注入模块，对原始输入进行多级下采样
    
    Args:
        downsample_ratio (int): 下采样倍数
    """
    def __init__(self, downsample_ratio: int):
        super().__init__()
        self.pools = nn.ModuleList([
            nn.MaxPool2d(3, stride=2, padding=1) 
            for _ in range(downsample_ratio)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        for pool in self.pools:
            x = pool(x)
        return x


# ============================================================================
# 空间和通道注意力模块 (Spatial and Channel Attention Modules)
# ============================================================================

class SEM_B(nn.Module):
    """
    空间增强模块 - B型
    将输入在通道维度上分割，分别进行常规卷积和空洞卷积，然后拼接
    
    Args:
        in_channels (int): 输入通道数
        dilation_rate (int): 膨胀率
        kernel_size (int): 卷积核大小
        dilated_kernel_size (int): 膨胀卷积核大小
    """
    def __init__(self, 
                 in_channels: int, 
                 dilation_rate: int = 1, 
                 kernel_size: int = 3, 
                 dilated_kernel_size: int = 3):
        super().__init__()

        # 初始3x3卷积，将通道数减半
        self.conv3x3 = Conv(
            in_channels, in_channels // 2, 
            kernel_size, 1, padding=1, 
            with_bn_activation=True
        )

        # 左分支：常规深度可分离卷积
        self.dconv_left = Conv(
            in_channels // 4, in_channels // 4, 
            (dilated_kernel_size, dilated_kernel_size), 1,
            padding=(1, 1), 
            groups=in_channels // 4, 
            with_bn_activation=True
        )

        # 右分支：膨胀深度可分离卷积
        self.dconv_right = Conv(
            in_channels // 4, in_channels // 4, 
            (dilated_kernel_size, dilated_kernel_size), 1,
            padding=(1 * dilation_rate, 1 * dilation_rate), 
            groups=in_channels // 4, 
            dilation=(dilation_rate, dilation_rate), 
            with_bn_activation=True
        )

        # 恢复通道数的卷积
        self.conv_resume = Conv3x3Resume(
            in_channels, in_channels, 
            (dilated_kernel_size, dilated_kernel_size), 1,
            padding=(1, 1), 
            with_bn_activation=True
        )
        
        # 最终的BN和激活
        self.final_bn_prelu = BNPReLU(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        identity = x
        
        # 初始卷积
        out = self.conv3x3(x)

        # 通道分割
        left, right = channel_split(out)

        # 分别处理左右分支
        left = self.dconv_left(left)
        right = self.dconv_right(right)

        # 拼接并恢复通道数
        out = torch.cat([left, right], dim=1)
        out = self.conv_resume(out)

        # 残差连接
        return self.final_bn_prelu(out + identity)


class SENetBlock(nn.Module):
    """
    Squeeze-and-Excitation 网络块
    
    Args:
        in_channels (int): 输入通道数
        reduction (int): 通道压缩比例
    """
    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        
        self.squeeze_excite = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.PReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        b, c, _, _ = x.size()
        # 全局平均池化 -> 挤压
        y = x.view(b, c)
        # 激励
        y = self.squeeze_excite(y).view(b, c, 1, 1)
        return y


class MEDCAM(nn.Module):
    """
    多尺度高效双重通道注意力机制
    Multi-scale Efficient Dual Channel Attention Mechanism
    
    Args:
        in_channels (int): 输入通道数
        reduction (int): SE模块的通道压缩比例
    """
    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()

        # 分割池化：将特征图池化到2x2
        self.partition_pool = nn.AdaptiveMaxPool2d((2, 2))
        
        # 2x2深度可分离卷积
        self.conv2x2 = Conv(
            in_channels, in_channels, 
            2, 1, padding=0, 
            groups=in_channels, 
            with_bn_activation=False
        )
        
        # 全局池化
        self.global_pool = nn.AdaptiveMaxPool2d(1)
        
        # SE注意力模块
        self.se_block = SENetBlock(in_channels=in_channels, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 分割池化路径：2x2池化 + 2x2卷积
        partition_out = self.partition_pool(x) 
        partition_out = self.conv2x2(partition_out)

        # 全局池化路径
        global_out = self.global_pool(x)

        # 两路径特征融合
        fused_features = partition_out + global_out
        
        # 通过SE模块生成通道注意力权重
        attention_weights = self.se_block(fused_features)
        
        # 应用注意力权重
        return attention_weights * x


# ============================================================================
# 特征融合模块 (Feature Fusion Modules)
# ============================================================================

class RFFM_A(nn.Module):
    """
    富特征融合模块 - A型 (Rich Feature Fusion Module - Type A)
    将两个特征图拼接后进行1x1卷积，并使用残差连接
    
    Args:
        total_channels (int): 拼接后的总通道数
    """
    def __init__(self, total_channels: int):
        super().__init__()
        self.bn_prelu = BNPReLU(total_channels)
        self.conv1x1 = Conv(
            total_channels, total_channels, 
            1, 1, padding=0, 
            with_bn_activation=False
        )

    def forward(self, features: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features: (主特征, 辅助特征)
        
        Returns:
            融合后的特征
        """
        main_feature, aux_feature = features
        
        # 拼接特征
        fused = torch.cat([main_feature, aux_feature], dim=1)
        
        # BN + 激活 + 1x1卷积
        fused = self.bn_prelu(fused)
        output = self.conv1x1(fused)
        
        # 残差连接（如果通道数匹配）
        if main_feature.size(1) == output.size(1):
            output = output + main_feature
        elif main_feature.size(1) < output.size(1):
            # 如果主特征通道数较少，进行零填充
            pad_channels = output.size(1) - main_feature.size(1)
            main_feature_padded = F.pad(main_feature, (0, 0, 0, 0, 0, pad_channels))
            output = output + main_feature_padded
        else:
            # 如果输出通道数较少，对输出进行零填充
            pad_channels = main_feature.size(1) - output.size(1)
            output_padded = F.pad(output, (0, 0, 0, 0, 0, pad_channels))
            output = output_padded + main_feature

        return output


class RFFM_B(nn.Module):
    """
    富特征融合模块 - B型 (Rich Feature Fusion Module - Type B)
    融合三个特征图，其中一个通过MEDCAM注意力机制增强
    
    Args:
        total_channels (int): 拼接后的总通道数
        medcam_channels (int): MEDCAM处理的特征通道数
    """
    def __init__(self, total_channels: int, medcam_channels: int):
        super().__init__()
        self.medcam = MEDCAM(in_channels=medcam_channels, reduction=8)
        self.bn_prelu = BNPReLU(total_channels)
        self.conv1x1 = Conv(
            total_channels, total_channels, 
            1, 1, padding=0, 
            with_bn_activation=False
        )

    def forward(self, features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features: (主特征, MEDCAM特征, 辅助特征)
        
        Returns:
            融合后的特征
        """
        main_feature, medcam_feature, aux_feature = features
        
        # 对中间特征应用MEDCAM注意力
        enhanced_medcam_feature = self.medcam(medcam_feature)
        
        # 拼接所有特征
        fused = torch.cat([main_feature, enhanced_medcam_feature, aux_feature], dim=1)
        
        # BN + 激活 + 1x1卷积
        fused = self.bn_prelu(fused)
        output = self.conv1x1(fused)

        # 残差连接（如果通道数匹配）
        if main_feature.size(1) == output.size(1):
            output = output + main_feature
        elif main_feature.size(1) < output.size(1):
            # 如果主特征通道数较少，进行零填充
            pad_channels = output.size(1) - main_feature.size(1)
            main_feature_padded = F.pad(main_feature, (0, 0, 0, 0, 0, pad_channels))
            output = output + main_feature_padded
        else:
            # 如果输出通道数较少，对输出进行零填充
            pad_channels = main_feature.size(1) - output.size(1)
            output_padded = F.pad(output, (0, 0, 0, 0, 0, pad_channels))
            output = output_padded + main_feature
            
        return output


# ============================================================================
# 组合模块 (Composite Modules)
# ============================================================================

class SEM_B_Block(nn.Module):
    """
    SEM_B模块组合块
    
    Args:
        in_channels (int): 输入通道数
        num_blocks (int): SEM_B模块数量
        dilation_rates (List[int]): 膨胀率列表
        block_id (int): 块标识符
    """
    def __init__(self, 
                 in_channels: int, 
                 num_blocks: int, 
                 dilation_rates: List[int], 
                 block_id: int):
        super().__init__()
        
        self.sem_blocks = nn.Sequential()
        for i in range(num_blocks):
            self.sem_blocks.add_module(
                f"SEM_Block_{block_id}_{i}", 
                SEM_B(in_channels, dilation_rate=dilation_rates[i])
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.sem_blocks(x)


class MAD(nn.Module):
    """
    多尺度注意力解码器 (Multi-scale Attention Decoder)
    
    Args:
        mid_channels (int): 中层特征处理后的通道数
        deep_channels (int): 深层特征处理后的通道数
        num_classes (int): 分类数量
        num_features (int): 基础特征通道数
    """
    def __init__(self, 
                 mid_channels: int = 32, 
                 deep_channels: int = 64, 
                 num_classes: int = 19, 
                 num_features: int = 32):
        super().__init__()
        
        self.mid_channels = mid_channels
        self.deep_channels = deep_channels
        self.num_features = num_features

        # 中层特征处理
        self.mid_layer_conv = Conv(
            4 * self.num_features + 3, mid_channels, 
            1, 1, padding=0, 
            with_bn_activation=False
        )

        # 深层特征处理
        self.deep_layer_conv = Conv(
            8 * self.num_features + 3, deep_channels, 
            1, 1, padding=0, 
            with_bn_activation=False
        )

        # 融合特征的深度可分离卷积
        self.fused_dwconv = Conv(
            self.mid_channels + self.deep_channels, 
            self.mid_channels + self.deep_channels, 
            3, 1, padding=1,
            groups=self.mid_channels + self.deep_channels, 
            with_bn_activation=True
        )

        # 融合特征的逐点卷积（生成注意力图）
        self.fused_pwconv = Conv(
            self.mid_channels + self.deep_channels, num_classes, 
            1, 1, padding=0, 
            with_bn_activation=False
        )

        # 深层特征的深度可分离卷积
        self.deep_dwconv = Conv(
            8 * self.num_features + 3, 
            8 * self.num_features + 3, 
            3, 1, padding=1,
            groups=8 * self.num_features + 3, 
            with_bn_activation=True
        )
        
        # 深层特征的逐点卷积（生成最终输出）
        self.deep_pwconv = Conv(
            8 * self.num_features + 3, num_classes, 
            1, 1, padding=0, 
            with_bn_activation=False
        )

    def forward(self, features: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features: (中层特征, 深层特征)
        
        Returns:
            解码后的特征图
        """
        mid_features, deep_features = features
        # mid_features: [B, 4*num_features+3, H, W]
        # deep_features: [B, 8*num_features+3, H/2, W/2]
        
        deep_size = deep_features.size()[2:]

        # 处理中层和深层特征
        mid_processed = self.mid_layer_conv(mid_features)
        deep_processed = self.deep_layer_conv(deep_features)

        # 上采样深层特征以匹配中层特征尺寸
        deep_upsampled = F.interpolate(
            deep_processed, 
            size=[deep_size[0] * 2, deep_size[1] * 2], 
            mode='bilinear', 
            align_corners=False
        )

        # 拼接并生成注意力图
        fused_features = torch.cat([mid_processed, deep_upsampled], dim=1)
        fused_features = self.fused_dwconv(fused_features)
        attention_map = self.fused_pwconv(fused_features)
        attention_weights = torch.sigmoid(attention_map)

        # 处理深层特征生成最终输出
        deep_output = self.deep_dwconv(deep_features)
        deep_output = self.deep_pwconv(deep_output)
        deep_output = F.interpolate(
            deep_output, 
            size=[deep_size[0] * 2, deep_size[1] * 2], 
            mode='bilinear', 
            align_corners=False
        )

        # 应用注意力权重
        attended_output = deep_output * attention_weights

        # 最终上采样到原始输入尺寸
        final_output = F.interpolate(
            attended_output, 
            size=[deep_size[0] * 8, deep_size[1] * 8], 
            mode='bilinear', 
            align_corners=False
        )

        return final_output


# ============================================================================
# 主网络 (Main Network)
# ============================================================================

class TTTSNet(nn.Module):
    """
    TTTS网络 - 时间测试时训练分割网络
    Temporal Test-Time Training Segmentation Network
    
    Args:
        num_classes (int): 分类数量，默认为2（二分类）
        stage1_blocks (int): 第一阶段SEM_B块数量
        stage2_blocks (int): 第二阶段SEM_B块数量  
        num_features (int): 基础特征通道数
        img_size (int): 输入图像尺寸
    """
    def __init__(self, 
                 num_classes: int = 2, 
                 stage1_blocks: int = 3, 
                 stage2_blocks: int = 8, 
                 num_features: int = 64,
                 img_size: int = 1024):
        super().__init__()
        
        # 网络参数
        self.img_size = img_size
        self.num_classes = num_classes
        self.stage1_blocks = stage1_blocks
        self.stage2_blocks = stage2_blocks
        self.num_features = num_features
        
        # 初始化块
        self.init_block = InitBlock(num_features=self.num_features)

        # 输入注入模块（多尺度下采样）
        self.input_injection_1x = InputInjection(downsample_ratio=1)  # 1倍下采样
        self.input_injection_2x = InputInjection(downsample_ratio=2)  # 2倍下采样
        self.input_injection_4x = InputInjection(downsample_ratio=3)  # 4倍下采样

        # 第一阶段特征融合
        self.stage1_fusion = RFFM_A(total_channels=self.num_features + 3)

        # 第一次下采样
        self.downsample_stage1 = DownSamplingBlock(
            in_channels=self.num_features + 3, 
            out_channels=2 * self.num_features
        )

        # 第一阶段SEM_B块组合
        self.stage1_sem_blocks = SEM_B_Block(
            in_channels=2 * self.num_features, 
            num_blocks=self.stage1_blocks,
            dilation_rates=[2, 2, 2], 
            block_id=1
        )

        # 第一阶段特征融合（包含MEDCAM）
        self.stage1_medcam_fusion = RFFM_B(
            total_channels=4 * self.num_features + 3, 
            medcam_channels=2 * self.num_features
        )

        # 第二次下采样
        self.downsample_stage2 = DownSamplingBlock(
            in_channels=4 * self.num_features + 3, 
            out_channels=4 * self.num_features
        )

        # 第二阶段SEM_B块组合  
        self.stage2_sem_blocks = SEM_B_Block(
            in_channels=4 * self.num_features, 
            num_blocks=self.stage2_blocks,
            dilation_rates=[4, 4, 8, 8, 16, 16, 32, 32], 
            block_id=2
        )

        # 第二阶段特征融合（包含MEDCAM）
        self.stage2_medcam_fusion = RFFM_B(
            total_channels=8 * self.num_features + 3, 
            medcam_channels=4 * self.num_features
        )

        # 多尺度注意力解码器
        self.decoder = MAD(
            num_classes=self.num_classes, 
            num_features=self.num_features
        )

        # 用于特征提取的1x1卷积（当作为backbone使用时）
        self.mid_feature_conv = Conv(
            4 * self.num_features + 3, 2 * self.num_features, 
            1, 1, padding=0, 
            with_bn_activation=False
        )
        self.deep_feature_conv = Conv(
            8 * self.num_features + 3, 2 * self.num_features, 
            1, 1, padding=0, 
            with_bn_activation=False
        )

    def forward(self, input): ##1*3*448*448
        # Init Block
        out_init_block = self.Init_Block(input) # [1, 64, 224, 224]

        #max pool做一次
        down_1 = self.down_1(input) #[1, 3, 224, 224]
        input_RFFM_a = out_init_block, down_1
        # RFFM-A
        # concat -> bn -> relu -> conv1*1 ->残差
        out_RFFM_a = self.RFFM_A(input_RFFM_a) #[1, 67, 224, 224]

        # SEM-B Block1
        #3*3卷积 和 max pool concat，然后 BN+ relu
        out_downsample_1 = self.downsample_1(out_RFFM_a) #[1, 128, 112, 112]

        #将输入在channel维度上均分为二，一部分使用常规卷积， 另一部分进空洞卷积， 再concat到一起，使用残差连接
        out_sem_block1 = self.SEM_B_Block1(out_downsample_1)#[1, 128, 112, 112]

        # RFFM-B1
        # 两次maxpooling
        down_2 = self.down_2(input) #[1, 3, 112, 112]

        input_sem1_MEDCAM1 = out_sem_block1, out_downsample_1, down_2

        #对out_downsample_1计算通道间的注意力权重 ， 然后三个输入无脑concat到一起，再BN、relu、conv1*1
        out_RFFM_b1 = self.RFFM_B1(input_sem1_MEDCAM1) #[1, 259, 112, 112] 128+128+3  浅层特征
        # print("out_RFFM_b1 shape: %s"%str(out_RFFM_b1.shape))


        # SEM-B Block2
        #继续卷积， 减小size，同时将channel从259变回256
        out_downsample_2 = self.downsample_2(out_RFFM_b1) #[1, 256, 56, 56] 

        out_se_block2 = self.SEM_B_Block2(out_downsample_2) #[1, 256, 56, 56] 

        # RFFM-B2
        down_3 = self.down_3(input) #[1, 3, 56, 56]
        input_sem2_MEDCAM2 = out_se_block2, out_downsample_2, down_3
        out_RFFM_b2 = self.RFFM_B2(input_sem2_MEDCAM2) #[1, 515, 56, 56]  深层特征

        mid_feature = self.mid_layer_1x1(out_RFFM_b1)
        deep_feature = self.deep_layer_1x1(out_RFFM_b2)

        mid_feature = F.max_pool2d(mid_feature, kernel_size=2, stride=2) 
        all_features = torch.cat([mid_feature, deep_feature], 1)
        all_features = F.max_pool2d(all_features, kernel_size=2, stride=2) 
        return  all_features# [1, 256, 56, 56]

        # MAD

        # input_RFFMb1_RFFMb2 = out_RFFM_b1, out_RFFM_b2
        # out_mad = self.MAD(input_RFFMb1_RFFMb2) #[1, 4, 448, 448]
        # return out_mad


# if __name__ == "__main__":
#     def count_parameters(model):
#         return sum(p.numel() for p in model.parameters() if p.requires_grad)

#     image = torch.rand((1, 3, 448, 448))
#     model = TTTSNet(classes=2, num_features=64)
#     print(model)
#     print(model.forward(input=image))
#     print(count_parameters(model))
