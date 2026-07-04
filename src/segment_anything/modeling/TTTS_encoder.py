# TTTSNet - 优化版本 #
# Time-aware Two-Tower Transformer Segmentation Network
# 
# 主要优化：
# 1. 模块化设计，清晰的代码结构
# 2. 规范的命名约定和类型注解
# 3. 改进的网络架构和特征融合策略
# 4. 更好的Transformer集成
# 5. 详细的文档和注释

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
from abc import ABC, abstractmethod

__all__ = ["TTTSNet", "TTTSNetConfig"]


class TTTSNetConfig:
    """TTTSNet配置类，统一管理模型参数"""
    
    def __init__(
        self,
        num_classes: int = 2,
        base_channels: int = 64,
        block_1_layers: int = 3,
        block_2_layers: int = 8,
        img_size: int = 1024,
        transformer_heads: int = 4,
        reduction_ratio: int = 8,
        dilation_rates_1: List[int] = None,
        dilation_rates_2: List[int] = None,
    ):
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.block_1_layers = block_1_layers
        self.block_2_layers = block_2_layers
        self.img_size = img_size
        self.transformer_heads = transformer_heads
        self.reduction_ratio = reduction_ratio
        
        # 默认膨胀率配置
        self.dilation_rates_1 = dilation_rates_1 or [2, 2, 2]
        self.dilation_rates_2 = dilation_rates_2 or [4, 4, 8, 8, 16, 16, 32, 32]


# ==================== 基础工具函数 ====================

def channel_split(x: torch.Tensor, ratio: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将输入张量在通道维度按比例分割
    
    Args:
        x: 输入张量 [B, C, H, W]
        ratio: 分割比例，第一部分占总通道数的比例
        
    Returns:
        两个分割后的张量
    """
    channels = x.size(1)
    split_point = int(channels * ratio)
    return x[:, :split_point].contiguous(), x[:, split_point:].contiguous()


def ensure_channel_compatibility(x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    确保两个张量在通道维度兼容，通过padding调整
    
    Args:
        x1, x2: 输入张量
        
    Returns:
        调整后的张量对
    """
    c1, c2 = x1.size(1), x2.size(1)
    if c1 < c2:
        pad_channels = c2 - c1
        x1 = F.pad(x1, (0, 0, 0, 0, 0, pad_channels))
    elif c1 > c2:
        pad_channels = c1 - c2
        x2 = F.pad(x2, (0, 0, 0, 0, 0, pad_channels))
    return x1, x2

# ==================== 基础神经网络模块 ====================

class ConvBlock(nn.Module):
    """
    优化的卷积块，支持多种配置选项
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
        stride: 步长
        padding: 填充
        dilation: 膨胀率
        groups: 分组数
        use_activation: 是否使用激活函数
        activation_type: 激活函数类型
        norm_type: 归一化类型
        bias: 是否使用偏置
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        use_activation: bool = False,
        activation_type: str = "prelu",
        norm_type: str = "batch",
        bias: bool = False
    ):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias
        )
        
        self.use_activation = use_activation
        if use_activation:
            self.norm_activation = NormActivation(out_channels, norm_type, activation_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.use_activation:
            x = self.norm_activation(x)
        return x


class NormActivation(nn.Module):
    """
    归一化 + 激活函数模块
    
    Args:
        channels: 通道数
        norm_type: 归一化类型 ('batch', 'group', 'instance', 'layer')
        activation_type: 激活函数类型 ('prelu', 'relu', 'gelu', 'swish')
    """
    
    def __init__(self, channels: int, norm_type: str = "batch", activation_type: str = "prelu"):
        super().__init__()
        
        # 归一化层
        if norm_type == "batch":
            self.norm = nn.BatchNorm2d(channels, eps=1e-3)
        elif norm_type == "group":
            self.norm = nn.GroupNorm(min(32, channels), channels, eps=1e-3)
        elif norm_type == "instance":
            self.norm = nn.InstanceNorm2d(channels, eps=1e-3)
        elif norm_type == "layer":
            self.norm = nn.LayerNorm([channels, 1, 1], eps=1e-3)
        else:
            self.norm = nn.Identity()
        
        # 激活函数
        if activation_type == "prelu":
            self.activation = nn.PReLU(channels)
        elif activation_type == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation_type == "gelu":
            self.activation = nn.GELU()
        elif activation_type == "swish":
            self.activation = nn.SiLU()
        else:
            self.activation = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(x))


class ChannelRecoveryConv(nn.Module):
    """
    通道恢复卷积模块：3x3卷积 + 1x1卷积进行通道维度调整
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        
        self.conv3x3 = ConvBlock(
            mid_channels, mid_channels, kernel_size=3, 
            padding=1, use_activation=True
        )
        self.conv1x1 = ConvBlock(
            mid_channels, in_channels, kernel_size=1, 
            padding=0, use_activation=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv3x3(x)
        return self.conv1x1(x)

class InitialFeatureExtractor(nn.Module):
    """
    初始特征提取模块，使用渐进式特征提取
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        num_layers: 卷积层数
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 32, num_layers: int = 3):
        super().__init__()
        
        layers = []
        current_channels = in_channels
        
        for i in range(num_layers):
            target_channels = out_channels if i > 0 else out_channels
            stride = 2 if i == 0 else 1  # 第一层下采样
            
            layers.append(
                ConvBlock(
                    current_channels, target_channels,
                    kernel_size=3, stride=stride, padding=1,
                    use_activation=True
                )
            )
            current_channels = target_channels
        
        self.feature_extractor = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)

class SEM_B(nn.Module):
    """
    空间增强模块 - 通道分组后分别用普通卷积和空洞卷积，增强多尺度特征提取
    
    Args:
        channels: 输入通道数
        dilation: 膨胀率
        kernel_size: 卷积核大小
    """
    
    def __init__(self, channels: int, d: int = 1, kernel_size: int = 3):
        super().__init__()
        
        # 通道压缩
        self.channel_compress = ConvBlock(
            channels, channels // 2, kernel_size=kernel_size, 
            padding=1, use_activation=True
        )
        
        # 左分支：普通分组卷积
        self.left_conv = ConvBlock(
            channels // 4, channels // 4, kernel_size=kernel_size,
            padding=1, groups=channels // 4, use_activation=True
        )
        
        # 右分支：膨胀分组卷积
        self.right_conv = ConvBlock(
            channels // 4, channels // 4, kernel_size=kernel_size,
            padding=d, dilation=d, groups=channels // 4, use_activation=True
        )
        
        # 通道恢复
        self.channel_recover = ChannelRecoveryConv(channels, channels)
        
        # 最终归一化激活
        self.final_norm_activation = NormActivation(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # 通道压缩
        compressed = self.channel_compress(x)
        
        # 通道分割
        left, right = channel_split(compressed, ratio=0.5)
        
        # 分别处理
        left_out = self.left_conv(left)
        right_out = self.right_conv(right)
        
        # 拼接
        combined = torch.cat([left_out, right_out], dim=1)
        
        # 通道恢复
        recovered = self.channel_recover(combined)
        
        # 残差连接 + 归一化激活
        return self.final_norm_activation(recovered + identity)

class DownSamplingBlock(nn.Module):
    """
    高效下采样模块，融合卷积和池化
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 计算卷积输出通道数
        conv_channels = out_channels - in_channels if in_channels < out_channels else out_channels
        
        # 下采样卷积
        self.downsample_conv = ConvBlock(
            in_channels, conv_channels, kernel_size=3, 
            stride=2, padding=1, use_activation=False
        )
        
        # 最大池化（如果需要）
        if in_channels < out_channels:
            self.max_pool = nn.MaxPool2d(2, stride=2)
        else:
            self.max_pool = None
        
        # 最终归一化激活
        self.norm_activation = NormActivation(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 卷积下采样
        conv_out = self.downsample_conv(x)
        
        # 如果需要池化分支
        if self.max_pool is not None:
            pool_out = self.max_pool(x)
            output = torch.cat([conv_out, pool_out], dim=1)
        else:
            output = conv_out
        
        # 归一化激活
        return self.norm_activation(output)

class InputInjection(nn.Module):
    """
    多尺度输入处理模块，通过多次池化生成不同分辨率的输入
    
    Args:
        ratio: 下采样次数
        pool_type: 池化类型 ('max', 'avg')
    """
    
    def __init__(self, ratio: int, pool_type: str = "max"):
        super().__init__()
        
        if pool_type == "max":
            pool_layer = nn.MaxPool2d(3, stride=2, padding=1)
        elif pool_type == "avg":
            pool_layer = nn.AvgPool2d(3, stride=2, padding=1)
        else:
            raise ValueError(f"Unsupported pool_type: {pool_type}")
        
        self.pools = nn.ModuleList([
            pool_layer for _ in range(ratio)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for pool in self.pools:
            x = pool(x)
        return x

class SEAttention(nn.Module):
    """
    改进的Squeeze-and-Excitation注意力模块
    
    Args:
        channels: 输入通道数
        reduction: 通道缩减比例
        activation: 激活函数类型
    """
    
    def __init__(self, channels: int, reduction: int = 8, activation: str = "prelu"):
        super().__init__()
        
        reduced_channels = max(channels // reduction, 8)  # 确保最小通道数
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_attention = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.PReLU() if activation == "prelu" else nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        
        # 全局平均池化
        y = self.global_pool(x).view(b, c)
        
        # 通道注意力计算
        attention_weights = self.channel_attention(y).view(b, c, 1, 1)
        
        # 应用注意力权重
        return x * attention_weights

class MultiScaleChannelAttention(nn.Module):
    """
    多尺度通道注意力机制（MEDCAM的改进版本）
    结合全局池化和分区池化进行多尺度特征融合
    
    Args:
        channels: 输入通道数
        reduction: 通道缩减比例
        partition_size: 分区池化尺寸
    """
    
    def __init__(self, channels: int, reduction: int = 8, partition_size: int = 2):
        super().__init__()
        
        # 分区池化分支
        self.partition_pool = nn.AdaptiveMaxPool2d((partition_size, partition_size))
        self.partition_conv = ConvBlock(
            channels, channels, kernel_size=partition_size, 
            padding=0, groups=channels, use_activation=False
        )
        
        # 全局池化分支
        self.global_pool = nn.AdaptiveMaxPool2d(1)
        
        # SE注意力模块
        self.se_attention = SEAttention(channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 分区池化分支
        partition_features = self.partition_pool(x)
        partition_features = self.partition_conv(partition_features)
        
        # 全局池化分支  
        global_features = self.global_pool(x)
        
        # 特征融合
        combined_features = partition_features + global_features
        
        # SE注意力计算
        attention_weights = self.se_attention(combined_features)
        
        # 应用注意力权重
        return attention_weights * x

class ResidualFeatureFusion(nn.Module):
    """
    基础残差特征融合模块
    
    Args:
        in_channels: 输入通道数
        use_attention: 是否使用注意力机制
    """
    
    def __init__(self, in_channels: int, use_attention: bool = False):
        super().__init__()
        
        self.norm_activation = NormActivation(in_channels)
        self.channel_adapter = ConvBlock(
            in_channels, in_channels, kernel_size=1, 
            padding=0, use_activation=False
        )
        
        if use_attention:
            self.attention = SEAttention(in_channels)
        else:
            self.attention = None

    def forward(self, features: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Args:
            features: 输入特征元组，第一个作为残差连接的基础
        """
        primary_feature = features[0]
        
        # 特征对齐和拼接
        if len(features) > 1:
            # 确保所有特征的空间尺寸与第一个特征匹配
            aligned_features = [primary_feature]
            target_size = primary_feature.shape[2:]
            
            for feat in features[1:]:
                if feat.shape[2:] != target_size:
                    # 调整空间尺寸
                    feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
                aligned_features.append(feat)
            
            concatenated = torch.cat(aligned_features, dim=1)
        else:
            concatenated = primary_feature
            
        # 归一化和激活
        fused = self.norm_activation(concatenated)
        
        # 通道调整
        fused = self.channel_adapter(fused)
        
        # 应用注意力
        if self.attention is not None:
            fused = self.attention(fused)
        
        # 残差连接（确保通道兼容）
        primary_feature, fused = ensure_channel_compatibility(primary_feature, fused)
        
        return primary_feature + fused


class AttentionFeatureFusion(nn.Module):
    """
    带注意力的特征融合模块
    
    Args:
        in_channels: 输入通道数
        attention_channels: 注意力模块的通道数
        reduction: 注意力模块的缩减比例
    """
    
    def __init__(self, in_channels: int, attention_channels: int, reduction: int = 8):
        super().__init__()
        
        self.attention = MultiScaleChannelAttention(attention_channels, reduction)
        self.base_fusion = ResidualFeatureFusion(in_channels, use_attention=True)

    def forward(self, features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: 三个输入特征 (primary, attention_target, auxiliary)
        """
        primary, attention_target, auxiliary = features
        
        # 对第二个特征应用多尺度注意力
        enhanced_attention_target = self.attention(attention_target)
        
        # 使用基础融合模块处理所有特征
        return self.base_fusion((primary, enhanced_attention_target, auxiliary))

# SEM_B_Block已被StackedSEMBlocks替代，该类定义在主网络之后

class MultiScaleAggregationDecoder(nn.Module):
    """
    多尺度聚合解码器（MAD的改进版本）
    
    Args:
        mid_channels: 中层特征通道数
        deep_channels: 深层特征通道数
        num_classes: 分类数
        base_features: 基础特征数
    """
    
    def __init__(
        self, 
        mid_channels: int = 32, 
        deep_channels: int = 64, 
        num_classes: int = 19, 
        base_features: int = 32
    ):
        super().__init__()
        
        self.mid_channels = mid_channels
        self.deep_channels = deep_channels
        self.base_features = base_features
        
        # 中层特征适配
        self.mid_adapter = ConvBlock(
            4 * base_features + 3, mid_channels, 
            kernel_size=1, use_activation=False
        )
        
        # 深层特征适配
        self.deep_adapter = ConvBlock(
            8 * base_features + 3, deep_channels,
            kernel_size=1, use_activation=False
        )
        
        # 深度可分离卷积 1
        self.depthwise_conv1 = ConvBlock(
            mid_channels + deep_channels, mid_channels + deep_channels,
            kernel_size=3, groups=mid_channels + deep_channels, use_activation=True
        )
        
        self.pointwise_conv1 = ConvBlock(
            mid_channels + deep_channels, num_classes,
            kernel_size=1, use_activation=False
        )
        
        # 深度可分离卷积 2
        self.depthwise_conv2 = ConvBlock(
            8 * base_features + 3, 8 * base_features + 3,
            kernel_size=3, groups=8 * base_features + 3, use_activation=True
        )
        
        self.pointwise_conv2 = ConvBlock(
            8 * base_features + 3, num_classes,
            kernel_size=1, use_activation=False
        )

    def forward(self, features: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: (mid_features, deep_features)
        """
        mid_features, deep_features = features
        deep_size = deep_features.size()[2:]
        
        # 特征适配
        mid_adapted = self.mid_adapter(mid_features)
        deep_adapted = self.deep_adapter(deep_features)
        
        # 上采样深层特征与中层特征匹配
        deep_upsampled = F.interpolate(
            deep_adapted, 
            size=[deep_size[0] * 2, deep_size[1] * 2], 
            mode='bilinear', 
            align_corners=False
        )
        
        # 特征融合
        fused_features = torch.cat([mid_adapted, deep_upsampled], dim=1)
        fused_features = self.depthwise_conv1(fused_features)
        attention_map = torch.sigmoid(self.pointwise_conv1(fused_features))
        
        # 深层分支处理
        deep_processed = self.depthwise_conv2(deep_features)
        deep_output = self.pointwise_conv2(deep_processed)
        
        # 上采样到中层尺寸
        deep_output = F.interpolate(
            deep_output, 
            size=[deep_size[0] * 2, deep_size[1] * 2], 
            mode='bilinear', 
            align_corners=False
        )
        
        # 应用注意力
        attended_output = deep_output * attention_map
        
        # 最终上采样
        final_output = F.interpolate(
            attended_output, 
            size=[deep_size[0] * 8, deep_size[1] * 8], 
            mode='bilinear', 
            align_corners=False
        )
        
        return final_output


class EfficientTransformerBlock(nn.Module):
    """
    高效的Transformer块，针对视觉任务优化
    
    Args:
        dim: 特征维度
        num_heads: 注意力头数
        mlp_ratio: MLP扩展比例
        dropout: dropout比例
        window_size: 窗口注意力大小
    """
    
    def __init__(
        self, 
        dim: int, 
        num_heads: int = 4, 
        mlp_ratio: float = 4.0, 
        dropout: float = 0.1,
        window_size: Optional[int] = None
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        
        # 层归一化
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # 多头自注意力
        self.attention = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # MLP层
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # 如果使用窗口注意力，则分割成窗口
        if self.window_size is not None and H > self.window_size and W > self.window_size:
            x = self._window_attention(x)
        else:
            x = self._global_attention(x)
        
        return x
    
    def _global_attention(self, x: torch.Tensor) -> torch.Tensor:
        """全局注意力"""
        B, C, H, W = x.shape
        
        # 重塑为序列格式 [B, HW, C]
        x_seq = x.flatten(2).transpose(1, 2)
        
        # 应用注意力
        x_norm = self.norm1(x_seq)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        x_seq = x_seq + attn_out
        
        # 应用MLP
        x_norm = self.norm2(x_seq)
        mlp_out = self.mlp(x_norm)
        x_seq = x_seq + mlp_out
        
        # 重塑回原始格式
        return x_seq.transpose(1, 2).reshape(B, C, H, W)
    
    def _window_attention(self, x: torch.Tensor) -> torch.Tensor:
        """窗口注意力（用于大特征图）"""
        B, C, H, W = x.shape
        ws = self.window_size
        
        # 分割成窗口
        x_windows = x.view(B, C, H // ws, ws, W // ws, ws)
        x_windows = x_windows.permute(0, 2, 4, 3, 5, 1).contiguous()
        x_windows = x_windows.view(-1, ws * ws, C)
        
        # 对每个窗口应用注意力
        x_norm = self.norm1(x_windows)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        x_windows = x_windows + attn_out
        
        # 应用MLP
        x_norm = self.norm2(x_windows)
        mlp_out = self.mlp(x_norm)
        x_windows = x_windows + mlp_out
        
        # 重塑回原始格式
        x_windows = x_windows.view(B, H // ws, W // ws, ws, ws, C)
        x = x_windows.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(B, C, H, W)
    

class TTTSNet(nn.Module):
    """
    时序感知双塔Transformer分割网络 (Time-aware Two-Tower Transformer Segmentation Network)
    
    优化的模块化架构，支持灵活配置和高效推理
    
    Args:
        config: TTTSNetConfig配置对象
    """
    
    def __init__(self, config: Optional[TTTSNetConfig] = None):
        super().__init__()
        
        # 使用默认配置如果未提供
        self.config = config or TTTSNetConfig()
        self.base_channels = self.config.base_channels
        
        # 为了向后兼容，添加img_size属性
        self.img_size = self.config.img_size
        
        # 初始特征提取器
        self.initial_extractor = InitialFeatureExtractor(
            in_channels=3,
            out_channels=self.base_channels,
            num_layers=3
        )
        
        # 多尺度输入处理
        self.multi_scale_inputs = nn.ModuleList([
            InputInjection(ratio=i) for i in range(1, 4)
        ])
        
        # 构建两塔架构
        self._build_tower_1()
        self._build_tower_2()
        
        # 特征融合模块
        self._build_feature_fusion()
        
        # Transformer模块
        self.transformer = EfficientTransformerBlock(
            dim=4 * self.base_channels,
            num_heads=self.config.transformer_heads,
            window_size=8  # 使用窗口注意力提高效率
        )
        
        # 输出投影层（确保输出通道数为256以兼容SAM）
        self.output_projection = ConvBlock(
            4 * self.base_channels, 256,  # SAM期望256通道输出
            kernel_size=1, use_activation=True
        )
        
        # 自适应池化层确保输出尺寸与SAM兼容
        self.adaptive_pool = nn.AdaptiveAvgPool2d((64, 64))  # SAM期望64x64的特征图
        
    def _build_tower_1(self):
        """构建第一个塔（浅层特征处理）"""
        # 浅层特征融合
        self.shallow_fusion = ResidualFeatureFusion(
            in_channels=self.base_channels + 3
        )
        
        # 第一次下采样
        self.downsample_1 = DownSamplingBlock(
            self.base_channels + 3, 
            2 * self.base_channels
        )
        
        # SEM_B模块组
        self.sem_block_1 = StackedSEMBlocks(
            channels=2 * self.base_channels,
            num_blocks=self.config.block_1_layers,
            dilation_rates=self.config.dilation_rates_1
        )
        
        # 带注意力的特征融合
        self.attention_fusion_1 = AttentionFeatureFusion(
            in_channels=4 * self.base_channels + 3,
            attention_channels=2 * self.base_channels,
            reduction=self.config.reduction_ratio
        )
    
    def _build_tower_2(self):
        """构建第二个塔（深层特征处理）"""
        # 第二次下采样
        self.downsample_2 = DownSamplingBlock(
            4 * self.base_channels + 3,
            4 * self.base_channels
        )
        
        # 深层SEM_B模块组
        self.sem_block_2 = StackedSEMBlocks(
            channels=4 * self.base_channels,
            num_blocks=self.config.block_2_layers,
            dilation_rates=self.config.dilation_rates_2
        )
        
        # 深层特征融合
        self.attention_fusion_2 = AttentionFeatureFusion(
            in_channels=8 * self.base_channels + 3,
            attention_channels=4 * self.base_channels,
            reduction=self.config.reduction_ratio
        )
    
    def _build_feature_fusion(self):
        """构建特征融合模块"""
        # 中层特征压缩
        self.mid_feature_adapter = ConvBlock(
            4 * self.base_channels + 3,
            2 * self.base_channels,
            kernel_size=1, use_activation=False
        )
        
        # 深层特征压缩
        self.deep_feature_adapter = ConvBlock(
            8 * self.base_channels + 3,
            2 * self.base_channels,
            kernel_size=1, use_activation=False
        )
        
        # 最终特征融合
        self.final_fusion = ResidualFeatureFusion(
            in_channels=4 * self.base_channels,
            use_attention=True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像 [B, 3, H, W]
            
        Returns:
            特征表示 [B, 256, 64, 64] - 兼容SAM的图像编码器输出格式
        """
        # 确保输入是float32类型（兼容uint8输入）
        if x.dtype == torch.uint8:
            x = x.float()
        
        # 初始特征提取
        initial_features = self.initial_extractor(x)  # [B, C, H/2, W/2]
        
        # 多尺度输入处理
        multi_scale_features = [
            injection(x) for injection in self.multi_scale_inputs
        ]  # [B, 3, H/2^i, W/2^i] for i in [1,2,3]
        
        # ==================== 第一个塔 ====================
        # 浅层特征融合
        tower1_input = self.shallow_fusion((initial_features, multi_scale_features[0]))
        
        # 下采样和SEM处理
        tower1_downsampled = self.downsample_1(tower1_input)  # [B, 2C, H/4, W/4]
        tower1_processed = self.sem_block_1(tower1_downsampled)  # [B, 2C, H/4, W/4]
        
        # 带注意力的特征融合
        tower1_output = self.attention_fusion_1((
            tower1_processed, 
            tower1_downsampled, 
            multi_scale_features[1]
        ))  # [B, 4C+3, H/4, W/4]
        
        # ==================== 第二个塔 ====================
        # 深层下采样和处理
        tower2_downsampled = self.downsample_2(tower1_output)  # [B, 4C, H/8, W/8]
        tower2_processed = self.sem_block_2(tower2_downsampled)  # [B, 4C, H/8, W/8]
        
        # 深层特征融合
        tower2_output = self.attention_fusion_2((
            tower2_processed,
            tower2_downsampled,
            multi_scale_features[2]
        ))  # [B, 8C+3, H/8, W/8]
        
        # ==================== 特征融合 ====================
        # 特征适配
        mid_features = self.mid_feature_adapter(tower1_output)  # [B, 2C, H/4, W/4]
        deep_features = self.deep_feature_adapter(tower2_output)  # [B, 2C, H/8, W/8]
        
        # 空间对齐（通过池化使中层特征与深层特征尺寸匹配）
        mid_features = F.max_pool2d(mid_features, kernel_size=2, stride=2)  # [B, 2C, H/8, W/8]
        
        # 最终特征融合
        fused_features = self.final_fusion((mid_features, deep_features))  # [B, 4C, H/8, W/8]
        
        # ==================== Transformer处理 ====================
        # 进一步压缩并应用Transformer
        compressed_features = F.avg_pool2d(fused_features, kernel_size=2, stride=2)  # [B, 4C, H/16, W/16]
        
        # 应用Transformer进行全局建模
        enhanced_features = self.transformer(compressed_features)  # [B, 4C, H/16, W/16]
        
        # 最终输出投影
        output_features = self.output_projection(enhanced_features)  # [B, 256, H/16, W/16]
        
        # 自适应池化确保输出尺寸为64x64（SAM兼容）
        output_features = self.adaptive_pool(output_features)  # [B, 256, 64, 64]
        
        return output_features


class StackedSEMBlocks(nn.Module):
    """
    堆叠的SEM_B模块
    
    Args:
        channels: 通道数
        num_blocks: 模块数量
        dilation_rates: 膨胀率列表
    """
    
    def __init__(self, channels: int, num_blocks: int, dilation_rates: List[int]):
        super().__init__()
        
        # 确保膨胀率数量匹配模块数量
        if len(dilation_rates) != num_blocks:
            # 循环使用膨胀率
            dilation_rates = (dilation_rates * ((num_blocks // len(dilation_rates)) + 1))[:num_blocks]
        
        self.blocks = nn.Sequential(*[
            SEM_B(channels, d=dilation_rates[i])
            for i in range(num_blocks)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)