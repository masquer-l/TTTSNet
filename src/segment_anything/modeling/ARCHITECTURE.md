# MultiFrameMaskDecoder 架构说明

## 整体架构对比

### 原始 MaskDecoder
```
输入: [B, C, H, W]
  ↓
MaskDecoder
  ├── Transformer
  ├── Output Upscaling (4x)
  └── Hypernetwork MLPs
  ↓
输出: [B, num_masks, H×4, W×4]
```

### 多帧 MultiFrameMaskDecoder
```
输入: [B, T, C, H, W]
  ↓
MultiFrameMaskDecoder
  ├── Temporal Position Embedding
  ├── Per-Frame Processing
  │   ├── Frame-wise Transformer
  │   ├── Output Upscaling (4x)
  │   └── Hypernetwork MLPs
  ├── [可选] Temporal Attention
  └── Temporal Aggregation
  ↓
输出: [B, T, num_masks, H×4, W×4]
```

## 详细架构

### 1. 输入层
```
输入组件:
┌────────────────────────────────────┐
│ image_embeddings: [B, T, C, H, W] │ ← 时序图像特征
│ image_pe: [1, C, H, W]             │ ← 空间位置编码
│ sparse_prompts: [B, N, C]          │ ← 点/框提示
│ dense_prompts: [B, C, H, W]        │ ← mask提示
└────────────────────────────────────┘
```

### 2. 时序位置编码模块
```
TemporalPositionEmbedding
├── Input: num_frames = T
├── Position: [0, 1, 2, ..., T-1]
├── Sinusoidal Encoding:
│   ├── pe[:, 0::2] = sin(position × div_term)
│   └── pe[:, 1::2] = cos(position × div_term)
└── Output: [T, C]
```

### 3. 逐帧处理流程
```
for t in range(T):
  ┌──────────────────────────────┐
  │ 1. 获取第t帧特征             │
  │    frame_emb = img_emb[:, t] │
  │    shape: [B, C, H, W]       │
  └──────────────────────────────┘
          ↓
  ┌──────────────────────────────┐
  │ 2. 添加时序位置编码          │
  │    frame_emb += temporal_pe[t]│
  └──────────────────────────────┘
          ↓
  ┌──────────────────────────────┐
  │ 3. 添加密集提示              │
  │    frame_emb += dense_prompt │
  └──────────────────────────────┘
          ↓
  ┌──────────────────────────────┐
  │ 4. Transformer处理           │
  │    hs, src = transformer()   │
  │    hs: [B, N_tokens, C]      │
  │    src: [B, H×W, C]          │
  └──────────────────────────────┘
          ↓
  ┌──────────────────────────────┐
  │ 5. 分离tokens                │
  │    iou_token = hs[:, 0, :]   │
  │    mask_tokens = hs[:, 1:5]  │
  └──────────────────────────────┘
          ↓
  ┌──────────────────────────────┐
  │ 6. 上采样 (4倍)              │
  │    upscaled = upscaling(src) │
  │    shape: [B, C/8, H×4, W×4] │
  └──────────────────────────────┘
          ↓
  ┌──────────────────────────────┐
  │ 7. 预测mask                  │
  │    hyper_in = MLPs(mask_tok) │
  │    masks = hyper_in @ upscaled│
  │    shape: [B, 4, H×4, W×4]   │
  └──────────────────────────────┘
          ↓
  ┌──────────────────────────────┐
  │ 8. 预测IoU                   │
  │    iou = iou_head(iou_token) │
  │    shape: [B, 4]             │
  └──────────────────────────────┘
```

### 4. 时序注意力模块 (可选)
```
TemporalAttention
├── Input: [B, T, N_tokens, C]
├── Reshape: [B×N_tokens, T, C]
├── Multi-Head Self-Attention:
│   ├── Q = Linear(x)  [B×N, T, C]
│   ├── K = Linear(x)  [B×N, T, C]
│   ├── V = Linear(x)  [B×N, T, C]
│   ├── Split into heads: [B×N, num_heads, T, C/num_heads]
│   ├── Attention: softmax(QK^T / √d) × V
│   └── Concat heads: [B×N, T, C]
├── Residual + LayerNorm
└── Output: [B, T, N_tokens, C]
```

### 5. 输出层
```
┌────────────────────────────────────┐
│ 拼接所有帧的输出                   │
│ masks = concat([mask_0, ..., mask_T])│
│ iou = concat([iou_0, ..., iou_T])  │
└────────────────────────────────────┘
          ↓
┌────────────────────────────────────┐
│ 选择mask类型                       │
│ if multimask_output:               │
│   masks = masks[:, :, 1:4]  # 3个  │
│ else:                              │
│   masks = masks[:, :, 0:1]  # 1个  │
└────────────────────────────────────┘
          ↓
┌────────────────────────────────────┐
│ 最终输出                           │
│ masks: [B, T, num_masks, H×4, W×4] │
│ iou_pred: [B, T, num_masks]        │
└────────────────────────────────────┘
```

## 核心模块详解

### TemporalPositionEmbedding

**作用**: 为每一帧生成唯一的时序标识

**数学公式**:
```
position = [0, 1, 2, ..., T-1]
div_term = exp(2i / d_model × log(10000))

PE(t, 2i) = sin(t / 10000^(2i/d_model))
PE(t, 2i+1) = cos(t / 10000^(2i/d_model))

其中: t是时间步, i是维度索引, d_model是嵌入维度
```

**代码实现**:
```python
class TemporalPositionEmbedding(nn.Module):
    def forward(self, num_frames, device):
        position = torch.arange(num_frames, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, device=device) * 
                            -(math.log(10000.0) / embed_dim))
        
        pe = torch.zeros(num_frames, embed_dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
```

**特性**:
- 每一帧有唯一的编码
- 相邻帧的编码相近
- 编码可外推到更长的序列

### TemporalAttention

**作用**: 在时间维度上建模帧间依赖关系

**处理流程**:
```
1. 输入: [B, T, N_tokens, C]
   - B: batch size
   - T: 帧数
   - N_tokens: 每帧的token数
   - C: 通道数

2. Reshape: [B×N_tokens, T, C]
   - 将空间tokens合并到batch维度
   - 在时间维度上应用注意力

3. Multi-Head Attention:
   Q, K, V = Linear(x)
   Attention(Q, K, V) = softmax(QK^T / √d_k)V

4. Reshape回: [B, T, N_tokens, C]
```

**注意力图示**:
```
帧间注意力权重矩阵 (T×T):
     t0   t1   t2   t3   t4
t0 [1.0  0.3  0.1  0.0  0.0]
t1 [0.3  1.0  0.4  0.1  0.0]
t2 [0.1  0.4  1.0  0.4  0.1]
t3 [0.0  0.1  0.4  1.0  0.3]
t4 [0.0  0.0  0.1  0.3  1.0]

注: 相邻帧之间注意力权重更高
```

### 逐帧Transformer处理

**TwoWayTransformer结构**:
```
TwoWayTransformer (每一帧)
├── Input:
│   ├── image_emb: [B, C, H, W]
│   ├── image_pe: [B, C, H, W]
│   └── tokens: [B, N_tokens, C]
│
├── Flatten image_emb: [B, H×W, C]
│
├── Transformer Layers (depth=2):
│   └── TwoWayAttentionBlock:
│       ├── Self-Attention (tokens)
│       ├── Cross-Attention (tokens → image)
│       ├── MLP
│       └── Cross-Attention (image → tokens)
│
└── Output:
    ├── hs: [B, N_tokens, C]  (处理后的tokens)
    └── src: [B, H×W, C]      (处理后的image特征)
```

## 数据流示例

假设: B=2, T=3, C=256, H=W=64

```
Step 1: 输入
  image_embeddings: [2, 3, 256, 64, 64]
  image_pe: [1, 256, 64, 64]
  sparse_prompts: [2, 10, 256]
  dense_prompts: [2, 256, 64, 64]

Step 2: 时序位置编码
  temporal_pe = TemporalPE(T=3)
  temporal_pe: [3, 256]
  
  frame 0: temporal_pe[0] = [sin/cos pattern for t=0]
  frame 1: temporal_pe[1] = [sin/cos pattern for t=1]
  frame 2: temporal_pe[2] = [sin/cos pattern for t=2]

Step 3: 逐帧处理
  
  Frame 0:
    frame_emb: [2, 256, 64, 64]
    + temporal_pe[0]: [256] (broadcast)
    + dense_prompts: [2, 256, 64, 64]
    → Transformer
    → masks_0: [2, 4, 256, 256]
    → iou_0: [2, 4]
  
  Frame 1:
    frame_emb: [2, 256, 64, 64]
    + temporal_pe[1]: [256]
    + dense_prompts: [2, 256, 64, 64]
    → Transformer
    → masks_1: [2, 4, 256, 256]
    → iou_1: [2, 4]
  
  Frame 2:
    frame_emb: [2, 256, 64, 64]
    + temporal_pe[2]: [256]
    + dense_prompts: [2, 256, 64, 64]
    → Transformer
    → masks_2: [2, 4, 256, 256]
    → iou_2: [2, 4]

Step 4: 时序注意力 (可选)
  mask_tokens: [2, 3, 4, 256]  (B, T, N_masks, C)
  → TemporalAttention
  → refined_tokens: [2, 3, 4, 256]

Step 5: 拼接输出
  all_masks = stack([masks_0, masks_1, masks_2])
  all_masks: [2, 3, 4, 256, 256]
  
  all_iou = stack([iou_0, iou_1, iou_2])
  all_iou: [2, 3, 4]

Step 6: 选择mask
  if multimask_output = False:
    masks = all_masks[:, :, 0:1]  → [2, 3, 1, 256, 256]
    iou = all_iou[:, :, 0:1]      → [2, 3, 1]
```

## 参数量分析

假设: C=256, num_heads=8, mlp_dim=2048

### TemporalPositionEmbedding
- **可学习参数**: 0 (纯函数计算)
- **计算量**: O(T × C)

### TemporalAttention
- **参数量**:
  ```
  Q_proj: C × C = 256 × 256 = 65,536
  K_proj: C × C = 256 × 256 = 65,536
  V_proj: C × C = 256 × 256 = 65,536
  out_proj: C × C = 256 × 256 = 65,536
  LayerNorm: 2 × C = 512
  ──────────────────────────────
  Total: ~262K 参数
  ```
- **计算量**: O(B × N × T² × C)

### MultiFrameMaskDecoder (不含Transformer)
- **参数量**:
  ```
  iou_token: C = 256
  mask_tokens: 4 × C = 1,024
  output_upscaling: ~262K
  hypernetwork_mlps: 4 × ~16K = 64K
  iou_prediction_head: ~66K
  temporal_attention: ~262K (可选)
  ──────────────────────────────
  Total: ~655K 参数 (含时序注意力)
        ~393K 参数 (不含时序注意力)
  ```

## 计算效率对比

| 操作 | 单帧MaskDecoder | 多帧MultiFrameMaskDecoder |
|------|----------------|--------------------------|
| Transformer | O(B × H × W × C²) | O(B × T × H × W × C²) |
| Upscaling | O(B × H × W × C) | O(B × T × H × W × C) |
| Temporal Attention | - | O(B × N × T² × C) |
| **总计** | O(B × H × W × C²) | O(B × T × (H × W × C² + N × T × C)) |

**结论**: 
- 不含时序注意力: ~T倍计算量
- 含时序注意力: ~T倍 + O(T²)

## 总结

MultiFrameMaskDecoder通过以下设计实现时序处理:

1. **时序位置编码** - 为每帧添加时序信息
2. **逐帧处理** - 保持单帧处理的效率
3. **时序注意力** - 可选的跨帧建模
4. **模块化设计** - 易于扩展和修改

这种设计在保持原始MaskDecoder核心功能的同时,有效地扩展到了多帧场景。

