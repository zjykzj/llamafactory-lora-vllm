# Qwen3.5

[Qwen3.5](https://modelscope.cn/organization/qwen) 是阿里通义千问团队于 2026 年 2-3 月发布的开源多模态语言模型系列。其核心创新是**混合注意力架构**——将 Gated DeltaNet（线性注意力）与标准 Full Attention 交替堆叠，在长上下文场景下实现 8.6-19 倍的解码加速，同时保持甚至超越纯 Transformer 的建模质量。

## Model Family

Qwen3.5 共 8 个开源规格，覆盖从端侧到数据中心的全部场景，均采用 Apache 2.0 协议。

| Model | Type | Total Params | Active Params | Layers | Context | VRAM (bf16) |
|-------|------|-------------|---------------|--------|---------|-------------|
| Qwen3.5-0.8B | Dense | 0.8B | 0.8B | 24 | 262K | ~2 GB |
| Qwen3.5-2B | Dense | 2B | 2B | 24 | 262K | ~4 GB |
| Qwen3.5-4B | Dense | 4B | 4B | 32 | 262K | ~8 GB |
| Qwen3.5-9B | Dense | 9B | 9B | 32 | 262K | ~18 GB |
| Qwen3.5-27B | Dense | 27B | 27B | 64 | 262K | ~54 GB |
| Qwen3.5-35B-A3B | MoE | 35B | ~3B | 40 | 262K | ~70 GB |
| Qwen3.5-122B-A10B | MoE | 122B | ~10B | 48 | 262K | ~244 GB |
| Qwen3.5-397B-A17B | MoE | 397B | ~17B | 60 | 262K (1M w/ YaRN) | ~780 GB |

**Dense vs MoE 的分工逻辑**：
- **Dense（≤27B）**：所有参数在每个 token 上都参与计算，质量稳定，适合本地部署和低延迟场景。4B 已被社区广泛用于本地代码生成，9B 在知识基准上超越了 GPT-OSS-120B。
- **MoE（≥35B）**：总参数量巨大但每个 token 仅激活一小部分（如 35B-A3B 每个 token 仅激活 ~3B 参数），推理速度极快——RTX 4090 上可达 **196 tok/s**。代价是需要足够显存存放全部 expert 权重。

此外，还有一个未开源的内部分支 **397B-A17B-Thinking**，在 SWE-Bench Verified 上达到 **78.2%**，GPQA Diamond 90.3%，MathArena Apex 43.4%，处于开发者工具第一梯队。

## Architecture: Hybrid Attention

Qwen3.5 最关键的架构决策是用 **Gated DeltaNet（线性注意力）** 替代了纯 Transformer 中 75% 的自注意力层，两者沿网络深度方向交替排列：

```
Layer  1:  Gated DeltaNet (linear)
Layer  2:  Gated DeltaNet (linear)
Layer  3:  Gated DeltaNet (linear)
Layer  4:  Full Attention        ← 每 4 层一次
Layer  5:  Gated DeltaNet (linear)
...
```

**设计动机**：标准 Softmax Attention 的计算和显存开销随序列长度平方增长（O(n²)），而 DeltaNet 作为线性注意力只需 O(n)。将大多数层替换为线性注意力后，长序列（256K tokens）的解码速度提升了 **8.6-19 倍**，而定期插入的 Full Attention 层保障了长程依赖不丢失。

### Gated DeltaNet 机制

DeltaNet 的核心思想是将注意力建模为 **RNN 式的递归状态更新**，每个 token 维护一个固定大小的状态矩阵，而非与所有历史 token 逐一交互：

```
α     = exp(-g[t])                    // decay gate：控制旧状态衰减速率
s_dec = α · s[t-1]                    // 先对旧状态施加衰减
δ     = v[t] - s_dec @ k[t]           // 计算"惊喜量"（delta）：新信息 vs 预测
s[t]  = s_dec + β[t] · δ · k[t]^T     // 按需更新状态矩阵
o[t]  = s[t] @ q[t]                   // 从更新后的状态中读出输出
```

直觉解释：
- **g[t]（遗忘门）** 决定历史信息以多快的速度被遗忘
- **δ（delta）** 衡量当前 token 有多"意外"——如果状态矩阵已经能很好预测 v[t]，delta ≈ 0，状态几乎不变
- **β[t]（写入门）** 控制新信息以多大的强度写入状态矩阵
- 状态矩阵 **s** 是固定尺寸的（每头 D_k × D_v），不随序列增长，因此每次更新的计算量是 O(1)

这与 Linear Attention 的经典形式（key-value 外积累加）一脉相承，但 DeltaNet 通过 gating 让模型学会了**选择性记忆**——只在对当前预测"意外"时才写入新信息。

### Full Attention 层

剩余 25% 的层使用标准 **Grouped Query Attention (GQA)**，结合：

- **Partial Rotary Position Embedding (RoPE)**：仅对每个 head 的 25% 维度施加旋转位置编码，剩余维度不编码位置信息
- **Sigmoid 输出门控**：对注意力输出施加 sigmoid 门控，类似 LSTM 式信息筛选
- **Sliding Window**（32K tokens）：限制注意力窗口为 32768 个 token
- **M-RoPE（多模态模式）**：处理图像+文本时，位置编码按 temporal/height/width 三个维度拆分（如 [11, 11, 10]）

### 为什么混合比纯线性更好

2025 年的一项组件消融研究（Borobia et al.）在 Qwen3.5-0.8B 上验证了几个关键结论：

1. **无法绕过任何一个组件**——在 5 个基准测试（MMLU, GSM8K, ARC-Challenge, HellaSwag, TruthfulQA）上移除任一组件都导致严重性能崩溃
2. **DeltaNet 是主力，Attention 是精修**——移除全部 18 个 DeltaNet 层，PPL 暴涨 **35,200 倍**（7.6 → 268K）；移除全部 6 个 Attention 层，PPL 涨 82 倍（7.6 → 625）
3. **混合架构天然抗噪声**——随机移除层时，Qwen3.5 比同规模的纯 Transformer（Qwen2.5-0.5B）稳定 **20-119 倍**

### 配置细节（Dense 模型）

| 参数 | 0.8B | 2B | 4B | 9B | 27B |
|------|------|----|----|----|------|
| Hidden Size | 1024 | 2048 | 2560 | 3584 | 5120 |
| Layers (Total) | 24 | 24 | 32 | 32 | 64 |
| — DeltaNet layers | 18 | 18 | 24 | 24 | 48 |
| — Full Attn layers | 6 | 6 | 8 | 8 | 16 |
| Attention Heads | 8 | 8 | 32 | 32 | 32 |
| KV Heads | 2 | 2 | 8 | 8 | 8 |
| Intermediate Size | 3584 | 6144 | 9728 | 14336 | 20480 |
| Vocabulary | 248K | 248K | 248K | 248K | 248K |

### 线性注意力头的配置

| 参数 | 默认值 |
|------|--------|
| Key Heads | 16 |
| Value Heads | 32（每 2 个 V head 共享 1 个 K/Q head，类 GQA 广播） |
| K/V Head Dim | 128 |
| Conv Kernel | 4（因果 conv1d 预处理） |
| 累加精度 | float32 |

## Vision Encoder

Qwen3.5 原生于多模态——**文本和视觉共享同一套 Transformer 参数**，不需要单独的 VL 变体。视觉编码器为 SigLIP：

| 参数 | 值 |
|------|-----|
| Architecture | SigLIP ViT |
| Depth | 24 layers |
| Hidden Size | 1024 |
| Intermediate Size | 4096 |
| Attention Heads | 16 |
| Patch Size | 16 |
| Output Projection | 1024 → 对齐 LM hidden size |
| Spatial Merge | 2×2 合并（减少 token 数） |

图像经 patch 切分、SigLIP 编码后，通过一个线性投影层映射到语言模型的 hidden size，然后直接作为 "visual tokens" 插入文本序列。在全注意力层中，M-RoPE 为 visual tokens 分配独立的位置编码维度。

### Image Preprocessing（图像预处理）

Qwen3.5 的 image processor 在图像进入 Vision Encoder 前执行以下流水线：

```
输入图像 → 转 RGB → 等比缩放 → 1/255 缩放到 [0,1] → normalize → [-1,1] → patchify+merge
```

**缩放规则**：保持宽高比，使总像素数落在 `[min_pixels, max_pixels]` 范围内。小于 min 的上采样，大于 max 的下采样。

**默认参数**（来自 `preprocessor_config.json`）：

| 参数 | 配置字段 | 默认值 | 等效分辨率 |
|------|---------|--------|-----------|
| min_pixels | `size.shortest_edge` | 65,536 | 256×256 |
| max_pixels | `size.longest_edge` | 16,777,216 | 4096×4096 |
| resample | — | 3 (bicubic) | — |
| rescale_factor | — | 1/255 | 映射 [0,255] → [0,1] |
| image_mean | — | [0.5, 0.5, 0.5] | — |
| image_std | — | [0.5, 0.5, 0.5] | 映射 [0,1] → [-1,1] |

> 正常化之后的值域是 [-1, 1]，而非常见的 ImageNet 均值和标准差。

**与 LLaMA Factory 的关系**：LLaMA Factory 在数据加载阶段也会做一次 resize（参数 `image_min_pixels`/`image_max_pixels`），之后 Qwen3.5 的 image processor 再做第二次 resize。对于 CIFAR 32×32（1024 像素）这种极小图，LLaMA Factory 层不缩放（1024 ≥ 默认的 1024 min），但到 Qwen3.5 层会被上采样到约 256×256（≥65536 min）。详见 [训练参数文档](training_config.md#图像预处理与两层-resize)。

## Key Features

**FP8 Native 训练与推理**：Qwen3.5 是首个全链路支持 FP8 的开源大模型。FP8 相比 BF16 将显存占用和计算带宽开销均减半，且 Qwen 团队在预训练阶段就使用 FP8，非后量化方案，精度损失极低。

**201 种语言**：Qwen3.5 是一个真正的多语言模型（非英/中双语偏置），覆盖 201 种语言和方言。

**256K 原生上下文 + YaRN 扩展**：原生 262K token 上下文窗口（得益于 DeltaNet 的 O(1) 状态），通过 YaRN 外推可扩展至超过 100 万 token。

**Apache 2.0 全开源**：所有权重、代码、技术报告均以 Apache 2.0 许可证发布，包括 MoE 旗舰模型。

**多 Token 预测 (MTP)**：训练时使用 multi-step prediction，1 个 MTP 层，提升推理时的 speculative decoding 效率。

## Qwen3 vs Qwen3.5 vs Qwen3-Next

| 特性 | Qwen3 (2025.04) | Qwen3.5 (2026.02) | Qwen3-Next (2026.05) |
|------|-----------------|--------------------|----------------------|
| 架构 | 纯 Transformer | Hybrid DeltaNet + Attention | 继承 3.5 |
| 模态 | 文本 | 文本 + 图像 + 视频 | 文本 + 图像 + 视频 + 语音 |
| 上下文 | 32K / 128K | 262K（原生） | 1M+ |
| 训练精度 | BF16 | FP8（原生） | FP8 |
| 开源范围 | 全部规格 | 全部规格 | 仅 0.8B/2B/4B |
| 代表亮点 | Thinking 模式首发 | 9B 超越 GPT-OSS-120B | — |

Qwen3.5 相比 Qwen3 的核心跨越：
1. **纯文本 → 原生多模态**：不再需要单独的 VL 模型
2. **纯 Transformer → 混合注意力**：长上下文性能数量级提升
3. **BF16 → FP8 native**：同等硬件跑更大模型或更快推理
4. **128K → 262K 上下文**，且 35B+ MoE 模型以更低激活参数量提供更强能力

## References

- [Qwen3.5 技术报告](https://arxiv.org/abs/2602.19444) — Hybrid Attention and Native FP8 Training
- [Gated DeltaNet 论文](https://arxiv.org/abs/2412.06464) — Linear Attention with Gating
- [Hybrid Architecture Ablation Study](https://ar5iv.labs.arxiv.org/html/2603.22473) — Borobia et al., 2025
- [Qwen3.5 HuggingFace](https://huggingface.co/Qwen)
- [Qwen3.5 ModelScope](https://modelscope.cn/organization/qwen)
