# LoRA / QLoRA

## LoRA（Low-Rank Adaptation）

LoRA 是一种参数高效微调（PEFT）方法，核心思想是：**不修改原始模型权重，而是在模型层旁路插入低秩矩阵进行训练**。

### 原理

对于原始权重矩阵 $W \in \mathbb{R}^{d \times k}$，LoRA 引入两个低秩矩阵 $A \in \mathbb{R}^{d \times r}$ 和 $B \in \mathbb{R}^{r \times k}$（其中 $r \ll d, k$），前向传播变为：

$$h = Wx + \alpha \cdot BAx$$

训练时冻结 $W$，仅更新 $A$ 和 $B$。

### 关键参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `r` (rank) | 低秩矩阵的秩，越大表达能力越强 | 8~64 |
| `alpha` | 缩放因子，通常设为 `2 * r` | 16~128 |
| `dropout` | LoRA dropout 比例 | 0.0~0.1 |
| `target_modules` | 应用 LoRA 的模块 | `all` (LLamaFactory) |

### LoRA Rank 深入理解

#### 数学直觉

原始权重矩阵 $W \in \mathbb{R}^{d \times k}$ 的完整更新需要 $d \times k$ 个参数。LoRA 将其分解为 $B_{d \times r} \cdot A_{r \times k}$，参数压缩为 $r \times (d + k)$：

$$\Delta W = BA, \quad \text{rank}(\Delta W) \le r$$

这意味着 **LoRA 更新矩阵最多只有 r 个非零奇异值**。rank 直接决定了 adapter 能表达多么复杂的权重修正。

#### 为什么大模型需要更高 rank

核心矛盾：**模型内部维度 d 随参数量增长，但 rank 不变意味着 adapter 覆盖的比例在下降**。

以 Qwen3.5 系列为例，每层的 hidden dimension 和 adapter 覆盖能力：

| 模型 | hidden dim | 全量更新参数量 | rank=8 adapter | rank=32 adapter |
|------|-----------|-------------|---------------|----------------|
| 0.8B | ~1536 | 2.4M | 24K（1.0%） | 96K（4.0%） |
| 2B | ~2048 | 4.2M | 33K（0.8%） | 131K（3.1%） |
| 4B | ~2560 | 6.6M | 41K（0.6%） | 164K（2.5%） |
| 9B | ~4096 | 16.8M | 66K（0.4%） | 262K（1.6%） |

> 表中数据为**每层单个线性层**的近似值。`d` 取 hidden_dim，adapter 参数量 = r × (d + k)，其中 k 为另一维度（通常接近 d）。

可以看到：
- 0.8B 用 rank=8 已经覆盖了 1% 的更新空间，足够
- 9B 用 rank=8 只覆盖了 0.4%，**信息瓶颈严重**
- 9B 提到 rank=32 后覆盖 1.6%，才接近 0.8B 用 rank=8 的水平

换句话说：**同样 rank=8，在 9B 上起到的效果大约只有 0.8B 上的 40%**。

#### 内在维度视角

研究表明，大模型微调时参数的实际变化集中在一个低维子空间中（即"内在维度"）。但这个内在维度**随模型增大而增大**：

- 小模型（<1B）的内在维度通常 <10，rank=8 足够捕获
- 大模型（>7B）的内在维度可达几十甚至上百，rank=8 成为瓶颈

这也是为什么更高 rank 对大模型精度提升明显，而小模型 rank=8 就可能饱和。

#### rank 与 alpha 的关系

LoRA 的实际输出为：

$$h = Wx + \frac{\alpha}{r} \cdot BAx$$

缩放因子 $\alpha/r$ 控制 LoRA 输出的强度。增大 rank 时**必须同步增大 alpha** 才能保持相同的有效强度。如果只改 rank 不动 alpha：
- rank=8, alpha=16 → 缩放 2.0
- rank=32, alpha=16 → 缩放 0.5（LoRA 输出被压到原来的 1/4）

这也是推荐配置中 `alpha = 2 × rank` 的理论依据：保持缩放因子恒定为 2.0，让不同 rank 之间可比较。

#### 总结

| 因素 | rank 低（如 8） | rank 高（如 32） |
|------|----------------|-----------------|
| 表达能力 | 受限，只能学简单修正 | 强，能学复杂任务模式 |
| 过拟合风险 | 低（天然正则化） | 高（需要 dropout 配合） |
| 参数量 | ~32 MB | ~128 MB |
| 适用模型 | 0.8B~2B | 4B~9B |

> **经验法则**：rank 应该使 adapter 参数量达到基座模型的 0.5%~2%。对于 9B 模型，rank=32 的 128MB adapter 约占 0.7%，处于合理区间。

### 优势

- **参数量极小**：2B 模型的 LoRA 权重仅 ~10MB
- **训练快**：只需更新少量参数
- **可插拔**：同一个基座模型可挂载多个 LoRA adapter
- **显存友好**：2B 模型 LoRA 训练仅需 ~8GB 显存

## QLoRA（Quantized LoRA）

QLoRA 在 LoRA 基础上将基座模型 **4-bit 量化**，进一步降低显存需求。

- 基座模型以 NF4 格式存储
- 训练时反量化为 BF16 计算
- 2B 模型 QLoRA 训练仅需 ~4GB 显存

### QLoRA 专属参数

```yaml
### quantization (QLoRA)
quantization_method: bitsandbytes
quantization_bit: 4
double_quantization: true         # 二次量化——连量化常量也量化，再省 ~0.4 bit/param
bnb_4bit_compute_dtype: bfloat16  # 反量化后的计算精度
bnb_4bit_quant_type: nf4          # 量化数据类型：nf4（推荐）或 fp4
```

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `quantization_bit` | 量化位宽 | `4`（QLoRA 标准） |
| `double_quantization` | 二次量化——将量化常量 `c1` 再用 `c2` 量化 | `true`（省显存，几乎无损） |
| `bnb_4bit_compute_dtype` | 反量化后的计算精度 | `bfloat16`（与训练精度一致） |
| `bnb_4bit_quant_type` | 量化数据类型 | `nf4`（NormalFloat4，对正态分布权重最优） |

**NF4 vs FP4**：FP4 是标准 4-bit 浮点（等距分区），NF4 假设权重服从正态分布——在概率密度高的区域分配更多量化台阶，信息损失更小。大模型预训练权重经 LayerNorm 归一化后恰好近似正态分布。

**为什么 double_quantization 几乎零成本**：量化常量本身很小（每 64 个参数共享 1 个 c1 = 32bit），二次量化为 8bit 节省了 `32-8=24` bits per 64 params ≈ 0.38 bit/param。对于 9B 模型，这省了约 0.4GB 显存。

### QLoRA vs LoRA 精度对比

核心问题：4-bit 量化会不会损害最终精度？答案是**基本不会**。

| 维度 | LoRA (bf16) | QLoRA (nf4) |
|------|------------|-------------|
| 基座模型存储 | BF16（2 bytes/param） | NF4（0.5 bytes/param） |
| 前向/反向计算 | BF16 | BF16（反量化后） |
| 梯度存储 | BF16（仅 LoRA 参数） | BF16（仅 LoRA 参数） |
| 优化器状态 | BF16 × 2（仅 LoRA 参数） | BF16 × 2（仅 LoRA 参数） |
| 对精度的影响 | 0（无量化） | 量化噪声（NF4 → BF16 的误差） |

**关键发现**：

- **同等 rank**：QLoRA 精度通常比 LoRA 低 0.5~2%。但通过**提高 rank** 可以**完全补偿**——QLoRA rank=16 精度 ≈ LoRA rank=8 精度
- **本项目实践**：QLoRA 配置默认 `rank=16`（对比 LoRA 默认 `rank=8`），正是基于这个经验
- **对于 RTX 5090 32GB**：LoRA 显存足够，直接选 LoRA。只有 9B + 高分辨率图像显存吃紧时才考虑 QLoRA

### 什么时候不该用 LoRA

LoRA 并非万能。以下场景应优先考虑全量微调：

| 场景 | 问题 | 建议 |
|------|------|------|
| **任务分布与预训练差异极大** | 低秩约束限制模型"转向"幅度 | 全量微调（数据足够时）或换更贴近的基座模型 |
| **需要大幅改变模型行为** | 如从英文主模型切换到小众语种 | 全量微调 + 大学习率 |
| **数据量极大（百万级）且质量高** | LoRA 的 rank 瓶颈压制了数据潜力 | 逐步提高 rank（32→64→128）直至接近全量微调效果 |
| **需要合并后分发** | LoRA 权重需额外文件管理 | 这也正是 LoRA 的优势——不合并反而更灵活 |

**实际判断法则**：先跑 LoRA——如果 loss 不降、精度不涨，且确认数据没问题——再考虑全量微调。对于 CIFAR 分类这种与视觉预训练高度对齐的任务，LoRA 完全够用。

## LLamaFactory 中的 LoRA 配置

```yaml
finetuning_type: lora
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05
lora_target: all
```

如果要使用 QLoRA：

```yaml
finetuning_type: lora
quantization_bit: 4
```

## 分模型调参建议

不同参数规模的模型需要不同的 LoRA 超参数来发挥最佳性能。核心原则：**大模型容量更大，需要更高 rank 来匹配，同时需要更保守的学习率和更强的正则化来防止过拟合**。

### 推荐配置

| 参数 | 0.8B | 2B | 4B | 9B |
|------|------|----|----|----|
| `lora_rank` | 8 | 8 | 16 | 32 |
| `lora_alpha` | 16 | 16 | 32 | 64 |
| `lora_dropout` | 0.05 | 0.05 | 0.05 | 0.1 |
| `learning_rate` | 2e-4 | 2e-4 | 1e-4 | 5e-5 |
| `lora_target` | `all` | `all` | `all` | `all` |

> **使用方式**：通过 `run.sh` 的 `--` 传参覆盖 YAML 默认值：
> ```bash
> # 9B 训练示例
> bash scripts/run.sh train --dataset cifar10 --model Qwen/Qwen3.5-9B \
>     -- lora_rank=32 lora_alpha=64 lora_dropout=0.1 learning_rate=5e-5
> ```

### 重要前提：推荐配置 ≠ 所有任务通用

上表的推荐配置基于**复杂任务**（NLU、推理、代码生成等）的通用经验。对于与预训练数据高度对齐的简单任务（如 CIFAR 图像分类），实测结果是另一回事：

| 配置 | 4B CIFAR10 | 4B CIFAR100 | 9B CIFAR10 | 9B CIFAR100 |
|------|-----------|------------|-----------|------------|
| default (rank=8, α=16, dropout=0.05, lr=2e-4) | 97.76% | **88.02%** | **98.05%** | **88.96%** |
| tuned (rank↑, α↑, dropout=0.1, lr↓) | **97.83%** | 87.81% | 97.93% | 88.25% |

> **结论：对于 CIFAR 级别的简单分类任务，默认配置（rank=8）在所有模型尺寸上都是最佳或接近最佳。** 提高 rank 并配套调整其他参数后，9B CIFAR100 反而下降了 0.71%，4 组对比中仅 1 组有微弱提升（+0.07%，噪声级别）。
>
> **为什么会这样？** 上述 rank 理论假设任务需要复杂的权重修正。但 CIFAR 分类只需将预训练中已有的视觉特征映射到 10/100 个类别——这是极浅层的适配。rank=8 在 9B 上的 0.4% 覆盖率对这个任务而言不是瓶颈，反而是保护（天然低秩正则化，防止小数据集过拟合）。提高 rank 给了 adapter 更多自由度，但 CIFAR10 只有 2000 张训练图，CIFAR100 (200/cls) 只有 20000 张，这些自由度没有被有效约束，反而引入了噪声。
>
> **原则：先判断任务复杂度。** 如果任务与预训练数据高度对齐（分类、简单 QA、情感分析），rank=8 对所有模型尺寸都够用。只有任务需要模型学习全新的模式（代码生成、数学推理、小众语种）时，才需要按上表提高 rank。

### 各参数说明

#### `lora_rank` — 直接影响 adapter 表达能力

> 理论推导见 [LoRA Rank 深入理解](#lora-rank-深入理解)。这里是实践层面的总结。

LoRA adapter 是一个低秩瓶颈。rank 决定了 adapter 能学到多复杂的修正：

| rank | adapter 参数量 | 适用 |
|------|---------------|------|
| 8 | ~32 MB | 0.8B/2B：模型内部维度小（1536~2048），rank=8 已覆盖约 1% 的更新空间 |
| 16 | ~64 MB | 4B：内部维度增大（~2560），rank=8 覆盖降到 0.6%，需翻倍 |
| 32 | ~128 MB | 9B：内部维度 ~4096，rank=32 才覆盖 ~1.6%，接近小模型的覆盖水平 |

**直觉类比**：rank 相当于 adapter 的"像素"。同样 8 像素在 100×100 画布（小模型）上能画清一个图形，放到 400×400 画布（大模型）上就糊了。需要提高像素数来匹配画布尺寸。

#### `lora_alpha` — 与 rank 联动

LoRA 输出的缩放因子是 `alpha / rank`。增大 rank 时需同步增大 alpha，否则 LoRA 输出的有效强度会降低：

| rank | alpha | 缩放因子 | 
|------|-------|---------|
| 8 | 16 | 2.0 |
| 16 | 32 | 2.0 |
| 32 | 64 | 2.0 |

保持 `alpha = 2 × rank` 是最保守且经过广泛验证的做法。

#### `learning_rate` — 大模型对学习率更敏感

LoRA 学习中，大模型需要更低的 lr 来避免破坏预训练表征：

| 模型 | lr | 原因 |
|------|-----|------|
| 0.8B/2B | 2e-4 | 通用默认值，稳定收敛 |
| 4B | 1e-4 | 适度降低，减少震荡 |
| 9B | 5e-5 | 大幅降低，保护预训练知识 |

lr 过高在 9B 上可能表现为 loss 抖动、最终精度不升反降。

> **CIFAR 实测**：以上推荐针对复杂任务。CIFAR 分类中，9B 使用默认 lr=2e-4 配合 rank=8 取得了最佳精度（98.05%），降到 5e-5 配合 rank=32 反而下降。简单任务中，低 lr 配合高 rank 可能导致"参数多但学不动"——高 rank 给了更多自由度，但 lr 太低导致这些自由度被噪声填充而非有效利用。

#### `lora_dropout` — 正则化防过拟合

以下推荐基于高 rank 场景（rank ≥ 16）。如果使用默认 rank=8，dropout=0.05 对所有场景都足够：

| 场景 | dropout | 原因 |
|------|---------|------|
| rank=8, 所有模型/数据集 | 0.05 | rank 低 = 天然正则化，不需额外 dropout |
| rank≥16, 数据量充足（>20K） | 0.05~0.1 | 过拟合风险可控 |
| rank≥16, 小数据集（如 CIFAR10 2K） | **0.1** | 高 rank + 小数据 = 必须强正则化 |

> **CIFAR 实测**：9B 默认配置（rank=8, dropout=0.05）在 CIFAR10 上取得了 98.05%，优于 tuned 配置（rank=32, dropout=0.1）的 97.93%。这验证了**与其提高 rank 再加 dropout 压制过拟合，不如直接用低 rank 享受天然正则化**——尤其在小数据集上，低 rank 是更优雅的解决方案。

#### `lora_target` — 可实验的方向

`all` 表示在所有线性层（attention + MLP + vision encoder）上加 LoRA。也可以尝试：

- `q_proj,k_proj,v_proj,o_proj` — **仅 attention 层**，参数量减半。对分类任务可能更好：attention 学任务模式，MLP 保留原始知识
- `all` — 基准配置，适用范围最广

> 当前推荐保持 `all`。如果想追求极致精度，可以对比实验 attention-only vs all。

### 对比总结

以下为**复杂任务**的推荐策略。对于 CIFAR 级别的简单分类任务，`rank=8, α=16, dropout=0.05, lr=2e-4` 对所有模型尺寸均适用——详见上方「重要前提」小节。

| 维度 | 0.8B | 2B | 4B | 9B |
|------|------|----|----|----|
| 适配思路（复杂任务） | 轻量 rank，快速收敛 | 基准配置 | 适度提升 rank，降 lr | 高 rank，强正则化，低 lr |
| 适配思路（简单任务） | rank=8 足够 | rank=8 足够 | rank=8 足够 | rank=8 足够 |
| adapter 大小 | ~32 MB | ~32 MB | ~64 MB | ~128 MB |
| 主要风险 | rank 不足导致欠拟合（仅复杂任务） | — | 过拟合（小数据集 + 高 rank） | 过拟合，lr 过高震荡 |

## 参考

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [LLamaFactory LoRA Documentation](https://llamafactory.readthedocs.io/)
