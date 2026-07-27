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

#### `lora_dropout` — 正则化防过拟合

| 场景 | dropout | 原因 |
|------|---------|------|
| 0.8B~4B, CIFAR100 | 0.05 | 数据量 20K，过拟合风险低 |
| 9B, CIFAR10 | **0.1** | 仅 2K 样本，大模型容易记住训练集 |
| 9B, CIFAR100 | 0.1 | 高 rank 带来更多参数，需要更强的正则化 |

CIFAR10 只有 10 类 2000 张训练图，用 9B + rank=32 训练时必须加大 dropout。

#### `lora_target` — 可实验的方向

`all` 表示在所有线性层（attention + MLP + vision encoder）上加 LoRA。也可以尝试：

- `q_proj,k_proj,v_proj,o_proj` — **仅 attention 层**，参数量减半。对分类任务可能更好：attention 学任务模式，MLP 保留原始知识
- `all` — 基准配置，适用范围最广

> 当前推荐保持 `all`。如果想追求极致精度，可以对比实验 attention-only vs all。

### 对比总结

| 维度 | 0.8B | 2B | 4B | 9B |
|------|------|----|----|----|
| 适配思路 | 轻量 rank，快速收敛 | 基准配置 | 适度提升 rank，降 lr | 高 rank，强正则化，低 lr |
| adapter 大小 | ~32 MB | ~32 MB | ~64 MB | ~128 MB |
| 主要风险 | rank 不足导致欠拟合 | — | 过拟合（小数据集） | 过拟合，lr 过高震荡 |

## 参考

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [LLamaFactory LoRA Documentation](https://llamafactory.readthedocs.io/)
