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

## 参考

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [LLamaFactory LoRA Documentation](https://llamafactory.readthedocs.io/)
