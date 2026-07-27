# 自回归语言模型架构

> 以 GPT / Qwen3.5 为代表的 decoder-only 自回归架构，是当前大语言模型的主流范式。

自回归（Autoregressive）的核心思想极其简洁：**给定前文，预测下一个 token**。把模型生成的 token 拼回输入，再预测下一个——循环往复，直到生成结束符。ChatGPT、Claude、Qwen、DeepSeek、Llama 等所有主流 LLM 都遵循这一范式。

## 自回归生成原理

### 数学定义

自回归模型将文本的联合概率分解为条件概率的连乘：

$$P(x_1, x_2, \dots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, x_2, \dots, x_{t-1})$$

- $x_t$：第 $t$ 个 token
- 模型每次只看**过去**的 token，预测当前 token
- 总的文本概率 = 每一步条件概率的乘积

### 训练：Teacher Forcing

训练时使用"老师强制"策略——喂入**真实的历史 token**，让模型预测下一个：

```
输入:  [今天, 天气, ___]  → 预测: 很
输入:  [今天, 天气, 很, ___] → 预测: 好
输入:  [今天, 天气, 很, 好, ___] → 预测: 。
```

每一步都并行计算（因为历史是已知的），loss 为交叉熵：

$$\mathcal{L} = -\sum_{t} \log P(x_t \mid x_{<t})$$

### 推理：逐 Token 生成

推理时模型**只能看到自己生成的 token**，必须串行：

```
Step 1: [BOS] → 输出 "今天"
Step 2: [BOS, 今天] → 输出 "天气"
Step 3: [BOS, 今天, 天气] → 输出 "很"
Step 4: [BOS, 今天, 天气, 很] → 输出 "好"
...
```

这是自回归的根本约束：**训练时并行，推理时串行**。也是为什么 LLM 推理受显存带宽而非算力限制——每生成一个 token 都需要重新加载整个模型权重。

## Transformer Decoder 架构

现代自回归 LLM 几乎全部基于 Transformer Decoder（GPT 架构）。以 Qwen3.5 为例，每个 Decoder 层由以下组件构成：

```
输入 Token Embedding (+ Positional Encoding)
    ↓
┌─────────────────────────────┐
│  RMS Norm                   │
│  Causal Self-Attention      │ ← 核心：只能看到当前位置之前的信息
│  Residual Connection (+)     │
├─────────────────────────────┤
│  RMS Norm                   │
│  Feed-Forward Network (FFN) │ ← SwiGLU / GELU 激活
│  Residual Connection (+)     │
└─────────────────────────────┘
    ↓  × N layers
  LM Head (vocab projection)
    ↓
Softmax → 下一个 token 的概率分布
```

### 1. Causal Self-Attention（因果自注意力）

这是 Transformer 架构最核心的组件。对于输入序列 $X \in \mathbb{R}^{n \times d}$，注意力计算：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

其中 $M$ 是**因果掩码**（causal mask）——一个上三角为 -∞ 的矩阵，确保位置 $i$ 只能看到位置 $1 \dots i$：

```
M[i,j] = 0   if j ≤ i    (可以看到)
M[i,j] = -∞  if j > i    (不可看到，softmax 后权重为 0)
```

**直觉**：每个 token 的 Query 去"问"所有 Key，但只能问到当前位置及之前的内容。"未来的词不能看"——这就是 causal 的含义。

**实际实现**通常使用 **Grouped Query Attention (GQA)** 来节省 KV cache：
- Q heads 比 KV heads 多 N 倍（如 8 个 Q head 共享 2 个 KV head）
- 大幅减少推理时的 KV cache 显存占用
- Qwen3.5-9B：32 attention heads，8 KV heads（4:1 比例）

### 2. Positional Encoding（位置编码）

Transformer 的 self-attention 本身**对位置不敏感**——"A 看 B"和"A 看 C"的权重计算完全相同。位置编码告诉模型每个 token 在哪里。

| 方法 | 原理 | 代表 |
|------|------|------|
| **Absolute PE** | 每个位置有一个唯一向量，直接加到 token embedding | 原始 Transformer、BERT |
| **RoPE** (Rotary Position Embedding) | 通过旋转矩阵对 Q/K 施加相对位置偏置 | Qwen、Llama、DeepSeek |
| **ALiBi** | 在注意力分数上直接加线性偏置 | BLOOM、早期 MPT |

当前主流是 **RoPE**：将位置信息编码为旋转角度，使得注意力分值自然包含 Q 和 K 之间的**相对**距离：

$$\langle f_q(x_m, m), f_k(x_n, n) \rangle = g(x_m, x_n, m-n)$$

RoPE 可以做外推——训练时没见过那么长的序列，推理时照样能用（Qwen3.5 训练 32K，通过 YaRN 外推到 1M+）。

### 3. Feed-Forward Network (FFN)

每个 token 独立通过一个两层的全连接网络（位置间无交互）：

$$\text{FFN}(x) = W_2 \cdot \phi(W_1 \cdot x)$$

现代 LLM 使用 **SwiGLU** 激活（而非原始 ReLU/GELU）：

$$\text{SwiGLU}(x) = (x \cdot W_{\text{gate}}) \odot \text{SiLU}(x \cdot W_{\text{up}}) \cdot W_{\text{down}}$$

- SiLU = x·σ(x)，是 GELU 的近似，实现更简单
- 门控机制让网络学会选择性传递信息

**FFN 存储了模型的大部分知识**（参数占 2/3），Attention 负责"从上下文中提取相关模式"。

### 4. RMS Norm & Residual Connection

**Pre-Norm**（当前主流）：Norm 放在子层之前而非之后，训练更稳定：

```
输出 = x + SubLayer(RMSNorm(x))
```

**RMSNorm** 是 LayerNorm 的简化版——去掉减均值的步骤，只做缩放，计算更快且效果相当。

## 推理关键技术

### KV Cache

自回归推理的核心优化。每次生成新 token 时，之前所有 token 的 K、V 矩阵不变——缓存起来不要重算：

| 无 KV Cache | 有 KV Cache |
|------------|------------|
| 每步重新计算所有 token 的 KQV | 只算新 token，历史 K/V 复用 |
| O(n²) 计算 | O(n) 计算 |
| 无法支持长序列 | 显存换速度 |

KV Cache 的显存占用 = $2 \times \text{layers} \times \text{kv\_heads} \times \text{head\_dim} \times \text{seq\_len} \times \text{bytes}$。

以 Qwen3.5-2B 为例：2 × 24 × 2 × 128 × 4096 × 2 ≈ 100MB（4K 序列），但 256K 序列时膨胀到 ~6GB。

### 采样策略

| 方法 | 原理 | 适用场景 |
|------|------|---------|
| Greedy | 每步选最高概率 token | 确定性输出 |
| Temperature | 缩放 logits 后采样，T>1 更随机 | 创造性写作 |
| Top-K | 只从概率最高的 K 个中采样 | 过滤低概率噪声 |
| Top-P (nucleus) | 累积概率达到 P 时截断 | 动态调整候选集 |
| Beam Search | 维护 K 条候选路径 | 翻译、代码生成 |

本项目 CIFAR 分类 eval 使用 `temperature=0`（即 greedy），因为分类任务需要确定性输出。

## 架构对比

### 全景对比

| 架构 | 代表模型 | Attention 方向 | 典型任务 | 优势 | 劣势 |
|------|---------|---------------|---------|------|------|
| **Decoder-only** | GPT, Qwen, Llama, Claude | Causal（单向） | 文本生成、对话 | 统一范式，zero-shot 强 | 推理串行，KV cache 膨胀 |
| **Encoder-only** | BERT, RoBERTa | Bidirectional（双向） | 分类、NER、检索 | 上下文理解深 | 不能生成，需加任务头 |
| **Encoder-Decoder** | T5, BART, GLM | 双向 Enc + 单向 Dec | 翻译、摘要 | 输入输出长度解耦 | 参数多一份，两套逻辑 |
| **SSM** | Mamba, Mamba-2, H3 | N/A（状态空间） | 长序列、DNA | O(1) 推理，长序列成本恒定 | 复制/召回任务弱于 Attention |
| **Hybrid** | Qwen3.5, Jamba | Attention + SSM 混合 | 通用 | 兼顾质量与长序列效率 | 架构复杂，调参经验少 |

### Decoder-only vs Encoder-only

Decoder-only 为什么最终胜出？

1. **统一范式**：所有 NLP 任务都可以改写为"序列→序列"（分类→"这张图是__"，翻译→"翻译：EN→ZH：__"）
2. **Zero-shot 涌现**：单向建模迫使模型学会"向前看"，训练规模足够后自动涌现推理能力
3. **硬件友好**：causal mask 的规则性让 GPU kernel 可以极致优化（FlashAttention 的核心假设）
4. **BERT 的局限**：双向编码器做生成依然需要拼接一个 Decoder，且无法统一 prompt 范式

但这不意味着 Encoder-only 过时了——BERT 系在**嵌入检索、序列表征**场景仍然高效（如 BGE、E5 等 embedding 模型）。

### Decoder-only vs SSM（状态空间模型）

SSM 是最值得关注的非 Transformer 路线：

**SSM 原理**：将序列建模为连续时间状态空间的离散化

$$h'(t) = A h(t) + B x(t), \quad y(t) = C h(t)$$

- $h(t)$：固定大小的隐藏状态（类似 RNN）
- 每个 token 只需 O(1) 时间更新状态
- 不存储 KV cache，长序列推理成本与短序列相同

| 维度 | Transformer (Attention) | SSM (Mamba) |
|------|------------------------|-------------|
| 计算复杂度 | O(n²) → 需 KV cache 配合 | O(n) → 无需 KV cache |
| 长序列推理成本 | 随长度平方增长 | 随长度线性增长 |
| 复制/精确召回 | 强：精准注意力匹配 | 弱：状态压缩丢失细节 |
| 多跳推理 | 强 | 中等 |
| 生态成熟度 | 极高 | 快速追赶中 |

**Qwen3.5 的取巧方案**：把 DeltaNet（线性注意力的变体，SSM 的近亲）作为主力，75% 的层跑 O(1)；25% 的层保留 Full Attention 处理需要精密匹配的长程依赖。做到了鱼和熊掌兼得。

### MoE（混合专家）

MoE 不是替代 Attention 的架构，而是对 FFN 层的改造：

```
常规 FFN：所有 token 走同一套 W1, W2
MoE FFN：Router 选择 top-K 个 Expert，每个 Expert 是一套独立的 FFN 参数
```

- 总参数量巨大（122B、397B），但每个 token 只激活一小部分（~10B）
- 推理速度极快——9B dense 和 35B-A3B MoE 的推理速度可以相当
- 代价：**需要足够显存放全部 expert 权重**

## 关键设计选择总结

| 设计选择 | 早期方案 | 现代主流（Qwen3.5/Llama 系） |
|---------|---------|---------------------------|
| Norm 位置 | Post-Norm | **Pre-Norm** + RMSNorm |
| 激活函数 | ReLU / GELU | **SwiGLU** |
| 位置编码 | 可学习 Absolute PE | **RoPE** |
| Attention 类型 | MHA（全量多头） | **GQA**（分组查询） |
| FFN 结构 | 单门控或无门控 | **SwiGLU 门控** |
| 架构混合 | 纯 Transformer | **DeltaNet + Attention** 混合 |

这些选择并非一开始就确定的——是 GPT-2 → GPT-3 → Llama → Chinchilla → Llama2 → Qwen3.5 的渐进式优化累积。

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Transformer 原始论文
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- [Qwen3.5 Technical Report](https://arxiv.org/abs/2602.19444)
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
