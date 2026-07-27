# 扩散模型架构

> 以 Stable Diffusion / Seedance 2.0 为代表的扩散模型，是当前图像和视频生成领域的主流范式。

扩散模型的核心直觉：**逐步给一张图片加噪，直到变成纯噪声；然后训练一个神经网络学会将这个过程逆转——从噪声中一步步还原出图片**。这就像学会如何从一团混沌中"雕刻"出有意义的内容。

## 扩散原理

### 前向过程（加噪）

前向过程是**固定的、无需学习**的马尔可夫链。从真实图像 $x_0$ 开始，逐步加入高斯噪声：

$$q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} \cdot x_{t-1}, \beta_t \cdot \mathbf{I})$$

- $\beta_t$：噪声调度（noise schedule），控制每步加多少噪声
- $t = 1, 2, \dots, T$：通常 $T = 1000$
- 到 $x_T$ 时，图像已经完全变成各向同性的高斯噪声 $\mathcal{N}(0, \mathbf{I})$

**重参数化技巧**：可以从 $x_0$ 一步算出任意时刻的 $x_t$：

$$x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

其中 $\bar{\alpha}_t = \prod_{s=1}^t (1 - \beta_s)$。

### 反向过程（去噪）

反向过程是从噪声逐步还原图像，由一个**可学习的神经网络** $\epsilon_\theta$ 驱动：

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

关键：$\mu_\theta$ 不直接预测去噪后的均值，而是**预测噪声** $\epsilon$——模型学会"识别当前 $x_t$ 里混了多少噪声，把它抽出来"。

### 训练目标

扩散模型的核心 loss 极其简洁：

$$\mathcal{L} = \mathbb{E}_{x_0, \epsilon, t} \left[ \|\epsilon - \epsilon_\theta(x_t, t)\|^2 \right]$$

翻译成人话：**随机挑一张图 $x_0$、一个时间步 $t$、一个噪声 $\epsilon$，把 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$ 喂给模型，让模型猜出噪声 $\epsilon$ 是什么。**

这就是著名的 **$\epsilon$-prediction**（噪声预测）目标。

```
训练循环：
  x_0 = 真实图片
  t   = random(1, T)            # 随机时间步
  ε   = random normal           # 随机噪声
  x_t = sqrt(ᾱ_t)·x_0 + sqrt(1-ᾱ_t)·ε    # 加噪后的图片
  loss = MSE(ε, model(x_t, t))  # 让模型预测噪声
  loss.backward()
```

### 为什么加噪去噪能 work？

直觉类比——雕塑：

- 前向过程 = 把石膏粉均匀撒在雕像上，直到完全看不见
- 反向过程 = 雕塑家一笔一笔把石膏去掉，露出原始形状
- 模型学会的是 **"这块石膏下面大概是什么"**——它不记住具体图片，而是学会数据分布中的统计规律

关键洞察：**预测噪声等价于学习数据分布的梯度（score function）**：

$$\nabla_{x_t} \log q(x_t) \approx -\frac{1}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)$$

所以扩散模型的本质是**学习数据密度函数的梯度场，然后沿梯度场迭代走向高概率区域**。

## 扩散模型架构演进

### 第一代：U-Net（基于卷积）

Stable Diffusion 1.x / 2.x / XL 使用的经典架构：

```
输入 (latent, 4×H×W)
    ↓
┌──────────────────────┐
│  ResBlock + Self-Attn│  ← Encoder（下采样）
│  ↓ 2× downsample     │
│  ResBlock + Self-Attn│
│  ↓ 2× downsample     │
│  ResBlock + Self-Attn│
│  ↓ 2× downsample     │
├──────────────────────┤
│  ResBlock + Cross-Attn│ ← Bottleneck（中间层）
├──────────────────────┤
│  ResBlock + Self-Attn│  ← Decoder（上采样）
│  ↑ 2× upsample       │
│  ResBlock + Self-Attn│
│  ↑ 2× upsample       │
│  ResBlock + Self-Attn│
│  ↑ 2× upsample       │
└──────────────────────┘
    ↓
输出 (噪声预测, 4×H×W)
```

U-Net 的特点：
- **Skip connection**：Encoder 每层的输出直接拼接到 Decoder 对应层（保留细节信息）
- **Cross-attention**：文本条件通过交叉注意力注入每一层
- **时间嵌入**：时间步 $t$ 编码为正弦位置编码后加到 ResBlock 中

### 第二代：DiT（Diffusion Transformer）

SD3、Flux、Seedance、Sora 使用的架构——用 Transformer 替代 U-Net：

```
输入 (noisy latent patches)
    ↓
Patch Embedding + Positional Encoding
    ↓
┌──────────────────────────────┐
│  Adaptive Layer Norm         │ ← 时间 t 调制 scale/shift
│  Multi-Head Self-Attention   │
│  Residual Connection (+)     │
├──────────────────────────────┤
│  Adaptive Layer Norm         │ ← 时间 t + 文本 c 调制
│  MLP / FFN                   │
│  Residual Connection (+)     │
└──────────────────────────────┘
    ↓  × N layers
  Layer Norm → Linear Projection
    ↓
噪声预测
```

DiT 的优势：
1. **完全摒弃卷积**：全部用 Attention + FFN 处理
2. **adaLN-Zero**（自适应层归一化）：时间步和条件通过 scale/shift 参数注入每一层，而非简单相加
3. **可扩展性强**：参数量随 hidden dim 和层数扩展，类似 LLM 的 scaling law
4. **统一架构**：和 LLM 使用相同的 Transformer 组件，可复用 FlashAttention 等优化

### U-Net vs DiT 对比

| 维度 | U-Net (SD 1.x/XL) | DiT (SD3/Flux/Seedance) |
|------|-------------------|------------------------|
| 基础算子 | 卷积 (ResBlock) | 注意力 (Transformer Block) |
| 下/上采样 | 逐级 2× 下采样+上采样 | 补丁化 (patchify) + 固定分辨率 |
| 条件注入 | Cross-attention | adaLN (scale/shift 调制) |
| Skip connection | Encoder→Decoder 拼接 | 无（纯 Transformer，同分辨率） |
| 参数量 | ~0.8B (SDXL) | ~2B~8B (Flux, Seedance) |
| 生成质量 | 尚可，细节有时模糊 | 显著提升，文字渲染能力强 |
| 训练效率 | 收敛快（卷积 inductive bias） | 收敛慢但上限高 |

## 关键技术与概念

### 1. Latent Diffusion（潜在空间扩散）

直接在像素空间做扩散极其昂贵（512×512×3 = 786K 维）。**Stable Diffusion** 的关键创新：

```
像素空间 (512×512×3) → VAE Encoder → 潜在空间 (64×64×4) → 扩散模型 → VAE Decoder → 像素空间
```

- 压缩比约 48:1
- VAE 单独预训练，扩散模型只在 latent 上训练
- 所有现代图像/视频扩散模型都采用这一范式

### 2. Classifier-Free Guidance (CFG)

无条件生成 vs 有条件生成之间的差值，用来**放大条件信号**：

$$\tilde{\epsilon}(x_t, c) = \epsilon_\theta(x_t, \varnothing) + w \cdot \left[\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \varnothing)\right]$$

- $c$：文本/图像条件
- $\varnothing$：空条件（随机丢弃，训练时同时学习有条件和无条件预测）
- $w$：guidance scale，通常 3~7.5

**直觉**：$w$ 越大，生成结果越"听话"（跟随 prompt），但也越不自然。$w>1$ 外推了条件信号的方向——相当于"超分辨率"了条件。

### 3. 采样调度器（Scheduler）

DDPM 原版需要 $T=1000$ 步采样。后来的改进大幅减少了步数：

| 方法 | 原理 | 典型步数 |
|------|------|---------|
| **DDPM** | 马尔可夫链逐步去噪 | 1000 |
| **DDIM** | 去噪过程可以是确定性的（非马尔可夫），跨步跳跃 | 50~200 |
| **DPM-Solver** | 将扩散过程视为 ODE，用高阶数值求解器 | 20~50 |
| **Euler Ancestral** | 欧拉方法 + 每步注入少量随机噪声 | 20~30 |
| **LCM** (Latent Consistency Model) | 蒸馏后的模型，直接预测 $x_0$ | 1~4 |

现代实践：**DDIM 或 DPM-Solver，20~50 步即可**，再多步数边际收益极低。

### 4. 条件机制

扩散模型支持多种条件输入方式：

| 条件类型 | 注入方式 | 示例 |
|---------|---------|------|
| **文本** | Cross-attention（U-Net）或 adaLN（DiT） | Prompt → 图像 |
| **图像** | 拼接为 latent 输入 | img2img, ControlNet, IP-Adapter |
| **结构** | 额外编码器输出拼接 | depth, canny, pose |
| **时间** | 时间步正弦编码 + 额外维度 | 视频帧索引 |
| **身份** | 面部特征注入 | IP-Adapter-FaceID, InstantID |

## 视频扩散模型（Seedance 2.0）

视频生成的本质挑战：**既要每帧好看（空间质量），又要帧间连贯（时间一致性）**。

### 核心思路

视频扩散模型在图像模型基础上增加了**时间维度**：

```
图像:  [C, H, W]           →  latent: [C, H/8, W/8]
视频:  [C, T, H, W]        →  latent: [C, T/4, H/8, W/8]
```

- 空间维度：H×W，走 8× 压缩（和图像相同）
- 时间维度：T，走 4× 压缩（帧间冗余大，压缩比可以更高）

### 架构扩展

在 DiT 基础上，视频模型增加了两类注意力：

```
┌───────────────────────────────────┐
│  Spatial Self-Attention           │ ← 同一帧内各 patch 互相关注
│  (每个 patch 只看自己帧里的 patch) │
├───────────────────────────────────┤
│  Temporal Self-Attention          │ ← 跨帧同位置 patch 互相关注
│  (同一空间位置，不同时间帧互看)    │
├───────────────────────────────────┤
│  Cross-Attention (text condition) │ ← 文本条件注入
├───────────────────────────────────┤
│  MLP / FFN                        │
└───────────────────────────────────┘
```

Spatial Attention 保证**帧内质量**，Temporal Attention 保证**帧间连贯**。两者分工明确。

### Seedance 2.0 的关键特性

Seedance 2.0（字节跳动，2025）是目前视频生成的第一梯队：

| 特性 | 说明 |
|------|------|
| 架构 | DiT-based，Spatial + Temporal Attention |
| 分辨率 | 原生 1080p，支持 720p/1080p 输出 |
| 时长 | 最长 10 秒，24fps |
| 关键帧控制 | 支持给定首帧/尾帧做插值 |
| 相机控制 | 可控运镜（推拉摇移） |
| 推理速度 | 优化后 ~30s 生成 5s 视频（H20 GPU） |

> 截至 2025 年末，Seedance 2.0、Kling 2.0、Sora 三者处于视频生成的头部梯队，各有优势。

## 其他生成式架构对比

### 全景对比

| 架构 | 原理 | 生成质量 | 生成速度 | 可控性 | 代表 |
|------|------|---------|---------|--------|------|
| **GAN** | Generator vs Discriminator 对抗训练 | 高（单次前向） | **极快** | 弱（模式坍塌） | StyleGAN, BigGAN |
| **VAE** | Encoder→Latent→Decoder，最大化 ELBO | 中（偏模糊） | 快 | 中 | VAE, VQ-VAE |
| **Autoregressive** | 逐 token 生成图像 patch | 高 | **慢**（串行） | 强（prompt 跟随好） | DALL-E 1, Parti |
| **Diffusion** | 迭代去噪 | **最高** | 中等（20~50 步） | 强 | SD, Flux, Seedance |
| **Flow Matching** | 学习 ODE 路径，更自由的噪声调度 | 高 | 快（10~20 步） | 强 | SD3, Flux (部分) |
| **Masked Models** | 随机 mask 掉 patch 再预测回来 | 中高 | 中 | 中 | MaskGIT, MAGVIT |

### 为什么扩散模型胜出

1. **训练稳定**：不需要像 GAN 那样平衡生成器和判别器，loss 就是简单的 MSE
2. **质量天花板高**：迭代式生成 + CFG 引导，可以在质量和 prompt 跟随之间精确折衷
3. **条件注入灵活**：Cross-attention 和 adaLN 天然支持多模态条件
4. **架构复用**：DiT 之后和 LLM 共享大量基础设施（FlashAttention、序列并行等）

### 扩散 vs 自回归

这是当前生成式 AI 最核心的路线之争：

| 维度 | 自回归 (AR) | 扩散 (Diffusion) |
|------|------------|-----------------|
| 生成方式 | 逐个 token，串行 | 并行去噪，多步迭代 |
| 最适合 | 离散序列（文本、代码） | 连续信号（图像、视频、音频） |
| 推理速度 | kV cache，每 token O(n) | 每步 O(1)，但需要 20~50 步 |
| 多样性 | 强（温度、top-p 采样） | 通过初始噪声控制 |
| 精确控制 | prompt 跟随精准 | 需要 CFG 或 ControlNet 辅助 |
| 统一趋势 | GPT-4o（图像理解+生成） | 视频生成 + 多模态扩散 |

**融合趋势**：最新的 SOTA 开始打破边界——
- **VAR (Visual Autoregressive)**：用自回归生成图像的"下一尺度残差"而非"下一个像素"
- **Muse / MAGVIT**：把图像量化成离散 token，用 Masked + Autoregressive 混合生成
- **Unified AR-Diffusion**：端到端用 diffusion 做视觉 token 生成，AR 做文本 token 生成

## 总结

| 概念 | 一句话 |
|------|--------|
| 扩散原理 | 学习逆向去噪过程——从随机噪声逐步还原数据 |
| 训练目标 | MSE——让模型学会预测噪声 |
| U-Net 时代 | 卷积骨架 + Cross-attention 条件注入 |
| DiT 时代 | 全 Transformer，adaLN 调制，Scaling Law 友好 |
| Latent 扩散 | 在压缩后的潜在空间扩散，大幅降低计算量 |
| CFG | 有条件 vs 无条件的差值来引导生成方向 |
| 视频扩散 | Spatial + Temporal Attention，空间质量和时间连贯分工 |
| 扩散 vs AR | 连续信号用扩散，离散序列用 AR，两者正在融合 |

## References

- [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239) — 扩散模型奠基论文
- [Denoising Diffusion Implicit Models (DDIM)](https://arxiv.org/abs/2010.02502) — 确定性采样，大幅减少步数
- [High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion)](https://arxiv.org/abs/2112.10752)
- [Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) — Transformer 替代 U-Net
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [Seedance 2.0 Technical Report](https://arxiv.org/abs/2506.18283)
- [Sora Technical Report](https://openai.com/index/video-generation-models-as-world-simulators/)
