# 训练参数配置指南

## 核心概念

### 有效 batch size

训练时真正的 batch size 由三个参数共同决定：

```
有效 batch size = per_device_train_batch_size × gradient_accumulation_steps × GPU 数量
```

| 参数 | 控制什么 | 代价 |
|------|---------|------|
| `per_device_train_batch_size` | 单次前向+反向的样本数 | **吃显存**，越大显存占用越高 |
| `gradient_accumulation_steps` | 梯度累积 N 次后才更新一次权重 | **吃时间**，越大训练越慢 |
| GPU 数量 | 数据并行，拆分 batch 到多卡 | 吃硬件 |

三者可以换着用：有效 batch 相同的前提下，**增大 batch_size、减小 accum 步数** → 显存换速度。

### 为什么 accum 步数越多越慢

每步 gradient accumulation 都要跑一次完整的前向+反向传播，但不更新权重。accum=4 意味着做 4 次前向+反向才更新一次，而 accum=1 只需 1 次。虽然有效 batch 相同，但 accum 的开销叠加会显著拖慢每个 step。

---

## 显存分布参考（Qwen3.5-2B + LoRA）

| 组件 | 显存占用 |
|------|---------|
| 基座模型 (bf16) | ~4 GB |
| LoRA 权重 | ~32 MB |
| 优化器状态 (AdamW, LoRA only) | ~134 MB |
| 视觉编码器激活值 | 取决于 batch × image_max_pixels |
| 文本序列激活值 | 取决于 batch × cutoff_len |

> **关键**：对于多模态模型，视觉编码器的激活值是最大的显存消耗源。`image_max_pixels: 262144`（512×512）下，单张图片的视觉 token 大约 ~256 个，batch 增大后激活值线性增长。

### 分模型显存估算

以下为 LoRA（bf16）模式下不同模型的**基座显存 + LoRA + 优化器**基础开销（不含激活值）：

| 模型 | 基座 BF16 | LoRA (rank=8) | LoRA (rank=16) | LoRA (rank=32) | 优化器 | **rank=8 基础总计** |
|------|----------|---------------|----------------|----------------|--------|-------------------|
| 0.8B | ~2 GB | 16 MB | 32 MB | 64 MB | ~67 MB | **~2.1 GB** |
| 2B | ~4 GB | 32 MB | 64 MB | 128 MB | ~134 MB | **~4.2 GB** |
| 4B | ~8 GB | 52 MB | 104 MB | 208 MB | ~218 MB | **~8.3 GB** |
| 9B | ~18 GB | 66 MB | 132 MB | 264 MB | ~276 MB | **~18.4 GB** |

> 优化器状态 ≈ LoRA 参数量 × 4.2（AdamW 需存储 fp32 的 param + exp_avg + exp_avg_sq，加上 bf16 副本）。

**在此基础上叠加激活值**（batch × image tokens 驱动）：

| image_max_pixels | 分辨率 | 单张图 visual tokens | batch=4 时激活值估算 |
|-----------------|--------|---------------------|---------------------|
| 65536 (256²) | 256×256 | ~64 | ~4-6 GB |
| 131072 (362²) | ~362×362 | ~100 | ~7-10 GB |
| 262144 (512²) | 512×512 | ~256 | ~12-18 GB |

**关键结论**：9B + LoRA rank=32（18.7 GB 基础）+ batch=4 + image=262144（~15 GB 激活）≈ **33.7 GB**，RTX 5090 32GB 刚好放不下。此时有两个选择：
1. **降 image_max_pixels** → 65536（省 ~10 GB 激活）→ ~23.7 GB，轻松装下
2. **降 batch_size** → 2（激活减半）→ ~26.2 GB，也够

### 图像预处理与两层 Resize

图像在进入 Vision Encoder 前，会经历**两层 resize**：

```
原始图像 → [LLaMA Factory _regularize_images] → [Qwen3.5 ImageProcessor] → Vision Encoder
```

**第一层：LLaMA Factory（数据加载阶段）**

LLaMA Factory 在 `ProcessorArguments` 中提供了 min/max 两个参数（本项目 YAML 中只配置了 max）：

| 参数 | 默认值 | 本项目配置 | 说明 |
|------|--------|-----------|------|
| `image_min_pixels` | 1,024 (32×32) | 未设置（用默认） | 小于此值的图会被放大 |
| `image_max_pixels` | 589,824 (768×768) | **262,144** (512×512) | 大于此值的图会被缩小 |

该层保持宽高比、等比缩放。

**第二层：Qwen3.5 ImageProcessor（模型内部）**

Qwen3.5 的 `preprocessor_config.json` 自带独立的 resize 参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| min_pixels | 65,536 (256×256) | 小于此值的图会被上采样到约 256² |
| max_pixels | 16,777,216 (4096×4096) | 远大于本项目配置，实际不会触发 |

**CIFAR 32×32 的实际路径**：

```
32×32 (1024 px)
  → LLaMA Factory: 1024 ≥ min(1024)，不放大；1024 ≤ max(262144)，不缩小
  → Qwen3.5 ImageProcessor: 1024 < min(65536)，上采样到约 256×256
  → 最终 visual tokens: ~64 个 (256/16=16, 16²=256, 2×2 merge → 64)
```

对于 CIFAR 32×32 这种极小图片，实际 token 数取决于**两层的 resize 叠加结果**。`image_max_pixels: 262144` 限制的是 LLaMA Factory 这层的上限，但最终分辨率由 Qwen3.5 的 `min_pixels: 65536` 决定了下限。即使降了 LLaMA Factory 的 max，Qwen3.5 内部也会把图放大到至少 256×256（除非能修改模型的 preprocessor_config）。

---

## 参数配置方案

以下配置基于 Qwen3.5-2B LoRA 训练，以 CIFAR-100（20000 训练样本）为例。

### 方案 A：小显存（≤8 GB）— 勉强能跑

适用于 RTX 3060/4060 8GB、笔记本 GPU、免费 Colab。

```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
# 有效 batch = 16，共 6250 steps / epoch

dataloader_num_workers: 2
preprocessing_num_workers: 4
cutoff_len: 1024          # 缩短文本长度
image_max_pixels: 65536   # 降低图片分辨率 (256×256)
bf16: true                # 或 fp16
```

> 也可开启 QLoRA（`quantization_bit: 4`）进一步压缩到 ~4GB。

| 效果 | 预估 |
|------|------|
| 显存使用 | ~7-8 GB |
| Step 耗时 | ~8-12 s/it |
| CIFAR-100 1 epoch | ~14-20 小时 |

### 方案 B：入门级（12~16 GB）— 正常训练

适用于 RTX 4070 12GB、RTX 3080 10GB、T4 16GB。

```yaml
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
# 有效 batch = 16，共 6250 steps

dataloader_num_workers: 4
cutoff_len: 2048
image_max_pixels: 131072  # 362×362
bf16: true
```

| 效果 | 预估 |
|------|------|
| 显存使用 | ~10-14 GB |
| Step 耗时 | ~3-5 s/it |
| CIFAR-100 1 epoch | ~5-8 小时 |

### 方案 C：主流卡（24 GB）— 基准配置

适用于 RTX 3090/4090 24GB、A5000 24GB。当前项目的默认配置。

```yaml
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
# 有效 batch = 16，共 6250 steps

dataloader_num_workers: 4
cutoff_len: 2048
image_max_pixels: 262144  # 512×512
bf16: true
```

| 效果 | 预估 |
|------|------|
| 显存使用 | ~14-18 GB |
| Step 耗时 | ~2-3 s/it |
| CIFAR-100 1 epoch | ~3-5 小时 |

### 方案 D：大显存（32 GB+）— 满速训练

适用于 RTX 5090 32GB、A6000 48GB、A100 40/80GB。

```yaml
per_device_train_batch_size: 16
gradient_accumulation_steps: 1
# 有效 batch = 16，共 6250 steps（step 数相同，但每步快得多）

dataloader_num_workers: 8
cutoff_len: 2048
image_max_pixels: 262144  # 512×512
bf16: true
```

| 效果 | 预估 |
|------|------|
| 显存使用 | ~20-26 GB |
| Step 耗时 | ~0.5-1 s/it |
| CIFAR-100 1 epoch | ~0.6-1.2 小时 |

### 方案 E：超大显存（48 GB+）— 暴力堆 batch

适用于 A6000 48GB、A100 80GB、H100 80GB。

```yaml
per_device_train_batch_size: 32
gradient_accumulation_steps: 1
# 有效 batch = 32，仅 3125 steps

dataloader_num_workers: 8
cutoff_len: 2048
image_max_pixels: 262144
bf16: true
```

> 有效 batch 翻倍到 32，总 step 数减半。对于 CIFAR-100 这种 100 分类任务，大 batch 通常不影响最终精度。

| 效果 | 预估 |
|------|------|
| 显存使用 | ~28-40 GB |
| Step 耗时 | ~0.3-0.6 s/it |
| CIFAR-100 1 epoch | ~0.2-0.5 小时 |

---

## 如何找到自己的最佳 batch_size

逐步翻倍法，每次翻倍后启动训练，观察显存：

```bash
# 监控显存
watch -n 1 nvidia-smi
```

| 步骤 | batch_size | accum | 操作 |
|------|-----------|-------|------|
| 1 | 4 | 4 | 基准，能跑 |
| 2 | 8 | 2 | 翻倍 batch，减半 accum |
| 3 | 16 | 1 | 再翻倍 |
| 4 | 24 | 1 | 激进尝试（改 accum=1 保持有效 batch 不变的话...注意有效 batch 变大） |
| 5 | 32 | 1 | 极限 |

> **OOM 了怎么办**：退回到上一级，或者降低 `image_max_pixels`（视觉激活值是显存大户）。

**注意**：如果保持 `gradient_accumulation_steps: 1` 不变，增大 batch_size 会增大有效 batch，从而减少总 step 数。对于简单任务（CIFAR 分类），有效 batch 16~64 都可以正常收敛。

---

## 多卡配置

### 单机多卡（数据并行）

LLaMA Factory 基于 HuggingFace Trainer，天然支持多卡：

```bash
# 自动使用所有可见 GPU（需设置 CUDA_VISIBLE_DEVICES）
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train configs/cifar100_lora_train.yaml
```

多卡时有效 batch 会乘以卡数：

```
有效 batch = per_device_train_batch_size × gradient_accumulation_steps × GPU 数量
```

因此多卡时可以**大幅减小 `gradient_accumulation_steps`**：

```yaml
# 4 × RTX 5090 配置
per_device_train_batch_size: 16
gradient_accumulation_steps: 1
# 有效 batch = 16 × 1 × 4 = 64，仅 ~782 steps
```

### DeepSpeed（更大规模）

如果需要 ZeRO 优化（显存不够 or 模型更大），使用 DeepSpeed：

```bash
llamafactory-cli train configs/cifar100_lora_train.yaml \
    --deepspeed configs/ds_z2_config.json
```

> 对于 2B 模型 LoRA 训练，一般不需要 DeepSpeed。8B+ 模型或多模态全量微调时才考虑。

---

## 其他加速建议

| 参数 | 建议 | 说明 |
|------|------|------|
| `dataloader_num_workers` | 设为 CPU 核心数的一半 | 加快图片加载，减少 GPU 空闲等待 |
| `preprocessing_num_workers` | 4~8 | 首次预处理并行度 |
| `bf16: true` | 启用 | BF16 与 FP32 精度几乎一致，显存减半 |
| `save_only_model: true` | 不需要断点续训时 | 跳过优化器状态保存，写盘更快 |
| `cutoff_len` | 按需调整 | CIFAR 任务文本很短，1024 足够 |
| `image_max_pixels` | 按需调整 | 降低分辨率显著减少视觉激活值 |
| `save_steps` | 适当加大（如 500→1000） | 减少 checkpoint 写入频率 |

---

## 验证集与 Early Stopping

当前项目 config 中 eval 部分全部注释掉了。启用验证集可以帮助判断是否过拟合，对于高 rank 大模型训练尤其重要。

### 启用验证集

取消注释 eval 部分并配置：

```yaml
### eval
val_size: 0.1                  # 从训练集划出 10% 做验证
per_device_eval_batch_size: 4
eval_strategy: steps           # 按步数评估（也可用 epoch）
eval_steps: 200                # 每 200 步评估一次
load_best_model_at_end: true   # 训练结束时加载最佳 checkpoint
metric_for_best_model: loss    # 以验证 loss 为标准
greater_is_better: false       # loss 越低越好
```

### 如何判断过拟合

观察 `eval_loss` 曲线：

| 现象 | 判断 | 处理 |
|------|------|------|
| train_loss ↘, eval_loss ↘ | ✅ 正常 | 继续训练 |
| train_loss ↘, eval_loss ↗ | 🔴 过拟合 | 加大 dropout，降低 rank，或减少 epoch |
| train_loss →, eval_loss → | ⚠️ 收敛 | 可提前停止 |
| train_loss ↗ | ❌ 异常 | 降低 lr，检查数据 |

当前项目没有启用验证集，判断过拟合主要靠"eval 精度不涨反跌"。加上验证集可以让这个判断更精确。

### 为什么之前没启用

CIFAR 数据集本身有独立的 test set（10K 张图），当前做法是**训完直接测 test set**，相当于 test set 承担了验证和最终评估双重角色。这在实验初期可以接受（快速迭代），但正式调参时建议启用验证集——避免多次测 test set 导致**间接数据泄露**（你根据 test 精度调了参数，然后再测同一条 test set）。

---

## CIFAR10 vs CIFAR100 训练差异

两个数据集虽共用同一套训练 pipeline，但数据量和任务难度差异显著：

| 维度 | CIFAR10 | CIFAR100 (200/cls) | CIFAR100 (Full) |
|------|---------|--------------------|-----------------|
| 类别数 | 10 | 100 | 100 |
| 训练样本 | 2,000 | 20,000 | 50,000 |
| 当前 epochs | 3 | 5 | 5 |
| 总 training steps（batch=16） | ~375 | ~6,250 | ~15,625 |
| 单 epoch 时间（2B, RTX 5090） | ~2-5 min | ~20-50 min | ~50-120 min |
| 总训练时间（2B, RTX 5090） | **~10-15 min** | **~2-4 h** | **~4-10 h** |
| 过拟合风险 | 高（2K 样本） | 中（20K 样本） | 低（50K 样本） |
| 建议 rank | 8（2K 样本，高 rank 易过拟合） | 8-32（取决于模型） | 16-32（数据充足） |

**CIFAR10 的特殊性**：
- 仅 2000 张训练图，9B 高 rank 极容易过拟合
- epoch 数从 5 降到 3 是实践中收敛的结果——再多 epoch train loss 降到 0 但 test 精度不涨
- 如果启用验证集，CIFAR10 的 eval_steps 应设得较小（如 50）——总共才 375 steps，200 step 才测一次就看不到早期过拟合信号

**CIFAR100 的 epoch 数**：5 个 epoch 对于 200/cls（20K 样本）是合理的。但对于 full dataset（50K 样本、~15K steps），5 epoch 可能偏多——建议启用验证集观察是否在 3-4 epoch 就已收敛。

### 推理时间估算（vLLM）

| 模型 | 10K 张图耗时 | 吞吐量 | 备注 |
|------|-----------|--------|------|
| Qwen3.5-0.8B | ~12 min | ~14 im/s | |
| Qwen3.5-2B | ~17 min | ~10 im/s | |
| Qwen3.5-4B | ~18 min | ~9 im/s | 从 xx4.md 数据估算 |
| Qwen3.5-9B | ~25 min | ~7 im/s | 比训练快几个数量级 |

推理时间远小于训练时间（25 min vs 数小时），因此**瓶颈始终在训练侧**。优化训练参数（batch size、accum steps）的收益远大于优化推理。

## 参考

- 当前项目各方案对应的 config 文件：`configs/cifar10_lora_train.yaml`、`configs/cifar100_lora_train.yaml`
- [提示词设计策略](./prompt_strategy.md) — 训练/评估提示词的对齐原则
- [LLaMA Factory 训练文档](https://llamafactory.readthedocs.io/zh-cn/latest/getting_started/training.html)
- [LoRA/QLoRA 说明](./lora_qlora.md)
