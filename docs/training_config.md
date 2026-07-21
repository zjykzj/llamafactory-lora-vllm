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

## 参考

- 当前项目各方案对应的 config 文件：`configs/cifar10_lora_train.yaml`、`configs/cifar100_lora_train.yaml`
- [LLaMA Factory 训练文档](https://llamafactory.readthedocs.io/zh-cn/latest/getting_started/training.html)
- [LoRA/QLoRA 说明](./lora_qlora.md)
