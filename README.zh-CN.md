# llamafactory-lora-vllm

> 🦙 基于 LLaMA-Factory 的 LoRA 微调 → 通过 vLLM 进行高性能推理部署。

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="Python 3.12+"></a>
  <a href="https://github.com/zjykzj/llamafactory-lora-vllm/releases"><img src="https://img.shields.io/github/v/release/zjykzj/llamafactory-lora-vllm" alt="Release"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
  <a href="https://deepwiki.com/zjykzj/llamafactory-lora-vllm"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"></a>
</p>

在 **CIFAR10 / CIFAR100** 图像分类任务上进行微调。架构理论、训练指南与部署文档见 [docs/](docs/README.md)。

| 模型 | CIFAR10 零样本 | LoRA (200/类) | LoRA (全量) | 备注 |
|------|-----------|----------------|-------------|------|
| Qwen3.5-0.8B | 90.72% | 95.51% | 97.13% | 默认配置 |
| Qwen3.5-2B | 96.26% | 97.67% | **98.51%** | 默认配置 |
| Qwen3.5-4B | 96.52% | 97.77% | — | 默认配置 |
| Qwen3.5-4B | 96.52% | 97.74% | — | 调参: rank=16, α=32, dropout=0.1, lr=1e-4 |
| Qwen3.5-9B | 95.83% | **98.01%** | — | 默认配置 |
| Qwen3.5-9B | 95.83% | 97.93% | — | 调参: rank=32, α=64, dropout=0.1, lr=5e-5 |

| 模型 | CIFAR100 零样本 | LoRA (200/类) | LoRA (全量) | 备注 |
|------|-----------|----------------|-------------|------|
| Qwen3.5-0.8B | 51.80% | 82.12% | 84.41% | 默认配置 |
| Qwen3.5-2B | 75.11% | 88.37% | **89.70%** | 默认配置 |
| Qwen3.5-4B | 77.77% | 88.18% | — | 默认配置 |
| Qwen3.5-4B | 77.77% | 88.05% | — | 调参: rank=16, α=32, dropout=0.1, lr=1e-4 |
| Qwen3.5-9B | 79.16% | **89.19%** | — | 默认配置 |
| Qwen3.5-9B | 79.16% | 88.38% | — | 调参: rank=32, α=64, dropout=0.1, lr=5e-5 |

*所有结果均通过 vLLM 在完整测试集（10,000 张图片）上评估。零样本指未微调的基座模型。数据规模：200/类 = 每类 200 张图片（子集，CIFAR10: 2K / CIFAR100: 20K 张），全量 = 全部训练图片（CIFAR10: 5,000/类共 50K / CIFAR100: 500/类共 50K 张）。*

**默认配置：** `lora_rank=8, lora_alpha=16, lora_dropout=0.05, lr=2e-4, batch_size=4×4`。CIFAR10 训练 3 个 epoch，CIFAR100 训练 5 个 epoch。完整 YAML 设置见 [configs/](configs/)。

> CIFAR-10 SOTA 参考（Papers with Code）
>
> 以下为 [CIFAR-10 排行榜](https://paperswithcode.com/sota/image-classification-on-cifar-10) 的前列结果，仅作背景参考——我们的 LoRA 微调小模型以少量精度换取通用视觉-语言能力。

| 排名 | 模型 | 准确率 | 论文 | 年份 |
|------|-------|----------|-------|------|
| 1 | ViT-H/14 | 99.5% | [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) | 2020 |
| 1 | DINOv2 (ViT-g/14) | 99.5% | [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) | 2023 |
| 9 | EfficientNet-B7 | 98.9% | [EfficientNet: Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946) | 2019 |
| 15 | CN-CLIP | 96.0% | [Chinese CLIP: Contrastive Vision-Language Pretraining in Chinese](https://arxiv.org/abs/2211.01335) | 2022 |
| 19 | ResNet-110 | 93.6% | [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) | 2015 |

*排名截至 2026-07，来源 Papers with Code。所有结果均使用完整 CIFAR-10 训练集（50,000 张图片）。*

> CIFAR-100 SOTA 参考（Papers with Code）
>
> 以下为 [CIFAR-100 排行榜](https://paperswithcode.com/sota/image-classification-on-cifar-100) 的前列结果。CN-CLIP 作为视觉-语言基线列入，便于与我们的多模态 LLM 方法对比。

| 排名 | 模型 | 准确率 | 论文 | 年份 |
|------|-------|----------|-------|------|
| 2 | ViT-H/14 | 94.5% | [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) | 2020 |
| 4 | DINOv2 (ViT-g/14) | 94.4% | [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) | 2023 |
| 10 | EfficientNet-B7 | 91.7% | [EfficientNet: Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946) | 2019 |
| 21 | CN-CLIP | 79.7% | [Chinese CLIP: Contrastive Vision-Language Pretraining in Chinese](https://arxiv.org/abs/2211.01335) | 2022 |

*排名截至 2026-07，来源 Papers with Code。所有结果均使用完整 CIFAR-100 训练集（50,000 张图片）。*

## 🚀 更新日志

### v0.2.0

- **优化训练提示词** — 在 `build_instructions.py` 中应用三项设计原则：所有变体都包含类别列表、统一以 "Answer with only the class name." 结尾、去除数据集特有的噪声。详见 [`docs/prompt_strategy.md`](docs/prompt_strategy.md)。
- **模糊标签匹配** — `normalize_label()` 处理下划线/连字符/空格变体，使 `"pickup_truck"` 能匹配 `"pickup truck"`。
- **并发评估** — `--workers N` 启用多线程推理，显著加速全量测试集评估。
- **全部结果重新评估** — 下方基准表反映 **v0.2.0 提示词**（重新训练的 LoRA 模型 + 全新的零样本运行）。

### v0.1.x

初始版本，使用基线提示词。在 v0.1.x 提示词下训练的 LoRA 模型与 v0.2.0 **结果不同**——两个版本因提示词变化而**不可直接比较**。

> 📎 **v0.1.x 结果保存在 [v0.1.1 tag](https://github.com/zjykzj/llamafactory-lora-vllm/releases/tag/v0.1.1)。** 如果你使用 v0.1.x 训练的 LoRA adapter，请参考该 tag 的 README 获取正确的准确率数字。

## 流水线

```
CIFAR Data → Build Instructions → LoRA Train ─┬─ LLaMA API ────────────→ Evaluate
   (PNG)         (ShareGPT JSON)               ├─ vLLM + LoRA (direct) → Evaluate
                                               └─ Merge → vLLM ────────→ Evaluate
```

## 项目结构

```
├── configs/          # LLaMA-Factory YAML 配置 + 数据集注册
├── scripts/          # 训练与合并包装脚本（与模型无关）
├── data/             # CIFAR 下载与指令数据构建脚本
├── serve/            # vLLM 服务启动（LoRA 与合并两种模式）
├── eval/             # 通过 OpenAI 兼容 API 评估
├── docs/             # 架构理论、LoRA、训练、部署、分词文档
├── CLAUDE.md         # AI 助手指南
└── README.md
```

## 快速开始

### 1. 安装 LLaMA Factory

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory && pip install -e ".[torch,metrics]" && cd ..
```

### 2. 数据准备

```bash
# 下载 CIFAR10 图片（每类 200 张，用于快速实验）
python data/download_cifar.py --dataset cifar10 --subset 200

# 构建 ShareGPT 格式指令数据
python data/build_instructions.py --dataset cifar10
```

### 3. LoRA 训练

使用 `scripts/run.sh` 训练——通过 `--model` 指定模型，脚本自动处理路径：

```bash
# 向 LLaMA Factory 注册数据集
cp configs/dataset_info.json LLaMA-Factory/data/

# 训练（输出自动保存至 models/lora/cifar10_qwen3.5-0.8B）
bash scripts/run.sh train --dataset cifar10 --model Qwen/Qwen3.5-0.8B

# 只需修改 --model 即可切换模型
bash scripts/run.sh train --dataset cifar100 --model Qwen/Qwen3.5-2B

# 在 -- 之后传入额外训练参数
bash scripts/run.sh train --dataset cifar10 --model Qwen/Qwen3.5-0.8B -- --num_train_epochs=5
```

> **提示：** `scripts/run.sh` 只是一个薄封装——它调用 `llamafactory-cli` 并覆盖 `--model_name_or_path` 与 `--output_dir`。你仍然可以直接运行 `llamafactory-cli train configs/cifar10_lora_train.yaml`，如果你更习惯手改 YAML 的话。
>
> **ModelScope：** 如果 HuggingFace Hub 不可达，设置 `export USE_MODELSCOPE_HUB=1` 并将 `model_name_or_path` 替换为本地 ModelScope 下载的模型目录。

### 4. 推理（两种方式）

#### 方式 A：LLaMA Factory API（直挂 LoRA，无需合并，快速验证）

LLaMA Factory 的 `ChatModel` 直接加载 LoRA adapter 进行推理，无需合并。

```bash
# 启动 OpenAI 兼容 API 服务（端口 8000）
llamafactory-cli api configs/cifar10_infer.yaml
```

#### 方式 B：vLLM（高吞吐，生产环境）

两种模式——直挂 LoRA（推荐）或合并后模型。

```bash
# LoRA 直挂模式（无需合并，推荐）
bash serve/serve.sh --lora models/lora/cifar10_qwen3.5-0.8B

# 或者：先合并，再启动服务
bash scripts/run.sh merge --dataset cifar10 --model Qwen/Qwen3.5-0.8B
bash serve/serve.sh --model models/merged/cifar10_qwen3.5-0.8B
```

详见 `serve/README.md` 与 `docs/vllm_deployment.md`，包括多 adapter 配置与 ModelScope 设置。

### 5. 评估

所有推理方式均提供 OpenAI 兼容 API（`http://localhost:8000/v1`），因此同一套评估脚本适用于所有后端。

```bash
# CIFAR10 准确率评估（100 个样本快速验证）
python eval/eval_cifar10.py --max-samples 100

# CIFAR100 准确率评估
python eval/eval_cifar100.py --max-samples 200

# 延迟与吞吐基准测试
python eval/bench.py --num-samples 100
```

## 环境

由于依赖冲突（LLaMA Factory 与 vLLM 依赖范围不兼容），训练与部署使用**独立的 Python 环境**。

### 训练环境

| 包 | 版本 |
|---------|---------|
| Python | 3.12.13 |
| LLaMA Factory | 0.9.5 |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |
| Transformers | 5.6.0 |
| PEFT | 0.18.1 |
| Accelerate | 1.11.0 |

### 部署环境

| 包 | 版本 |
|---------|---------|
| Python | 3.12.3 |
| vLLM | 0.20.0 |
| PyTorch | 2.11.0+cu130 |
| CUDA | 13.0 |
| Transformers | 5.10.2 |

> **为什么需要独立环境？** LLaMA Factory 与 vLLM 的依赖版本范围互不兼容（CUDA 工具链版本不同、Transformers 主版本不同）。隔离环境可以避免版本冲突。

## 参考资料

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [vLLM](https://github.com/vllm-project/vllm)
- [Qwen3.5](https://modelscope.cn/organization/qwen)

## 许可证

本项目基于 [MIT 许可证](LICENSE) 开源。
