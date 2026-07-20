# llamafactory-lora-vllm

Workflow for LoRA fine-tuning with LLamaFactory and high-performance inference deployment via vLLM.

Based on **Qwen3.5-2B** (multimodal vision-language model), fine-tuned on **CIFAR10 / CIFAR100** image classification tasks.

## Pipeline

```
CIFAR Data → Build Instructions → LoRA Train → Merge Export → vLLM Serve → Evaluate
   (PNG)         (ShareGPT JSON)    (llamafactory)   (export)     (OpenAI API)  (accuracy)
```

## Project Structure

```
├── configs/          # LLamaFactory YAML configs + dataset registry
├── data/             # CIFAR download & instruction building scripts
├── serve/            # vLLM serve startup
├── eval/             # Evaluation via vLLM OpenAI-compatible API
├── docs/             # LoRA/QLoRA, vLLM, multimodal format documentation
└── README.md
```

## Quick Start

### 1. llamafactory 安装

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory && pip install -e ".[torch,metrics]" && cd ..
```

### 2. 数据准备

```bash
# 下载 CIFAR10 图片（每类 200 张，快速实验）
python data/download_cifar.py --dataset cifar10 --subset 200

# 构建 ShareGPT 格式训练数据
python data/build_instructions.py --dataset cifar10
```

### 3. LoRA 训练

```bash
# 注册数据集
cp configs/dataset_info.json LLaMA-Factory/data/

# 使用 llamafactory-cli train 命令训练
llamafactory-cli train configs/cifar10_lora_train.yaml
```

### 4. 模型导出/合并

训练完成后，使用 `llamafactory-cli export` 命令将 LoRA adapter 合并到基座模型：

```bash
llamafactory-cli export -c configs/cifar10_merge.yaml
```

合并后的完整模型在 `models/merged/cifar10/` 目录。

### 5. vLLM 部署

```bash
pip install vllm

bash serve/serve.sh models/merged/cifar10
```

服务启动后提供 OpenAI 兼容 API：`http://localhost:8000/v1`

### 6. 评估

```bash
# CIFAR10 评估（100 张快速验证）
python eval/eval_cifar10.py --max-samples 100

# CIFAR100 评估
python eval/eval_cifar100.py --max-samples 200
```

## Hardware Requirements

| Stage | Min GPU Memory |
|-------|---------------|
| LoRA Train | ~8 GB |
| vLLM Serve | ~6 GB |

Qwen3.5-2B only 2B parameters, trainable on single consumer GPU (RTX 3060+).

## References

- [LLamaFactory](https://github.com/hiyouga/LLaMA-Factory)
- [vLLM](https://github.com/vllm-project/vllm)
- [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)
