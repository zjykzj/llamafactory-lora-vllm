# llamafactory-lora-vllm

Workflow for LoRA fine-tuning with LLamaFactory and high-performance inference deployment via vLLM.

Based on **Qwen3.5-2B** (multimodal vision-language model), fine-tuned on **CIFAR10 / CIFAR100** image classification tasks.

## Pipeline

```
CIFAR Data → Build Instructions → LoRA Train ─┬─ LLaMA API ──→ Evaluate
   (PNG)         (ShareGPT JSON)    (llamafactory) └─ Export → vLLM → Evaluate
```

## Project Structure

```
├── configs/          # LLamaFactory YAML configs + dataset registry
├── data/             # CIFAR download & instruction building scripts
├── serve/            # vLLM serve startup
├── eval/             # Evaluation via OpenAI-compatible API
├── docs/             # LoRA/QLoRA, vLLM, multimodal format documentation
└── README.md
```

## Quick Start

### 1. Install LLaMA Factory

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory && pip install -e ".[torch,metrics]" && cd ..
```

### 2. Data Preparation

```bash
# Download CIFAR10 images (200 per class for quick experiments)
python data/download_cifar.py --dataset cifar10 --subset 200

# Build ShareGPT-format instruction data
python data/build_instructions.py --dataset cifar10
```

### 3. LoRA Training

```bash
# Register dataset with LLaMA Factory
cp configs/dataset_info.json LLaMA-Factory/data/

# Train with llamafactory-cli
llamafactory-cli train configs/cifar10_lora_train.yaml
```

### 4. Inference (two options)

#### Option A: LLaMA Factory API (LoRA direct, no merge needed, quick validation)

LLaMA Factory's `ChatModel` loads LoRA adapters directly for inference without merging.

```bash
# Start OpenAI-compatible API server (port 8000)
llamafactory-cli api configs/cifar10_infer.yaml
```

#### Option B: vLLM (merge first, higher throughput, production)

Merge LoRA first, then deploy with vLLM for maximum inference throughput.

```bash
# Merge LoRA adapter into base model
llamafactory-cli export configs/cifar10_merge.yaml

# Start vLLM server
bash serve/serve.sh models/merged/cifar10
```

### 5. Evaluation

Both inference methods provide OpenAI-compatible APIs (`http://localhost:8000/v1`), so the same eval scripts work for both.

```bash
# CIFAR10 evaluation (100 samples for quick validation)
python eval/eval_cifar10.py --max-samples 100

# CIFAR100 evaluation
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
