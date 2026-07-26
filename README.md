# llamafactory-lora-vllm

> Workflow for LoRA fine-tuning with LLamaFactory and high-performance inference deployment via vLLM.

Fine-tuned on **CIFAR10 / CIFAR100** image classification tasks. See [docs/qwen3.5.md](docs/qwen3.5.md) for model configuration.

| Model | CIFAR10 Zero-shot | CIFAR10 LoRA | CIFAR100 Zero-shot | CIFAR100 LoRA (200/cls) | CIFAR100 LoRA (Full) |
|-------|-------------------|-------------|--------------------|------------------------|-----------------------|
| Qwen3.5-0.8B | 91.18% | 95.39% | 52.71% | 82.07% | 84.01% |
| Qwen3.5-2B | 96.47% | **97.58%** | 75.67% | 88.22% | **89.53%** |
| Qwen3.5-4B | 96.44% | — | 77.83% | — | — |

*All results evaluated on the full test set (10,000 images) via vLLM. Zero-shot refers to the base model without fine-tuning. CIFAR100 LoRA shows two training data sizes: 200 images per class (subset, 20K total) and the full training set (500 images per class, 50K total).*

## Pipeline

```
CIFAR Data → Build Instructions → LoRA Train ─┬─ LLaMA API ────────────→ Evaluate
   (PNG)         (ShareGPT JSON)               ├─ vLLM + LoRA (direct) → Evaluate
                                               └─ Merge → vLLM ────────→ Evaluate
```

## Project Structure

```
├── configs/          # LLamaFactory YAML configs + dataset registry
├── scripts/          # Train & merge wrapper (model-agnostic)
├── data/             # CIFAR download & instruction building scripts
├── serve/            # vLLM serve startup (LoRA & merged modes)
├── eval/             # Evaluation via OpenAI-compatible API
├── docs/             # Model config, LoRA/QLoRA, vLLM, training, multimodal format
├── CLAUDE.md         # AI assistant guidance
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

Use `scripts/run.sh` to train — specify the model with `--model` and the script handles paths automatically:

```bash
# Register dataset with LLaMA Factory
cp configs/dataset_info.json LLaMA-Factory/data/

# Train (output auto-saved to models/lora/cifar10_qwen3.5-0.8B)
bash scripts/run.sh train --dataset cifar10 --model Qwen/Qwen3.5-0.8B

# Try a different model by changing --model only
bash scripts/run.sh train --dataset cifar100 --model Qwen/Qwen3.5-2B

# Pass extra training args after --
bash scripts/run.sh train --dataset cifar10 --model Qwen/Qwen3.5-0.8B -- --num_train_epochs=5
```

> **Tip:** `scripts/run.sh` is a thin wrapper — it simply calls `llamafactory-cli` with `--model_name_or_path` and `--output_dir` overrides. You can still call `llamafactory-cli train configs/cifar10_lora_train.yaml` directly if you prefer editing YAML by hand.
>
> **ModelScope:** If HuggingFace Hub is unreachable, set `export USE_MODELSCOPE_HUB=1` and replace `model_name_or_path` with a local ModelScope-downloaded model directory.

### 4. Inference (two options)

#### Option A: LLaMA Factory API (LoRA direct, no merge needed, quick validation)

LLaMA Factory's `ChatModel` loads LoRA adapters directly for inference without merging.

```bash
# Start OpenAI-compatible API server (port 8000)
llamafactory-cli api configs/cifar10_infer.yaml
```

#### Option B: vLLM (high throughput, production)

Two modes — LoRA direct (recommended) or merged model.

```bash
# LoRA direct mode (no merge needed, recommended)
bash serve/serve.sh --lora models/lora/cifar10_qwen3.5-0.8B

# Or: merge first, then serve
bash scripts/run.sh merge --dataset cifar10 --model Qwen/Qwen3.5-0.8B
bash serve/serve.sh --model models/merged/cifar10_qwen3.5-0.8B
```

See `serve/README.md` and `docs/vllm_deployment.md` for details, including multi-adapter setup and ModelScope configuration.

### 5. Evaluation

All inference methods provide OpenAI-compatible APIs (`http://localhost:8000/v1`), so the same eval scripts work for all backends.

```bash
# CIFAR10 accuracy evaluation (100 samples for quick validation)
python eval/eval_cifar10.py --max-samples 100

# CIFAR100 accuracy evaluation
python eval/eval_cifar100.py --max-samples 200

# Benchmark latency & throughput
python eval/bench.py --num-samples 100
```

## Environment

Training and deployment use **separate Python environments** due to conflicting dependency requirements (LLaMA Factory vs vLLM).

### Training Environment

| Package | Version |
|---------|---------|
| Python | 3.12.13 |
| LLaMA Factory | 0.9.5 |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |
| Transformers | 5.6.0 |
| PEFT | 0.18.1 |
| Accelerate | 1.11.0 |

### Deployment Environment

| Package | Version |
|---------|---------|
| Python | 3.12.3 |
| vLLM | 0.20.0 |
| PyTorch | 2.11.0+cu130 |
| CUDA | 13.0 |
| Transformers | 5.10.2 |

> **Why separate environments?** LLaMA Factory and vLLM have incompatible dependency ranges (different CUDA toolkit versions and Transformers major versions). Keeping them isolated avoids version conflicts.

## References

- [LLamaFactory](https://github.com/hiyouga/LLaMA-Factory)
- [vLLM](https://github.com/vllm-project/vllm)
- [Qwen3.5](https://modelscope.cn/organization/qwen)

## License

This project is licensed under the [MIT License](LICENSE).
