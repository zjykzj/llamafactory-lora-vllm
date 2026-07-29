# llamafactory-lora-vllm

> Workflow for LoRA fine-tuning with LLamaFactory and high-performance inference deployment via vLLM.

Fine-tuned on **CIFAR10 / CIFAR100** image classification tasks. See [docs/](docs/README.md) for architecture theory, training guides, and deployment docs.

| Model | CIFAR10 Zero-shot | LoRA (200/cls) | LoRA (Full) | Notes |
|-------|-----------|----------------|-------------|-------|
| Qwen3.5-0.8B | 91.18% | 95.39% | 97.10% | default |
| Qwen3.5-2B | 96.47% | 97.58% | **98.55%** | default |
| Qwen3.5-4B | 96.44% | 97.76% | — | default |
| Qwen3.5-4B | 96.44% | 97.83% | — | tuned: rank=16, α=32, dropout=0.1, lr=1e-4 |
| Qwen3.5-9B | 96.12% | **98.05%** | — | default |
| Qwen3.5-9B | 96.12% | 97.93% | — | tuned: rank=32, α=64, dropout=0.1, lr=5e-5 |

| Model | CIFAR100 Zero-shot | LoRA (200/cls) | LoRA (Full) | Notes |
|-------|-----------|----------------|-------------|-------|
| Qwen3.5-0.8B | 52.71% | 82.07% | 84.01% | default |
| Qwen3.5-2B | 75.67% | 88.22% | **89.53%** | default |
| Qwen3.5-4B | 77.83% | 88.02% | — | default |
| Qwen3.5-4B | 77.83% | 87.81% | — | tuned: rank=16, α=32, dropout=0.1, lr=1e-4 |
| Qwen3.5-9B | 79.26% | **88.96%** | — | default |
| Qwen3.5-9B | 79.26% | 88.25% | — | tuned: rank=32, α=64, dropout=0.1, lr=5e-5 |

*All results evaluated on the full test set (10,000 images) via vLLM. Zero-shot refers to the base model without fine-tuning. Data sizes: 200/cls = 200 images per class (subset, CIFAR10: 2K / CIFAR100: 20K total), Full = all training images (CIFAR10: 5,000/cls 50K / CIFAR100: 500/cls 50K total).*

**Default config:** `lora_rank=8, lora_alpha=16, lora_dropout=0.05, lr=2e-4, batch_size=4×4`. CIFAR10 uses 3 epochs, CIFAR100 uses 5 epochs. See [configs/](configs/) for full YAML settings.

> CIFAR-10 SOTA Reference (Papers with Code)
>
> Top results from the [CIFAR-10 leaderboard](https://paperswithcode.com/sota/image-classification-on-cifar-10). Listed for context — our LoRA fine-tuned small LLMs trade some accuracy for general-purpose vision-language capability.

| Rank | Model | Accuracy | Paper | Year |
|------|-------|----------|-------|------|
| 1 | ViT-H/14 | 99.5% | [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) | 2020 |
| 1 | DINOv2 (ViT-g/14) | 99.5% | [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) | 2023 |
| 9 | EfficientNet-B7 | 98.9% | [EfficientNet: Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946) | 2019 |
| 15 | CN-CLIP | 96.0% | [Chinese CLIP: Contrastive Vision-Language Pretraining in Chinese](https://arxiv.org/abs/2211.01335) | 2022 |
| 19 | ResNet-110 | 93.6% | [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) | 2015 |

*Rankings as of 2026-07 by Papers with Code. All results use full CIFAR-10 training set (50,000 images).*

> CIFAR-100 SOTA Reference (Papers with Code)
>
> Top results from the [CIFAR-100 leaderboard](https://paperswithcode.com/sota/image-classification-on-cifar-100). CN-CLIP included as a vision-language baseline for comparison with our multimodal LLM approach.

| Rank | Model | Accuracy | Paper | Year |
|------|-------|----------|-------|------|
| 2 | ViT-H/14 | 94.5% | [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) | 2020 |
| 4 | DINOv2 (ViT-g/14) | 94.4% | [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) | 2023 |
| 10 | EfficientNet-B7 | 91.7% | [EfficientNet: Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946) | 2019 |
| 21 | CN-CLIP | 79.7% | [Chinese CLIP: Contrastive Vision-Language Pretraining in Chinese](https://arxiv.org/abs/2211.01335) | 2022 |

*Rankings as of 2026-07 by Papers with Code. All results use full CIFAR-100 training set (50,000 images).*

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
├── docs/             # Architecture theory, LoRA, training, deployment, tokenization
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
