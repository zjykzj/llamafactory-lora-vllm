# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

> Workflow for LoRA fine-tuning with LLaMA Factory and high-performance inference deployment via vLLM.

Fine-tuned on CIFAR10/CIFAR100 image classification tasks. Supports Qwen3.5-0.8B/2B/4B. See [docs/qwen3.5.md](docs/qwen3.5.md) for model configuration.

Based on [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) and [vLLM](https://github.com/vllm-project/vllm). Models from [ModelScope](https://modelscope.cn/organization/qwen).

## Pipeline

```
CIFAR Data → Build Instructions → LoRA Train → (Merge optional) → Serve → Evaluate
```

## Key commands

```bash
# Data prep (subset: 200 images per class for quick experiments)
python data/download_cifar.py --dataset cifar10 --subset 200
python data/build_instructions.py --dataset cifar10

# Full dataset: omit --subset to download all images (CIFAR10: 5,000/cls, CIFAR100: 500/cls)
python data/download_cifar.py --dataset cifar10
python data/build_instructions.py --dataset cifar10

# Train & merge — use the wrapper script (switches models without editing YAML)
bash scripts/run.sh train --dataset cifar10 --model Qwen/Qwen3.5-0.8B
bash scripts/run.sh train --dataset cifar100 --model Qwen/Qwen3.5-2B
bash scripts/run.sh merge --dataset cifar10 --model Qwen/Qwen3.5-0.8B

# Full dataset training: increase per_device_train_batch_size to fill VRAM, reduce gradient_accumulation_steps
bash scripts/run.sh train --dataset cifar10 --model Qwen/Qwen3.5-0.8B -- per_device_train_batch_size=16 gradient_accumulation_steps=1
bash scripts/run.sh train --dataset cifar10 --model Qwen/Qwen3.5-2B -- per_device_train_batch_size=16 gradient_accumulation_steps=1

# Or call llamafactory-cli directly (YAML defaults must match your model)
llamafactory-cli train configs/cifar10_lora_train.yaml
llamafactory-cli export configs/cifar10_merge.yaml

# Serve — three options
llamafactory-cli api configs/cifar10_infer.yaml          # LLaMA Factory API (debug, no concurrency)
bash serve/serve.sh                                      # vLLM + merged model (default)
bash serve/serve.sh --lora models/lora/cifar10_qwen3.5-0.8B  # vLLM + LoRA adapter (recommended)

# Eval (all backends expose OpenAI-compatible /v1/chat/completions)
python eval/eval_cifar10.py --max-samples 100 --model cifar10
python eval/eval_cifar100.py --max-samples 100 --model cifar100

# Zero-shot baseline (base model without fine-tuning)
bash serve/serve.sh --model /root/.cache/modelscope/Qwen/Qwen3.5-2B
python eval/eval_cifar10.py --model Qwen3.5-2B

python eval/bench.py --num-samples 100  # latency & throughput benchmark
```

## Config file conventions

All YAML configs live in `configs/`. Three config types per dataset:

| Pattern | Purpose | Key fields |
|---------|---------|------------|
| `*_lora_train.yaml` | LoRA SFT training | `finetuning_type: lora`, `dataset`, `output_dir`, `per_device_train_batch_size`, `gradient_accumulation_steps` |
| `*_infer.yaml` | LLaMA Factory API inference | `model_name_or_path`, `adapter_name_or_path`, `infer_backend` (huggingface/vllm/sglang) |
| `*_merge.yaml` | Export/merge LoRA into base model | `adapter_name_or_path`, `export_dir`, `export_device` |

`model_name_or_path` uses ModelScope model IDs (e.g. `Qwen/Qwen3.5-2B`). For HuggingFace users, omit `USE_MODELSCOPE_HUB`.

## Training performance tuning

Effective batch = `per_device_train_batch_size × gradient_accumulation_steps × num_gpus`. Maximize `per_device_train_batch_size` to fill VRAM and minimize `gradient_accumulation_steps` for throughput. For Qwen3.5-2B multimodal, vision encoder activations (controlled by `image_max_pixels`) are the main VRAM consumer. Detailed per-GPU configurations in `docs/training_config.md`.

## Key architecture decisions

- **All backends expose identical `/v1/chat/completions` API** — eval scripts are backend-agnostic, only `--model` changes between modes.
- **LLaMA Factory API** loads LoRA adapter in-process, single-request only, good for quick validation.
- **vLLM native** supports `--enable-lora --lora-modules name=path` to load LoRA adapters directly without merging. Multiple adapters can be loaded simultaneously, selected by the `model` field in each request.
- **vLLM merged** requires `llamafactory-cli export` first, use when adapters won't change.
- **ModelScope default**: models are loaded from ModelScope by default (`USE_MODELSCOPE_HUB=1`). For HuggingFace, omit the env var.
- **Template `qwen3_5_nothink`** suppresses Qwen's reasoning tags — required for classification tasks. Eval scripts pass `enable_thinking: False` via `extra_body` to disable thinking on base models.

## Docs

- `docs/qwen3.5.md` — Qwen3.5 model configuration and architecture notes
- `docs/lora_qlora.md` — LoRA/QLoRA theory and parameter guide
- `docs/multimodal_dataset.md` — ShareGPT-format multimodal data for LLaMA Factory
- `docs/vllm_deployment.md` — vLLM deployment options (LoRA and merged)
- `docs/training_config.md` — Training batch size / VRAM / multi-GPU configuration guide
- `serve/README.md` — Serve script usage for all deployment modes

## Git Operations

Commit and release workflows follow the conventions below. When the `maestro` plugin is installed, its skills trigger automatically (`maestro:commit`, `maestro:release`) and execute these workflows; without it, apply them manually. This project has no tests or lint — skip those steps.

### Commit conventions

- Conventional Commits format: `<type>(<scope>): <subject>` — types: feat/fix/docs/build/test/refactor/style/perf/ci/chore; imperative subject, ≤50 chars.
- Every commit ends with a mandatory `Co-Authored-By: {{AI_MODEL_NAME}} <{{AI_MODEL_EMAIL}}>` line (values below).
- User-facing commits update `CHANGELOG.md` in the same commit, under `## [Unreleased]`, in the subsection matching the type: feat→`### Added`, fix→`### Fixed`, docs→`### Documentation`, chore→`### Changed`. Entry style: `- **Bold summary**: description.` Subsection order: Added → Changed → Fixed → Removed → Security → Documentation. Breaking changes must be marked `(breaking)`.
- Skip CHANGELOG for: `test`/`ci`/`build`/`style` commits, pure-internal `refactor`/`perf` commits, version bump commits, and commits that only touch `CHANGELOG.md`.

### Release conventions

1. Pick the bump from commits since the last tag: breaking change → MAJOR; any `feat` → MINOR; otherwise PATCH. Cross-check: empty `### Added` → PATCH.
2. Update `VERSION` to `X.Y.Z`; rename `## [Unreleased]` → `## [X.Y.Z] — YYYY-MM-DD` in `CHANGELOG.md`, then add a fresh empty `## [Unreleased]` on top.
3. Commit as `chore: bump version to X.Y.Z`, body = `[X.Y.Z]` section content (omitting `###` headers) + `Co-Authored-By` line.
4. Tag and push: `git tag -a vX.Y.Z -m "vX.Y.Z"`, then `git push && git push --tags`.
5. Create a GitHub Release for tag `vX.Y.Z`: title `vX.Y.Z: <one-line summary>`, body = the `[X.Y.Z]` CHANGELOG section + citation link to `{{REPO_URL}}/blob/main/CHANGELOG.md`.

### AI Model Configuration

```
{{AI_MODEL_NAME}} = DeepSeek-V4.0
{{AI_MODEL_EMAIL}} = noreply@deepseek.com
{{REPO_URL}} = https://github.com/zjykzj/llamafactory-lora-vllm
```
