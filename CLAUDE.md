# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

LoRA fine-tuning pipeline for Qwen3.5-0.8B/2B (multimodal VLM) on CIFAR10/CIFAR100 classification, with vLLM inference deployment. Based on [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory).

## Pipeline

```
CIFAR Data → Build Instructions → LoRA Train → (Merge optional) → Serve → Evaluate
```

## Key commands

```bash
# Data prep
python data/download_cifar.py --dataset cifar10 --subset 200
python data/build_instructions.py --dataset cifar10

# Train & merge — use the wrapper script (switches models without editing YAML)
bash scripts/run.sh train --dataset cifar10 --model Qwen/Qwen3.5-0.8B
bash scripts/run.sh merge --dataset cifar10 --model Qwen/Qwen3.5-0.8B

# Or call llamafactory-cli directly (YAML defaults must match your model)
llamafactory-cli train configs/cifar10_lora_train.yaml
llamafactory-cli export configs/cifar10_merge.yaml

# Serve — three options
llamafactory-cli api configs/cifar10_infer.yaml          # LLaMA Factory API (debug, no concurrency)
bash serve/serve.sh                                      # vLLM + merged model (default)
bash serve/serve.sh --lora models/lora/cifar10_qwen3.5-0.8b  # vLLM + LoRA adapter (recommended)

# Eval (all backends expose OpenAI-compatible /v1/chat/completions)
python eval/eval_cifar10.py --max-samples 100 --model cifar10
python eval/eval_cifar100.py --max-samples 100 --model cifar100
python eval/bench.py --num-samples 100  # latency & throughput benchmark
```

## Config file conventions

All YAML configs live in `configs/`. Three config types per dataset:

| Pattern | Purpose | Key fields |
|---------|---------|------------|
| `*_lora_train.yaml` | LoRA SFT training | `finetuning_type: lora`, `dataset`, `output_dir`, `per_device_train_batch_size`, `gradient_accumulation_steps` |
| `*_infer.yaml` | LLaMA Factory API inference | `model_name_or_path`, `adapter_name_or_path`, `infer_backend` (huggingface/vllm/sglang) |
| `*_merge.yaml` | Export/merge LoRA into base model | `adapter_name_or_path`, `export_dir`, `export_device` |

`model_name_or_path` uses HuggingFace model IDs by default. For ModelScope users, set `export USE_MODELSCOPE_HUB=1`.

## Training performance tuning

Effective batch = `per_device_train_batch_size × gradient_accumulation_steps × num_gpus`. Maximize `per_device_train_batch_size` to fill VRAM and minimize `gradient_accumulation_steps` for throughput. For Qwen3.5-2B multimodal, vision encoder activations (controlled by `image_max_pixels`) are the main VRAM consumer. Detailed per-GPU configurations in `docs/training_config.md`.

## Key architecture decisions

- **All backends expose identical `/v1/chat/completions` API** — eval scripts are backend-agnostic, only `--model` changes between modes.
- **LLaMA Factory API** loads LoRA adapter in-process, single-request only, good for quick validation.
- **vLLM native** supports `--enable-lora --lora-modules name=path` to load LoRA adapters directly without merging. Multiple adapters can be loaded simultaneously, selected by the `model` field in each request.
- **vLLM merged** requires `llamafactory-cli export` first, use when adapters won't change.
- **ModelScope fallback**: when HuggingFace is unreachable, set `USE_MODELSCOPE_HUB=1` and point `model_name_or_path` to a local ModelScope-downloaded model directory.
- **Template `qwen3_5_nothink`** suppresses Qwen's reasoning tags — required for classification tasks.

## Docs

- `docs/lora_qlora.md` — LoRA/QLoRA theory and parameter guide
- `docs/multimodal_dataset.md` — ShareGPT-format multimodal data for LLaMA Factory
- `docs/vllm_deployment.md` — vLLM deployment options (LoRA and merged)
- `docs/training_config.md` — Training batch size / VRAM / multi-GPU configuration guide
- `serve/README.md` — Serve script usage for all deployment modes

## Project Skills — MANDATORY invocation rules

Some workflows are defined as project skills in `.claude/skills/`. These MUST be invoked before the corresponding action — do NOT perform the action directly.

| Trigger | Skill | Action |
|---------|-------|--------|
| User says "提交", "commit", "git commit" | **`/commit`** | Invoke `Skill("commit")` first, then follow its format |
| User says "发布", "release", "bump version" | `/release` | ⏭️ Not configured yet — skip |
| User says "测试", "lint", "typecheck" | `/dev` | ⏭️ No test/lint commands in this project |

**Rule: when user asks to commit, the first and only correct response is `Skill("commit")`. Never run `git commit` directly.**

Available skills without triggers (invoke on demand): `/claude` (CLAUDE.md guide), `/spec` (spec authoring).

### AI Model Configuration

```
{{AI_MODEL_NAME}} = DeepSeek-V4.0
{{AI_MODEL_EMAIL}} = noreply@deepseek.com
```
