# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] — 2026-07-27

### Added

- vLLM serve flags: `--max-model-len`, `--max-num-seqs`, `--tensor-parallel` for inference performance tuning.
- `CHANGELOG.md` tracking all project changes since inception.

### Documentation

- README: add Qwen3.5-4B LoRA and 9B base/LoRA accuracy results to the results table.
- `docs/lora_qlora.md`: add LoRA rank theory deep-dive and per-model tuning recommendations.
- `docs/architecture.md` — project architecture overview.
- `docs/loss.md` — loss function design and formulation.
- `docs/tokenization.md` — tokenization strategy and configuration.
- `docs/README.md` — documentation index.
- `docs/training_config.md`: expanded training configuration guidance.
- README: point docs link to `docs/README.md` index.

## [0.1.0] — 2026-07-26

### Added

- Initial project scaffolding: CIFAR data pipeline, LoRA fine-tuning with LLaMA Factory, and vLLM inference deployment.
- CIFAR10/CIFAR100 dataset download and ShareGPT-format instruction building scripts.
- LoRA SFT training configs for Qwen3.5-0.8B/2B multimodal models.
- Model merge/export configs to fuse LoRA adapters into base weights.
- LLaMA Factory native API inference support (single-request, debug use).
- vLLM inference deployment: LoRA adapter mode (recommended) and merged model mode, both exposing OpenAI-compatible `/v1/chat/completions` API.
- `scripts/run.sh` wrapper for model-agnostic training and merging (switch models via CLI flags without editing YAML).
- `eval/eval_cifar10.py` and `eval/eval_cifar100.py` evaluation scripts with zero-shot baseline support.
- `eval/bench.py` latency and throughput benchmarking tool.
- QLoRA training support with 4-bit BitsAndBytes quantization configs.
- `VERSION` file tracking the current release.

### Changed

- Aligned all training/inference configs with LLaMA Factory official examples.
- Refactored serve module: replaced single-backend script with unified `serve/serve.sh` supporting vLLM LoRA, vLLM merged, and base model modes.
- Normalized model size suffix from lowercase `b` to uppercase `B` (e.g., `0.8b` → `0.8B`).

### Fixed

- Data pipeline: corrected `dataset_info.json` tags and image paths from absolute to relative.
- Added `--served-model-name` for merged model deployments, aligning model naming between serve and eval scripts.
- Corrected `.gitignore` patterns (LF line endings).
- Fixed passthrough argument handling in `run.sh` for benchmark extra args.
- QLoRA adapters now saved to `models/qlora/` directory.

### Documentation

- README translated to English with environment stack, accuracy results table, and MIT License section.
- `CLAUDE.md` with project overview, key commands, architecture decisions, and mandatory skill invocation rules.
- `docs/qwen3.5.md` — comprehensive Qwen3.5 model architecture reference.
- `docs/lora_qlora.md` — LoRA/QLoRA theory and parameter guide.
- `docs/multimodal_dataset.md` — ShareGPT-format multimodal data specification for LLaMA Factory.
- `docs/vllm_deployment.md` — vLLM deployment guide covering LoRA and merged modes.
- `docs/training_config.md` — training batch size, VRAM, and multi-GPU configuration guide.
- `serve/README.md` — serve script usage for all deployment modes.
- README accuracy results: added CIFAR100 full-dataset LoRA results, refined table layout with best accuracy highlighting.
