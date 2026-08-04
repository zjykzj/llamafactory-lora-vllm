# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-08-04

### Changed

- **Training prompts optimized with research-backed strategy.** Three design principles: (1) include the class list in every variant — avoids forcing the model to memorize classes from other samples; (2) end every variant with "Answer with only the class name." — consistent output format constraint; (3) drop dataset-specific phrasing ("CIFAR-10", "CIFAR-100") — removes noise and improves generalization. Documented in `docs/prompt_strategy.md`. ([0dc2704](https://github.com/zjykzj/llamafactory-lora-vllm/commit/0dc2704))
- **Eval label matching now uses fuzzy normalization.** `normalize_label()` collapses underscores, hyphens, and consecutive whitespace into a single space before comparison. This ensures model outputs like `"pickup_truck"`, `"pickup truck"`, and `"Pickup Truck"` all match the ground truth `"pickup truck"`. The same normalization is applied to both predictions and labels. ([622b960](https://github.com/zjykzj/llamafactory-lora-vllm/commit/622b960))
- **Accuracy results updated across all configurations** — re-evaluated all models after prompt + normalization changes. Updated both CIFAR-10 and CIFAR-100 result tables in README. ([622b960](https://github.com/zjykzj/llamafactory-lora-vllm/commit/622b960))

### Added

- **Concurrent evaluation with `--workers` argument.** Extracted shared `evaluate_dataset()` into `eval/__init__.py` with `ThreadPoolExecutor` support. Both `eval_cifar10.py` and `eval_cifar100.py` now accept `--workers N` for multi-threaded inference, significantly improving evaluation throughput. ([4c89ab6](https://github.com/zjykzj/llamafactory-lora-vllm/commit/4c89ab6))

### Documentation

- `docs/prompt_strategy.md` — comprehensive guide on the research-backed prompt design strategy, covering three design principles, experimental ablation motivation, and template comparison. ([0dc2704](https://github.com/zjykzj/llamafactory-lora-vllm/commit/0dc2704))
- `docs/qwen3.5.md` — expanded with model architecture details and configuration notes. ([0dc2704](https://github.com/zjykzj/llamafactory-lora-vllm/commit/0dc2704))
- `docs/training_config.md` — expanded training configuration guidance. ([0dc2704](https://github.com/zjykzj/llamafactory-lora-vllm/commit/0dc2704))
- `docs/lora_qlora.md` — expanded LoRA/QLoRA theory. ([0dc2704](https://github.com/zjykzj/llamafactory-lora-vllm/commit/0dc2704))
- `docs/preprocessing.md` — image preprocessing pipeline documentation covering the two-layer resize strategy. ([5a9bbd4](https://github.com/zjykzj/llamafactory-lora-vllm/commit/5a9bbd4))
- README: restructured benchmark tables to show tuned vs default configs side-by-side. ([c8a2c5a](https://github.com/zjykzj/llamafactory-lora-vllm/commit/c8a2c5a))
- README: added CIFAR-10 full-dataset LoRA results. ([216adbb](https://github.com/zjykzj/llamafactory-lora-vllm/commit/216adbb))

### Fixed

- Restored CIFAR-100 class list that was accidentally removed during prompt refactoring. ([622b960](https://github.com/zjykzj/llamafactory-lora-vllm/commit/622b960))

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
