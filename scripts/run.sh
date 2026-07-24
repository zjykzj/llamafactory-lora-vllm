#!/bin/bash
# ============================================================
# run.sh — thin wrapper around llamafactory-cli
# ============================================================
#
# Usage:
#   bash scripts/run.sh train --dataset cifar10 --model Qwen/Qwen3.5-0.8B
#   bash scripts/run.sh merge --dataset cifar10 --model Qwen/Qwen3.5-0.8B
#
# The script overrides model_name_or_path and output/export dirs
# based on --model, so you can switch models without editing YAML.
# All other params (lora_rank, batch_size, etc.) stay in the YAML.
#
# Extra args after -- are passed through to llamafactory-cli:
#   bash scripts/run.sh train --dataset cifar10 --model Qwen/Qwen3.5-0.8B -- --num_train_epochs 5
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Help ───────────────────────────────────────────────────

show_help() {
    cat << 'EOF'
Usage: bash scripts/run.sh <command> [OPTIONS] [-- extra args]

Commands:
  train     Run LoRA fine-tuning
  merge     Merge LoRA adapter into base model (export)

Common options:
  --dataset NAME    Dataset name: cifar10, cifar100 (required)
  --model ID        HuggingFace model ID or local path (required)

Examples:
  bash scripts/run.sh train --dataset cifar10 --model Qwen/Qwen3.5-0.8B
  bash scripts/run.sh train --dataset cifar100 --model Qwen/Qwen3.5-2B
  bash scripts/run.sh merge --dataset cifar10 --model Qwen/Qwen3.5-0.8B

  # Pass extra training args after --
  bash scripts/run.sh train --dataset cifar10 --model Qwen/Qwen3.5-0.8B -- --num_train_epochs 5

Auto-derived paths (no need to specify):
  train → output_dir:   models/lora/{dataset}_{model_short}
  merge → adapter:      models/lora/{dataset}_{model_short}
          export_dir:   models/merged/{dataset}_{model_short}

EOF
}

# ── Parse command ──────────────────────────────────────────

if [[ $# -eq 0 ]]; then
    show_help
    exit 0
fi

COMMAND="$1"; shift

case "$COMMAND" in
    train|merge) ;;
    --help|-h)
        show_help; exit 0 ;;
    *)
        echo "Error: unknown command '$COMMAND'"
        echo "Run 'bash scripts/run.sh --help' for usage."
        exit 1 ;;
esac

# ── Parse options ──────────────────────────────────────────

DATASET=""
MODEL=""
PASSTHRU_ARGS=()
PARSE_PASSTHRU=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            DATASET="$2"; shift 2 ;;
        --model)
            MODEL="$2"; shift 2 ;;
        --)
            PARSE_PASSTHRU=true; shift ;;
        --help|-h)
            show_help; exit 0 ;;
        *)
            if $PARSE_PASSTHRU; then
                PASSTHRU_ARGS+=("$1")
            else
                echo "Unknown option: $1"
                echo "Run 'bash scripts/run.sh --help' for usage."
                exit 1
            fi
            shift ;;
    esac
done

# ── Validation ─────────────────────────────────────────────

if [ -z "$DATASET" ]; then
    echo "Error: --dataset is required (cifar10 or cifar100)"
    exit 1
fi

if [ -z "$MODEL" ]; then
    echo "Error: --model is required (e.g. Qwen/Qwen3.5-0.8B)"
    exit 1
fi

# Check llamafactory-cli is available
if ! command -v llamafactory-cli &>/dev/null; then
    echo "Error: llamafactory-cli not found. Is LLaMA Factory installed?"
    exit 1
fi

# ── Derive paths from model name ───────────────────────────

# Extract short name: "Qwen/Qwen3.5-0.8B" → "qwen3.5-0.8b"
# For local paths: "/path/to/Qwen3.5-0.8B" → "qwen3.5-0.8b"
MODEL_SHORT=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')

OUTPUT_DIR="models/lora/${DATASET}_${MODEL_SHORT}"
EXPORT_DIR="models/merged/${DATASET}_${MODEL_SHORT}"

# ── Execute ────────────────────────────────────────────────

cd "$PROJECT_ROOT"

case "$COMMAND" in
    train)
        YAML="configs/${DATASET}_lora_train.yaml"
        if [ ! -f "$YAML" ]; then
            echo "Error: config not found: $YAML"
            exit 1
        fi

        echo "=========================================="
        echo " LoRA Training"
        echo "=========================================="
        echo "  Config:    $YAML"
        echo "  Model:     $MODEL"
        echo "  Output:    $OUTPUT_DIR"
        echo "=========================================="

        llamafactory-cli train "$YAML" \
            --model_name_or_path "$MODEL" \
            --output_dir "$OUTPUT_DIR" \
            "${PASSTHRU_ARGS[@]}"
        ;;

    merge)
        YAML="configs/${DATASET}_merge.yaml"
        if [ ! -f "$YAML" ]; then
            echo "Error: config not found: $YAML"
            exit 1
        fi

        ADAPTER_DIR="$OUTPUT_DIR"

        if [ ! -d "$ADAPTER_DIR" ]; then
            echo "Error: LoRA adapter not found: $ADAPTER_DIR"
            echo "Run training first: bash scripts/run.sh train --dataset $DATASET --model $MODEL"
            exit 1
        fi

        echo "=========================================="
        echo " Merge LoRA → Merged Model"
        echo "=========================================="
        echo "  Config:    $YAML"
        echo "  Model:     $MODEL"
        echo "  Adapter:   $ADAPTER_DIR"
        echo "  Export:    $EXPORT_DIR"
        echo "=========================================="

        llamafactory-cli export "$YAML" \
            --model_name_or_path "$MODEL" \
            --adapter_name_or_path "$ADAPTER_DIR" \
            --export_dir "$EXPORT_DIR" \
            "${PASSTHRU_ARGS[@]}"
        ;;
esac
