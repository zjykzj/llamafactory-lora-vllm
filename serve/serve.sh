#!/bin/bash
# ============================================================
# serve.sh — vLLM model serving with merged or LoRA models
# ============================================================
#
# Usage:
#   bash serve/serve.sh [OPTIONS]
#
# Merged model (default):
#   bash serve/serve.sh
#   bash serve/serve.sh --model models/merged/cifar100
#
# LoRA adapter (no merge needed):
#   bash serve/serve.sh --lora models/lora/cifar10
#   bash serve/serve.sh --lora models/lora/cifar10 --base-model /path/to/Qwen3.5-2B
#
# Env vars: SERVE_HOST, SERVE_PORT, SERVE_MODEL, SERVE_ADAPTER,
#           SERVE_BASE_MODEL, SERVE_GPU_MEMORY, SERVE_API_KEY
# ============================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

HOST="${SERVE_HOST:-0.0.0.0}"
PORT="${SERVE_PORT:-8000}"
MODEL_PATH="${SERVE_MODEL:-models/merged/cifar10}"
ADAPTER_PATH="${SERVE_ADAPTER:-}"
BASE_MODEL="${SERVE_BASE_MODEL:-}"
GPU_MEMORY="${SERVE_GPU_MEMORY:-0.9}"
API_KEY="${SERVE_API_KEY:-not-needed}"
LORA_MODE=false

# ── Help ───────────────────────────────────────────────────

show_help() {
    cat << 'EOF'
Usage: bash serve/serve.sh [OPTIONS]

Start a vLLM OpenAI-compatible API server.

Options:
  --model PATH        Merged model directory (default: models/merged/cifar10)
  --lora PATH         LoRA adapter directory (no merge needed)
  --base-model PATH   Base model for LoRA mode
  --host ADDR         Bind address (default: 0.0.0.0, env: SERVE_HOST)
  --port PORT         Listen port (default: 8000, env: SERVE_PORT)
  --gpu-memory FLOAT  GPU memory utilization (default: 0.9, env: SERVE_GPU_MEMORY)
  --api-key KEY       API key for auth (default: not-needed, env: SERVE_API_KEY)
  --help, -h          Show this help message

Examples:
  # Merged model (default)
  bash serve/serve.sh
  bash serve/serve.sh --model models/merged/cifar100

  # LoRA adapter (no merge needed)
  bash serve/serve.sh --lora models/lora/cifar10
  bash serve/serve.sh --lora models/lora/cifar10 --base-model /path/to/Qwen3.5-2B

  # Custom host/port
  bash serve/serve.sh --port 8080
  SERVE_PORT=8080 bash serve/serve.sh
EOF
}

# ── Parse arguments ────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL_PATH="$2"; shift 2 ;;
        --lora)
            LORA_MODE=true
            ADAPTER_PATH="$2"; shift 2 ;;
        --base-model)
            BASE_MODEL="$2"; shift 2 ;;
        --host)
            HOST="$2"; shift 2 ;;
        --port)
            PORT="$2"; shift 2 ;;
        --gpu-memory)
            GPU_MEMORY="$2"; shift 2 ;;
        --api-key)
            API_KEY="$2"; shift 2 ;;
        --help|-h)
            show_help; exit 0 ;;
        *)
            echo "Unknown option: $1"
            echo "Run 'bash serve/serve.sh --help' for usage."
            exit 1 ;;
    esac
done

# ── Validation ─────────────────────────────────────────────

# Check vLLM is installed
if ! command -v vllm &>/dev/null; then
    echo "Error: vLLM is not installed."
    echo "Install with: pip install vllm"
    exit 1
fi

if $LORA_MODE; then
    # ── LoRA mode ───────────────────────────────────────────

    ADAPTER_PATH="$(cd "$PROJECT_ROOT" && realpath "$ADAPTER_PATH")"

    if [ ! -d "$ADAPTER_PATH" ]; then
        echo "Error: LoRA adapter directory not found: $ADAPTER_PATH"
        echo "Make sure LoRA training has completed and weights are saved."
        exit 1
    fi

    if [ ! -f "$ADAPTER_PATH/adapter_config.json" ]; then
        echo "Error: $ADAPTER_PATH is not a valid LoRA adapter (missing adapter_config.json)."
        exit 1
    fi

    # Derive adapter name from directory basename
    ADAPTER_NAME="$(basename "$ADAPTER_PATH")"

    if [ -z "$BASE_MODEL" ]; then
        # Try to read base_model_name_or_path from adapter_config.json
        BASE_MODEL=$(python3 -c "
import json, sys
try:
    with open('$ADAPTER_PATH/adapter_config.json') as f:
        cfg = json.load(f)
    print(cfg.get('base_model_name_or_path', ''))
except: pass
" 2>/dev/null || true)

        if [ -z "$BASE_MODEL" ]; then
            echo "Error: Could not determine base model from adapter config."
            echo "Please specify with: --base-model /path/to/Qwen3.5-2B"
            exit 1
        fi
        echo "Base model from adapter config: $BASE_MODEL"
    fi

    echo "=========================================="
    echo " vLLM Server (LoRA mode)"
    echo "=========================================="
    echo "  Base model:    $BASE_MODEL"
    echo "  LoRA adapter:  $ADAPTER_PATH"
    echo "  Adapter name:  $ADAPTER_NAME"
    echo "  Host:          $HOST"
    echo "  Port:          $PORT"
    echo "  GPU memory:    $GPU_MEMORY"
    echo "=========================================="

    vllm serve "$BASE_MODEL" \
        --host "$HOST" \
        --port "$PORT" \
        --trust-remote-code \
        --enable-lora \
        --lora-modules "$ADAPTER_NAME=$ADAPTER_PATH" \
        --gpu-memory-utilization "$GPU_MEMORY" \
        --api-key "$API_KEY"

else
    # ── Merged model mode (default) ─────────────────────────

    MODEL_PATH="$(cd "$PROJECT_ROOT" && realpath "$MODEL_PATH")"

    if [ ! -d "$MODEL_PATH" ]; then
        echo "Error: Merged model directory not found: $MODEL_PATH"
        echo ""
        echo "The LoRA adapter needs to be merged into the base model first:"
        echo "  llamafactory-cli export configs/cifar10_merge.yaml"
        echo ""
        echo "Or use LoRA mode (no merge needed):"
        echo "  bash serve/serve.sh --lora models/lora/cifar10"
        exit 1
    fi

    echo "=========================================="
    echo " vLLM Server (merged model)"
    echo "=========================================="
    echo "  Model:  $MODEL_PATH"
    echo "  Host:   $HOST"
    echo "  Port:   $PORT"
    echo "  GPU:    $GPU_MEMORY"
    echo "=========================================="

    vllm serve "$MODEL_PATH" \
        --host "$HOST" \
        --port "$PORT" \
        --trust-remote-code \
        --gpu-memory-utilization "$GPU_MEMORY" \
        --api-key "$API_KEY"
fi
