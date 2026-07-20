#!/bin/bash
# Start vLLM server with merged LoRA model

set -e

MODEL_PATH="${1:-models/merged/cifar10}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model directory '$MODEL_PATH' not found."
    echo "Run 'llamafactory-cli export' first to merge LoRA into the base model."
    exit 1
fi

echo "Starting vLLM server..."
echo "  Model: $MODEL_PATH"
echo "  Host:  $HOST"
echo "  Port:  $PORT"

vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --trust-remote-code \
    --api-key "not-needed"
