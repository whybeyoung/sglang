#!/bin/bash
# Start SGLang Router in gRPC mode
# gRPC mode requires tokenizer configuration

cd "$(dirname "$0")"

# Default values
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
WORKER_URLS="${WORKER_URLS:-grpc://10.109.185.20:8001}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/Users/yangyanbo/Downloads/qwen3coder/tokenizer.json}"
MODEL_PATH="${MODEL_PATH:-/Users/yangyanbo/Downloads/qwen3coder}"
POLICY="${POLICY:-round_robin}"
LOG_LEVEL="${LOG_LEVEL:-info}"

# Build command
# Use ./sglang-router if it exists, otherwise ./bin/sglang-router
if [ -f "./sglang-router" ]; then
    CMD="./sglang-router"
elif [ -f "./bin/sglang-router" ]; then
    CMD="./bin/sglang-router"
else
    echo "Error: sglang-router binary not found"
    echo "Please build it first: go build -o sglang-router ./cmd/router"
    exit 1
fi

CMD="$CMD --host=$HOST"
CMD="$CMD --port=$PORT"

if [ -n "$WORKER_URLS" ]; then
    CMD="$CMD --worker-urls=$WORKER_URLS"
fi

if [ -n "$TOKENIZER_PATH" ] && [ "$TOKENIZER_PATH" != "<none>" ]; then
    CMD="$CMD --tokenizer-path=$TOKENIZER_PATH"
fi

if [ -n "$MODEL_PATH" ] && [ "$MODEL_PATH" != "<none>" ]; then
    CMD="$CMD --model-path=$MODEL_PATH"
fi

CMD="$CMD --policy=$POLICY"
CMD="$CMD --log-level=$LOG_LEVEL"

# Print configuration
echo "=========================================="
echo "Starting SGLang Router (gRPC Mode)"
echo "=========================================="
echo "Host: $HOST"
echo "Port: $PORT"
echo "Worker URLs: ${WORKER_URLS:-<none>}"
echo "Tokenizer Path: ${TOKENIZER_PATH:-/Users/yangyanbo/Downloads/qwen3coder/tokenizer.json}"
echo "Model Path: ${MODEL_PATH:-<none>}"
echo "Policy: $POLICY"
echo "Log Level: $LOG_LEVEL"
echo "=========================================="
echo ""

# Start the router
exec $CMD

