#!/bin/bash

# Start SGLang Router HTTP Server
# Usage: ./scripts/start-router.sh [options]

# Default values
HOST="10.109.185.20"
PORT="8002"
WORKER_URLS=""
TOKENIZER_PATH=""
MODEL_PATH=""
POLICY="round_robin"
LOG_LEVEL="info"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --worker-urls)
            WORKER_URLS="$2"
            shift 2
            ;;
        --tokenizer-path)
            TOKENIZER_PATH="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --policy)
            POLICY="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --host HOST              Host to bind to (default: 10.109.185.20)"
            echo "  --port PORT              Port to bind to (default: 8002)"
            echo "  --worker-urls URLS       Comma-separated worker URLs (grpc://host:port)"
            echo "  --tokenizer-path PATH    Path to tokenizer.json"
            echo "  --model-path PATH        Path to model (HuggingFace ID or local path)"
            echo "  --policy POLICY          Load balancing policy (default: round_robin)"
            echo "  --log-level LEVEL        Log level (default: info)"
            echo "  -h, --help               Show this help message"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Build the router if not exists
ROUTER_BIN="$PROJECT_DIR/sglang-router"
if [ ! -f "$ROUTER_BIN" ]; then
    echo "Building router..."
    cd "$PROJECT_DIR"
    go build -o sglang-router ./cmd/router
    if [ $? -ne 0 ]; then
        echo "Failed to build router"
        exit 1
    fi
fi

# Build command
CMD="$ROUTER_BIN"
CMD="$CMD --host=$HOST"
CMD="$CMD --port=$PORT"

if [ -n "$WORKER_URLS" ]; then
    CMD="$CMD --worker-urls=$WORKER_URLS"
fi

if [ -n "$TOKENIZER_PATH" ]; then
    CMD="$CMD --tokenizer-path=$TOKENIZER_PATH"
fi

if [ -n "$MODEL_PATH" ]; then
    CMD="$CMD --model-path=$MODEL_PATH"
fi

CMD="$CMD --policy=$POLICY"
CMD="$CMD --log-level=$LOG_LEVEL"

# Print configuration
echo "=========================================="
echo "Starting SGLang Router"
echo "=========================================="
echo "Host: $HOST"
echo "Port: $PORT"
echo "Worker URLs: ${WORKER_URLS:-<none>}"
echo "Tokenizer Path: ${TOKENIZER_PATH:-<none>}"
echo "Model Path: ${MODEL_PATH:-<none>}"
echo "Policy: $POLICY"
echo "Log Level: $LOG_LEVEL"
echo "=========================================="
echo ""

# Start the router
exec $CMD

