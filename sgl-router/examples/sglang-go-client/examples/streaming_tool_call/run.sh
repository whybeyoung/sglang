#!/bin/bash

# Streaming tool call example runner
# Usage: ./run.sh [tokenizer_path] [endpoint] [parser_type]

# Set library path for Rust FFI library
CARGO_BUILD_DIR="${CARGO_BUILD_DIR:-/Users/yangyanbo/cargobuild}"
LIB_DIR="${CARGO_BUILD_DIR}/release"

# Get Python LDFLAGS (needed for Rust FFI that depends on Python)
PYTHON_LDFLAGS=$(python3-config --ldflags --embed 2>/dev/null || python3-config --ldflags 2>/dev/null || echo "")

# Set CGO_LDFLAGS to link with the Rust library
export CGO_LDFLAGS="-L${LIB_DIR} -lsglang_router_rs ${PYTHON_LDFLAGS} -ldl"

# macOS uses DYLD_LIBRARY_PATH, Linux uses LD_LIBRARY_PATH
if [[ "$OSTYPE" == "darwin"* ]]; then
    export DYLD_LIBRARY_PATH="${LIB_DIR}:${DYLD_LIBRARY_PATH}"
else
    export LD_LIBRARY_PATH="${LIB_DIR}:${LD_LIBRARY_PATH}"
fi

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default configuration (can be overridden by command line arguments or environment variables)
# Default tokenizer path: examples/tokenizer directory
DEFAULT_TOKENIZER_PATH="${SGL_TOKENIZER_PATH:-/Users/yangyanbo/projects/iflytek/code/opensource/pd/official/sglang/sgl-router/examples/sglang-go-client/examples/tokenizer}"
# Default endpoint
DEFAULT_ENDPOINT="${SGL_GRPC_ENDPOINT:-grpc://10.109.185.20:8001}"
# Default parser type
DEFAULT_PARSER_TYPE="qwen_coder"

TOKENIZER_PATH="${1:-${DEFAULT_TOKENIZER_PATH}}"
ENDPOINT="${2:-${DEFAULT_ENDPOINT}}"
PARSER_TYPE=qwen_coder

echo "Running streaming tool call example..."
echo "Library path: ${LIB_DIR}"
echo "Tokenizer: $TOKENIZER_PATH"
echo "Endpoint: $ENDPOINT"
echo "Parser: $PARSER_TYPE"
echo ""

go run main.go -tokenizer "$TOKENIZER_PATH" -endpoint "$ENDPOINT" -parser "$PARSER_TYPE"

