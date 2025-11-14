#!/bin/bash

# Simple example runner
# Usage: ./run.sh [tokenizer_path] [endpoint]

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
DEFAULT_TOKENIZER_PATH="${SGL_TOKENIZER_PATH:-/Users/yangyanbo/projects/iflytek/code/opensource/pd/official/sglang/sgl-router/examples/sglang-go-client/examples/tokenizer}"
DEFAULT_ENDPOINT="${SGL_GRPC_ENDPOINT:-grpc://10.109.185.20:8001}"

TOKENIZER_PATH="${1:-${DEFAULT_TOKENIZER_PATH}}"
ENDPOINT="${2:-${DEFAULT_ENDPOINT}}"

echo "Running simple example..."
echo "Library path: ${LIB_DIR}"
echo "Tokenizer: $TOKENIZER_PATH"
echo "Endpoint: $ENDPOINT"
echo ""

cd "$SCRIPT_DIR"
SGL_TOKENIZER_PATH="$TOKENIZER_PATH" SGL_GRPC_ENDPOINT="$ENDPOINT" go run main.go

