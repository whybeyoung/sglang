#!/bin/bash
# Run script for function_call example with proper environment setup

set -e

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Default configuration (can be overridden by command line arguments or environment variables)
# Default tokenizer path: examples/tokenizer directory
DEFAULT_TOKENIZER_PATH="${SGL_TOKENIZER_PATH:-/Users/yangyanbo/projects/iflytek/code/opensource/pd/official/sglang/sgl-router/examples/sglang-go-client/examples/tokenizer}"
# Default endpoint
DEFAULT_ENDPOINT="${SGL_GRPC_ENDPOINT:-grpc://10.109.185.20:8001}"

# Set Rust library path
CARGO_BUILD_DIR="${CARGO_BUILD_DIR:-/Users/yangyanbo/cargobuild}"
RUST_LIB_PATH="$CARGO_BUILD_DIR/release"

# Check if library exists
if [ ! -f "$RUST_LIB_PATH/libsglang_router_rs.dylib" ] && [ ! -f "$RUST_LIB_PATH/libsglang_router_rs.so" ]; then
    echo "Error: Rust library not found at $RUST_LIB_PATH"
    echo "Please build the Rust library first:"
    echo "  cd $PROJECT_ROOT/../.."
    echo "  cargo build --release --target-dir $CARGO_BUILD_DIR"
    exit 1
fi

# Get Python LDFLAGS
PYTHON_LDFLAGS=$(python3-config --ldflags --embed 2>/dev/null || python3-config --ldflags 2>/dev/null || echo "")

# Set environment variables for CGO
export CGO_LDFLAGS="-L$RUST_LIB_PATH -lsglang_router_rs $PYTHON_LDFLAGS -ldl"
export DYLD_LIBRARY_PATH="$RUST_LIB_PATH:$DYLD_LIBRARY_PATH"  # macOS
export LD_LIBRARY_PATH="$RUST_LIB_PATH:$LD_LIBRARY_PATH"      # Linux

# Check if arguments are provided, otherwise use defaults
ARGS=("$@")
if [ ${#ARGS[@]} -eq 0 ]; then
    # No arguments provided, use defaults
    echo "Using default configuration:"
    echo "  Tokenizer: $DEFAULT_TOKENIZER_PATH"
    echo "  Endpoint: $DEFAULT_ENDPOINT"
    echo ""
    ARGS=("-tokenizer" "$DEFAULT_TOKENIZER_PATH" "-endpoint" "$DEFAULT_ENDPOINT")
elif [ ${#ARGS[@]} -eq 1 ] && [[ "$1" == "-tokenizer" || "$1" == "-endpoint" ]]; then
    # Only one flag provided, fill in the missing one
    if [[ "$1" == "-tokenizer" ]]; then
        ARGS=("-tokenizer" "${ARGS[1]:-$DEFAULT_TOKENIZER_PATH}" "-endpoint" "$DEFAULT_ENDPOINT")
    else
        ARGS=("-tokenizer" "$DEFAULT_TOKENIZER_PATH" "-endpoint" "${ARGS[1]:-$DEFAULT_ENDPOINT}")
    fi
fi

# Run the Go program
cd "$SCRIPT_DIR"
CGO_LDFLAGS="$CGO_LDFLAGS" go run main.go "${ARGS[@]}"

