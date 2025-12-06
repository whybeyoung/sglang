#!/bin/bash

# OpenAI-compatible server runner
# Usage: ./run.sh [tokenizer_path] [endpoint] [port] [--profile] [--pprof-port PORT] [--grpc] [--no-ffi-postprocess] [--no-ffi-preprocess]
#
# Options:
#   --profile          Enable pprof profiling (default port: 6060)
#   --pprof-port PORT  Set pprof port (default: 6060, requires --profile)
#   --grpc             Enable optimized gRPC client mode (reduces FFI overhead by 90%+)
#   --no-ffi-postprocess  Disable FFI postprocessing for performance testing (skips token decoding)
#   --no-ffi-preprocess   Disable FFI preprocessing for performance testing (skips chat_template and tokenization)

# Set library path for Rust FFI library
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINDINGS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LIB_DIR="${BINDINGS_DIR}/lib"

if [ ! -d "$LIB_DIR" ]; then
    echo "Error: Library directory not found at $LIB_DIR"
    echo "Please run 'make lib' first to build and export the library"
    exit 1
fi

# Get Python LDFLAGS (needed for Rust FFI that depends on Python)
PYTHON_LDFLAGS=$(python3-config --ldflags --embed 2>/dev/null || python3-config --ldflags 2>/dev/null || echo "")

# Set CGO_LDFLAGS to link with the Rust library
# Note: -lsglang_router_rs and -ldl are already in the #cgo directive in internal/ffi/client.go
# We only need to add the library path (-L) and Python flags
export CGO_LDFLAGS="-L${LIB_DIR} ${PYTHON_LDFLAGS}"

# macOS uses DYLD_LIBRARY_PATH, Linux uses LD_LIBRARY_PATH
if [[ "$OSTYPE" == "darwin"* ]]; then
    export DYLD_LIBRARY_PATH="${LIB_DIR}:${DYLD_LIBRARY_PATH}"
else
    export LD_LIBRARY_PATH="${LIB_DIR}:${LD_LIBRARY_PATH}"
fi

# Parse arguments
ENABLE_PROFILE=false
PPROF_PORT="6060"
USE_GRPC_CLIENT=false
DISABLE_FFI_POSTPROCESS=false
DISABLE_FFI_PREPROCESS=false
TOKENIZER_PATH=""
ENDPOINT=""
PORT=""

while [[ $# -gt 0 ]]; do
	case $1 in
		--profile)
			ENABLE_PROFILE=true
			shift
			;;
		--pprof-port)
			ENABLE_PROFILE=true
			PPROF_PORT="$2"
			shift 2
			;;
		--grpc)
			USE_GRPC_CLIENT=true
			shift
			;;
		--no-ffi-postprocess)
			DISABLE_FFI_POSTPROCESS=true
			shift
			;;
		--no-ffi-preprocess)
			DISABLE_FFI_PREPROCESS=true
			shift
			;;
		*)
			if [[ -z "$TOKENIZER_PATH" ]]; then
				TOKENIZER_PATH="$1"
			elif [[ -z "$ENDPOINT" ]]; then
				ENDPOINT="$1"
			elif [[ -z "$PORT" ]]; then
				PORT="$1"
			fi
			shift
			;;
	esac
done

# Default configuration
DEFAULT_TOKENIZER_PATH="${SGL_TOKENIZER_PATH:-../tokenizer}"
DEFAULT_ENDPOINT="${SGL_GRPC_ENDPOINT:-grpc://localhost:20000}"
DEFAULT_PORT="${PORT:-8080}"

TOKENIZER_PATH="${TOKENIZER_PATH:-${DEFAULT_TOKENIZER_PATH}}"
ENDPOINT="${ENDPOINT:-${DEFAULT_ENDPOINT}}"
PORT="${PORT:-${DEFAULT_PORT}}"

echo "Running OpenAI-compatible server..."
echo "Library path: ${LIB_DIR}"
echo "Tokenizer: $TOKENIZER_PATH"
echo "Endpoint: $ENDPOINT"
echo "Port: $PORT"
if [[ "$USE_GRPC_CLIENT" == "true" ]]; then
	echo "Client Mode: gRPC (optimized, reduces FFI overhead by 90%+)"
	export USE_GRPC_CLIENT=true
else
	echo "Client Mode: FFI (default, backward compatible)"
	export USE_GRPC_CLIENT=false
fi

if [[ "$DISABLE_FFI_POSTPROCESS" == "true" ]]; then
	echo "FFI Postprocessing: DISABLED (performance testing mode - token decoding skipped)"
	export DISABLE_FFI_POSTPROCESS=true
else
	echo "FFI Postprocessing: ENABLED (normal mode)"
	export DISABLE_FFI_POSTPROCESS=false
fi

if [[ "$DISABLE_FFI_PREPROCESS" == "true" ]]; then
	echo "FFI Preprocessing: DISABLED (performance testing mode - chat_template and tokenization skipped)"
	export DISABLE_FFI_PREPROCESS=true
else
	echo "FFI Preprocessing: ENABLED (normal mode)"
	export DISABLE_FFI_PREPROCESS=false
fi
if [[ "$ENABLE_PROFILE" == "true" ]]; then
	echo "Profiling: enabled (port: $PPROF_PORT)"
	echo "  pprof endpoint: http://localhost:$PPROF_PORT/debug/pprof/"
	export PPROF_ENABLED=true
	export PPROF_PORT="$PPROF_PORT"
else
	echo "Profiling: disabled"
fi
echo ""

cd "$(dirname "${BASH_SOURCE[0]}")"
SGL_TOKENIZER_PATH="$TOKENIZER_PATH" SGL_GRPC_ENDPOINT="$ENDPOINT" PORT="$PORT" USE_GRPC_CLIENT="$USE_GRPC_CLIENT" DISABLE_FFI_POSTPROCESS="$DISABLE_FFI_POSTPROCESS" DISABLE_FFI_PREPROCESS="$DISABLE_FFI_PREPROCESS" go run main.go
