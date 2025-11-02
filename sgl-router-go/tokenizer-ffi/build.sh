#!/bin/bash
# Build script for Rust tokenizer FFI library

set -e

cd "$(dirname "$0")"

echo "Building Rust tokenizer FFI library..."

# Build the library
cargo build --release

# Generate C header file
if command -v cbindgen &> /dev/null; then
    echo "Generating C header file..."
    cbindgen --config cbindgen.toml --crate tokenizer-ffi --output tokenizer_ffi.h
    echo "✅ Header file generated: tokenizer_ffi.h"
else
    echo "⚠️  cbindgen not found, skipping header generation"
    echo "Install it with: cargo install cbindgen"
fi

# Copy library to appropriate location
if [[ "$OSTYPE" == "darwin"* ]]; then
    LIB_NAME="libtokenizer_ffi.dylib"
    TARGET="target/release/libtokenizer_ffi.dylib"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    LIB_NAME="libtokenizer_ffi.so"
    TARGET="target/release/libtokenizer_ffi.so"
else
    echo "Unsupported OS: $OSTYPE"
    exit 1
fi

if [ -f "$TARGET" ]; then
    echo "✅ Library built: $TARGET"
    echo "Copy it to your Go project directory or add to library path"
else
    echo "❌ Library not found: $TARGET"
    exit 1
fi

