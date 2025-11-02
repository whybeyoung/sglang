.PHONY: build clean header

build:
	@echo "Building Rust tokenizer FFI library..."
	cargo build --release

header:
	@if command -v cbindgen >/dev/null 2>&1; then \
		echo "Generating C header file..."; \
		cbindgen --config cbindgen.toml --crate tokenizer-ffi --output tokenizer_ffi.h; \
		echo "✅ Header file generated: tokenizer_ffi.h"; \
	else \
		echo "⚠️  cbindgen not found, skipping header generation"; \
		echo "Install it with: cargo install cbindgen"; \
	fi

clean:
	cargo clean

all: build header

