.PHONY: build test clean fmt lint examples run-simple run-streaming help

# Variables
# Support custom cargo build directory (set via CARGO_BUILD_DIR env var or use default)
CARGO_BUILD_DIR ?= /Users/yangyanbo/cargobuild
RUST_LIB_PATH := $(CARGO_BUILD_DIR)/release
PYTHON_LDFLAGS := $(shell python3-config --ldflags --embed 2>/dev/null || python3-config --ldflags 2>/dev/null || echo "")

# Default configuration
DEFAULT_TOKENIZER_PATH ?= /Users/yangyanbo/projects/iflytek/code/opensource/pd/official/sglang/sgl-router/examples/sglang-go-client/examples/tokenizer
DEFAULT_ENDPOINT ?= grpc://10.109.185.20:8001

# Build the SDK library
build:
	@echo "Building SGLang Go gRPC SDK..."
	@echo "Python LDFLAGS: $(PYTHON_LDFLAGS)"
	@CGO_LDFLAGS="-L$(RUST_LIB_PATH) -lsglang_router_rs $(PYTHON_LDFLAGS) -ldl" \
	go build ./...

# Build examples
build-examples: build
	@echo "Building examples..."
	@CGO_LDFLAGS="-L$(RUST_LIB_PATH) -lsglang_router_rs $(PYTHON_LDFLAGS) -ldl" \
	go build -o examples/simple/simple ./examples/simple
	@CGO_LDFLAGS="-L$(RUST_LIB_PATH) -lsglang_router_rs $(PYTHON_LDFLAGS) -ldl" \
	go build -o examples/streaming/streaming ./examples/streaming
	@echo "✓ Examples built"

# Run tests
test:
	@echo "Running tests..."
	@CGO_LDFLAGS="-L$(RUST_LIB_PATH) -lsglang_router_rs $(PYTHON_LDFLAGS) -ldl" \
	go test ./...

# Run simple example
run-simple:
	@echo "Running simple example..."
	@if [ -z "$(TOKENIZER)" ]; then \
		TOKENIZER="$(DEFAULT_TOKENIZER_PATH)"; \
	fi
	@if [ -z "$(ENDPOINT)" ]; then \
		ENDPOINT="$(DEFAULT_ENDPOINT)"; \
	fi
	@cd examples/simple && \
	CARGO_BUILD_DIR="$(CARGO_BUILD_DIR)" \
	SGL_TOKENIZER_PATH="$(TOKENIZER)" \
	SGL_GRPC_ENDPOINT="$(ENDPOINT)" \
	bash run.sh

# Run streaming example
run-streaming:
	@echo "Running streaming example..."
	@if [ -z "$(TOKENIZER)" ]; then \
		TOKENIZER="$(DEFAULT_TOKENIZER_PATH)"; \
	fi
	@if [ -z "$(ENDPOINT)" ]; then \
		ENDPOINT="$(DEFAULT_ENDPOINT)"; \
	fi
	@cd examples/streaming && \
	CARGO_BUILD_DIR="$(CARGO_BUILD_DIR)" \
	SGL_TOKENIZER_PATH="$(TOKENIZER)" \
	SGL_GRPC_ENDPOINT="$(ENDPOINT)" \
	bash run.sh

# Run all examples
examples: run-simple run-streaming

# Format code
fmt:
	@echo "Formatting code..."
	@go fmt ./...
	@echo "✓ Formatted"

# Lint code (requires golangci-lint)
lint:
	@echo "Linting code..."
	@if command -v golangci-lint >/dev/null 2>&1; then \
		golangci-lint run ./...; \
	else \
		echo "golangci-lint not found, skipping lint"; \
	fi
	@echo "✓ Linted"

# Install dependencies
deps:
	@echo "Installing dependencies..."
	@go mod download
	@go mod tidy
	@echo "✓ Dependencies installed"

# Clean build artifacts
clean:
	@echo "Cleaning..."
	@rm -rf examples/simple/simple
	@rm -rf examples/streaming/streaming
	@go clean
	@echo "✓ Cleaned"

# Setup development environment
setup: deps
	@echo "✓ Development environment setup complete"

# Verify build (check if everything compiles)
verify: build
	@echo "✓ Build verification passed"

# Help
help:
	@echo "SGLang Go gRPC SDK - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  build          - Build the SDK library"
	@echo "  build-examples - Build all example binaries"
	@echo "  test           - Run tests"
	@echo "  run-simple     - Run the simple example"
	@echo "  run-streaming  - Run the streaming example"
	@echo "  examples       - Run all examples"
	@echo "  fmt            - Format code"
	@echo "  lint           - Lint code (requires golangci-lint)"
	@echo "  deps           - Install dependencies"
	@echo "  clean          - Clean build artifacts"
	@echo "  setup          - Setup development environment"
	@echo "  verify         - Verify build"
	@echo "  help           - Show this help message"
	@echo ""
	@echo "Environment variables:"
	@echo "  CARGO_BUILD_DIR      - Rust build directory (default: /Users/yangyanbo/cargobuild)"
	@echo "  TOKENIZER            - Tokenizer path (default: $(DEFAULT_TOKENIZER_PATH))"
	@echo "  ENDPOINT             - gRPC endpoint (default: $(DEFAULT_ENDPOINT))"
	@echo ""
	@echo "Examples:"
	@echo "  make run-simple TOKENIZER=/path/to/tokenizer ENDPOINT=grpc://localhost:8001"
	@echo "  make run-streaming"
	@echo "  make build-examples"

