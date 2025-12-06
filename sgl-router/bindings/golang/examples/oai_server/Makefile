# Makefile for OAI Server
# Builds binary, runs tests, and creates distribution packages

# Configuration
APP_NAME = oai_server
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
BUILD_TIME := $(shell date -u '+%Y-%m-%d_%H:%M:%S')
GIT_COMMIT := $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Paths
ROOT_DIR := $(shell pwd)
BINDINGS_DIR := $(shell cd $(ROOT_DIR)/../.. && pwd)
PARENT_MAKEFILE := $(BINDINGS_DIR)/Makefile
BUILD_DIR := $(ROOT_DIR)/build
DIST_DIR := $(ROOT_DIR)/dist
BINARY := $(BUILD_DIR)/$(APP_NAME)

# Rust FFI library paths
LIB_DIR := $(BINDINGS_DIR)/lib
LIB_NAME = libsglang_router_rs

# Detect OS
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    LIB_EXT = .so
    LD_LIBRARY_PATH_VAR = LD_LIBRARY_PATH
    ARCH := $(shell uname -m)
    ifeq ($(ARCH),x86_64)
        GOARCH = amd64
    else ifeq ($(ARCH),aarch64)
        GOARCH = arm64
    endif
endif
ifeq ($(UNAME_S),Darwin)
    LIB_EXT = .dylib
    LD_LIBRARY_PATH_VAR = DYLD_LIBRARY_PATH
    ARCH := $(shell uname -m)
    ifeq ($(ARCH),x86_64)
        GOARCH = amd64
    else ifeq ($(ARCH),arm64)
        GOARCH = arm64
    endif
endif

# Build flags
LDFLAGS = -X main.Version=$(VERSION) -X main.BuildTime=$(BUILD_TIME) -X main.GitCommit=$(GIT_COMMIT)
GO_BUILD_FLAGS = -ldflags "$(LDFLAGS)"

# Python LDFLAGS (needed for Rust FFI that depends on Python)
PYTHON_LDFLAGS := $(shell python3-config --ldflags --embed 2>/dev/null || python3-config --ldflags 2>/dev/null || python-config --ldflags --embed 2>/dev/null || python-config --ldflags 2>/dev/null || echo "")

# CGO flags
# Note: -lsglang_router_rs and -ldl are already in the #cgo directive in internal/ffi/client.go
# We only need to add the library path (-L) and Python flags
CGO_LDFLAGS = -L$(LIB_DIR) $(PYTHON_LDFLAGS)

.PHONY: all build build-dev test benchmark benchmark-quick e2e e2e-quick clean dist dist-all help check-rust-lib check-server run-profile analyze-tpot

# E2E test configuration
E2E_HOST ?= localhost
E2E_PORT ?= 8080
E2E_MODEL ?= default
E2E_TOKENIZER ?= $(shell echo $$SGL_TOKENIZER_PATH || echo "./examples/tokenizer")
E2E_NUM_PROMPTS ?= 100
E2E_INPUT_LEN ?= 1024
E2E_OUTPUT_LEN ?= 512
E2E_REQUEST_RATE ?= 20
E2E_MAX_CONCURRENCY ?= 20
E2E_BASE_URL ?= http://$(E2E_HOST):$(E2E_PORT)

help:
	@echo "OAI Server Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  build           - Build binary (release mode)"
	@echo "  build-dev       - Build binary (debug mode)"
	@echo "  test            - Run tests"
	@echo "  benchmark       - Run performance benchmarks"
	@echo "  benchmark-quick - Run quick benchmarks (subset)"
	@echo "  e2e             - Run end-to-end test with bench_serving.py"
	@echo "  e2e-quick       - Run quick e2e test (smaller workload)"
	@echo "  clean           - Clean build artifacts"
	@echo "  dist             - Create distribution package for current OS/ARCH"
	@echo "  dist-all         - Create distribution packages for all platforms"
	@echo "  run              - Run the server (development)"
	@echo "  run-profile      - Run the server with pprof enabled"
	@echo "  analyze-tpot     - Analyze TPOT performance bottlenecks"
	@echo "  check-rust-lib   - Check if Rust FFI library exists"
	@echo "  check-server     - Check if OAI Server is running"
	@echo ""
	@echo "Environment variables:"
	@echo "  VERSION          - Version string (default: git describe or 'dev')"
	@echo "  BUILD_MODE       - Build mode: release or debug (default: release)"
	@echo ""
	@echo "E2E test variables:"
	@echo "  E2E_HOST          - OAI Server host (default: localhost)"
	@echo "  E2E_PORT          - OAI Server port (default: 8080)"
	@echo "  E2E_MODEL         - Model name (default: default)"
	@echo "  E2E_TOKENIZER     - Tokenizer path (optional, defaults to model tokenizer if not set)"
	@echo "  E2E_NUM_PROMPTS   - Number of prompts (default: 100)"
	@echo "  E2E_INPUT_LEN     - Input token length (default: 512)"
	@echo "  E2E_OUTPUT_LEN    - Output token length (default: 256)"
	@echo "  E2E_REQUEST_RATE - Request rate per second (default: 10, use 'inf' for max)"
	@echo "  E2E_MAX_CONCURRENCY - Max concurrent requests (default: 10)"

all: build

# Check if Rust FFI library exists
check-rust-lib:
	@if [ ! -f "$(LIB_DIR)/$(LIB_NAME)$(LIB_EXT)" ]; then \
		echo "Error: Rust FFI library not found at $(LIB_DIR)/$(LIB_NAME)$(LIB_EXT)"; \
		echo "Building Rust library..."; \
		cd $(BINDINGS_DIR) && $(MAKE) lib; \
	fi
	@echo "✓ Rust FFI library found"

# Build Rust FFI library if needed
rust-lib:
	@echo "Building Rust FFI library..."
	@cd $(BINDINGS_DIR) && $(MAKE) lib
	@echo "✓ Rust FFI library built"

# Build binary (release)
build: check-rust-lib
	@echo "Building $(APP_NAME) (release mode)..."
	@mkdir -p $(BUILD_DIR)
	@CGO_ENABLED=1 \
	 CGO_LDFLAGS="$(CGO_LDFLAGS)" \
	 GOOS=$(shell go env GOOS) \
	 GOARCH=$(GOARCH) \
	 go build $(GO_BUILD_FLAGS) -o $(BINARY) .
	@echo "✓ Binary built: $(BINARY)"
	@echo "  Version: $(VERSION)"
	@echo "  Build time: $(BUILD_TIME)"
	@echo "  Git commit: $(GIT_COMMIT)"

# Build binary (debug)
build-dev: check-rust-lib
	@echo "Building $(APP_NAME) (debug mode)..."
	@mkdir -p $(BUILD_DIR)
	@CGO_ENABLED=1 \
	 CGO_LDFLAGS="$(CGO_LDFLAGS)" \
	 go build -o $(BINARY) .
	@echo "✓ Binary built (debug): $(BINARY)"

# Run tests
test: check-rust-lib
	@echo "Running tests..."
	@CGO_ENABLED=1 \
	 CGO_LDFLAGS="$(CGO_LDFLAGS)" \
	 export $(LD_LIBRARY_PATH_VAR)="$(LIB_DIR):$$$(LD_LIBRARY_PATH_VAR)" && \
	 go test -v ./...
	@echo "✓ Tests completed"

# Run benchmarks
benchmark:
	@echo "Running performance benchmarks..."
	@cd benchmark && bash run_benchmarks.sh
	@echo "✓ Benchmarks completed"

# Run quick benchmarks
benchmark-quick:
	@echo "Running quick benchmarks..."
	@cd benchmark && bash run_benchmarks.sh --quick
	@echo "✓ Quick benchmarks completed"

# Check if OAI Server is running
check-server:
	@echo "Checking if OAI Server is running at $(E2E_BASE_URL)..."
	@if curl -s -f $(E2E_BASE_URL)/health > /dev/null 2>&1; then \
		echo "✓ OAI Server is running"; \
		exit 0; \
	else \
		echo "✗ OAI Server is not running at $(E2E_BASE_URL)"; \
		echo "  Start it with: make run"; \
		exit 1; \
	fi

# Find sglang project root (4 levels up from oai_server: oai_server -> examples -> golang -> bindings -> sgl-router -> sglang root)
SGLANG_ROOT := $(shell cd $(ROOT_DIR)/../../../../.. && pwd)

# Check if bench_serving.py is available
check-bench-script:
	@echo "Checking if bench_serving.py is available..."
	@if python -m sglang.bench_serving --help > /dev/null 2>&1; then \
		echo "✓ bench_serving.py is available"; \
	elif [ -f "$(SGLANG_ROOT)/python/sglang/bench_serving.py" ]; then \
		echo "✓ bench_serving.py found at $(SGLANG_ROOT)/python/sglang/bench_serving.py"; \
	else \
		echo "✗ bench_serving.py is not available"; \
		echo "  Install dependencies: pip install aiohttp numpy datasets transformers tqdm pillow pybase64"; \
		echo "  Or ensure you're in the sglang project root"; \
		exit 1; \
	fi

# Run end-to-end test with bench_serving.py
e2e: check-server
	@echo "Checking if bench_serving.py is available..."
	@if python -m sglang.bench_serving --help > /dev/null 2>&1; then \
		echo "✓ Using installed bench_serving.py module"; \
		USE_SGLANG_ROOT=false; \
	elif [ -f "$(SGLANG_ROOT)/python/sglang/bench_serving.py" ]; then \
		echo "✓ Using bench_serving.py from $(SGLANG_ROOT)"; \
		USE_SGLANG_ROOT=true; \
	else \
		echo "✗ bench_serving.py is not available"; \
		echo "  Install dependencies: pip install aiohttp numpy datasets transformers tqdm pillow pybase64"; \
		echo "  Or ensure you're in the sglang project root"; \
		exit 1; \
	fi
	@echo "Running end-to-end test with bench_serving.py..."
	@echo "Configuration:"
	@echo "  Server: $(E2E_BASE_URL)"
	@if [ "$(E2E_MODEL)" != "default" ]; then \
		echo "  Model: $(E2E_MODEL)"; \
	else \
		echo "  Model: (will be fetched from server)"; \
	fi
	@if [ -n "$(E2E_TOKENIZER)" ]; then \
		echo "  Tokenizer: $(E2E_TOKENIZER)"; \
	else \
		echo "  Tokenizer: (will be inferred from model)"; \
		echo "  ⚠ Warning: If server returns model='default', tokenizer loading may fail."; \
		echo "  💡 Tip: Set E2E_TOKENIZER to avoid this issue."; \
	fi
	@echo "  Prompts: $(E2E_NUM_PROMPTS)"
	@echo "  Input/Output: $(E2E_INPUT_LEN)/$(E2E_OUTPUT_LEN) tokens"
	@echo "  Request rate: $(E2E_REQUEST_RATE) req/s"
	@echo "  Max concurrency: $(E2E_MAX_CONCURRENCY)"
	@echo ""
	@if [ -z "$(E2E_TOKENIZER)" ] && [ "$(E2E_MODEL)" = "default" ]; then \
		echo "⚠ Warning: E2E_MODEL='default' and E2E_TOKENIZER is not set."; \
		echo "  The server may return model='default' which is not a valid HuggingFace model ID."; \
		echo "  Please set E2E_TOKENIZER to a valid tokenizer path."; \
		echo "  Example: make e2e E2E_TOKENIZER=/path/to/tokenizer"; \
		echo ""; \
	fi
	@TOKENIZER_ABS=$$(cd $(ROOT_DIR) && python3 -c "import os; path='$(E2E_TOKENIZER)'; print(os.path.abspath(path) if not os.path.isabs(path) else path)" 2>/dev/null || echo "$(E2E_TOKENIZER)"); \
	if [ -n "$(E2E_TOKENIZER)" ]; then \
		if [ -n "$$TOKENIZER_ABS" ] && ([ -d "$$TOKENIZER_ABS" ] || [ -f "$$TOKENIZER_ABS" ]); then \
			TOKENIZER_ARG="--tokenizer $$TOKENIZER_ABS"; \
		else \
			echo "⚠ Warning: Tokenizer path '$(E2E_TOKENIZER)' resolved to '$$TOKENIZER_ABS' but not found"; \
			echo "  Using as-is (may download from HuggingFace if it's a model name)"; \
			TOKENIZER_ARG="--tokenizer $(E2E_TOKENIZER)"; \
		fi; \
	else \
		TOKENIZER_ARG=""; \
	fi; \
	if [ "$$USE_SGLANG_ROOT" = "true" ]; then \
		cd $(SGLANG_ROOT) && PYTHONPATH=$(SGLANG_ROOT)/python:$$PYTHONPATH python python/sglang/bench_serving.py \
			--backend sglang-oai-chat \
			--base-url $(E2E_BASE_URL) \
			$$([ "$(E2E_MODEL)" != "default" ] && echo "--model $(E2E_MODEL)") \
			$$TOKENIZER_ARG \
			--dataset-name random \
			--num-prompts $(E2E_NUM_PROMPTS) \
			--random-input-len $(E2E_INPUT_LEN) \
			--random-output-len $(E2E_OUTPUT_LEN) \
			--request-rate $(E2E_REQUEST_RATE) \
			--max-concurrency $(E2E_MAX_CONCURRENCY) \
			--warmup-requests 5 \
			--disable-tqdm || (echo "✗ E2E test failed"; exit 1); \
	else \
		python -m sglang.bench_serving \
			--backend sglang-oai-chat \
			--base-url $(E2E_BASE_URL) \
			$$([ "$(E2E_MODEL)" != "default" ] && echo "--model $(E2E_MODEL)") \
			$$TOKENIZER_ARG \
			--dataset-name random \
			--num-prompts $(E2E_NUM_PROMPTS) \
			--random-input-len $(E2E_INPUT_LEN) \
			--random-output-len $(E2E_OUTPUT_LEN) \
			--request-rate $(E2E_REQUEST_RATE) \
			--max-concurrency $(E2E_MAX_CONCURRENCY) \
			--warmup-requests 5 \
			--disable-tqdm || (echo "✗ E2E test failed"; exit 1); \
	fi
	@echo ""
	@echo "✓ E2E test completed"

# Run quick end-to-end test (smaller workload)
e2e-quick: check-server
	@echo "Checking if bench_serving.py is available..."
	@if python -m sglang.bench_serving --help > /dev/null 2>&1; then \
		echo "✓ Using installed bench_serving.py module"; \
		USE_SGLANG_ROOT=false; \
	elif [ -f "$(SGLANG_ROOT)/python/sglang/bench_serving.py" ]; then \
		echo "✓ Using bench_serving.py from $(SGLANG_ROOT)"; \
		USE_SGLANG_ROOT=true; \
	else \
		echo "✗ bench_serving.py is not available"; \
		echo "  Install dependencies: pip install aiohttp numpy datasets transformers tqdm pillow pybase64"; \
		echo "  Or ensure you're in the sglang project root"; \
		exit 1; \
	fi
	@echo "Running quick end-to-end test..."
	@echo "Configuration:"
	@echo "  Server: $(E2E_BASE_URL)"
	@if [ "$(E2E_MODEL)" != "default" ]; then \
		echo "  Model: $(E2E_MODEL)"; \
	else \
		echo "  Model: (will be fetched from server)"; \
	fi
	@if [ -n "$(E2E_TOKENIZER)" ]; then \
		echo "  Tokenizer: $(E2E_TOKENIZER)"; \
	else \
		echo "  Tokenizer: (will be inferred from model)"; \
		echo "  ⚠ Warning: If server returns model='default', tokenizer loading may fail."; \
		echo "  💡 Tip: Set E2E_TOKENIZER to avoid this issue."; \
	fi
	@echo "  Prompts: 10"
	@echo "  Input/Output: 128/64 tokens"
	@echo ""
	@if [ -z "$(E2E_TOKENIZER)" ] && [ "$(E2E_MODEL)" = "default" ]; then \
		echo "⚠ Warning: E2E_MODEL='default' and E2E_TOKENIZER is not set."; \
		echo "  The server may return model='default' which is not a valid HuggingFace model ID."; \
		echo "  Please set E2E_TOKENIZER to a valid tokenizer path."; \
		echo "  Example: make e2e-quick E2E_TOKENIZER=/path/to/tokenizer"; \
		echo ""; \
	fi
	@if [ "$$USE_SGLANG_ROOT" = "true" ]; then \
		cd $(SGLANG_ROOT) && PYTHONPATH=$(SGLANG_ROOT)/python:$$PYTHONPATH python python/sglang/bench_serving.py \
			--backend sglang-oai-chat \
			--base-url $(E2E_BASE_URL) \
			$$([ "$(E2E_MODEL)" != "default" ] && echo "--model $(E2E_MODEL)") \
			$$([ -n "$(E2E_TOKENIZER)" ] && echo "--tokenizer $(E2E_TOKENIZER)") \
			--dataset-name random \
			--num-prompts 10 \
			--random-input-len 128 \
			--random-output-len 64 \
			--request-rate 5 \
			--max-concurrency 5 \
			--warmup-requests 2 \
			--disable-tqdm || (echo "✗ Quick E2E test failed"; exit 1); \
	else \
		python -m sglang.bench_serving \
			--backend sglang-oai-chat \
			--base-url $(E2E_BASE_URL) \
			$$([ "$(E2E_MODEL)" != "default" ] && echo "--model $(E2E_MODEL)") \
			$$([ -n "$(E2E_TOKENIZER)" ] && echo "--tokenizer $(E2E_TOKENIZER)") \
			--dataset-name random \
			--num-prompts 10 \
			--random-input-len 128 \
			--random-output-len 64 \
			--request-rate 5 \
			--max-concurrency 5 \
			--warmup-requests 2 \
			--disable-tqdm || (echo "✗ Quick E2E test failed"; exit 1); \
	fi
	@echo ""
	@echo "✓ Quick E2E test completed"

# Run the server (development)
run: build-dev
	@echo "Running server..."
	@export $(LD_LIBRARY_PATH_VAR)="$(LIB_DIR):$$$(LD_LIBRARY_PATH_VAR)" && \
	 $(BINARY)

# Run the server with pprof enabled
run-profile: build-dev
	@echo "Running server with pprof enabled..."
	@echo "pprof endpoint: http://localhost:6060/debug/pprof/"
	@export $(LD_LIBRARY_PATH_VAR)="$(LIB_DIR):$$$(LD_LIBRARY_PATH_VAR)" && \
	 export PPROF_ENABLED=true && \
	 export PPROF_PORT=6060 && \
	 $(BINARY)

# Analyze TPOT performance bottlenecks
# Usage: make analyze-tpot [DURATION=60] [REQUESTS=100] [CONCURRENCY=20] [PPROF_PORT=6060] [SERVER_URL=http://localhost:8080]
analyze-tpot:
	@echo "Running TPOT performance analysis..."
	@echo "Make sure the server is running with profiling enabled:"
	@echo "  make run-profile"
	@echo ""
	@bash scripts/analyze_tpot.sh \
		--duration $(if $(DURATION),$(DURATION),60) \
		--requests $(if $(REQUESTS),$(REQUESTS),100) \
		--concurrency $(if $(CONCURRENCY),$(CONCURRENCY),20) \
		--pprof-port $(if $(PPROF_PORT),$(PPROF_PORT),6060) \
		--server-url $(if $(SERVER_URL),$(SERVER_URL),http://localhost:8080)

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR)
	@rm -rf $(DIST_DIR)
	@echo "✓ Clean complete"

# Create distribution package for current platform
dist: build
	@echo "Creating distribution package..."
	@mkdir -p $(DIST_DIR)
	@PACKAGE_NAME="$(APP_NAME)-$(VERSION)-$(shell go env GOOS)-$(GOARCH)" && \
	 PACKAGE_DIR="$(DIST_DIR)/$$PACKAGE_NAME" && \
	 mkdir -p $$PACKAGE_DIR && \
	 cp $(BINARY) $$PACKAGE_DIR/$(APP_NAME) && \
	 cp $(LIB_DIR)/$(LIB_NAME)$(LIB_EXT) $$PACKAGE_DIR/ && \
	 cp README.md $$PACKAGE_DIR/ 2>/dev/null || true && \
	 echo "#!/bin/bash" > $$PACKAGE_DIR/run.sh && \
	 echo "export $(LD_LIBRARY_PATH_VAR)=\"$$(dirname \"$$0\"):$$$(LD_LIBRARY_PATH_VAR)\"" >> $$PACKAGE_DIR/run.sh && \
	 echo "./$(APP_NAME) \"$$@\"" >> $$PACKAGE_DIR/run.sh && \
	 chmod +x $$PACKAGE_DIR/run.sh && \
	 cd $(DIST_DIR) && \
	 if command -v zip >/dev/null 2>&1; then \
	   zip -r $$PACKAGE_NAME.zip $$PACKAGE_NAME; \
	   echo "✓ Package created: $(DIST_DIR)/$$PACKAGE_NAME.zip"; \
	 elif command -v tar >/dev/null 2>&1; then \
	   tar -czf $$PACKAGE_NAME.tar.gz $$PACKAGE_NAME; \
	   echo "✓ Package created: $(DIST_DIR)/$$PACKAGE_NAME.tar.gz"; \
	 else \
	   echo "✓ Package directory: $$PACKAGE_DIR"; \
	 fi
	@echo ""
	@echo "Package contents:"
	@ls -lh $(DIST_DIR)/$(APP_NAME)-$(VERSION)-$(shell go env GOOS)-$(GOARCH)* 2>/dev/null || true

# Create distribution packages for multiple platforms
dist-all: clean
	@echo "Building distribution packages for all platforms..."
	@echo ""
	@echo "Note: Cross-compilation requires setting up cross-compilation toolchains."
	@echo "This will build for: linux-amd64, linux-arm64, darwin-amd64, darwin-arm64"
	@echo ""
	@mkdir -p $(DIST_DIR)
	@for platform in linux-amd64 linux-arm64 darwin-amd64 darwin-arm64; do \
		OS=$$(echo $$platform | cut -d'-' -f1); \
		ARCH=$$(echo $$platform | cut -d'-' -f2); \
		echo "Building for $$OS-$$ARCH..."; \
		PACKAGE_NAME="$(APP_NAME)-$(VERSION)-$$OS-$$ARCH" && \
		PACKAGE_DIR="$(DIST_DIR)/$$PACKAGE_NAME" && \
		mkdir -p $$PACKAGE_DIR && \
		if [ "$$OS" = "darwin" ] && [ "$$ARCH" = "arm64" ] && [ "$(UNAME_S)" != "Darwin" ]; then \
			echo "  Skipping darwin-arm64 (requires macOS)"; \
			continue; \
		fi && \
		CGO_ENABLED=1 \
		CGO_LDFLAGS="$(CGO_LDFLAGS)" \
		GOOS=$$OS \
		GOARCH=$$ARCH \
		go build $(GO_BUILD_FLAGS) -o $$PACKAGE_DIR/$(APP_NAME) . && \
		if [ "$$OS" = "$(shell go env GOOS)" ] && [ "$$ARCH" = "$(GOARCH)" ]; then \
			cp $(LIB_DIR)/$(LIB_NAME)$(LIB_EXT) $$PACKAGE_DIR/; \
		else \
			echo "  Warning: Rust library not copied (cross-compilation)"; \
			echo "  You need to build Rust library for $$OS-$$ARCH separately"; \
		fi && \
		cp README.md $$PACKAGE_DIR/ 2>/dev/null || true && \
		echo "#!/bin/bash" > $$PACKAGE_DIR/run.sh && \
		echo "export $(LD_LIBRARY_PATH_VAR)=\"$$(dirname \"$$0\"):$$$(LD_LIBRARY_PATH_VAR)\"" >> $$PACKAGE_DIR/run.sh && \
		echo "./$(APP_NAME) \"$$@\"" >> $$PACKAGE_DIR/run.sh && \
		chmod +x $$PACKAGE_DIR/run.sh && \
		cd $(DIST_DIR) && \
		if command -v zip >/dev/null 2>&1; then \
			zip -r $$PACKAGE_NAME.zip $$PACKAGE_NAME >/dev/null; \
		elif command -v tar >/dev/null 2>&1; then \
			tar -czf $$PACKAGE_NAME.tar.gz $$PACKAGE_NAME >/dev/null; \
		fi && \
		echo "  ✓ Built: $$PACKAGE_NAME"; \
	done
	@echo ""
	@echo "✓ All packages created in $(DIST_DIR)"
	@ls -lh $(DIST_DIR)/*.zip $(DIST_DIR)/*.tar.gz 2>/dev/null | head -10 || true

# Install binary to system (optional)
install: build
	@echo "Installing $(APP_NAME) to /usr/local/bin..."
	@sudo cp $(BINARY) /usr/local/bin/$(APP_NAME)
	@echo "✓ Installed: /usr/local/bin/$(APP_NAME)"

# Uninstall binary from system
uninstall:
	@echo "Uninstalling $(APP_NAME)..."
	@sudo rm -f /usr/local/bin/$(APP_NAME)
	@echo "✓ Uninstalled"

