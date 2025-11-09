.PHONY: build proto clean test run example deps

# Variables
# Support custom cargo build directory (set via CARGO_BUILD_DIR env var or use default)
CARGO_BUILD_DIR ?= /Users/yangyanbo/cargobuild
RUST_LIB_PATH := $(CARGO_BUILD_DIR)/release
PROTO_PATH := ../../src/proto
PROTO_OUT := proto
PYTHON_LDFLAGS := $(shell python3-config --ldflags --embed 2>/dev/null || python3-config --ldflags 2>/dev/null || echo "")

# Build the sglang-client
build: proto
	@echo "Building sglang-client..."
	@echo "Python LDFLAGS: $(PYTHON_LDFLAGS)"
	@CGO_LDFLAGS="-L$(RUST_LIB_PATH) -lsglang_router_rs $(PYTHON_LDFLAGS) -ldl" \
	go build -o bin/sglang-client ./cmd/sglang-client

# Generate protobuf code
proto:
	@echo "Generating protobuf code..."
	@mkdir -p $(PROTO_OUT)
	@export PATH="$$(go env GOPATH)/bin:$$PATH" && \
	protoc \
		--go_out=$(PROTO_OUT) \
		--go_opt=paths=source_relative \
		--go_opt=Msglang_scheduler.proto=github.com/sglang/sgl-router-go-client/proto \
		--go-grpc_out=$(PROTO_OUT) \
		--go-grpc_opt=paths=source_relative \
		--go-grpc_opt=Msglang_scheduler.proto=github.com/sglang/sgl-router-go-client/proto \
		--proto_path=$(PROTO_PATH) \
		$(PROTO_PATH)/sglang_scheduler.proto
	@echo "✓ Protobuf code generated"

# Install dependencies
deps:
	@echo "Installing dependencies..."
	@go mod download
	@go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
	@go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
	@echo "✓ Dependencies installed"

# Run tests
test:
	@echo "Running tests..."
	@go test ./...

# Build function_call example
build-function-call: proto
	@echo "Building function_call example..."
	@echo "Python LDFLAGS: $(PYTHON_LDFLAGS)"
	@CGO_LDFLAGS="-L$(RUST_LIB_PATH) -lsglang_router_rs $(PYTHON_LDFLAGS) -ldl" \
	go build -o bin/function_call ./examples/function_call

# Run sglang-client
run: build
	@echo "Running sglang-client..."
	@if [ -z "$(TOKENIZER)" ] || [ -z "$(ENDPOINT)" ]; then \
		echo "Usage: make run TOKENIZER=/path/to/tokenizer.json ENDPOINT=grpc://localhost:20000"; \
		echo "Or set environment variables: SGL_TOKENIZER_PATH and SGL_GRPC_ENDPOINT"; \
		exit 1; \
	fi
	@DYLD_LIBRARY_PATH="$(RUST_LIB_PATH):$$DYLD_LIBRARY_PATH" \
	LD_LIBRARY_PATH="$(RUST_LIB_PATH):$$LD_LIBRARY_PATH" \
	./bin/sglang-client -tokenizer "$(TOKENIZER)" -endpoint "$(ENDPOINT)"

# Run function_call example
run-function-call: build-function-call
	@echo "Running function_call example..."
	@if [ -z "$(TOKENIZER)" ] || [ -z "$(ENDPOINT)" ]; then \
		echo "Usage: make run-function-call TOKENIZER=/path/to/tokenizer ENDPOINT=grpc://localhost:20000"; \
		exit 1; \
	fi
	@DYLD_LIBRARY_PATH="$(RUST_LIB_PATH):$$DYLD_LIBRARY_PATH" \
	LD_LIBRARY_PATH="$(RUST_LIB_PATH):$$LD_LIBRARY_PATH" \
	./bin/function_call -tokenizer "$(TOKENIZER)" -endpoint "$(ENDPOINT)"

# Alias for backward compatibility
example: run

# Clean build artifacts
clean:
	@echo "Cleaning..."
	@rm -rf bin/
	@rm -rf $(PROTO_OUT)/
	@go clean
	@echo "✓ Cleaned"

# Format code
fmt:
	@echo "Formatting code..."
	@go fmt ./...
	@echo "✓ Formatted"

# Lint code
lint:
	@echo "Linting code..."
	@golangci-lint run ./...
	@echo "✓ Linted"

# Setup development environment
setup: deps proto
	@echo "✓ Development environment setup complete"

# Help
help:
	@echo "Available targets:"
	@echo "  build              - Build the sglang-client"
	@echo "  build-function-call - Build the function_call example"
	@echo "  proto              - Generate protobuf code"
	@echo "  deps               - Install dependencies"
	@echo "  test               - Run tests"
	@echo "  run                - Build and run sglang-client"
	@echo "  run-function-call  - Build and run function_call example"
	@echo "  example            - Alias for run (backward compatibility)"
	@echo "  clean              - Clean build artifacts"
	@echo "  fmt                - Format code"
	@echo "  lint               - Lint code"
	@echo "  setup              - Setup development environment"
	@echo "  help               - Show this help message"

