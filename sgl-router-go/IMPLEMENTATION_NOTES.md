# Implementation Notes

This document describes the Go implementation of sgl-router and notes differences from the Rust version.

## Architecture

The Go implementation follows the same pipeline-based architecture as the Rust version:

1. **Request Preparation**: Tokenizes inputs, processes messages, filters tools
2. **Worker Selection**: Selects appropriate worker(s) based on routing policy
3. **Client Acquisition**: Acquires gRPC client connections
4. **Request Building**: Builds gRPC request messages
5. **Dispatch Metadata**: Prepares dispatch metadata
6. **Request Execution**: Executes requests via gRPC streams
7. **Response Processing**: Processes and formats responses

## Completed Components

### Core Components
- ✅ Worker interface and BasicWorker implementation
- ✅ WorkerRegistry with indexing by model, type, URL
- ✅ Pipeline architecture with stage interface
- ✅ Basic pipeline stages (Preparation, WorkerSelection, ClientAcquisition, etc.)
- ✅ gRPC Router structure
- ✅ Configuration loading from flags
- ✅ Basic load balancing policies (Random, RoundRobin)

## TODO: Components Requiring Implementation

### 1. Tokenizer Integration

**Rust Version**: Uses `tokenizers` crate and custom implementations in `src/tokenizer/`

**Go Implementation**: 
- Option A: Use CGO bindings to Rust tokenizers (maintains compatibility)
- Option B: Use Go tokenizer libraries (e.g., `github.com/sugarme/tokenizer`)
- Option C: Implement tokenizer interface that calls external service

**Location**: `internal/router/pipeline/stages/preparation.go` - TODO comments marked

**Recommended Approach**: Start with Option B, add Option A for production if needed.

### 2. gRPC Client Pool

**Rust Version**: Uses `tonic` for gRPC clients with connection pooling

**Go Implementation**:
- Use `google.golang.org/grpc` with connection pooling
- Implement connection reuse in `internal/router/pipeline/stages/client_acquisition.go`
- Consider using `google.golang.org/grpc/connectivity` for connection state management

**Location**: `internal/router/pipeline/stages/client_acquisition.go`

### 3. Proto Request Building

**Rust Version**: Uses `prost` to build protobuf messages

**Go Implementation**:
- Generate Go code from proto files using `protoc`
- Implement request building in `internal/router/pipeline/stages/request_building.go`
- Reference: `proto/sglang_scheduler.proto`

**Steps**:
1. Run `make generate` to generate proto code
2. Implement `buildGenerateRequest()` function
3. Handle PD metadata injection

### 4. Request Execution Stage

**Rust Version**: `src/routers/grpc/stages/request_execution.rs`

**Go Implementation**:
- Implement streaming gRPC calls
- Handle both single and dual (PD) dispatch modes
- Manage stream cancellation and timeouts
- Store `ExecutionResult` in context

**Location**: `internal/router/pipeline/stages/request_execution.go` (to be created)

### 5. Response Processing Stage

**Rust Version**: `src/routers/grpc/responses/` and `src/routers/grpc/processing.rs`

**Go Implementation**:
- Implement streaming response processing
- Handle stop sequence decoding
- Process tool calls and reasoning (if applicable)
- Convert proto responses to OpenAI-compatible format

**Location**: `internal/router/pipeline/stages/response_processing.go` (to be created)

### 6. Load Balancing Policies

**Completed**: Random, RoundRobin

**TODO**: 
- CacheAware: Requires prefix tree implementation (Rust: `src/tree.rs`)
- PowerOfTwo: Requires load monitoring (Rust: `src/policies/power_of_two.rs`)

**Location**: `internal/policy/`

**Note**: CacheAware policy is complex - may need to adapt Rust's prefix tree implementation or use a simplified version.

### 7. Reliability Mechanisms

**Circuit Breaker**:
- Rust: `src/core/circuit_breaker.rs`
- Go: Implement similar state machine (closed/open/half-open)
- Location: `internal/reliability/circuit_breaker.go` (to be created)

**Retry**:
- Rust: `src/core/retry.rs`
- Go: Implement exponential backoff with jitter
- Location: `internal/reliability/retry.go` (to be created)

**Rate Limiting**:
- Rust: `src/core/token_bucket.rs`
- Go: Use `golang.org/x/time/rate` or implement token bucket
- Location: `internal/reliability/rate_limiter.go` (to be created)

### 8. HTTP Server

**Rust Version**: Uses `axum` for HTTP server

**Go Implementation**:
- Use `net/http` or framework like `gin` or `chi`
- Implement OpenAI-compatible endpoints:
  - `POST /v1/chat/completions`
  - `POST /generate`
  - `GET /health`
  - `GET /workers`
  - `POST /workers`
  - `DELETE /workers/{url}`

**Location**: `internal/server/http.go` (to be created)

### 9. Protocol Types

**Rust Version**: `src/protocols/`

**Go Implementation**:
- Define Go structs for:
  - `ChatCompletionRequest`
  - `ChatCompletionResponse`
  - `GenerateRequest`
  - `GenerateResponse`
- Use JSON tags for serialization

**Location**: `internal/protocols/` (to be created)

### 10. Tool Parser and Reasoning Parser

**Rust Version**: 
- Tool parser: `src/tool_parser/`
- Reasoning parser: `src/reasoning_parser/`

**Go Implementation**:
- These may be significantly simplified or require external libraries
- For tool parsing, JSON parsing may be sufficient
- For reasoning parsers (DeepSeek-R1, Qwen3, etc.), may need specialized implementations

**Location**: 
- `internal/parser/tool.go` (to be created)
- `internal/parser/reasoning.go` (to be created)

## Differences from Rust Version

### Memory Management
- Rust: Uses `Arc` for shared ownership
- Go: Uses pointers and sync primitives (mutexes, channels)

### Error Handling
- Rust: Uses `Result<T, E>` and `?` operator
- Go: Uses multiple return values `(result, error)`

### Async/Await
- Rust: Native async/await with `tokio`
- Go: Uses goroutines and channels, or `context.Context` for cancellation

### Type System
- Rust: Traits for interfaces, enum variants
- Go: Interfaces (duck typing), no enum variants (use constants or iota)

### Performance Considerations
- The Go version may have different performance characteristics
- Some optimizations from Rust (zero-copy, SIMD) may not be directly applicable
- Consider profiling and optimization after basic implementation

## Testing

Add unit tests for:
- Worker registry operations
- Pipeline stages
- Load balancing policies
- Reliability mechanisms

Integration tests should verify:
- End-to-end request processing
- Worker health checking
- Load balancing behavior

## Documentation

Update README.md with:
- Complete usage examples
- Configuration options
- API documentation
- Performance benchmarks (once available)

## Getting Started with Completion

To complete the implementation:

1. **Start with Protocol Types** (`internal/protocols/`): Define request/response types
2. **Implement gRPC Client** (`internal/grpc/`): Create client pool and connection management
3. **Complete Pipeline Stages**: Implement RequestExecution and ResponseProcessing
4. **Add HTTP Server**: Expose REST API endpoints
5. **Implement Reliability**: Add circuit breaker, retry, rate limiting
6. **Add Advanced Policies**: Implement CacheAware and PowerOfTwo
7. **Integrate Tokenizer**: Choose and integrate tokenizer solution
8. **Add Tests**: Write comprehensive tests

Each component should follow the patterns established in the Rust version, adapted to Go idioms and best practices.
