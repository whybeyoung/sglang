# SGLang Router (Go Implementation)

> ⚠️ **重要提示：这是一个学习实验分支，仅用于学习目的**
> 
> 本项目是为了学习和理解 Rust 版本的 `sgl-router` 架构而创建的 Go 语言实现。这不是生产环境的实现，仅用于：
> - 学习 Rust router 的架构设计
> - 理解 gRPC/HTTP 路由器的实现原理
> - 实验不同的实现方式和技术栈
> - 教育和研究目的
> 
> **请勿在生产环境中使用此实现。** 生产环境请使用官方 Rust 版本的 `sgl-router`。

This is a Go implementation of the SGLang Router, a high-performance model routing control and data plane for large-scale LLM deployments. This implementation follows the architecture of the [Rust version](../sgl-router) and provides gRPC-based routing capabilities.

## Overview

- Unified control plane for registering, monitoring, and orchestrating workers
- Data plane that routes requests across gRPC backends
- High-performance gRPC pipeline with request processing stages
- Multiple load balancing strategies (random, round-robin, cache-aware, power-of-two)
- Reliability features: retries, circuit breakers, rate limiting
- Observable with structured logging and Prometheus metrics

## Architecture

The router uses a pipeline-based architecture similar to the Rust implementation:

1. **Preparation Stage**: Tokenizes inputs, processes messages, filters tools
2. **Worker Selection Stage**: Selects appropriate worker(s) based on routing policy
3. **Client Acquisition Stage**: Acquires gRPC client connections
4. **Request Building Stage**: Builds gRPC request messages
5. **Dispatch Metadata Stage**: Prepares dispatch metadata
6. **Request Execution Stage**: Executes requests via gRPC streams
7. **Response Processing Stage**: Processes and formats responses

## Features

- gRPC-based routing with streaming support
- Worker registry with health checking
- Load balancing policies with per-model overrides
- Circuit breakers and retry mechanisms
- Rate limiting with token bucket
- Prometheus metrics
- Structured logging with zap

## Building

```bash
# Generate protobuf code
make generate

# Build the router
make build

# Run tests
make test
```

## Running

```bash
./bin/sglang-router \
  --worker-urls grpc://worker1:31001 grpc://worker2:31002 \
  --tokenizer-path /path/to/tokenizer.json \
  --policy cache_aware
```

## Differences from Rust Implementation

Some features from the Rust implementation may not be fully implemented or may work differently:

1. **Tokenizer**: The Go version uses a simplified tokenizer interface. Full tokenizer implementations may require external libraries or CGO bindings to Rust tokenizers.
2. **Reasoning Parser**: Reasoning parsers for DeepSeek-R1, Qwen3, etc. may not be fully implemented. See code comments for details.
3. **Tool Parser**: Tool call parsing may be simplified compared to the Rust version.
4. **Performance**: While optimized, the Go version may have different performance characteristics than the Rust version.

See individual source files for detailed comments on implementation differences.

## License

See LICENSE file in the parent directory.
