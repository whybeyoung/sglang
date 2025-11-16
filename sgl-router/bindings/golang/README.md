# SGLang Go gRPC SDK

A high-level Go SDK for interacting with SGLang gRPC API, designed with an OpenAI-style API for familiarity and ease of use.

## Location

This SDK is located at `sgl-router/bindings/golang/` and provides Go bindings for the SGLang router FFI.

## Features

- **OpenAI-style API**: Familiar interface similar to OpenAI Go SDK
- **Streaming Support**: Real-time streaming chat completions
- **Non-streaming Support**: Simple request/response API
- **Tool Calling**: Support for function calling and tool use
- **Type-safe**: Full Go type definitions for requests and responses

## Installation

```bash
go get github.com/sglang/sglang-go-grpc-sdk
```

## Quick Start

### Basic Usage

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/sglang/sglang-go-grpc-sdk"
)

func main() {
    // Create client
    client, err := sglang.NewClient(sglang.ClientConfig{
        Endpoint:      "grpc://localhost:20000",
        TokenizerPath: "/path/to/tokenizer",
    })
    if err != nil {
        log.Fatal(err)
    }
    defer client.Close()

    // Create completion
    req := sglang.ChatCompletionRequest{
        Model: "default",
        Messages: []sglang.ChatMessage{
            {Role: "user", Content: "Hello!"},
        },
        Stream: false,
    }

    resp, err := client.CreateChatCompletion(context.Background(), req)
    if err != nil {
        log.Fatal(err)
    }

    fmt.Println(resp.Choices[0].Message.Content)
}
```

### Streaming Usage

```go
stream, err := client.CreateChatCompletionStream(ctx, sglang.ChatCompletionRequest{
    Model: "default",
    Messages: []sglang.ChatMessage{
        {Role: "user", Content: "Tell me a story"},
    },
    Stream: true,
})
if err != nil {
    log.Fatal(err)
}
defer stream.Close()

for {
    chunk, err := stream.Recv()
    if err == io.EOF {
        break
    }
    if err != nil {
        log.Fatal(err)
    }
    
    for _, choice := range chunk.Choices {
        if choice.Delta.Content != "" {
            fmt.Print(choice.Delta.Content)
        }
    }
}
```

## Examples

The SDK includes several examples in the `examples/` directory:

- **simple**: Basic non-streaming chat completion
- **streaming**: Real-time streaming with metrics

### Running Examples

```bash
# Run simple example
cd bindings/golang/examples/simple
bash run.sh

# Run streaming example
cd bindings/golang/examples/streaming
bash run.sh

# Or use Makefile from bindings/golang directory
cd bindings/golang
make run-simple
make run-streaming
```

## Configuration

### Environment Variables

- `SGL_GRPC_ENDPOINT`: Default gRPC endpoint (e.g., `grpc://localhost:20000`)
- `SGL_TOKENIZER_PATH`: Default tokenizer directory path
- `CARGO_BUILD_DIR`: Rust build directory (default: `/Users/yangyanbo/cargobuild`)

### Build Requirements

- Go 1.21 or later
- Rust toolchain (for building the FFI library)
- Python 3.x (for Python bindings in Rust FFI)

## API Reference

### Client

```go
type Client struct {
    // ...
}

func NewClient(config ClientConfig) (*Client, error)
func (c *Client) Close() error
func (c *Client) CreateChatCompletion(ctx context.Context, req ChatCompletionRequest) (*ChatCompletionResponse, error)
func (c *Client) CreateChatCompletionStream(ctx context.Context, req ChatCompletionRequest) (*ChatCompletionStream, error)
```

### Request Types

- `ChatCompletionRequest`: Main request type for chat completions
- `ChatMessage`: Individual message in a conversation
- `Tool`: Tool/function definition for function calling

### Response Types

- `ChatCompletionResponse`: Non-streaming response
- `ChatCompletionStreamResponse`: Streaming response chunk
- `Message`: Complete message with content and tool calls
- `ToolCall`: Tool call information

## Development

### Building

```bash
cd bindings/golang
make build
```

### Running Tests

```bash
cd bindings/golang
make test
```

### Formatting

```bash
cd bindings/golang
make fmt
```

## License

See LICENSE file for details.
