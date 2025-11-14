# Complete SDK Example

This example demonstrates the complete request-response flow using the Rust FFI SDK. It shows how to:

1. Create a client handle (connects to gRPC endpoint and initializes tokenizer)
2. Send an OpenAI-format ChatCompletionRequest
3. Receive and process streaming responses in OpenAI format
4. Display metrics (TTFT, tokens per second)

## Features

- **Full SDK Integration**: Uses the complete Rust FFI API (`sgl_client_create`, `sgl_client_chat_completion_stream`, `sgl_stream_read_next`)
- **OpenAI Format**: Accepts OpenAI-format requests and receives OpenAI-format responses
- **Streaming Support**: Real-time streaming response processing
- **Metrics**: Tracks Time To First Token (TTFT) and throughput

## Usage

```bash
# Using default configuration
./run.sh

# With custom tokenizer and endpoint
./run.sh /path/to/tokenizer grpc://localhost:20000

# Using environment variables
export SGL_TOKENIZER_PATH=/path/to/tokenizer
export SGL_GRPC_ENDPOINT=grpc://localhost:20000
./run.sh
```

## Command Line Options

```bash
go run main.go -tokenizer <path> -endpoint <endpoint> [options]

Options:
  -tokenizer <path>    Path to tokenizer directory (required)
  -endpoint <endpoint> gRPC endpoint, e.g., grpc://localhost:20000 (required)
  -model <name>        Model name (default: 'default')
  -help                Show help message
```

## How It Works

1. **Client Creation**: Creates a client handle that manages both the gRPC connection and tokenizer
2. **Request Preparation**: Builds an OpenAI-format JSON request
3. **Stream Initiation**: Sends the request and receives a stream handle
4. **Response Processing**: Reads chunks from the stream, converts them to OpenAI format, and displays content
5. **Cleanup**: Properly frees all handles and resources

## Architecture

This example uses the complete FFI API that handles:
- Message processing and chat template application
- Tokenization
- Tool constraint generation (if needed)
- gRPC request building
- Response conversion to OpenAI format

All of this is handled by the Rust SDK, making it a true "complete SDK" solution.

