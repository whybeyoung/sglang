# Streaming Tool Call Example

This example demonstrates streaming text generation with incremental tool call parsing using Rust FFI.

## Features

- **Real-time streaming**: Text chunks are received and processed as they arrive
- **Incremental tool parsing**: Uses Rust FFI tool parser to parse tool calls incrementally
- **Tool call detection**: Detects and tracks tool calls as they are being built
- **Performance metrics**: Shows TTFT (Time To First Token) and throughput

## Usage

```bash
# Basic usage
go run main.go -tokenizer /path/to/tokenizer -endpoint grpc://localhost:20000

# With specific parser type
go run main.go -tokenizer /path/to/tokenizer -endpoint grpc://localhost:20000 -parser json

# Using environment variables
export SGL_TOKENIZER_PATH=/path/to/tokenizer
export SGL_GRPC_ENDPOINT=grpc://localhost:20000
go run main.go
```

## Parser Types

The `-parser` flag accepts the following parser types:

- `auto` (default): Automatically detect parser type based on model
- `json`: JSON format parser
- `llama`: Llama-style parser
- `mistral`: Mistral-style parser
- `step3`: Step3 parser
- `qwen`: Qwen parser
- `deepseek`: DeepSeek parser
- `kimik2`: KimiK2 parser
- `gpt_oss`: GPT OSS parser
- `pythonic`: Pythonic parser

## How It Works

1. **Streaming Generation**: The client sends a request with tools and receives streaming chunks
2. **Incremental Parsing**: Each text chunk is passed to the Rust FFI tool parser
3. **Tool Call Detection**: The parser detects tool calls as they are being built
4. **Real-time Updates**: Tool call arguments are updated incrementally as more text arrives

## Example Output

```
Starting streaming generation with tool calls...
---
✓ First token received (TTFT: 150ms)
📝 Normal text: I'll help you check the weather in San Francisco.

🔧 New Tool Call Detected:
   ID: call_abc123
   Type: function
   Function: get_weather
   Arguments: {"location":"San Francisco"}

🔧 Tool Call Updated:
   ID: call_abc123
   Arguments: {"location":"San Francisco","unit":"fahrenheit"}

---
✓ Generation complete!
Finish reason: tool_calls
Prompt tokens: 45
Completion tokens: 23
Cached tokens: 0

--- Performance Metrics ---
TTFT (Time To First Token): 150ms
Total generation time: 1.2s
Throughput: 19.17 tokens/second

--- All Tool Calls (Final) ---

Tool Call #1:
  ID: call_abc123
  Type: function
  Function: get_weather
  Arguments: {"location":"San Francisco","unit":"fahrenheit"}
  Arguments (formatted):
    {
      "location": "San Francisco",
      "unit": "fahrenheit"
    }
```

