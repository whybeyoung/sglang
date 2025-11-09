# Streaming Example

This example demonstrates real-time streaming text generation.

## Features

- **Real-time streaming**: Text is printed as it arrives from the server
- **Performance metrics**: Shows TTFT (Time To First Token) and throughput
- **Token usage**: Displays prompt, completion, and cached token counts

## Usage

```bash
# Basic usage
go run main.go -tokenizer /path/to/tokenizer -endpoint grpc://localhost:20000

# Using environment variables
export SGL_TOKENIZER_PATH=/path/to/tokenizer
export SGL_GRPC_ENDPOINT=grpc://localhost:20000
go run main.go

# Using config file
go run main.go -config config.json
```

## Example Output

```
Starting streaming generation...
---
✓ First token received (TTFT: 120ms)
---
Once upon a time, in a world where technology and art intertwined, there lived a robot named Pixel. Pixel was not like other robots...

---
✓ Generation complete!
Finish reason: stop
Prompt tokens: 25
Completion tokens: 150
Cached tokens: 0

--- Performance Metrics ---
TTFT (Time To First Token): 120ms
Total generation time: 2.5s
Throughput: 60.00 tokens/second
Time per token: 16.67ms
```

