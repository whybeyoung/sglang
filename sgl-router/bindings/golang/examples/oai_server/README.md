# Go SGLang Router README

## 整体架构

Go SGLang Router 是一个高性能的 OpenAI 兼容 API 服务器，使用 gRPC 与 SGLang 后端通信，并通过 Rust FFI 进行高效的预处理和后处理。

**重要说明**：gRPC 模式**仍然会调用 FFI**，FFI 用于：
- **预处理**：chat_template 和 tokenization（请求阶段）
- **后处理**：token decoding 和 tool parsing（响应阶段）

gRPC 仅用于与 SGLang 后端通信，输入输出的处理完全依赖 Rust FFI。

```
┌─────────────────────────────────────────────────────────────────┐
│                        HTTP Client                               │
│                    (OpenAI API Format)                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastHTTP Server                               │
│              handlers/chat.go:HandleChatCompletion               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              SGLang Client (client.go)                           │
│         CreateChatCompletionStream(req)                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│          gRPC Client (internal/grpc/client_grpc.go)             │
│         CreateChatCompletionStream(ctx, reqJSON)                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────┴────────────────────┐
        │                                           │
        ▼                                           ▼
┌──────────────────┐                    ┌──────────────────────┐
│  FFI Preprocess  │                    │  Build gRPC Request  │
│  (Rust FFI)      │                    │  (GenerateRequest)   │
│  - chat_template │                    │  - SamplingParams   │
│  - tokenization  │                    │  - max_tokens        │
│  (Always enabled │                    │                      │
│   in gRPC mode)  │                    │                      │
└────────┬─────────┘                    └──────────┬───────────┘
         │                                          │
         └──────────────────┬───────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              gRPC Stream (client.Generate())                    │
│              SGLang Backend (Rust)                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│         GrpcChatCompletionStream.readLoop()                     │
│         (Background Goroutine)                                   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Recv() Goroutine (Dedicated)                            │  │
│  │  - Continuously calls stream.Recv()                      │  │
│  │  - Sends results to recvChan                            │  │
│  └────────────────────┬─────────────────────────────────────┘  │
│                       │                                          │
│                       ▼                                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Main Loop                                                │  │
│  │  - Reads from recvChan                                    │  │
│  │  - Converts proto to JSON (protoToJSON)                  │  │
│  │  - Calls FFI BatchPostprocessor.AddChunk()              │  │
│  │  - Sends JSON to resultJSONChan                          │  │
│  └────────────────────┬─────────────────────────────────────┘  │
└────────────────────────┼────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│         resultJSONChan (Channel)                                  │
│         - Buffered (10000, configurable)                        │
│         - Contains OpenAI-format JSON strings                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│         ChatCompletionStream.RecvJSON()                          │
│         - Direct wrapper around grpcStream.RecvJSON()            │
│         - No intermediate channels or parsing                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│         FastHTTP SetBodyStreamWriter                             │
│         - SSE streaming                                          │
│         - Immediate flush                                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        HTTP Client                               │
│                    (SSE Stream)                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 关键数据结构

### 1. GrpcClient
**位置**: `internal/grpc/client_grpc.go`

```go
type GrpcClient struct {
    conn            *grpc.ClientConn
    client          proto.SglangSchedulerClient
    tokenizerPath   string
    tokenizerHandle *ffi.TokenizerHandle // Pre-created at startup, thread-safe
}
```

**职责**:
- 管理 gRPC 连接
- 预创建 tokenizer handle（启动时创建，线程安全）
- 创建 gRPC stream

### 2. GrpcChatCompletionStream
**位置**: `internal/grpc/client_grpc.go`

```go
type GrpcChatCompletionStream struct {
    stream             grpcClientStream
    converterHandle    *ffi.GrpcResponseConverterHandle
    batchPostprocessor *ffi.BatchPostprocessor
    batchSize          int
    ctx                context.Context
    closed             int32 // Atomic flag
    resultJSONChan     chan string // Processed JSON responses
    errChan            chan error  // Errors
    readLoopDone       chan struct{} // Completion signal
    requestID          string
    model              string
}
```

**职责**:
- 管理 gRPC stream 生命周期
- 处理 proto 响应并转换为 OpenAI 格式
- 通过 channel 异步传递结果

### 3. ChatCompletionStream
**位置**: `client.go`

```go
type ChatCompletionStream struct {
    grpcStream *grpcclient.GrpcChatCompletionStream
    ctx        context.Context
    cancel     context.CancelFunc
}
```

**职责**:
- 提供高级 API 接口
- 管理 context 生命周期
- 直接委托给 `GrpcChatCompletionStream`

**设计简化**:
- 移除了冗余的 `chunksChan` 和 `readGrpcLoop`
- `RecvJSON()` 直接调用 `grpcStream.RecvJSON()`
- 避免了重复读取和 JSON 解析/序列化

### 4. Channel 设计

#### resultJSONChan
- **类型**: `chan string`
- **缓冲**: 10000 (可配置，默认值)
- **用途**: 存储 FFI 处理后的 OpenAI 格式 JSON 字符串
- **发送者**: `readLoop` (processAndSendResponse)
- **接收者**: `RecvJSON()`

#### errChan
- **类型**: `chan error`
- **缓冲**: 100 (可配置)
- **用途**: 传递错误信息
- **发送者**: `readLoop`
- **接收者**: `RecvJSON()`

#### recvChan
- **类型**: `chan recvResult`
- **缓冲**: 2000 (可配置，默认值)
- **用途**: Recv() goroutine 向 readLoop 传递 gRPC 响应
- **发送者**: Recv() goroutine
- **接收者**: `readLoop`

#### readLoopDone
- **类型**: `chan struct{}`
- **缓冲**: 0 (unbuffered)
- **用途**: 信号 channel，表示 readLoop 已完成
- **关闭者**: `readLoop` (defer)

## 数据流程

### 输入流程 (Request Processing)

1. **HTTP 请求接收** (`handlers/chat.go`)
   - FastHTTP 接收 OpenAI 格式请求
   - 解析 JSON 到 `ChatRequest` 结构

2. **请求转换** (`handlers/chat.go`)
   - `ChatRequest` → `sglang.ChatCompletionRequest`
   - 支持 `max_tokens` 和 `max_completion_tokens`

3. **FFI 预处理** (`client_grpc.go:CreateChatCompletionStream`)
   - 调用 `ffi.PreprocessChatRequestWithTokenizer()`
   - 使用预创建的 tokenizer handle（线程安全，无锁）
   - 执行 `chat_template` 和 tokenization
   - 返回 `PreprocessedRequest`:
     - `PromptText`: 处理后的文本
     - `TokenIDs`: token ID 数组
     - `ToolConstraintsJSON`: 工具约束（如果有）
     - `PromptTokens`: prompt token 数量

4. **构建 gRPC 请求** (`client_grpc.go`)
   - 创建 `proto.GenerateRequest`
   - 设置 `SamplingParams`:
     - `Temperature`, `TopP`, `TopK`
     - `MaxNewTokens` (从 `max_completion_tokens` 或 `max_tokens`)
     - `Constraint` (从 tool constraints)
   - 调用 `client.Generate()` 创建 gRPC stream

5. **创建 Converter** (`client_grpc.go`)
   - 调用 `ffi.CreateGrpcResponseConverterWithTokenizer()`
   - 创建 `BatchPostprocessor` (batchSize=1, 立即处理)

### 输出流程 (Response Processing)

1. **gRPC Stream 读取** (`readLoop`)
   - 启动专门的 Recv() goroutine
   - 持续调用 `stream.Recv()` 获取 proto 响应
   - 通过 `recvChan` 传递给主循环

2. **Proto 到 JSON 转换** (`processAndSendResponse`)
   - 调用 `protoToJSON()` 将 proto 转换为 JSON 字符串
   - 优化：手动构建 JSON，减少 `json.Marshal` 调用

3. **FFI 后处理** (`processAndSendResponse`)
   - 调用 `batchPostprocessor.AddChunk(protoJSON)`
   - Rust FFI 处理：
     - Token decoding
     - Tool call parsing
     - Usage calculation
   - 返回 OpenAI 格式 JSON 字符串数组

4. **Channel 传递** (`processAndSendResponse`)
   - 将 JSON 字符串发送到 `resultJSONChan`
   - 检查 `closed` 标志和 `ctx.Done()` 以避免死锁

5. **客户端读取** (`ChatCompletionStream.RecvJSON()`)
   - 直接调用 `grpcStream.RecvJSON()`
   - 从 `resultJSONChan` 读取 JSON 字符串
   - 无需中间层解析或序列化

6. **SSE 流式传输** (`handlers/chat.go`)
   - 使用 `SetBodyStreamWriter` 实现真正的流式传输
   - 每个 chunk 立即 flush
   - 检测客户端断开连接并取消 stream

## 关键设计决策

### 1. 线程安全的 Tokenizer
- **问题**: Tokenizer 需要并发访问
- **解决方案**: 启动时预创建 `TokenizerHandle`，Rust 端使用 `Arc<dyn TokenizerTrait>`，线程安全
- **优势**: 无锁并发，消除锁竞争

### 2. 可取消的 Recv()
- **问题**: `stream.Recv()` 是阻塞的，无法通过 context 取消
- **解决方案**: 
  - 使用专门的 goroutine 执行 `Recv()`
  - 通过 `recvChan` 传递结果
  - Context 取消时调用 `CloseSend()` 使 `Recv()` 返回错误

### 3. Lazy JSON Parsing
- **问题**: 在 `readLoop` 中解析 JSON 会阻塞处理
- **解决方案**: 
  - `readLoop` 只处理 proto → JSON 转换和 FFI 调用
  - JSON 解析延迟到 `Recv()` 调用时
- **优势**: 减少 `readLoop` 阻塞，提高吞吐量

### 4. 简化的 Channel 设计
- **移除**: 
  - `resultChan` (非 FFI 模式不再需要)
  - `chunksChan` (client.go 中的冗余 channel)
  - `readGrpcLoop` (重复读取的 goroutine)
- **保留**: 
  - `resultJSONChan`: 主要数据通道（gRPC 层）
  - `errChan`: 错误通道（gRPC 层）
  - `recvChan`: 内部通信通道（gRPC 层）
- **优势**: 
  - 减少 channel 数量，降低死锁风险
  - 消除重复读取和解析
  - 简化数据流路径

### 5. 批量处理优化
- **batchSize=1**: 立即处理，无延迟
- **flushInterval=0**: 无超时，立即处理
- **权衡**: 更多 FFI 调用，但消除所有批处理延迟

## 错误处理

### Context 取消
1. 客户端断开连接 → `SetBodyStreamWriter` 检测到 flush 错误
2. 取消 `streamCtx` → `readLoop` 检测到 `ctx.Done()`
3. 调用 `stream.CloseSend()` → `Recv()` goroutine 返回错误
4. 清理资源并退出

### Stream 错误
- EOF: 正常结束，flush 剩余 chunks
- 其他错误: 发送到 `errChan`，`Recv()` 返回错误

### Channel 阻塞
- 所有 channel 发送都检查 `ctx.Done()` 和 `closed` 标志
- 如果 channel 满且 context 取消，立即退出避免死锁

## 性能优化

1. **预创建 Tokenizer**: 启动时创建，避免首次请求延迟
2. **无锁并发**: Tokenizer 线程安全，无需锁
3. **Lazy Parsing**: JSON 解析延迟到需要时
4. **直接 JSON 传递**: `RecvJSON()` 避免解析/序列化开销
5. **立即批处理**: batchSize=1，无延迟
6. **异步处理**: `readLoop` 在后台处理，不阻塞请求处理

## 文件结构

```
sgl-router/bindings/golang/
├── client.go                          # 高级客户端 API
├── internal/
│   ├── grpc/
│   │   └── client_grpc.go            # gRPC 客户端实现
│   ├── ffi/                          # FFI 绑定（Rust）
│   └── proto/                        # Protobuf 定义
└── examples/
    └── oai_server/
        ├── handlers/
        │   └── chat.go               # HTTP 请求处理
        ├── models/
        │   └── chat.go               # 请求/响应模型
        └── service/
            └── sglang_service.go      # 服务层
```

## 关键函数

### CreateChatCompletionStream
**位置**: `internal/grpc/client_grpc.go:119`
- 预处理请求（FFI）
- 构建 gRPC 请求
- 创建 converter 和 batch processor
- 启动 `readLoop`

### readLoop
**位置**: `internal/grpc/client_grpc.go:321`
- 启动 Recv() goroutine
- 处理 proto 响应
- 调用 FFI 后处理
- 发送结果到 `resultJSONChan`

### processAndSendResponse
**位置**: `internal/grpc/client_grpc.go:443`
- 转换 proto 到 JSON
- 调用 FFI batch processor
- 发送 JSON 到 channel

### RecvJSON
**位置**: 
- `internal/grpc/client_grpc.go:453`: gRPC 层实现
- `client.go:423`: 客户端包装层
- 从 `resultJSONChan` 读取
- 直接返回 JSON 字符串，无需解析

