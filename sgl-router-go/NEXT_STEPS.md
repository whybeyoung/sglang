# 下一步实现指南

## 当前状态

✅ **编译通过**: 项目已可成功编译  
✅ **基础架构**: 所有 Pipeline Stages 结构已完成  
⚠️ **占位符实现**: 部分功能使用占位符，需要完整实现

## 立即需要完成的任务

### 1. 生成 Proto 代码 (优先级: 最高)

```bash
# 安装必要的工具
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

# 生成代码
cd sgl-router-go
make generate
```

**完成后需要更新**:
- `internal/router/pipeline/stages/request_building.go` - 使用实际的 `proto.GenerateRequest`
- `internal/router/pipeline/stages/request_execution.go` - 使用实际的 `proto.NewSglangSchedulerClient`
- `internal/router/pipeline/stages/response_processing.go` - 处理实际的 `proto.GenerateResponse` stream

### 2. 集成 Tokenizer (优先级: 高)

**选项 A: 使用 Go tokenizer 库** (推荐)
```bash
go get github.com/sugarme/tokenizer
# 或
go get github.com/pkoukk/tiktoken-go
```

**选项 B: CGO 绑定到 Rust tokenizers**
- 需要创建 C bindings
- 更复杂但性能可能更好

**需要更新**:
- `internal/tokenizer/interface.go` - 实现实际的 tokenizer
- `internal/router/pipeline/stages/preparation.go` - 使用 tokenizer
- `internal/router/pipeline/stages/response_processing.go` - 使用 tokenizer 进行 detokenization

### 3. 完善请求执行阶段

位置: `internal/router/pipeline/stages/request_execution.go`

需要实现:
```go
// 替换占位符代码
client := proto.NewSglangSchedulerClient(conn)
stream, err := client.Generate(ctx, grpcRequest)
if err != nil {
    return nil, err
}
// 返回实际的 stream
```

### 4. 完善响应处理阶段

位置: `internal/router/pipeline/stages/response_processing.go`

需要实现:
1. **流式响应处理**:
   ```go
   for {
       resp, err := stream.Recv()
       if err == io.EOF {
           break
       }
       // 处理每个响应块
   }
   ```

2. **非流式响应处理**:
   - 收集所有响应块
   - 使用 tokenizer detokenize
   - 应用 stop sequence trimming
   - 格式化为 ChatCompletionResponse 或 GenerateResponse

### 5. 完善请求构建阶段

位置: `internal/router/pipeline/stages/request_building.go`

需要实现:
1. 从 ChatCompletionRequest 提取采样参数
2. 从 GenerateRequest 提取采样参数
3. 转换为 proto.SamplingParams（注意默认值不是 0）

### 6. Worker 健康检查

位置: `internal/core/worker.go` - `CheckHealth` 方法

需要实现:
- gRPC 模式: 调用 `proto.HealthCheck` RPC
- HTTP 模式: 调用 `/health` 端点

### 7. Worker 管理 API

位置: `internal/server/http.go` - `handleWorkers`

需要实现:
- `GET /workers` - 列出所有 workers
- `POST /workers` - 注册新 worker
- `DELETE /workers/{url}` - 删除 worker

## 代码中的关键 TODO

### 高优先级 TODO

1. **Proto 类型转换** (多处)
   - `internal/router/pipeline/stages/request_building.go`
   - `internal/router/pipeline/stages/request_execution.go`

2. **Tokenizer 集成**
   - `internal/router/pipeline/stages/preparation.go`
   - `internal/router/pipeline/stages/response_processing.go`

3. **gRPC 流处理**
   - `internal/router/pipeline/stages/response_processing.go`

### 中优先级 TODO

4. **消息处理**
   - Chat template 应用
   - 多模态输入处理

5. **工具调用处理**
   - Tool filtering
   - Tool constraint generation
   - Tool call parsing

6. **Stop sequence 解码**
   - Stop decoder 实现
   - Stop trimming 逻辑

## 实现建议

### Proto 代码生成后的步骤

1. **更新导入**:
   ```go
   import "github.com/sglang/sglang-router-go/internal/grpc/proto"
   ```

2. **替换占位符类型**:
   ```go
   // 之前
   type ProtoGenerateRequest struct { ... }
   
   // 之后
   // 使用 proto.GenerateRequest
   ```

3. **实现客户端调用**:
   ```go
   client := proto.NewSglangSchedulerClient(conn)
   stream, err := client.Generate(ctx, &proto.GenerateRequest{...})
   ```

### Tokenizer 集成步骤

1. **选择合适的库** (推荐 `github.com/pkoukk/tiktoken-go` 或 `github.com/sugarme/tokenizer`)

2. **实现接口**:
   ```go
   type TokenizerImpl struct {
       tokenizer *tiktoken.Encoding // 或选择的库
   }
   
   func (t *TokenizerImpl) Encode(text string) (*Encoding, error) {
       // 实现编码
   }
   ```

3. **在 main.go 中初始化**:
   ```go
   tokenizer := tokenizer.NewFromModelPath(cfg.ModelPath)
   ```

## 测试建议

1. **单元测试**: 为每个 Pipeline Stage 编写测试
2. **集成测试**: 测试完整的请求流程
3. **Mock Worker**: 创建测试用的 mock gRPC worker

## 性能优化点

1. **连接池**: gRPC 客户端池已实现，确保正确使用
2. **Tokenizer 缓存**: 考虑缓存常见文本的 tokenization 结果
3. **Stream 处理**: 优化流式响应的处理性能

## 与 Rust 版本的对应关系

参考 `IMPLEMENTATION_NOTES.md` 了解详细的对应关系和实现差异说明。

## 完成后的验证

完成上述步骤后，应该能够:
1. ✅ 启动 router 服务器
2. ✅ 接收 HTTP 请求
3. ✅ 路由到 gRPC workers
4. ✅ 处理响应并返回给客户端

继续完善实现！
