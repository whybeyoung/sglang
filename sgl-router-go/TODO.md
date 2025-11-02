# 待办事项清单

## 🔥 高优先级（影响核心功能）

### 1. 工具过滤实现 ✅ 框架完成，需实现逻辑
**文件**: `internal/router/pipeline/stages/utils.go`

**需要实现**:
- `FilterToolsForRequest` 函数
  - ToolChoice::AllowedTools -> 过滤到允许的工具列表
  - ToolChoice::Function -> 过滤到特定函数
  - 否则 -> 不过滤

**参考**: Rust `src/routers/grpc/utils.rs::filter_tools_for_request`

### 2. 工具约束生成 ✅ 框架完成，需实现逻辑
**文件**: `internal/router/pipeline/stages/utils.go`

**需要实现**:
- `GenerateToolConstraints` 函数
  - ToolChoice::Function -> 返回单个函数的 parameters schema
  - ToolChoice::Required -> 构建数组 schema (minItems: 1)
  - ToolChoice::AllowedTools (mode="required") -> 构建数组 schema
  - 其他 -> 返回 nil

**参考**: Rust `src/routers/grpc/utils.rs::generate_tool_constraints` 和 `build_required_array_schema`

### 3. Chat Template JSON 解析 ⚠️ 部分完成
**文件**: `internal/tokenizer/huggingface.go`

**需要实现**:
- 从 `tokenizer_config.json` 中解析 `chat_template` 字段
- 从 JSON 文件中提取 Jinja 模板字符串

**当前状态**: 函数框架存在，但 JSON 解析未实现

## ⚡ 中优先级（增强功能）

### 4. Stop Decoder 序列匹配优化 ⚠️ 基础实现，需优化
**文件**: `internal/tokenizer/stop_decoder.go`

**需要优化**:
- 跨 token 边界的 stop sequence 匹配（当前实现简化）
- 更准确的序列匹配算法
- 处理多字节字符的边界情况

### 5. Chat Template Jinja 渲染 ⚠️ 未实现
**文件**: `internal/router/pipeline/stages/utils.go` - `ProcessChatMessages`

**需要实现**:
- 集成 Go 的 Jinja 模板库（如 `github.com/noirbizarre/gotext/template` 或其他）
- 或者参考 Rust 实现自定义 Jinja 解析器
- 支持 chat template 的各种功能（循环、条件等）

**参考**: Rust `src/tokenizer/chat_template/`

### 6. HuggingFace Hub 下载支持 ⚠️ 未实现
**文件**: `internal/tokenizer/factory.go`

**需要实现**:
- 从 HuggingFace Hub 下载 tokenizer 文件
- 缓存管理
- 支持模型 ID（如 "meta-llama/Llama-2-7b-hf"）

**参考**: Rust `src/tokenizer/hub.rs`

## 🔧 低优先级（可选功能）

### 7. Tool Parser Factory
**位置**: `internal/router/grpc_router.go`
- 用于解析工具调用响应
- 当前: `nil` 占位符

### 8. Reasoning Parser Factory
**位置**: `internal/router/grpc_router.go`
- 用于解析 reasoning 内容
- 当前: `nil` 占位符

### 9. Prometheus Metrics
**位置**: `cmd/router/main.go`
- 指标收集和导出
- 当前: TODO 注释

### 10. 高级负载均衡策略
- CacheAware Policy（需要前缀树实现）
- PowerOfTwo Policy（需要负载监控）

### 11. Worker 管理 API 完善
- 删除 worker 端点
- Worker 更新端点

## 📊 当前完成度

| 功能模块 | 完成度 | 状态 |
|---------|--------|------|
| 核心架构 | 100% | ✅ |
| Pipeline 阶段 | 95% | ✅ |
| gRPC 通信 | 100% | ✅ |
| Tokenizer 集成 | 100% | ✅ |
| 流式响应 | 100% | ✅ |
| 非流式响应 | 100% | ✅ |
| Stop Decoder | 80% | ⚠️ 基础完成 |
| 工具过滤 | 30% | ⚠️ 框架完成 |
| 工具约束 | 30% | ⚠️ 框架完成 |
| Chat Template | 40% | ⚠️ 框架完成 |

## 🎯 建议优先级

1. **立即完成**: 工具过滤和约束生成（影响工具调用功能）
2. **短期完成**: Chat Template JSON 解析和基础渲染
3. **中期完成**: Stop Decoder 优化
4. **长期完成**: HuggingFace Hub 下载、Tool/Reasoning Parser

