# SGLang Router gRPC 模式启动指南

## gRPC 模式概述

gRPC 模式是 SGLang Router 的默认模式，它使用 gRPC 协议与后端 worker 通信。gRPC 模式提供了更高效的通信和更多的功能支持。

## 必需配置

gRPC 模式**必须**提供以下配置之一：
- `--tokenizer-path`: tokenizer.json 文件路径
- `--model-path`: 模型路径（HuggingFace ID 或本地路径，会自动查找 tokenizer.json）

## 基本启动命令

### 方式 1: 使用命令行参数

```bash
./sglang-router \
  --host=0.0.0.0 \
  --port=30000 \
  --worker-urls="grpc://10.109.185.20:8000" \
  --tokenizer-path=/path/to/tokenizer.json \
  --policy=round_robin \
  --log-level=info
```

### 方式 2: 使用模型路径

```bash
./sglang-router \
  --host=0.0.0.0 \
  --port=30000 \
  --worker-urls="grpc://10.109.185.20:8000" \
  --model-path=/path/to/model \
  --policy=round_robin \
  --log-level=info
```

### 方式 3: 使用启动脚本

```bash
# 使用环境变量
TOKENIZER_PATH=/path/to/tokenizer.json \
WORKER_URLS="grpc://10.109.185.20:8000" \
./start-grpc-router.sh

# 或使用模型路径
MODEL_PATH=/path/to/model \
WORKER_URLS="grpc://10.109.185.20:8000" \
./start-grpc-router.sh
```

## 配置说明

### Worker URLs

- **格式**: `grpc://host:port`
- **示例**: `grpc://10.109.185.20:8000`
- **多 worker**: `grpc://worker1:8000,grpc://worker2:8000`

### Tokenizer Path

- 可以是 `tokenizer.json` 文件的完整路径
- 也可以是一个目录（会自动查找 `tokenizer.json`）

### Model Path

- 可以是 HuggingFace 模型 ID（如 `meta-llama/Llama-2-7b-hf`）
- 也可以是本地模型路径
- 会自动在模型目录中查找 `tokenizer.json`

## 完整示例

```bash
# 示例 1: 单 worker，使用 tokenizer 路径
./sglang-router \
  --host=0.0.0.0 \
  --port=30000 \
  --worker-urls="grpc://10.109.185.20:8000" \
  --tokenizer-path=/models/llama2/tokenizer.json \
  --policy=round_robin \
  --log-level=info

# 示例 2: 多 worker，使用模型路径
./sglang-router \
  --host=0.0.0.0 \
  --port=30000 \
  --worker-urls="grpc://worker1:8000,grpc://worker2:8000" \
  --model-path=/models/llama2 \
  --policy=cache_aware \
  --log-level=debug

# 示例 3: 使用 HuggingFace 模型 ID
./sglang-router \
  --host=0.0.0.0 \
  --port=30000 \
  --worker-urls="grpc://10.109.185.20:8000" \
  --model-path=meta-llama/Llama-2-7b-hf \
  --policy=round_robin \
  --log-level=info
```

## gRPC 模式 vs HTTP 模式

| 特性 | gRPC 模式 | HTTP 模式 |
|------|-----------|-----------|
| 协议 | gRPC | HTTP/HTTPS |
| Worker URL 格式 | `grpc://host:port` | `http://host:port` |
| Tokenizer | **必需** | 不需要（worker 处理） |
| 性能 | 更高 | 标准 |
| 功能 | 完整支持（流式、工具调用等） | 基础支持 |
| 推荐场景 | 生产环境、高性能需求 | 简单代理、测试环境 |

## 负载均衡策略

gRPC 模式支持以下负载均衡策略：

- `random`: 随机选择
- `round_robin`: 轮询（默认）
- `cache_aware`: 缓存感知（考虑 worker 负载）
- `power_of_two`: 两个随机选择中负载较小的

## 验证启动

启动后，可以通过以下方式验证：

```bash
# 检查健康状态
curl http://localhost:30000/health

# 查看注册的 workers
curl http://localhost:30000/workers

# 测试 chat 请求
curl -X POST http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }'
```

## 常见问题

### Q: 启动失败，提示 "tokenizer-path or model-path is required"
**A**: gRPC 模式必须提供 tokenizer 配置。请使用 `--tokenizer-path` 或 `--model-path` 参数。

### Q: Worker 连接失败
**A**: 确保：
1. Worker 使用 gRPC 协议运行
2. Worker URL 格式正确（`grpc://host:port`）
3. 网络连接正常
4. Worker 的健康检查端点可访问

### Q: Tokenizer 加载失败
**A**: 检查：
1. tokenizer.json 文件路径是否正确
2. 文件是否存在且可读
3. 如果是模型路径，确保包含 tokenizer.json

