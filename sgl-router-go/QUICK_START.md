# SGLang Router 快速启动指南

## 在 10.109.185.20:8002 启动 HTTP 服务器

### 方式 1: 直接启动（基本配置）

```bash
cd /Users/yangyanbo/projects/iflytek/code/opensource/pd/official/sglang/sgl-router-go

# 基本启动（需要 tokenizer 路径）
./sglang-router \
  --host=10.109.185.20 \
  --port=8002 \
  --worker-urls="" \
  --tokenizer-path=/path/to/tokenizer.json \
  --policy=round_robin \
  --log-level=info
```

### 方式 2: 使用启动脚本

```bash
# 使用简单脚本
./start-router-simple.sh

# 或使用完整脚本
./scripts/start-router.sh \
  --host=10.109.185.20 \
  --port=8002 \
  --worker-urls=grpc://worker1:8000,grpc://worker2:8000 \
  --tokenizer-path=/path/to/tokenizer.json
```

### 方式 3: 完整配置示例

```bash
./sglang-router \
  --host=10.109.185.20 \
  --port=8002 \
  --worker-urls="grpc://10.109.185.21:8000,grpc://10.109.185.22:8000" \
  --tokenizer-path=/path/to/tokenizer.json \
  --policy=round_robin \
  --log-level=info \
  --log-dir=/var/log/sglang-router
```

## 端点说明

启动后，HTTP 服务器将在 `http://10.109.185.20:8002` 提供以下端点：

- **POST** `/v1/chat/completions` - OpenAI 兼容的 Chat API
- **POST** `/generate` - SGLang Generate API
- **GET** `/health` - 健康检查
- **GET** `/liveness` - 存活检查
- **GET** `/readiness` - 就绪检查
- **GET** `/workers` - Worker 管理（列出所有 workers）
- **POST** `/workers` - Worker 管理（注册新 worker）

## 测试启动

```bash
# 测试健康检查
curl http://10.109.185.20:8002/health

# 测试 worker 列表
curl http://10.109.185.20:8002/workers
```

## 重要提示

1. **Tokenizer 路径**: gRPC 模式需要提供 `--tokenizer-path` 或 `--model-path`
2. **Worker URLs**: 可以启动时为空，后续通过 `/workers` API 动态添加
3. **负载均衡策略**: 支持 `random`, `round_robin`, `cache_aware`, `power_of_two`

## 后台运行

```bash
nohup ./sglang-router \
  --host=10.109.185.20 \
  --port=8002 \
  --tokenizer-path=/path/to/tokenizer.json \
  > router.log 2>&1 &

# 查看日志
tail -f router.log
```

## 停止服务

```bash
# 查找进程
ps aux | grep sglang-router

# 优雅停止（发送 SIGTERM）
kill -TERM <PID>

# 或使用 pkill
pkill -TERM sglang-router
```

