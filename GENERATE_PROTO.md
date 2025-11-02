# 生成 Protobuf 代码

## 已完成的修改

### 1. Proto 文件更新
已更新以下 proto 文件，添加了 `GetTokenizerInfo` RPC：
- `python/sglang/srt/grpc/sglang_scheduler.proto`
- `sgl-router/src/proto/sglang_scheduler.proto`
- `sgl-router-go/proto/sglang_scheduler.proto`

### 2. Python gRPC Server 实现
已在 `python/sglang/srt/entrypoints/grpc_server.py` 中实现 `GetTokenizerInfo` 方法。

## 生成 Python Proto 代码

### 方法 1: 使用 compile_proto.py 脚本（推荐）

```bash
cd python/sglang/srt/grpc
pip install "grpcio==1.75.1" "grpcio-tools==1.75.1"
python compile_proto.py
```

### 方法 2: 手动使用 protoc

```bash
cd python/sglang/srt/grpc
protoc -I. \
  --python_out=. \
  --grpc_python_out=. \
  --pyi_out=. \
  sglang_scheduler.proto
```

这将生成：
- `sglang_scheduler_pb2.py` - Protobuf 消息类
- `sglang_scheduler_pb2_grpc.py` - gRPC 服务类
- `sglang_scheduler_pb2.pyi` - 类型提示

## 生成 Rust Proto 代码

Rust router 的 proto 代码会在构建时自动生成（通过 `build.rs`）：

```bash
cd sgl-router
cargo build
```

## 生成 Go Proto 代码

Go router 的 proto 代码通过 Makefile 生成：

```bash
cd sgl-router-go
make proto
```

## 验证

生成完成后，应该可以在 Python 代码中使用：
```python
from sglang.srt.grpc import sglang_scheduler_pb2, sglang_scheduler_pb2_grpc

# 使用 GetTokenizerInfoRequest 和 GetTokenizerInfoResponse
request = sglang_scheduler_pb2.GetTokenizerInfoRequest(requested_files=["tokenizer.json"])
```

