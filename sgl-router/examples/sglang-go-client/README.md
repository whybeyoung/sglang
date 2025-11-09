# SGLang Go Client

Go 客户端库，通过调用 Rust router 的 FFI 方法来与 SGLang 服务交互。

## 功能特性

- ✅ Tokenizer 支持（通过 FFI 调用 Rust 实现）
- ✅ Chat Template 处理
- ✅ gRPC 客户端（调用 SGLang 调度器）
- ✅ 流式和非流式生成
- ✅ 配置管理
- ✅ 错误处理

## 项目结构

```
sglang-go-client/
├── cmd/
│   └── sglang-client/    # 主程序
├── internal/
│   ├── ffi/              # FFI 包装层
│   ├── grpc/             # gRPC 客户端
│   ├── tokenizer/        # Tokenizer 高级封装
│   └── client/           # 统一客户端接口
├── config/               # 配置管理
├── examples/             # 更多示例
├── proto/                # Protobuf 生成代码（运行 make proto 后生成）
├── go.mod
├── Makefile
└── README.md
```

## 快速开始

### 1. 前置要求

- Go 1.21+
- Rust 和 Cargo（用于构建 FFI 库）
- protoc（用于生成 protobuf 代码）

### 2. 构建 Rust 库

```bash
cd ../..  # 回到 sgl-router 根目录
cargo build --release
```

### 3. 设置开发环境

```bash
cd examples/sglang-go-client
make setup  # 安装依赖并生成 protobuf 代码
```

或者手动执行：

```bash
# 安装 Go 依赖
go mod download

# 安装 protoc 插件
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

# 生成 protobuf 代码
make proto
```

### 4. 设置环境变量

```bash
# Linux/macOS
export CGO_LDFLAGS="-L$(pwd)/../../target/release -lsglang_router_rs -ldl"
export LD_LIBRARY_PATH="$(pwd)/../../target/release:$LD_LIBRARY_PATH"  # Linux
export DYLD_LIBRARY_PATH="$(pwd)/../../target/release:$DYLD_LIBRARY_PATH"  # macOS

# Windows (PowerShell)
$env:CGO_LDFLAGS="-L$PWD\..\..\target\release -lsglang_router_rs"
```

### 5. 运行示例

#### 使用命令行参数（推荐）

```bash
# 主程序（推荐：使用目录路径，自动发现 tokenizer 文件）
./bin/sglang-client -tokenizer /path/to/tokenizer -endpoint grpc://localhost:20000

# 或使用 make
make run TOKENIZER=/path/to/tokenizer ENDPOINT=grpc://localhost:20000

# 或直接运行
go run cmd/sglang-client/main.go -tokenizer /path/to/tokenizer -endpoint grpc://localhost:20000

# 也可以直接指定 tokenizer.json 文件
./bin/sglang-client -tokenizer /path/to/tokenizer.json -endpoint grpc://localhost:20000
```

#### 使用环境变量

```bash
# 设置配置（推荐：使用目录路径）
export SGL_TOKENIZER_PATH="/Users/yangyanbo/projects/iflytek/code/opensource/pd/official/sglang/sgl-router/examples/sglang-go-client/examples/tokenizer"
export SGL_GRPC_ENDPOINT="grpc://10.109.185.20:8001"

# 运行
make run
# 或
go run cmd/sglang-client/main.go
```

#### 查看帮助

```bash
./bin/sglang-client -help
go run cmd/sglang-client/main.go -help
```

详细使用说明请参考 [USAGE.md](USAGE.md)

## 使用示例

### 基本使用

```go
package main

import (
    "context"
    "time"
    
    "github.com/sglang/sgl-router-go-client/internal/client"
    "github.com/sglang/sgl-router-go-client/internal/grpc"
    "github.com/sglang/sgl-router-go-client/internal/tokenizer"
)

func main() {
    // 创建客户端（推荐：使用目录路径）
    cfg := client.Config{
        TokenizerPath: "/path/to/tokenizer",  // 目录路径，自动发现 tokenizer.json 和 tokenizer_config.json
        GRPCEndpoint:  "grpc://localhost:20000",
        Timeout:       30 * time.Second,
    }
    
    cl, err := client.NewClient(cfg)
    if err != nil {
        panic(err)
    }
    defer cl.Close()
    
    // 准备消息
    messages := []tokenizer.Message{
        {Role: "user", Content: "Hello!"},
    }
    
    // 生成选项
    options := grpc.DefaultGenerationOptions()
    maxTokens := int32(100)
    options.MaxNewTokens = &maxTokens
    
    // 生成
    ctx := context.Background()
    result, err := cl.Generate(ctx, messages, options)
    if err != nil {
        panic(err)
    }
    
    // 处理结果
    fmt.Printf("Generated %d tokens\n", len(result.TokenIds))
}
```

## 重要提示

⚠️ **Protobuf 代码生成**

在首次使用前，必须生成 protobuf 代码：

```bash
make proto
```

生成后，需要更新 `internal/grpc/client.go` 中的导入路径，将 stub 类型替换为实际的 protobuf 类型。

## Makefile 命令

- `make setup` - 设置开发环境（安装依赖 + 生成 protobuf）
- `make proto` - 生成 protobuf 代码
- `make build` - 构建 sglang-client 程序
- `make run` - 构建并运行 sglang-client（需要 TOKENIZER 和 ENDPOINT 参数）
- `make example` - `make run` 的别名（向后兼容）
- `make test` - 运行测试
- `make clean` - 清理构建产物
- `make fmt` - 格式化代码

## 配置

### 环境变量

- `SGL_TOKENIZER_PATH`: Tokenizer 文件路径
- `SGL_GRPC_ENDPOINT`: gRPC 服务端点

### 配置文件

创建 `config.json`（参考 `config.example.json`）:

```json
{
  "tokenizer": {
    "path": "/path/to/tokenizer"
  },
  "grpc": {
    "endpoint": "grpc://localhost:20000",
    "timeout": 30
  },
  "generation": {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": -1,
    "max_new_tokens": 100,
    "skip_special_tokens": true,
    "stream": false
  }
}
```

## 开发状态

⚠️ **当前状态**: 项目结构已创建，但需要：

1. ✅ 生成 protobuf 代码 (`make proto`)
2. ✅ 更新 `internal/grpc/client.go` 中的导入和类型
3. ✅ 测试 FFI 调用
4. ✅ 测试 gRPC 连接

## 常见问题

### 找不到库文件

确保设置了正确的库路径：

```bash
export LD_LIBRARY_PATH="$(pwd)/../../target/release:$LD_LIBRARY_PATH"
```

### Protobuf 代码未生成

运行：

```bash
make proto
```

然后更新 `internal/grpc/client.go` 中的导入路径。

### gRPC 连接失败

检查：
1. SGLang 服务是否运行
2. 端点地址是否正确
3. 防火墙设置

## 许可证

与 SGLang Router 项目相同。

## 贡献

欢迎提交 Issue 和 Pull Request！
