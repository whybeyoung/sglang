# 使用指南

## 命令行参数

所有示例程序都支持通过命令行参数传入 tokenizer path 和 gRPC endpoint。

### 基本用法

```bash
# 主程序（推荐：使用目录路径，自动发现 tokenizer.json 和 tokenizer_config.json）
./bin/sglang-client -tokenizer /path/to/tokenizer -endpoint grpc://localhost:20000

# 也可以直接指定 tokenizer.json 文件
./bin/sglang-client -tokenizer /path/to/tokenizer.json -endpoint grpc://localhost:20000

# 简单示例
go run examples/simple/main.go -tokenizer /path/to/tokenizer -endpoint grpc://localhost:20000

# 对话示例
go run examples/chat/main.go -tokenizer /path/to/tokenizer -endpoint grpc://localhost:20000
```

### 参数说明

#### 主程序 (cmd/sglang-client/main.go)

- `-tokenizer`: Tokenizer 目录或文件路径（必需）
  - **推荐**：指定目录（如 `/path/to/tokenizer`），会自动发现：
    - `tokenizer.json`（必需）
    - `tokenizer_config.json`（用于 chat template，可选）
  - 也可以直接指定 `tokenizer.json` 文件路径
- `-endpoint`: gRPC 端点地址（必需）
- `-config`: 配置文件路径（可选）
- `-help`: 显示帮助信息

#### 简单示例 (examples/simple/main.go)

- `-tokenizer`: Tokenizer 文件路径（必需）
- `-endpoint`: gRPC 端点地址（必需）
- `-prompt`: 生成提示文本（可选，默认："Hello, how are you?"）
- `-help`: 显示帮助信息

#### 对话示例 (examples/chat/main.go)

- `-tokenizer`: Tokenizer 文件路径（必需）
- `-endpoint`: gRPC 端点地址（必需）
- `-help`: 显示帮助信息

### 配置优先级

配置按以下优先级应用（高优先级覆盖低优先级）：

1. **命令行参数**（最高优先级）
2. **环境变量**
3. **配置文件**（如果使用 `-config`）
4. **默认配置**（最低优先级）

### 使用示例

#### 1. 仅使用命令行参数（推荐：使用目录路径）

```bash
# 推荐：指定目录，自动发现 tokenizer 文件
./bin/sglang-client \
  -tokenizer /path/to/tokenizer \
  -endpoint grpc://localhost:20000

# 也可以直接指定 tokenizer.json 文件
./bin/sglang-client \
  -tokenizer /path/to/tokenizer.json \
  -endpoint grpc://localhost:20000
```

#### 2. 使用环境变量

```bash
# 推荐：使用目录路径
export SGL_TOKENIZER_PATH=/path/to/tokenizer
export SGL_GRPC_ENDPOINT=grpc://localhost:20000
./bin/sglang-client
```

#### 3. 使用配置文件

```bash
./bin/sglang-client -config config.json
```

#### 4. 混合使用（配置文件 + 命令行覆盖）

```bash
./bin/sglang-client \
  -config config.json \
  -endpoint grpc://localhost:30000  # 覆盖配置文件中的 endpoint
```

#### 5. 简单示例（带自定义提示）

```bash
go run examples/simple/main.go \
  -tokenizer /path/to/tokenizer.json \
  -endpoint grpc://localhost:20000 \
  -prompt "What is artificial intelligence?"
```

### 查看帮助

所有程序都支持 `-help` 或 `-h` 参数：

```bash
./bin/sglang-client -help
go run examples/simple/main.go -help
go run examples/chat/main.go -help
```

### 错误处理

如果缺少必需参数，程序会显示错误信息并退出：

```bash
$ ./bin/sglang-client
Error: tokenizer path is required. Use -tokenizer flag or set SGL_TOKENIZER_PATH environment variable
```

### 完整示例

```bash
# 1. 构建程序
cd examples/sglang-go-client
make build

# 2. 运行主程序（推荐：使用目录路径）
./bin/sglang-client \
  -tokenizer /models/llama/tokenizer \
  -endpoint grpc://localhost:20000

# 3. 运行简单示例
go run examples/simple/main.go \
  -tokenizer /models/llama/tokenizer \
  -endpoint grpc://localhost:20000 \
  -prompt "Explain quantum computing"

# 4. 运行对话示例
go run examples/chat/main.go \
  -tokenizer /models/llama/tokenizer \
  -endpoint grpc://localhost:20000
```

