# 构建 Rust Tokenizer FFI

## 快速开始

### 1. 构建 Rust 库

```bash
cd tokenizer-ffi
./build.sh
# 或者
make build
```

### 2. 构建 Go 项目（启用 CGO）

```bash
CGO_ENABLED=1 go build -o sglang-router ./cmd/router
```

## 详细步骤

### 前置要求

1. **Rust 工具链**: 需要安装 Rust 和 Cargo
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **CGO**: Go 需要启用 CGO（默认启用）

### 构建 Rust FFI 库

```bash
cd sgl-router-go/tokenizer-ffi

# 首次构建（会下载依赖）
cargo build --release

# 生成 C 头文件（可选，需要 cbindgen）
cargo install cbindgen
cbindgen --config cbindgen.toml --crate tokenizer-ffi --output tokenizer_ffi.h
```

### 验证构建

构建成功后，应该看到：
- `target/release/libtokenizer_ffi.dylib` (macOS)
- `target/release/libtokenizer_ffi.so` (Linux)

### 集成到 Go 构建

Go 代码会自动尝试链接 Rust 库。如果库不存在或链接失败，会自动回退到 Go tokenizer。

## 故障排除

### 问题: `library not found`

**解决**: 确保 Rust 库已构建
```bash
cd tokenizer-ffi && cargo build --release
```

### 问题: `symbol not found`

**解决**: 检查库文件路径和链接器标志
```go
// 确保路径正确
#cgo LDFLAGS: -L${SRCDIR}/../../tokenizer-ffi/target/release -ltokenizer_ffi
```

### 问题: CGO 编译错误

**解决**: 确保 CGO 已启用
```bash
CGO_ENABLED=1 go build ...
```

### 问题: Rust 编译失败

**解决**: 检查 Rust 版本和依赖
```bash
rustc --version  # 应该 >= 1.70
cargo update
```

## 使用

启动 router 时，如果 Rust FFI tokenizer 可用，会自动使用。否则回退到 Go tokenizer。

查看日志确认使用的 tokenizer：
```
"Using Rust FFI tokenizer" - 使用 Rust FFI
"Rust FFI tokenizer not available, using Go tokenizer" - 使用 Go tokenizer
```

