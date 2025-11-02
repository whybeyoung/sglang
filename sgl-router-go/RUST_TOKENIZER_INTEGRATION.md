# Rust Tokenizer CGO 集成指南

## 概述

为了解决 Go tokenizer 库（`github.com/sugarme/tokenizer`）对某些正则表达式的兼容性问题，我们创建了一个 Rust FFI wrapper，通过 CGO 调用 Rust 的 `tokenizers` crate。

## 架构

```
Go Code → CGO → Rust FFI Library → HuggingFace tokenizers crate
```

## 目录结构

```
sgl-router-go/
├── tokenizer-ffi/          # Rust FFI 库
│   ├── Cargo.toml          # Rust 项目配置
│   ├── src/
│   │   └── lib.rs          # FFI 实现
│   ├── build.sh            # 构建脚本
│   └── cbindgen.toml       # 头文件生成配置
├── internal/
│   └── tokenizer/
│       └── rust_ffi.go     # Go CGO 绑定
└── Makefile                # 构建系统
```

## 构建步骤

### 1. 构建 Rust 库

```bash
cd tokenizer-ffi
./build.sh
# 或者
make build
```

这会生成：
- `target/release/libtokenizer_ffi.so` (Linux)
- `target/release/libtokenizer_ffi.dylib` (macOS)
- `tokenizer_ffi.h` (C 头文件，如果安装了 cbindgen)

### 2. 更新 Go 代码

确保 `rust_ffi.go` 中的链接路径正确：

```go
//#cgo LDFLAGS: -L${SRCDIR}/../../tokenizer-ffi/target/release -ltokenizer_ffi
```

### 3. 构建 Go 项目

```bash
go build ./cmd/router
```

## 使用

在 factory 中选择使用 Rust FFI tokenizer：

```go
// 在 factory.go 中添加
func CreateTokenizerWithChatTemplate(...) (Tokenizer, error) {
    // 尝试使用 Rust FFI tokenizer
    if rustTokenizer, err := NewRustFFITokenizer(filePath, logger); err == nil {
        return rustTokenizer, nil
    }
    
    // Fallback to Go tokenizer
    return NewHuggingFaceTokenizerWithChatTemplate(...)
}
```

## API 函数

### Rust FFI 函数

- `tokenizer_from_file(path)`: 从文件加载 tokenizer
- `tokenizer_encode(text, output, capacity, len)`: 编码文本到 token IDs
- `tokenizer_decode(ids, len, skip_special, output, capacity, len)`: 解码 token IDs 到文本
- `tokenizer_vocab_size()`: 获取词汇表大小
- `tokenizer_token_to_id(token)`: 获取 token 的 ID

### Go 接口

`RustFFITokenizer` 实现了 `Tokenizer` 接口：

```go
type RustFFITokenizer struct {
    initialized bool
    logger      *zap.Logger
}
```

## 当前状态

✅ **已实现:**
- 基础 FFI 框架
- Encode 功能
- 错误处理
- 内存管理

🚧 **进行中:**
- Decode 功能完善（需要处理 encoding 对象）
- Token-to-ID 映射
- 词汇表大小获取

## 优势

1. **兼容性**: 完全兼容 HuggingFace tokenizers（包括所有正则表达式）
2. **性能**: Rust tokenizer 性能优秀
3. **功能完整**: 支持所有 tokenizers 特性

## 注意事项

1. **CGO 开销**: CGO 调用有性能开销，但对于 tokenization 来说通常可接受
2. **内存管理**: C 字符串需要正确分配/释放
3. **线程安全**: 使用 Mutex 保护全局 tokenizer 实例
4. **编译依赖**: 需要 Rust 工具链和 CGO 支持

## 故障排除

### 编译错误: library not found

确保 Rust 库已构建，并且链接路径正确。

### 运行时错误: symbol not found

检查库文件是否在正确位置，以及链接器标志是否正确。

### Decode 失败

当前 decode 实现可能不完全，需要进一步完善 encoding 对象的处理。

## 下一步

1. 完善 decode 实现
2. 添加单元测试
3. 性能基准测试
4. 集成到主构建流程

