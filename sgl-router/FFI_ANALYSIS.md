# SGL-Router FFI 分析报告

## 概述

本文档分析了 sgl-router 的 Rust 实现，重点关注请求前处理和后处理功能（工具解析、JSON mode 等），并评估为 Golang 提供 FFI 接口的可行性。

## 一、核心功能分析

### 1.1 请求前处理（Preprocessing）

#### 1.1.1 Tokenization（分词）
- **位置**: `src/tokenizer/`
- **主要功能**:
  - 支持 HuggingFace tokenizer（tokenizer.json）
  - 支持 Tiktoken（OpenAI 模型）
  - 支持 Mock tokenizer（测试用）
- **关键接口**:
  ```rust
  pub trait Tokenizer: Send + Sync {
      fn encode(&self, input: &str) -> Result<Encoding>;
      fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String>;
      fn apply_chat_template(&self, messages: &[Value], params: ChatTemplateParams) -> Result<String>;
  }
  ```
- **工厂方法**: `create_tokenizer_from_file()`, `create_tokenizer_async()`

#### 1.1.2 Chat Template Processing（对话模板处理）
- **位置**: `src/tokenizer/chat_template.rs`
- **主要功能**:
  - 应用模型特定的对话模板（Jinja2 模板）
  - 处理多模态输入（图像、视频、音频）
  - 支持工具定义注入
  - 处理 `continue_final_message` 场景
- **关键函数**: `process_chat_messages()` in `src/routers/grpc/utils.rs:316`

#### 1.1.3 Tool Filtering & Constraint Generation（工具过滤和约束生成）
- **位置**: `src/routers/grpc/utils.rs`
- **主要功能**:
  - 根据 `tool_choice` 过滤可用工具
  - 生成 JSON Schema 约束（用于结构化输出）
  - 支持三种模式：
    - `Function { function }`: 特定函数调用
    - `Required`: 必须调用工具
    - `AllowedTools { tools, mode }`: 允许的工具列表
- **关键函数**:
  - `filter_tools_for_request()`: 过滤工具
  - `generate_tool_constraints()`: 生成 JSON Schema 约束
  - `build_required_array_schema()`: 构建必需工具调用的数组模式

#### 1.1.4 Message Processing（消息处理）
- **位置**: `src/routers/grpc/utils.rs`
- **主要功能**:
  - 处理多模态内容格式（String vs OpenAI format）
  - 转换工具调用参数（字符串 -> JSON 对象）
  - 处理 `reasoning_effort` 等模板参数
- **关键函数**: `process_content_format()`, `process_tool_call_arguments()`

### 1.2 请求后处理（Postprocessing）

#### 1.2.1 Tool Call Parsing（工具调用解析）
- **位置**: `src/tool_parser/`
- **支持的解析器**:
  - **JsonParser**: 纯 JSON 格式（OpenAI 兼容）
  - **LlamaParser**: Llama 格式（`<tool_call>` 标签）
  - **MistralParser**: Mistral 格式
  - **QwenParser**: Qwen 格式
  - **DeepSeekParser**: DeepSeek 格式
  - **PythonicParser**: Python 代码风格
  - **Step3Parser**: Step-3 模型格式
  - **Glm4MoeParser**: GLM4 MoE 格式
  - **KimiK2Parser**: Kimi K2 格式
  - **GptOssParser**: GPT-OSS 格式
- **关键接口**:
  ```rust
  pub trait ToolParser: Send + Sync {
      async fn parse_complete(&self, output: &str) -> ParserResult<(String, Vec<ToolCall>)>;
      async fn parse_incremental(&mut self, chunk: &str, tools: &[Tool]) -> ParserResult<StreamingParseResult>;
      fn has_tool_markers(&self, text: &str) -> bool;
  }
  ```
- **工厂模式**: `ParserFactory` 根据模型名称自动选择解析器

#### 1.2.2 JSON Schema Parsing（JSON Schema 解析）
- **位置**: `src/routers/grpc/utils.rs:528`
- **主要功能**:
  - 解析 JSON Schema 约束的输出（结构化输出）
  - 支持特定函数调用（返回参数对象）
  - 支持必需模式（返回工具调用数组）
- **关键函数**: `parse_json_schema_response()`

#### 1.2.3 Reasoning Parsing（推理内容解析）
- **位置**: `src/reasoning_parser/`
- **支持的解析器**:
  - **DeepSeekR1Parser**: DeepSeek-R1 推理格式
  - **Qwen3Parser**: Qwen3 推理格式
  - **Step3Parser**: Step-3 推理格式
  - **Glm45Parser**: GLM4.5 推理格式
  - **KimiParser**: Kimi 推理格式
- **主要功能**: 分离推理内容和正常文本（`separate_reasoning` 选项）

#### 1.2.4 Stop Sequence Decoding（停止序列解码）
- **位置**: `src/tokenizer/stop.rs`
- **主要功能**:
  - 处理停止序列（字符串和 token ID）
  - 支持部分匹配和提前停止
  - 处理特殊 token 跳过
- **关键类型**: `StopSequenceDecoder`

## 二、当前 FFI 状态

### 2.1 Python 绑定（PyO3）
- **状态**: ✅ 已实现
- **位置**: `src/lib.rs`
- **实现方式**: 使用 `pyo3` crate
- **导出类型**: `Router`, `PolicyType`, `BackendType`, `HistoryBackendType`
- **限制**: 主要面向完整路由器功能，而非单独的前/后处理功能

### 2.2 C FFI 支持
- **状态**: ⚠️ 部分支持
- **Cargo.toml**: 已配置 `crate-type = ["cdylib", "rlib"]`
- **问题**: 
  - 没有 `#[no_mangle]` 标记的 C 函数
  - 没有 `extern "C"` 接口
  - 没有 C 头文件

## 三、为 Golang 提供 FFI 的可行性分析

### 3.1 技术可行性 ✅

#### 优势：
1. **Rust 天然支持 C FFI**
   - Rust 可以导出 C 兼容的函数
   - 使用 `#[no_mangle]` 和 `extern "C"` 即可
   - Golang 的 `cgo` 可以直接调用 C 函数

2. **已有 cdylib 配置**
   - `Cargo.toml` 已配置 `cdylib`，可以编译为动态库

3. **模块化设计**
   - 前处理和后处理功能已经模块化
   - 可以独立导出为 FFI 函数

#### 挑战：
1. **异步处理**
   - 当前实现大量使用 `async/await`
   - FFI 需要同步接口或回调机制
   - 可能需要使用 `tokio::runtime::Handle` 或阻塞运行时

2. **内存管理**
   - Rust 的所有权系统与 C/Golang 的内存模型不同
   - 需要设计清晰的所有权转移机制
   - 可能需要使用 `Box` 和手动释放函数

3. **错误处理**
   - Rust 的 `Result<T, E>` 需要转换为 C 兼容的错误码
   - 需要设计错误码枚举和错误消息传递机制

4. **字符串处理**
   - Rust `String` 与 C `char*` 的转换
   - 需要处理 UTF-8 编码
   - 内存分配和释放需要明确

### 3.2 功能完整性评估

#### 可以导出的核心功能：

1. **Tokenizer 相关** ✅
   - 创建 tokenizer（从文件或 HuggingFace）
   - 编码/解码文本
   - 应用 chat template

2. **Tool Parser 相关** ✅
   - 创建工具解析器
   - 解析完整工具调用
   - 增量解析（流式）

3. **Reasoning Parser 相关** ✅
   - 创建推理解析器
   - 检测和解析推理内容

4. **JSON Schema 相关** ✅
   - 生成工具约束 JSON Schema
   - 解析 JSON Schema 响应

5. **Message Processing 相关** ✅
   - 处理消息格式
   - 过滤工具
   - 处理工具调用参数

#### 需要适配的功能：

1. **异步操作**
   - 需要提供同步包装或回调机制
   - 建议：提供阻塞版本和异步回调版本

2. **复杂类型序列化**
   - `ChatCompletionRequest` 等复杂类型需要 JSON 序列化
   - 建议：使用 JSON 字符串作为接口

3. **错误信息传递**
   - 需要设计错误码和错误消息机制
   - 建议：使用错误码 + 可选错误消息

## 四、推荐的 FFI 接口设计

### 4.1 接口设计原则

1. **使用 JSON 作为数据交换格式**
   - 简化复杂类型的传递
   - 与现有协议兼容

2. **提供同步和异步两种接口**
   - 同步：阻塞调用，简单易用
   - 异步：回调机制，高性能

3. **明确的内存管理**
   - 所有分配的内存由调用者释放
   - 提供明确的释放函数

4. **错误处理**
   - 返回错误码
   - 可选错误消息（需要释放）

### 4.2 示例接口设计

#### 4.2.1 Tokenizer 接口

```rust
// C FFI 接口
#[no_mangle]
pub extern "C" fn sgl_tokenizer_create_from_file(
    path: *const c_char,
    error_out: *mut *mut c_char
) -> *mut TokenizerHandle {
    // 实现
}

#[no_mangle]
pub extern "C" fn sgl_tokenizer_encode(
    handle: *mut TokenizerHandle,
    text: *const c_char,
    token_ids_out: *mut *mut u32,
    token_count_out: *mut usize,
    error_out: *mut *mut c_char
) -> i32 {
    // 实现
}

#[no_mangle]
pub extern "C" fn sgl_tokenizer_apply_chat_template(
    handle: *mut TokenizerHandle,
    messages_json: *const c_char,
    params_json: *const c_char,
    result_out: *mut *mut c_char,
    error_out: *mut *mut c_char
) -> i32 {
    // 实现
}
```

#### 4.2.2 Tool Parser 接口

```rust
#[no_mangle]
pub extern "C" fn sgl_tool_parser_create(
    parser_type: *const c_char,
    error_out: *mut *mut c_char
) -> *mut ToolParserHandle {
    // 实现
}

#[no_mangle]
pub extern "C" fn sgl_tool_parser_parse_complete(
    handle: *mut ToolParserHandle,
    text: *const c_char,
    result_json_out: *mut *mut c_char,
    error_out: *mut *mut c_char
) -> i32 {
    // 返回 JSON: {"normal_text": "...", "tool_calls": [...]}
}
```

#### 4.2.3 工具约束生成接口

```rust
#[no_mangle]
pub extern "C" fn sgl_generate_tool_constraints(
    tools_json: *const c_char,
    tool_choice_json: *const c_char,
    constraint_type_out: *mut *mut c_char,
    constraint_schema_out: *mut *mut c_char,
    error_out: *mut *mut c_char
) -> i32 {
    // 实现
}
```

### 4.3 Golang 绑定示例

```go
//go:build cgo

package sglrouter

/*
#cgo LDFLAGS: -L${SRCDIR}/../target/release -lsglang_router_rs -ldl
#include <stdlib.h>

typedef struct {
    void* handle;
} TokenizerHandle;

TokenizerHandle* sgl_tokenizer_create_from_file(const char* path, char** error_out);
int sgl_tokenizer_encode(TokenizerHandle* handle, const char* text, 
                         uint32_t** token_ids_out, size_t* token_count_out, 
                         char** error_out);
void sgl_tokenizer_free(TokenizerHandle* handle);
void sgl_free_string(char* str);
*/
import "C"
import (
    "unsafe"
    "encoding/json"
)

type Tokenizer struct {
    handle *C.TokenizerHandle
}

func NewTokenizerFromFile(path string) (*Tokenizer, error) {
    cPath := C.CString(path)
    defer C.free(unsafe.Pointer(cPath))
    
    var errorOut *C.char
    handle := C.sgl_tokenizer_create_from_file(cPath, &errorOut)
    
    if handle == nil {
        if errorOut != nil {
            errMsg := C.GoString(errorOut)
            C.sgl_free_string(errorOut)
            return nil, fmt.Errorf("failed to create tokenizer: %s", errMsg)
        }
        return nil, fmt.Errorf("failed to create tokenizer: unknown error")
    }
    
    return &Tokenizer{handle: handle}, nil
}

func (t *Tokenizer) Encode(text string) ([]uint32, error) {
    cText := C.CString(text)
    defer C.free(unsafe.Pointer(cText))
    
    var tokenIds *C.uint32_t
    var tokenCount C.size_t
    var errorOut *C.char
    
    result := C.sgl_tokenizer_encode(t.handle, cText, &tokenIds, &tokenCount, &errorOut)
    
    if result != 0 {
        if errorOut != nil {
            errMsg := C.GoString(errorOut)
            C.sgl_free_string(errorOut)
            return nil, fmt.Errorf("encode failed: %s", errMsg)
        }
        return nil, fmt.Errorf("encode failed: unknown error")
    }
    
    // 转换 C 数组为 Go slice
    tokens := (*[1 << 28]C.uint32_t)(unsafe.Pointer(tokenIds))[:tokenCount:tokenCount]
    result := make([]uint32, tokenCount)
    for i := range result {
        result[i] = uint32(tokens[i])
    }
    
    // 释放 C 分配的内存
    C.free(unsafe.Pointer(tokenIds))
    
    return result, nil
}
```

## 五、实施建议

### 5.1 分阶段实施

#### 阶段 1: 核心功能（MVP）
1. Tokenizer 基础功能
   - 创建 tokenizer
   - 编码/解码
   - Chat template 应用

2. Tool Parser 基础功能
   - 创建解析器
   - 完整解析

#### 阶段 2: 扩展功能
1. 工具约束生成
2. JSON Schema 解析
3. Reasoning parser
4. 消息处理

#### 阶段 3: 高级功能
1. 流式解析支持
2. 异步接口
3. 性能优化

### 5.2 技术选型

1. **FFI 绑定库**:
   - 直接使用 `cgo`（标准库）
   - 或使用 `c-for-go` 生成绑定

2. **错误处理**:
   - 使用错误码枚举
   - 提供错误消息获取函数

3. **内存管理**:
   - 所有分配使用 `malloc`/`free`
   - 提供明确的释放函数

4. **测试**:
   - 单元测试（Go 端）
   - 集成测试（端到端）

### 5.3 注意事项

1. **线程安全**
   - Rust 的 `Arc` 和 `Mutex` 需要正确暴露
   - 确保 FFI 函数是线程安全的

2. **生命周期管理**
   - Handle 的生命周期需要明确
   - 防止悬空指针

3. **性能考虑**
   - 减少不必要的内存拷贝
   - 考虑使用零拷贝技术（如 `bytes::Bytes`）

4. **兼容性**
   - 确保 ABI 稳定性
   - 版本管理策略

## 六、总结

### 6.1 可行性结论

✅ **高度可行**

sgl-router 的前处理和后处理功能设计良好，模块化程度高，非常适合通过 FFI 暴露给 Golang。主要优势：

1. 功能完整且独立
2. 已有 Python 绑定经验
3. Rust 天然支持 C FFI
4. 模块化设计便于导出

### 6.2 主要挑战

1. 异步到同步的转换
2. 内存管理设计
3. 错误处理机制
4. 类型系统映射

### 6.3 推荐方案

1. **使用 JSON 作为数据交换格式**（简化复杂类型）
2. **提供同步接口**（简化调用）
3. **明确的内存管理**（防止泄漏）
4. **分阶段实施**（降低风险）

### 6.4 预期收益

1. **性能**: Rust 实现的性能优势
2. **功能**: 完整的工具解析和 JSON mode 支持
3. **维护**: 单一 Rust 实现，多语言绑定
4. **生态**: 扩展 Golang LLM 服务生态

## 七、下一步行动

1. **设计详细的 FFI API**
   - 定义所有函数签名
   - 设计错误码体系
   - 编写 C 头文件

2. **实现核心功能**
   - Tokenizer FFI
   - Tool Parser FFI
   - 基础测试

3. **Golang 绑定实现**
   - 使用 cgo 或 c-for-go
   - 实现类型安全的包装
   - 编写文档和示例

4. **测试和优化**
   - 单元测试
   - 性能测试
   - 内存泄漏检测



