# Tokenizer 文件说明

## 概述

Rust router 在加载 tokenizer 时，主要使用 `tokenizer.json`，而 `tokenizer_config.json` 主要用于读取 chat template（对话模板）。

## 文件用途

### 1. `tokenizer.json`（主要文件）

**用途**：完整的 tokenizer 定义文件
- 包含所有词汇表（vocabulary）
- 包含 tokenizer 的完整配置
- 这是创建 tokenizer 实例的**主要文件**

**加载逻辑**：
- 如果指定的是目录，优先查找 `tokenizer.json`
- 如果指定的是文件路径且扩展名为 `.json`，直接加载

### 2. `tokenizer_config.json`（辅助文件）

**主要用途**：读取 chat template（对话模板）
- 当加载 `tokenizer.json` 时，会自动在同目录查找 `tokenizer_config.json`
- 从 `tokenizer_config.json` 中提取 `chat_template` 字段
- 如果找到，会使用其中的 Jinja2 模板用于格式化对话消息

**备选用途**：作为 tokenizer 文件的备选
- 仅在从 HuggingFace Hub 下载时，如果找不到 `tokenizer.json`，才会尝试使用 `tokenizer_config.json` 创建 tokenizer
- 本地文件加载时，**不会**使用 `tokenizer_config.json` 作为主要的 tokenizer 文件

### 3. `vocab.json`（备选文件）

**用途**：仅作为从 HuggingFace Hub 下载时的备选
- 如果找不到 `tokenizer.json` 和 `tokenizer_config.json`，会尝试使用 `vocab.json`
- 本地文件加载时不会使用

## 代码实现分析

### 目录加载逻辑（`factory.rs`）

```rust
// 如果路径是目录，优先查找 tokenizer.json
if path.is_dir() {
    let tokenizer_json = path.join("tokenizer.json");
    if tokenizer_json.exists() {
        // 使用 tokenizer.json
        // 会自动查找同目录的 tokenizer_config.json 读取 chat template
        return create_tokenizer_with_chat_template(...);
    }
    // 如果找不到，返回错误
    return Err(...);
}
```

### Chat Template 加载逻辑（`huggingface.rs`）

```rust
// 尝试从 tokenizer_config.json 加载 chat template
fn load_chat_template(tokenizer_path: &str) -> Option<String> {
    let path = std::path::Path::new(tokenizer_path);
    let dir = path.parent()?;
    let config_path = dir.join("tokenizer_config.json");
    
    if config_path.exists() {
        // 从 tokenizer_config.json 中提取 chat_template 字段
        return load_chat_template_from_config(config_path.to_str()?);
    }
    None
}
```

### HuggingFace Hub 下载逻辑（`factory.rs`）

```rust
// 从 HuggingFace 下载时，如果找不到 tokenizer.json
if !tokenizer_path.exists() {
    // 尝试其他常见文件名
    let possible_files = ["tokenizer_config.json", "vocab.json"];
    for file_name in &possible_files {
        let file_path = cache_dir.join(file_name);
        if file_path.exists() {
            // 使用备选文件
            return create_tokenizer_with_chat_template(...);
        }
    }
}
```

## 实际使用场景

### 场景 1：本地目录（**推荐**）

```
tokenizer/
├── tokenizer.json          # 主要文件（必需）
├── tokenizer_config.json   # 用于读取 chat template（可选）
└── vocab.json             # 通常不需要
```

**使用方式**：
```bash
# 指定目录，会自动查找 tokenizer.json 和 tokenizer_config.json
-tokenizer /path/to/tokenizer
```

**优势**：
- ✅ 自动发现所有相关文件
- ✅ 自动加载 chat template
- ✅ 符合 HuggingFace 标准目录结构
- ✅ 更灵活，易于管理

### 场景 2：仅指定 tokenizer.json

```
tokenizer.json  # 单独文件
```

**使用方式**：
```bash
-tokenizer /path/to/tokenizer.json
```

如果同目录有 `tokenizer_config.json`，会自动读取其中的 chat template。

**适用场景**：
- 只有单个 tokenizer.json 文件
- 不需要 chat template
- 文件不在标准目录结构中

### 场景 3：从 HuggingFace Hub 下载

如果模型没有 `tokenizer.json`，会按以下顺序尝试：
1. `tokenizer.json`（优先）
2. `tokenizer_config.json`（备选）
3. `vocab.json`（最后备选）

## 重要提示

1. **主要文件是 `tokenizer.json`**
   - 这是创建 tokenizer 实例的必需文件
   - 包含完整的词汇表和配置

2. **`tokenizer_config.json` 主要用于 chat template**
   - 不是创建 tokenizer 的主要文件
   - 仅用于提取对话模板
   - 如果不需要 chat template，可以没有这个文件

3. **目录 vs 文件路径**
   - **推荐：指定目录** - 自动发现机制更完善
   - 指定文件：直接加载该文件

4. **Chat Template 优先级**
   - 显式提供的 chat template 路径（最高优先级）
   - 从 `tokenizer_config.json` 自动发现
   - 从 `chat_template.jinja` 或 `chat_template.json` 自动发现
   - 无 chat template（最低优先级）

## 最佳实践

### ✅ 推荐方式：使用目录路径

```bash
# Go 代码中
-tokenizer /path/to/tokenizer

# 或环境变量
export SGL_TOKENIZER_PATH=/path/to/tokenizer
```

**目录结构**：
```
/path/to/tokenizer/
├── tokenizer.json          # 主要 tokenizer 文件
├── tokenizer_config.json   # 包含 chat_template
└── vocab.json              # 通常不需要
```

**优势**：
- 自动发现所有文件
- 自动加载 chat template
- 更符合标准实践
- 易于维护和更新

### ⚠️ 备选方式：直接指定文件

```bash
# 如果只有 tokenizer.json 文件
-tokenizer /path/to/tokenizer.json
```

**适用场景**：
- 只有单个文件
- 不需要 chat template
- 文件不在标准目录中

## 示例

### 示例 1：完整配置（推荐）

```bash
# 指定目录（会自动查找 tokenizer.json 和 tokenizer_config.json）
./bin/sglang-client \
  -tokenizer /path/to/tokenizer \
  -endpoint grpc://localhost:20000
```

目录结构：
```
/path/to/tokenizer/
├── tokenizer.json          # 主要 tokenizer 文件
├── tokenizer_config.json   # 包含 chat_template
└── vocab.json              # 通常不需要
```

### 示例 2：仅 tokenizer.json

```bash
# 直接指定 tokenizer.json 文件
./bin/sglang-client \
  -tokenizer /path/to/tokenizer.json \
  -endpoint grpc://localhost:20000
```

如果同目录有 `tokenizer_config.json`，会自动读取 chat template。

### 示例 3：使用项目中的 tokenizer 目录

```bash
# 使用 examples/tokenizer 目录
./bin/sglang-client \
  -tokenizer examples/tokenizer \
  -endpoint grpc://localhost:20000
```

## 总结

- ✅ **推荐**：使用目录路径（`/path/to/tokenizer`）
  - 自动发现 `tokenizer.json`
  - 自动发现并加载 `tokenizer_config.json` 中的 chat template
  - 更符合标准实践

- ⚠️ **备选**：直接指定文件路径（`/path/to/tokenizer.json`）
  - 适用于只有单个文件的情况
  - 如果同目录有 `tokenizer_config.json`，仍会自动读取

- 📝 **文件作用**：
  - `tokenizer.json`：主要 tokenizer 文件（必需）
  - `tokenizer_config.json`：主要用于 chat template（可选）
  - `vocab.json`：仅 HuggingFace 下载时使用（备选）
