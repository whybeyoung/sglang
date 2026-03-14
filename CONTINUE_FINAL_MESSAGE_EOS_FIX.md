# Continue Final Message EOS Token 问题修复

## 问题描述

当开启 `continue_final_message=True` 时，模型百分百吐出 EOS token，即使开启了续写。

### 测试用例

```json
{
  "messages": [
    {"role": "user", "content": "你是谁"},
    {"role": "assistant", "content": "爆火的哈尔滨旅游<end><end>"},
    {"role": "user", "content": "你会做什么"},
    {"role": "assistant", "content": "上班<end><end>"},
    {"role": "user", "content": "没劲"},
    {"role": "assistant", "content": "上班<end><end>"}
  ],
  "continue_final_message": true
}
```

## 问题分析

### 当前编码流程

1. **移除最后一个 assistant 消息**：
   - `messages` 不包含最后一个 assistant 消息
   - `assistant_prefix = "上班<end><end>"`

2. **`encode_messages()` 编码**：
   ```
   


   你是谁

   </think>
   爆火的哈尔滨旅游<end><end>
   

   你会做什么

   </think>
   上班<end><end>
   

   没劲

   </think>
   ```
   - 编码结果以 `</think>` 结尾（最后一个 user 消息）
   - **没有** EOS token `</think>` 结尾

3. **追加 `assistant_prefix`**：
   ```
   ...`</think>`上班<end><end>
   ```

### 问题根源

**关键问题**：
- 编码结果以 `</think>` 结尾，表示需要生成新的 assistant 回复
- 但是，我们想要的是继续生成，而不是生成新的回复
- 所以，应该移除编码结果末尾的 `</think>`，然后追加 `assistant_prefix`

**为什么模型会立即吐出 EOS**：
1. 编码结果以 `</think>` 结尾，表示需要生成新的 assistant 回复
2. 但是，`assistant_prefix` 被追加后，格式变成：`...`</think>`上班<end><end>`
3. 模型看到 `</think>` 后，可能会认为需要生成新的回复
4. 但是，`assistant_prefix` 的内容 `"上班<end><end>"` 包含 `<end><end>`，这可能被 tokenizer 识别为 EOS token
5. 或者，模型看到这个内容后，可能会认为应该停止生成

## 解决方案

### 修复方案：移除编码结果末尾的 `</think>`

当 `continue_final_message=True` 时，需要移除编码结果末尾的 `</think>`，然后追加 `assistant_prefix`。

**修改位置**：`python/sglang/srt/entrypoints/openai/serving_chat.py` 第 391-398 行

**修改前**：
```python
real_input = encode_messages(messages, thinking_mode=thinking_mode)
prompt_ids = self.tokenizer_manager.tokenizer.encode(real_input)

# Append assistant prefix if continue_final_message is enabled
if assistant_prefix:
    prompt_ids = self._append_assistant_prefix_to_prompt_ids(
        prompt_ids, assistant_prefix
    )
```

**修改后**：
```python
real_input = encode_messages(messages, thinking_mode=thinking_mode)
prompt_ids = self.tokenizer_manager.tokenizer.encode(real_input)

# Append assistant prefix if continue_final_message is enabled
if assistant_prefix:
    # For V3.2 encoding, if the prompt ends with </think> (thinking_end_token),
    # we need to remove it before appending the assistant_prefix.
    # This is because </think> indicates "generate new assistant reply",
    # but we want to continue from the assistant_prefix instead.
    if self.use_dpsk_v32_encoding:
        from sglang.srt.entrypoints.openai.encoding_dsv32 import thinking_end_token
        # Check if prompt ends with thinking_end_token
        if real_input.endswith(thinking_end_token):
            # Remove the thinking_end_token from prompt_ids
            # Find the token IDs for thinking_end_token
            thinking_end_token_ids = self.tokenizer_manager.tokenizer.encode(thinking_end_token)
            # Remove BOS token if present
            if thinking_end_token_ids and thinking_end_token_ids[0] == self.tokenizer_manager.tokenizer.bos_token_id:
                thinking_end_token_ids = thinking_end_token_ids[1:]
            # Remove thinking_end_token from the end of prompt_ids
            if prompt_ids[-len(thinking_end_token_ids):] == thinking_end_token_ids:
                prompt_ids = prompt_ids[:-len(thinking_end_token_ids)]
    
    prompt_ids = self._append_assistant_prefix_to_prompt_ids(
        prompt_ids, assistant_prefix
    )
```

## 修复后的流程

### 修复后的编码流程

1. **移除最后一个 assistant 消息**：
   - `messages` 不包含最后一个 assistant 消息
   - `assistant_prefix = "上班<end><end>"`

2. **`encode_messages()` 编码**：
   ```
   


   你是谁

   </think>
   爆火的哈尔滨旅游<end><end>
   

   你会做什么

   </think>
   上班<end><end>
   

   没劲

   </think>
   ```
   - 编码结果以 `</think>` 结尾

3. **移除 `</think>`**：
   ```
   


   你是谁

   </think>
   爆火的哈尔滨旅游<end><end>
   

   你会做什么

   </think>
   上班<end><end>
   

   没劲

   ```
   - 移除编码结果末尾的 `</think>`

4. **追加 `assistant_prefix`**：
   ```
   ...`</think>`上班<end><end>
   ```
   - 追加 `assistant_prefix` 到 `prompt_ids`
   - 模型会从 "上班<end><end>" 继续生成

## 测试验证

### 测试用例

```python
messages = [
    {"role": "user", "content": "没劲"},
    {"role": "assistant", "content": "上班<end><end>"}
]

# continue_final_message=True

# 期望结果：
# 1. 移除最后一个 assistant 消息
# 2. encode_messages() 编码剩余的消息（最后一个消息是 user）
# 3. 移除编码结果末尾的 `</think>`
# 4. 追加 assistant_prefix 到 prompt_ids
# 5. 模型会从 "上班<end><end>" 继续生成，而不是立即吐出 EOS
```

## 总结

### 问题根源

1. **编码结果以 `</think>` 结尾**：表示需要生成新的 assistant 回复
2. **但是，我们想要的是继续生成**：从 `assistant_prefix` 继续生成
3. **所以，需要移除 `</think>`**：然后追加 `assistant_prefix`

### 修复方案

1. **检查编码结果是否以 `</think>` 结尾**
2. **如果是，移除 `</think>` 的 token IDs**
3. **然后追加 `assistant_prefix` 的 token IDs**
4. **模型会从 `assistant_prefix` 继续生成**

### 修改的文件

- `python/sglang/srt/entrypoints/openai/serving_chat.py`
  - 第 391-415 行：添加移除 `</think>` 的逻辑

