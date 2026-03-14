#!/usr/bin/env python3
"""
测试脚本：完整测试 continue_final_message=True 时的编码流程
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

# 直接导入 encoding_dsv32 模块
import importlib.util
spec = importlib.util.spec_from_file_location(
    "encoding_dsv32",
    os.path.join(os.path.dirname(__file__), "python/sglang/srt/entrypoints/openai/encoding_dsv32.py")
)
encoding_dsv32 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(encoding_dsv32)

encode_messages = encoding_dsv32.encode_messages
eos_token = encoding_dsv32.eos_token
thinking_end_token = encoding_dsv32.thinking_end_token

# 测试用例：最后一个消息是 assistant
messages = [
    {"role": "user", "content": "你是谁"},
    {"role": "assistant", "content": "爆火的哈尔滨旅游<end><end>"},
    {"role": "user", "content": "你会做什么"},
    {"role": "assistant", "content": "上班<end><end>"},
    {"role": "user", "content": "没劲"},
    {"role": "assistant", "content": "上班<end><end>"}  # 最后一个消息是 assistant
]

print("=" * 80)
print("测试：continue_final_message=True 时的完整编码流程")
print("=" * 80)

# 步骤 1：移除最后一个 assistant 消息
messages_without_last_assistant = messages[:-1]
assistant_prefix = messages[-1]["content"]

print("步骤 1：移除最后一个 assistant 消息")
print(f"移除后的消息序列:")
for i, msg in enumerate(messages_without_last_assistant):
    print(f"  {i}: {msg['role']}: {msg['content']}")
print(f"\nassistant_prefix: {repr(assistant_prefix)}")

# 步骤 2：插入 system 消息（模拟 serving_chat.py 的行为）
if messages_without_last_assistant[0]["role"] != "system":
    messages_without_last_assistant.insert(0, {"role": "system", "content": ""})

print("\n步骤 2：插入 system 消息")
print(f"插入后的消息序列:")
for i, msg in enumerate(messages_without_last_assistant):
    print(f"  {i}: {msg['role']}: {msg['content']}")

# 步骤 3：编码消息
print("\n步骤 3：编码消息")
encoded = encode_messages(messages_without_last_assistant, thinking_mode="chat")
print(f"编码结果: {repr(encoded)}")
print(f"\n编码后的文本:")
print(encoded)

print(f"\n编码结果是否以 '{thinking_end_token}' 结尾: {encoded.endswith(thinking_end_token)}")
print(f"编码结果是否以 '{eos_token}' 结尾: {encoded.endswith(eos_token)}")

# 步骤 4：模拟追加 assistant_prefix
print("\n步骤 4：模拟追加 assistant_prefix")
print(f"assistant_prefix: {repr(assistant_prefix)}")
print(f"\n如果直接追加 assistant_prefix，最终格式会是:")
final_format = encoded + assistant_prefix
print(f"  {repr(final_format)}")
print(f"\n最终格式的文本:")
print(final_format)

print("\n" + "=" * 80)
print("问题分析")
print("=" * 80)
print(f"""
当 continue_final_message=True 时：

1. encode_messages() 编码结果：
   - 以 '{thinking_end_token}' 结尾（最后一个 user 消息）
   - 没有 '{eos_token}' 结尾

2. assistant_prefix：
   - 原始 content: {repr(assistant_prefix)}
   - 不包含 '{thinking_end_token}' 前缀
   - 不包含 '{eos_token}' 后缀

3. 如果直接追加 assistant_prefix：
   - 最终格式: {repr(final_format)}
   - 问题：编码结果以 '{thinking_end_token}' 结尾，然后直接追加 assistant_prefix
   - 这可能导致格式不正确

4. 可能的解决方案：
   - 检查编码结果是否以 '{thinking_end_token}' 结尾
   - 如果是，可能需要移除 '{thinking_end_token}'，然后追加 assistant_prefix
   - 或者，确保 assistant_prefix 的格式正确
""")

