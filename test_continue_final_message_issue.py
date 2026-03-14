#!/usr/bin/env python3
"""
测试脚本：验证 continue_final_message=True 时的问题
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
print("测试用例：最后一个消息是 assistant")
print("=" * 80)
print("原始消息序列:")
for i, msg in enumerate(messages):
    print(f"  {i}: {msg['role']}: {msg['content']}")

print("\n" + "=" * 80)
print("场景 1：continue_final_message=False（移除最后一个 assistant）")
print("=" * 80)
# 模拟 _handle_continue_final_message 的行为
messages_without_last_assistant = messages[:-1]
assistant_prefix = messages[-1]["content"]

print("移除最后一个 assistant 后的消息序列:")
for i, msg in enumerate(messages_without_last_assistant):
    print(f"  {i}: {msg['role']}: {msg['content']}")

print(f"\nassistant_prefix: {repr(assistant_prefix)}")

# 插入 system 消息（模拟 serving_chat.py 的行为）
if messages_without_last_assistant[0]["role"] != "system":
    messages_without_last_assistant.insert(0, {"role": "system", "content": ""})

print("\n编码结果 (thinking_mode='chat'):")
encoded = encode_messages(messages_without_last_assistant, thinking_mode="chat")
print(repr(encoded))
print("\n编码后的文本:")
print(encoded)
print(f"\n编码结果是否以 '{eos_token}' 结尾: {encoded.endswith(eos_token)}")

print("\n" + "=" * 80)
print("场景 2：continue_final_message=True（移除并继续生成）")
print("=" * 80)
print("移除最后一个 assistant 后的消息序列:")
for i, msg in enumerate(messages_without_last_assistant):
    print(f"  {i}: {msg['role']}: {msg['content']}")

print(f"\nassistant_prefix: {repr(assistant_prefix)}")
print(f"\n问题分析：")
print(f"1. assistant_prefix 是原始内容: {repr(assistant_prefix)}")
print(f"2. 这个内容会被 tokenize 后追加到 prompt_ids")
print(f"3. 但是，如果这个内容包含 EOS token ({repr(eos_token)})，模型看到 EOS 就会停止生成")
print(f"4. 检查 assistant_prefix 是否包含 EOS token: {eos_token in assistant_prefix}")

print("\n" + "=" * 80)
print("问题根源分析")
print("=" * 80)
print("""
当 continue_final_message=True 时：

1. 最后一个 assistant 消息被移除
2. 它的原始内容 "上班<end><end>" 被作为 assistant_prefix
3. encode_messages() 编码剩余的消息（最后一个消息是 user）
4. assistant_prefix 被 tokenize 后追加到 prompt_ids

问题：
- assistant_prefix 是原始内容，不包含 EOS token
- 但是，如果用户传入的内容本身包含 EOS token（比如 "上班<end><end>"），
  这个内容被 tokenize 后，可能会被识别为 EOS token
- 或者，在追加 assistant_prefix 时，需要确保它不包含 EOS token

但是，更可能的问题是：
- encode_messages() 编码的消息序列中，最后一个 user 消息后面
  没有添加生成提示（比如 ianhi</think>）
- 所以模型不知道需要继续生成

或者：
- assistant_prefix 的内容 "上班<end><end>" 被追加后，
  模型看到这个内容，可能会认为这是完整的回复，然后停止生成
""")

