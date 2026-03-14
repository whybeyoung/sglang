#!/usr/bin/env python3
"""
测试脚本：验证当最后一个消息是 assistant 且 continue_final_message=False 时的行为
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
find_last_user_index = encoding_dsv32.find_last_user_index

# 测试用例 1：最后一个消息是 assistant
messages_with_assistant_last = [
    {"role": "user", "content": "你是谁"},
    {"role": "assistant", "content": "爆火的哈尔滨旅游<end><end>"},
    {"role": "user", "content": "你会做什么"},
    {"role": "assistant", "content": "上班<end><end>"},
    {"role": "user", "content": "没劲"},
    {"role": "assistant", "content": "上班<end><end>"}  # 最后一个消息是 assistant
]

# 测试用例 2：最后一个消息是 user
messages_with_user_last = [
    {"role": "user", "content": "你是谁"},
    {"role": "assistant", "content": "爆火的哈尔滨旅游<end><end>"},
    {"role": "user", "content": "你会做什么"},
    {"role": "assistant", "content": "上班<end><end>"},
    {"role": "user", "content": "没劲"}  # 最后一个消息是 user
]

print("=" * 80)
print("测试用例 1：最后一个消息是 assistant")
print("=" * 80)
print("消息序列:")
for i, msg in enumerate(messages_with_assistant_last):
    print(f"  {i}: {msg['role']}: {msg['content']}")

last_user_idx = find_last_user_index(messages_with_assistant_last)
print(f"\n最后一个 user 消息的索引: {last_user_idx}")
print(f"最后一个消息的索引: {len(messages_with_assistant_last) - 1}")
print(f"最后一个消息的角色: {messages_with_assistant_last[-1]['role']}")

print("\n编码结果 (thinking_mode='chat'):")
try:
    encoded = encode_messages(messages_with_assistant_last, thinking_mode="chat")
    print(repr(encoded))
    print("\n编码后的文本:")
    print(encoded)
    print(f"\n编码结果是否以 '</think>' 结尾: {encoded.endswith('</think>')}")
    print(f"编码结果是否以 '<｜end▁of▁sentence｜>' 结尾: {encoded.endswith('<｜end▁of▁sentence｜>')}")
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("测试用例 2：最后一个消息是 user")
print("=" * 80)
print("消息序列:")
for i, msg in enumerate(messages_with_user_last):
    print(f"  {i}: {msg['role']}: {msg['content']}")

last_user_idx = find_last_user_index(messages_with_user_last)
print(f"\n最后一个 user 消息的索引: {last_user_idx}")
print(f"最后一个消息的索引: {len(messages_with_user_last) - 1}")
print(f"最后一个消息的角色: {messages_with_user_last[-1]['role']}")

print("\n编码结果 (thinking_mode='chat'):")
try:
    encoded = encode_messages(messages_with_user_last, thinking_mode="chat")
    print(repr(encoded))
    print("\n编码后的文本:")
    print(encoded)
    print(f"\n编码结果是否以 '</think>' 结尾: {encoded.endswith('</think>')}")
    print(f"编码结果是否以 '<｜end▁of▁sentence｜>' 结尾: {encoded.endswith('<｜end▁of▁sentence｜>')}")
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("问题分析")
print("=" * 80)
print("""
当最后一个消息是 assistant 且 continue_final_message=False 时：

1. 当前行为：
   - _handle_continue_final_message() 不会移除最后一个 assistant 消息
   - encode_messages() 会编码所有消息，包括最后一个 assistant
   - 编码结果以 '</think>上班<end><end>' 结尾
   - 模型看到这个结尾，可能认为对话已经结束，不会生成新的回复

2. 期望行为：
   - 自动移除最后一个 assistant 消息（如果 continue_final_message=False）
   - 或者抛出错误，提示用户移除最后一个 assistant 消息
   - 确保最后一个消息是 user，这样模型才能生成新的回复

3. 解决方案：
   - 修改 _handle_continue_final_message()，总是移除最后一个 assistant 消息
   - 如果 continue_final_message=False，记录警告日志
   - 如果 continue_final_message=True，将 assistant 内容作为 prefix 追加
""")

