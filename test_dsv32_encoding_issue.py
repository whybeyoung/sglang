#!/usr/bin/env python3
"""
测试脚本：分析 V3.2 encoding_dsv32.py 在处理最后一个消息是 assistant 时的问题
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

# 用户提供的消息序列
messages = [
    {"role": "user", "content": "你是谁"},
    {"role": "assistant", "content": "爆火的哈尔滨旅游<end><end>"},
    {"role": "user", "content": "你会做什么"},
    {"role": "assistant", "content": "上班<end><end>"},
    {"role": "user", "content": "没劲"},
    {"role": "assistant", "content": "上班<end><end>"}  # 最后一个消息是 assistant
]

print("=" * 80)
print("测试消息序列:")
print("=" * 80)
for i, msg in enumerate(messages):
    print(f"{i}: {msg['role']}: {msg['content']}")
print()

print("=" * 80)
print("V3.2 encoding_dsv32.py 编码结果 (thinking_mode='chat'):")
print("=" * 80)
try:
    encoded_chat = encode_messages(messages, thinking_mode="chat")
    print(repr(encoded_chat))
    print()
    print("编码后的文本:")
    print(encoded_chat)
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 80)
print("V3.2 encoding_dsv32.py 编码结果 (thinking_mode='thinking'):")
print("=" * 80)
try:
    encoded_thinking = encode_messages(messages, thinking_mode="thinking")
    print(repr(encoded_thinking))
    print()
    print("编码后的文本:")
    print(encoded_thinking)
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 80)
print("分析：")
print("=" * 80)
print("""
问题分析：

1. V3.1 的 chat_template 逻辑：
   - 使用 `is_last_user` 变量跟踪最后一个 user 消息
   - 在最后检查：`{%- if add_generation_prompt and ns.is_last_user and not ns.is_tool %}`
   - 如果最后一个消息是 user，会添加 `<｜Assistant｜>` 前缀，提示模型需要生成回复

2. V3.2 的 encoding_dsv32.py 逻辑：
   - `find_last_user_index()` 找到最后一个 user/developer 消息的索引
   - `render_message()` 处理 assistant 消息时：
     * 如果 `index > last_user_idx` 且 `thinking_mode == "thinking"`，会添加 thinking_part
     * 但**没有**检查是否需要添加生成提示（`<｜Assistant｜>`）
   - 当最后一个消息是 assistant 时，编码结果会直接以 `<｜end▁of▁sentence｜>` 结尾
   - **缺少**让模型知道需要继续生成下一个 assistant 回复的提示

3. 核心问题：
   - V3.2 的编码逻辑假设最后一个消息总是 user，或者通过 `continue_final_message` 机制处理
   - 当最后一个消息是 assistant 时，编码结果不完整，模型不知道需要继续生成
   - V3.1 的模板会在最后检查 `is_last_user`，如果为 True 则添加 assistant 前缀

4. 解决方案：
   - 在 `encode_messages()` 函数最后，检查最后一个消息是否是 user
   - 如果是 user，添加 `<｜Assistant｜>` 前缀（类似 V3.1 的 `add_generation_prompt` 逻辑）
   - 或者修改 `render_message()` 函数，在最后一个 user 消息后添加 assistant 前缀
""")

