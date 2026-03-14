#!/usr/bin/env python3
"""
测试脚本：验证 assistant_prefix 的格式问题
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
render_message = encoding_dsv32.render_message
thinking_end_token = encoding_dsv32.thinking_end_token
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
print("测试：原始 assistant 消息的编码格式")
print("=" * 80)

# 模拟完整的消息序列（包含最后一个 assistant）
full_messages = messages.copy()
if full_messages[0]["role"] != "system":
    full_messages.insert(0, {"role": "system", "content": ""})

# 找到最后一个 user 消息的索引
last_user_idx = -1
for idx in range(len(full_messages) - 1, -1, -1):
    if full_messages[idx].get("role") in ["user", "developer"]:
        last_user_idx = idx
        break

print(f"最后一个 user 消息的索引: {last_user_idx}")
print(f"最后一个 assistant 消息的索引: {len(full_messages) - 1}")

# 渲染最后一个 assistant 消息
last_assistant_idx = len(full_messages) - 1
last_assistant_msg = full_messages[last_assistant_idx]

print(f"\n最后一个 assistant 消息: {last_assistant_msg}")

# 渲染这个 assistant 消息
rendered = render_message(
    last_assistant_idx, full_messages, thinking_mode="chat"
)

print(f"\n渲染后的格式: {repr(rendered)}")
print(f"渲染后的文本:\n{rendered}")

print(f"\n是否以 '{thinking_end_token}' 开头: {rendered.startswith(thinking_end_token)}")
print(f"是否以 '{eos_token}' 结尾: {rendered.endswith(eos_token)}")

# 提取 content 部分（去掉 thinking_end_token 和 eos_token）
content_start = rendered.find(thinking_end_token)
if content_start != -1:
    content_start += len(thinking_end_token)
else:
    content_start = 0

content_end = rendered.rfind(eos_token)
if content_end != -1:
    extracted_content = rendered[content_start:content_end]
else:
    extracted_content = rendered[content_start:]

print(f"\n提取的 content 部分: {repr(extracted_content)}")

print("\n" + "=" * 80)
print("问题分析")
print("=" * 80)
print(f"""
当 continue_final_message=True 时：

1. 原始 assistant 消息编码后：
   {repr(rendered)}

2. assistant_prefix（原始 content）：
   {repr(last_assistant_msg['content'])}

3. 问题：
   - assistant_prefix 缺少 '{thinking_end_token}' 前缀
   - assistant_prefix 缺少 '{eos_token}' 后缀（但这是对的，因为要继续生成）
   
4. 解决方案：
   - 在追加 assistant_prefix 时，需要添加 '{thinking_end_token}' 前缀
   - 不需要添加 '{eos_token}' 后缀（因为要继续生成）
""")

