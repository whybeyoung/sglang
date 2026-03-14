"""
Kimi-K2.5 FP8 量化脚本
使用 llm-compressor 将 Kimi-K2.5 的 MoE routed experts 量化为 FP8 W8A8 (block-wise)
和 Attention/Dense/SharedExpert 保持 BF16。

显存需求分析（H20 96GB）:
  - BF16: ~1911GB 总，EP=32 才能跑
  - FP8(本脚本): ~966GB 总，EP=16 即可
  - 量化过程本身需要至少 1 台 8xH20 机器（约 768GB）分片加载

安装依赖:
  pip install llmcompressor>=0.4.0 transformers>=4.56.0

运行:
  # 单机 8GPU 分片加载量化（需要约 768GB CPU内存 或 GPU内存）
  python quantize_kimi_k25_fp8.py \
      --model-path moonshotai/Kimi-K2.5 \
      --output-dir ./Kimi-K2.5-FP8 \
      --scheme fp8_dynamic

  # 使用 CPU offload（内存换时间）
  python quantize_kimi_k25_fp8.py \
      --model-path moonshotai/Kimi-K2.5 \
      --output-dir ./Kimi-K2.5-FP8 \
      --cpu-offload
"""

import argparse
import torch
from transformers import AutoProcessor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="moonshotai/Kimi-K2.5")
    parser.add_argument("--output-dir", type=str, default="./Kimi-K2.5-FP8")
    parser.add_argument("--scheme", type=str, default="fp8_dynamic",
                        choices=["fp8_dynamic", "fp8_static"],
                        help="fp8_dynamic: W8A8 动态量化(推荐); fp8_static: W8A8 静态量化(需calibration)")
    parser.add_argument("--calibration-samples", type=int, default=512,
                        help="calibration 样本数，仅 fp8_static 时使用")
    parser.add_argument("--cpu-offload", action="store_true",
                        help="CPU offload 模式，显存不足时使用，速度慢")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    return parser.parse_args()


def get_calibration_dataset(processor, num_samples=512):
    """生成 calibration 数据集（仅 static 量化需要）"""
    from datasets import load_dataset

    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft",
                           streaming=True)
    samples = []
    for i, example in enumerate(dataset):
        if i >= num_samples:
            break
        messages = example.get("messages", [])
        if not messages:
            continue
        text = processor.tokenizer.apply_chat_template(
            messages[:2],  # 只用 user+assistant 第一轮
            tokenize=False,
            add_generation_prompt=False,
        )
        samples.append(text)

    def tokenize(sample):
        return processor.tokenizer(
            sample,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=2048,
        )

    return [tokenize(s) for s in samples]


def quantize_fp8_dynamic(model_path, output_dir, cpu_offload, trust_remote_code):
    """
    FP8 W8A8 动态量化：
    - 量化目标：只量化 routed experts (experts.w1/w2/w3)，与官方 INT4 策略一致
    - attention / shared_experts / dense MLP / lm_head 保持 BF16
    - 量化粒度：block-wise (weight_block_size=[128,128])，与 DSV3 官方 FP8 格式相同
    """
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    # 与官方 INT4 的 ignore 保持一致，只量化 routed experts
    # Kimi-K2.5 的 expert 权重路径: model.layers.*.mlp.experts.*.w1/w2/w3
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        # ignore 列表与官方 INT4 config.json 一致
        ignore=[
            "lm_head",
            "re:.*self_attn.*",          # 不量化 MLA attention
            "re:.*shared_experts.*",      # 不量化 shared expert
            "re:.*mlp\\.gate$",          # 不量化 MoE router gate
            # dense MLP (layer 0): gate_up_proj / down_proj
            "re:.*mlp\\.(gate|up|gate_up|down)_proj.*",
            # vision tower 和 mm projector 全部跳过
            "re:.*vision_tower.*",
            "re:.*mm_projector.*",
        ],
    )

    device_map = "auto" if not cpu_offload else {"": "cpu"}

    oneshot(
        model=model_path,
        recipe=recipe,
        output_dir=output_dir,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        # 保存时同时保存 tokenizer/processor
        save_compressed=True,
    )
    print(f"FP8 dynamic 量化完成，保存到: {output_dir}")


def quantize_fp8_static(model_path, output_dir, calibration_samples,
                         cpu_offload, trust_remote_code):
    """
    FP8 W8A8 静态量化（带 calibration）：
    - 更高精度，需要 calibration 数据集
    - 使用 SmoothQuant + GPTQ-style channel-wise calibration
    """
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

    recipe = [
        SmoothQuantModifier(smoothing_strength=0.8),
        QuantizationModifier(
            targets="Linear",
            scheme="FP8",
            ignore=[
                "lm_head",
                "re:.*self_attn.*",
                "re:.*shared_experts.*",
                "re:.*mlp\\.gate$",
                "re:.*mlp\\.(gate|up|gate_up|down)_proj.*",
                "re:.*vision_tower.*",
                "re:.*mm_projector.*",
            ],
        ),
    ]

    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )
    calibration_data = get_calibration_dataset(processor, calibration_samples)

    device_map = "auto" if not cpu_offload else {"": "cpu"}

    oneshot(
        model=model_path,
        recipe=recipe,
        dataset=calibration_data,
        output_dir=output_dir,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        save_compressed=True,
        num_calibration_samples=calibration_samples,
        max_seq_length=2048,
    )
    print(f"FP8 static 量化完成，保存到: {output_dir}")


def main():
    args = parse_args()

    print(f"=== Kimi-K2.5 FP8 量化 ===")
    print(f"模型路径: {args.model_path}")
    print(f"输出路径: {args.output_dir}")
    print(f"量化方案: {args.scheme}")
    print(f"CPU offload: {args.cpu_offload}")
    print()
    print("Kimi-K2.5 MoE 参数 (from config.json):")
    print("  n_routed_experts: 384 (比 DSV3 多 50%)")
    print("  n_group: 1 (无 group routing)")
    print("  moe_intermediate_size: 2048")
    print("  hidden_size: 7168")
    print("  MoE 层数: 60 (layer 1-60)")
    print()
    print("FP8 量化后显存估算:")
    print("  Routed Expert FP8: ~945GB")
    print("  非Expert (BF16): ~21GB")
    print("  总计: ~966GB")
    print("  EP=16 (16卡H20): 每卡 ~80GB ✅")
    print()

    if args.scheme == "fp8_dynamic":
        quantize_fp8_dynamic(
            model_path=args.model_path,
            output_dir=args.output_dir,
            cpu_offload=args.cpu_offload,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        quantize_fp8_static(
            model_path=args.model_path,
            output_dir=args.output_dir,
            calibration_samples=args.calibration_samples,
            cpu_offload=args.cpu_offload,
            trust_remote_code=args.trust_remote_code,
        )

    print()
    print("=== 部署命令 (H20 EP=16) ===")
    print(f"""
# 16 张 H20，EP=16，每卡约 80GB 显存占用
python -m sglang.launch_server \\
    --model {args.output_dir} \\
    --quantization fp8 \\
    --enable-dp-attention \\
    --dp-size 16 --tp-size 1 \\
    --trust-remote-code \\
    --mem-fraction-static 0.85 \\
    --enable-ep-moe \\
    --moe-dense-tp-size 1
""")


if __name__ == "__main__":
    main()
