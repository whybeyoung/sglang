# Kimi-K2.5 FP8 精度问题分析与修复方案

## 问题现象
- TP8PP2 部署 Kimi-K2.5 FP8 模型，GSM8K 精度为 0（Invalid 1.0），输出乱码

## 根因分析

### 1. shared_experts 被 fuse 进 FusedMoE，但 scale 不匹配

在 SGLang 的 DeepSeek-V2 模型实现中，`shared_experts` 默认被 fuse 进 `FusedMoE`（`num_fused_shared_experts=1`），即 `shared_experts.gate_proj.weight` 被 rename 为 `experts.384.gate_proj.weight` 后通过 FusedMoE weight_loader 加载。

当前 checkpoint 中 shared_experts 权重是 **BF16**（没有 `weight_scale_inv`），但 FusedMoE 的参数初始化为 `float8_e4m3fn`（因为 `is_checkpoint_fp8_serialized=True`）。加载时：
- BF16 weight 被 `copy_` 到 FP8 参数：值近似正确（FP8 精度内）
- **但 `weight_scale_inv` 保持初始值 `ones`(1.0)**

虽然对于 BF16→FP8 的值（范围约 [-1, 1]），scale=1.0 在 dequant 时值是近似正确的。但 w8a8_block_fp8 kernel 对 routed experts（scale≈0.001）和 shared expert（scale=1.0）的处理不一致，可能导致数值稳定性问题。

### 2. `ignored_layers` 中的 `shared_experts` 未生效

`is_layer_skipped` 函数有特殊逻辑：当 prefix 包含 `"experts"` 时，只检查 `ignored_layers` 中也包含 `"experts"` 的项，且要求 `prefix in layer_name`。对于 `"shared_experts"` 在 ignored_layers 中：
- `"experts" in "shared_experts"` → True
- `"model.layers.1.mlp.shared_experts.gate_proj" in "shared_experts"` → **False**

所以 `is_layer_skipped` 返回 `False`，shared_experts 不会被跳过。但因为 fuse，shared_experts 走的是 FusedMoE 路径（`Fp8MoEMethod`），而不是 `Fp8LinearMethod`。

### 3. 转换数学正确性已验证
- INT4→BF16 反量化：正确
- BF16→FP8 block-wise 量化：`weight_scale_inv = amax / 448`，mean relative error = 2.2%
- scale 语义与 SGLang 内部 `block_quant_dequant`（`fp8_val * scale`）一致

## 修复方案

### 方案一（推荐）：将 shared_experts 也量化为 FP8

修改 `step2_quant_fp8.py`，让 `is_expert_weight()` 同时匹配 `shared_experts`：

```python
def is_expert_weight(key):
    # 匹配 routed experts 和 shared_experts
    if ("experts." in key or "shared_experts." in key) and key.endswith(".weight"):
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            if proj in key:
                return True
    return False
```

同时从 `config.json` 的 `ignored_layers` 中**移除 `"shared_experts"`**。

### 方案二：禁用 shared_experts fusion

启动时加 `--disable-shared-experts-fusion`，shared_experts 走独立 `DeepseekV2MLP`（`LinearBase`），此时 `is_layer_skipped` 会因为 `"shared_experts" in prefix` 返回 True，使用 `UnquantizedLinearMethod`，BF16 权重正常处理。

但此方案牺牲了 shared_experts fusion 带来的性能优化。

### 需要在量化机上执行的修复脚本

```bash
# 在 26.5.27.241 上执行
python3 /data2/fix_add_shared_fp8.py
```

脚本需要：
1. 读取 bf16_out 中的 shared_experts BF16 权重
2. 执行 FP8 block-wise [128,128] 量化
3. 替换 Kimi-K2.5-FP8 中的 shared_experts weight 为 FP8
4. 添加对应的 weight_scale_inv
5. 更新 model.safetensors.index.json
6. 更新 config.json 从 ignored_layers 中移除 "shared_experts"
