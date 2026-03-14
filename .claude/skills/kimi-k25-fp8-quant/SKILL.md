---
name: kimi-k25-fp8-quant
description: Kimi-K2.5 BF16->FP8 block-wise W8A8 量化完整流程。包含机器连接、量化策略、历史踩坑、SGLang加载验证。量化源为纯BF16模型。
---

# Kimi-K2.5 FP8 量化 Skill（BF16 → FP8，最终正确版本）

## 机器连接信息

| 机器 | 用途 | 端口 | 密码 |
|------|------|------|------|
| 241 量化容器 | 量化脚本执行 | `5022`（经跳板机） | `Aipaasxylx1.t!@#` |
| 241 宿主机 | ossutil 上传 | `22`（经跳板机） | `Aipaasxylx1.t!@#` |
| 跳板机 | 中转 | `10.104.102.78:30022` | 免密 |

**⚠️ 关键区分**：
- **端口 5022** = Docker 量化容器，执行量化脚本
- **端口 22** = 宿主机，执行 ossutil 上传
- 两者共享同一个 `/data2` 目录（bind mount）

```bash
# 量化容器（5022）
sshpass -p 'Aipaasxylx1.t!@#' ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 \
  -o ServerAliveInterval=30 -p 5022 -o ProxyJump=root@10.104.102.78:30022 root@26.5.27.241 "<cmd>"

# 宿主机（22）- 用于 ossutil
sshpass -p 'Aipaasxylx1.t!@#' ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 \
  -o ServerAliveInterval=30 -p 22 -o ProxyJump=root@10.104.102.78:30022 root@26.5.27.241 "<cmd>"

# 上传脚本到量化容器
sshpass -p 'Aipaasxylx1.t!@#' scp -o StrictHostKeyChecking=no -P 5022 \
  -o ProxyJump=root@10.104.102.78:30022 <本地文件> root@26.5.27.241:<远程路径>
```

> SSH 长时间不操作会断连，加 `-o ServerAliveInterval=30` 保持心跳。

---

## 关键路径

- **BF16 源模型**：`/data2/Kimi-K2.5-BF16/`（64 shards，纯 BF16，无 index.json）
- **FP8 输出目录**：`/data2/Kimi-K25-FP8-new2/`
- **量化脚本**：`/data2/quant_bf16_to_fp8.py`
- **量化日志**：`/data2/quant_fp8_v2.log`
- **OSS 上传日志**：`/data2/oss_upload_Kimi-K25-FP8-new2.log`

---

## 模型基本信息

### Kimi-K2.5 架构

```
model_type:              kimi_k25（注意：不是 kimi_k2）
architectures:           DeepseekV3ForCausalLM
hidden_size:             7168
moe_intermediate_size:   2048
n_routed_experts:        384
n_shared_experts:        1
num_experts_per_tok:     8
num_hidden_layers:       61
first_k_dense_replace:   1（第0层是 dense MLP）
max_position_embeddings: 262144 (256K)
```

### BF16 源模型结构

```
Shard 01:       layers.0（dense MLP + attn）
Shard 02-61:    layers.1-60（MoE层，每层含384 experts + shared_experts）
Shard 62-63:    mm_projector.*（多模态投影，无 language_model. 前缀）
Shard 64:       vision_tower.*（视觉编码器，无 language_model. 前缀）

Key 前缀：大部分有 "language_model."，vision/mm_projector 无前缀
```

---

## 量化策略（对齐 DeepSeek-V3 / Kimi-K2 官方 W8A8 全量化）

### 量化为 FP8 的层

| 层类型 | key pattern |
|--------|-------------|
| Dense MLP layer0 | `layers.0.mlp.gate_proj/up_proj/down_proj.weight` |
| 所有 MoE routed experts | `mlp.experts.N.gate_proj/up_proj/down_proj.weight` |
| shared_experts | `mlp.shared_experts.gate_proj/up_proj/down_proj.weight` |
| 所有 attention 投影 | `self_attn.q_a_proj/q_b_proj/kv_a_proj_with_mqa/kv_b_proj/o_proj.weight` |

**判断条件**：`key.endswith(".weight") AND tensor.dim()==2 AND not in BF16白名单`

### 保持 BF16 的层

| 层类型 | 匹配方式 |
|--------|---------|
| `layernorm.weight/bias`, `norm.weight/bias` | key 子串匹配 |
| `embed_tokens.weight` | key 子串匹配 |
| `lm_head.weight` | key 子串匹配 |
| `mlp.gate.weight`（MoE router，2D `[384,7168]`） | key 子串匹配 `"mlp.gate.weight"` |
| `rotary_emb.inv_freq` | key 子串匹配 |
| `vision_tower.*` | stripped_key 前缀匹配 |
| `mm_projector.*` | stripped_key 前缀匹配 |
| 1D tensor（bias、norm）| `tensor.dim() != 2` |

---

## 量化脚本核心逻辑

```python
FP8_MAX = 448.0   # torch.finfo(torch.float8_e4m3fn).max
BLOCK = 128

BF16_KEY_PATTERNS = (
    "layernorm.weight", "layernorm.bias",
    "layer_norm.weight", "layer_norm.bias",
    "norm.weight", "norm.bias",
    "embed_tokens.weight",
    "lm_head.weight",
    "mlp.gate.weight",      # MoE router gate [384, 7168]，2D 但不量化
    "rotary_emb.inv_freq",
)

BF16_PREFIXES = (
    "vision_tower.", "mm_projector.",
    "vision_embed.", "image_newline", "multi_modal_projector.",
)

def should_keep_bf16(key: str) -> bool:
    stripped = key.removeprefix("language_model.")
    for pfx in BF16_PREFIXES:
        if stripped.startswith(pfx): return True
    for pat in BF16_KEY_PATTERNS:
        if pat in key: return True
    return False

def is_quantizable_weight(key: str, tensor) -> bool:
    if not key.endswith(".weight"): return False
    if tensor.dim() != 2: return False
    return not should_keep_bf16(key)

def quant_fp8_blockwise(w_bf16):
    w = w_bf16.float()
    rows, cols = w.shape
    # pad 到 128 倍数
    pad_r = (BLOCK - rows % BLOCK) % BLOCK
    pad_c = (BLOCK - cols % BLOCK) % BLOCK
    if pad_r > 0 or pad_c > 0:
        w = F.pad(w, (0, pad_c, 0, pad_r))
    pr, pc = w.shape
    nr, nc = pr // BLOCK, pc // BLOCK
    # 计算每块 amax
    blocks = w.reshape(nr, BLOCK, nc, BLOCK)
    amax = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-12)  # [nr, nc]
    scale_inv = (amax / FP8_MAX).to(torch.float32)          # [nr, nc] float32！
    # 量化
    scale_inv_exp = scale_inv.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
    fp8 = (w / scale_inv_exp).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return fp8[:rows, :cols].contiguous(), scale_inv.contiguous()

# 主循环
for k, v in raw.items():
    new_k = k.removeprefix("language_model.")   # 去掉 language_model. 前缀
    if is_quantizable_weight(k, v):
        fp8_w, scale_inv = quant_fp8_blockwise(v.cpu())
        out_tensors[new_k] = fp8_w                                       # dtype: float8_e4m3fn
        out_tensors[new_k.replace(".weight", ".weight_scale_inv")] = scale_inv  # dtype: float32
    else:
        out_tensors[new_k] = v.cpu()   # 保留 BF16
```

---

## config.json 配置（必须正确）

写入 `text_config.quantization_config`（因为 kimi_k25 有 text_config 嵌套）：

```json
{
  "quant_method": "fp8",
  "activation_scheme": "dynamic",
  "weight_block_size": [128, 128],
  "ignored_layers": ["lm_head"]
}
```

**关键说明**：
- `is_checkpoint_fp8_serialized` 不需要写，SGLang 从 `quant_method=fp8` 自动推断
- `ignored_layers` **只写 `"lm_head"`**，原因见下方踩坑记录
- config.json 写入位置：`cfg["text_config"]["quantization_config"] = fp8_qcfg`

---

## 运行量化

```bash
# 后台运行，约 64 shards × 40s ≈ 43 分钟
sshpass -p 'Aipaasxylx1.t!@#' ssh -o StrictHostKeyChecking=no -p 5022 \
  -o ProxyJump=root@10.104.102.78:30022 root@26.5.27.241 \
  "mkdir -p /data2/Kimi-K25-FP8-new2 && \
   nohup python3 /data2/quant_bf16_to_fp8.py > /data2/quant_fp8_v2.log 2>&1 & echo PID=\$!"

# 监控进度
sshpass -p 'Aipaasxylx1.t!@#' ssh -o StrictHostKeyChecking=no -p 5022 \
  -o ProxyJump=root@10.104.102.78:30022 root@26.5.27.241 \
  "tail -3 /data2/quant_fp8_v2.log && df -h /data2 | tail -1"
```

正常日志：
```
[1/64]  model-00001-of-000064.safetensors: 20 tensors,   fp8=8    bf16=4  (2.0s)   # dense+attn
[2/64]  model-00002-of-000064.safetensors: 2326 tensors, fp8=1160 bf16=6  (40s)    # MoE层
...
[63/64] model-00063-of-000064.safetensors: xx tensors,   fp8=0    bf16=xx (xs)     # mm_projector
[64/64] model-00064-of-000064.safetensors: 329 tensors,  fp8=0    bf16=329 (xs)    # vision_tower
```

**⚠️ 磁盘空间**：BF16 约 1.9T，FP8 约 1T，需要至少 1.2T 空闲空间。量化前先清理旧的 FP8 目录。

---

## OSS 上传（宿主机端口22）

```bash
# 等量化完成后自动触发上传（在宿主机22端口执行）
sshpass -p 'Aipaasxylx1.t!@#' ssh -o StrictHostKeyChecking=no -p 22 \
  -o ProxyJump=root@10.104.102.78:30022 root@26.5.27.241 \
  "nohup ossutil sync --force --jobs=200 \
    /data2/Kimi-K25-FP8-new2/ \
    oss://maas-resource-bj/Kimi-K25-FP8-new2/ \
    -e oss-cn-beijing-internal.aliyuncs.com \
    > /data2/oss_upload_Kimi-K25-FP8-new2.log 2>&1 & echo PID=\$!"
```

---

## 验证量化结果

```python
from safetensors import safe_open
import json

# 1. 验证权重 dtype
with safe_open('/data2/Kimi-K25-FP8-new2/model-00002-of-000064.safetensors', framework='pt') as f:
    # routed expert -> FP8
    assert f.get_tensor('model.layers.1.mlp.experts.0.gate_proj.weight').dtype == torch.float8_e4m3fn
    # shared_experts -> FP8（SGLang 加载时 rename 为 experts.384）
    assert f.get_tensor('model.layers.1.mlp.shared_experts.gate_proj.weight').dtype == torch.float8_e4m3fn
    # scale_inv -> float32
    s = f.get_tensor('model.layers.1.mlp.experts.0.gate_proj.weight_scale_inv')
    assert s.dtype == torch.float32

with safe_open('/data2/Kimi-K25-FP8-new2/model-00001-of-000064.safetensors', framework='pt') as f:
    # attn -> FP8
    assert f.get_tensor('model.layers.0.self_attn.q_a_proj.weight').dtype == torch.float8_e4m3fn
    # dense MLP layer0 -> FP8
    assert f.get_tensor('model.layers.0.mlp.gate_proj.weight').dtype == torch.float8_e4m3fn
    # lm_head -> BF16
    assert f.get_tensor('lm_head.weight').dtype == torch.bfloat16

# 2. 验证 config.json
cfg = json.load(open('/data2/Kimi-K25-FP8-new2/config.json'))
qcfg = cfg['text_config']['quantization_config']
assert qcfg['quant_method'] == 'fp8'
assert qcfg['activation_scheme'] == 'dynamic'
assert qcfg['weight_block_size'] == [128, 128]
assert qcfg['ignored_layers'] == ['lm_head']
print("All checks passed!")
```

---

## 历史踩坑记录（必读！）

### ❌ 坑1：weight_scale_inv 用 BF16 存储
**现象**：推理输出乱码  
**根因**：scale 精度不够，dequant 时数值严重偏差  
**修复**：`scale_inv = (amax / FP8_MAX).to(torch.float32)`，强制 float32

### ❌ 坑2：ignored_layers 用 regex 字符串
**现象**：配置了 `"re:.*self_attn.*"` 等 regex，ignore 不生效  
**根因**：SGLang `is_layer_skipped` 用纯子串匹配 `any(ignored in prefix)`，不支持 regex  
**修复**：只用简单子串，如 `["lm_head"]`

### ❌ 坑3：ignored_layers 中放 "shared_experts" 或 "experts"
**现象**：`shared_experts` skip 不生效（或意外跳过所有 experts）  
**根因**：`is_layer_skipped` 对含 `"experts"` 的 prefix 有特殊逻辑：
```python
elif "experts" in prefix:
    is_skipped = any(
        prefix in layer_name          # 注意：是 prefix IN layer_name（反向！）
        for layer_name in ignored_layers
        if "experts" in layer_name
    )
```
`"model.layers.1.mlp.shared_experts.gate_proj" in "shared_experts"` → **False**，skip 失效  
**修复**：`ignored_layers` 里**不放任何含 "experts" 的字符串**

### ❌ 坑4：INT4 双重量化精度损失
**现象**：即使配置正确，输出质量低于预期  
**根因**：INT4（group_size=32）→ BF16 反量化 → FP8，两次量化误差叠加，有效精度约 3~3.5 bit  
**修复**：从 BF16 原始模型直接量化，只经历一次量化，有效精度 ~7 bit

### ❌ 坑5：磁盘空间不足
**现象**：`SafetensorError: No space left on device`  
**根因**：BF16(1.9T) + FP8(~1T) 同时存在，超出磁盘  
**修复**：量化前删除旧的 FP8 目录，保留 ~1.2T 空闲

### ❌ 坑6：BF16 源模型没有 model.safetensors.index.json
**现象**：`FileNotFoundError: model.safetensors.index.json`  
**根因**：`/data2/Kimi-K2.5-BF16/` 没有 index 文件  
**修复**：用 `glob.glob("model-*-of-*.safetensors")` 直接枚举 shards，量化结束后生成新的 index.json

### ✅ shared_experts 正确处理方式
- **checkpoint 里**：`model.layers.N.mlp.shared_experts.{gate,up,down}_proj.weight` → **FP8**
- **SGLang 加载时**：weight_loader 自动 rename 为 `mlp.experts.384.{gate,up,down}_proj.*`，走 `Fp8MoEMethod` 路径
- **无需** 在 ignored_layers 里排除，也**无需** `--disable-shared-experts-fusion`

---

## SGLang 加载路径分析

| checkpoint key | SGLang 加载路径 | 量化方法 |
|---------------|----------------|---------|
| `mlp.experts.N.*` | `Fp8MoEMethod` → w13/w2 packed | block FP8 |
| `mlp.shared_experts.*` | rename → `experts.384.*` → `Fp8MoEMethod` | block FP8 |
| `self_attn.*.weight` | `Fp8LinearMethod` (block_quant=True) | block FP8 |
| `layers.0.mlp.*.weight` | `Fp8LinearMethod` (block_quant=True) | block FP8 |
| `lm_head.weight` | `UnquantizedLinearMethod`（ignored） | BF16 |
| `embed_tokens.weight` | 直接加载 | BF16 |
| `*.layernorm.weight` | 直接加载（1D，LinearBase 不匹配） | BF16 |
