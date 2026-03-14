# FlashInfer PR #2607：qk_nope_head_dim=192（GLM-5）在 SGLang 中的使用

PR [flashinfer#2607](https://github.com/flashinfer-ai/flashinfer/pull/2607) 放宽了 FlashInfer 中 TRT-LLM MLA 的 shape 检查，允许 `qk_nope_head_dim == 192`（除原来的 128 外），用于支持 GLM-5 等模型。

---

## 1. PR 做了什么

- **仓库**：flashinfer-ai/flashinfer（不是 SGLang）。
- **修改**：`flashinfer/mla.py` 里 `_check_trtllm_gen_mla_shape` 从只允许 `qk_nope_head_dim == 128` 改为允许 `128` 或 `192`。
- **原因**：GLM-5 使用 `qk_nope_head_dim=192`，旧版 FlashInfer 会报错。

---

## 2. SGLang 里 qk_nope_head_dim 从哪来

- SGLang **不提供** 启动参数来设置 `qk_nope_head_dim`。
- 该值来自 **模型配置**：`model_config.qk_nope_head_dim` 由 `configs/model_config.py` 从 HuggingFace 的 `hf_text_config.qk_nope_head_dim` 读取。
- 使用 TRT-LLM MLA 时，`trtllm_mla_backend.py` 中会使用 `config.qk_nope_head_dim` 并传给 FlashInfer。

因此：**只要模型 config 里是 192（例如 GLM-5），SGLang 会自动用 192，无需在启动命令里写任何与 head_dim 相关的参数。**

---

## 3. 在 SGLang 启动时如何用上 PR #2607

### 3.1 使用包含该 PR 的 FlashInfer

需要安装**已合入或包含 PR #2607 的 FlashInfer**，否则 GLM-5（qk_nope_head_dim=192）会在 FlashInfer 里报错。

- **方式 A**：等 FlashInfer 合并后发新版本，然后升级：
  ```bash
  pip install flashinfer -U
  ```
- **方式 B**：从 PR 分支本地安装：
  ```bash
  pip install 'git+https://github.com/flashinfer-ai/flashinfer.git@dev/mla-glm-5'
  ```
  或 clone 后在该分支上 `pip install -e .`。

### 3.2 正常启动 SGLang（GLM-5 或其它 qk_nope_head_dim=192 的 MLA 模型）

无需为 head_dim 加任何启动参数，按你当前用法即可，例如：

```bash
python -m sglang.launch_server \
  --model-path <GLM-5 或 MLA 模型路径> \
  --attention-backend trtllm_mla
```

- `--attention-backend trtllm_mla`：在支持的环境下（如 SM100）显式使用 TRT-LLM MLA；若 SGLang 已根据架构自动选到 `trtllm_mla`，可省略。
- 模型路径指向的目录里需有 HuggingFace 格式的 `config.json`，且其中 `qk_nope_head_dim` 为 192（GLM-5 官方权重一般已配好）。

### 3.3 若 SGLang 尚未支持 GLM-5 的 architecture 名

若 GLM-5 的 `config.json` 里 `architectures` 是 SGLang 目前未识别的（例如新的 `Glm5ForCausalLM`），需要在 `python/sglang/srt/configs/model_config.py` 里为 MLA 增加对应分支，并设置 `attention_arch = AttentionArch.MLA` 以及从 `hf_text_config` 读取 `qk_nope_head_dim`（与现有 DeepseekV3 / Glm4MoeLite / GlmMoeDsa 等分支一致）。  
若 GLM-5 复用已有架构（如 `GlmMoeDsaForCausalLM`），则无需改 SGLang，只要 FlashInfer 用上 PR #2607 即可。

---

## 4. 小结

| 项目 | 说明 |
|------|------|
| PR 归属 | FlashInfer 仓库，不是 SGLang |
| SGLang 启动参数 | 无需为 qk_nope_head_dim 增加任何参数 |
| 条件 | 安装包含 PR #2607 的 FlashInfer；模型 config 中 `qk_nope_head_dim=192`；MLA 使用 trtllm_mla 时生效 |
| 可选 | 显式指定 `--attention-backend trtllm_mla`（若未自动选用） |

这样即可在 SGLang 启动时用上 PR #2607 对 GLM-5（qk_nope_head_dim=192）的支持。
