# 国内头部AI公司 LLM 推理优化工程师招聘调研（2025年）

> 数据来源：基于公开招聘平台（Boss直聘、牛客、官网招聘）信息整理，薪资为市场参考区间，实际以 offer 为准。
> 更新时间：2025年3月

---

## 一、岗位概览

| 公司 | 岗位名称 | 类型 | 薪资范围（月薪·K） | 城市 |
|------|---------|------|-----------------|------|
| 字节跳动 | LLM Inference 工程师 / 大模型推理优化 | 正编 | 40–80K × 16 | 北京/上海 |
| 阿里巴巴（通义/云） | 大模型推理优化工程师 | 正编 | 35–70K × 15–16 | 杭州/北京 |
| 腾讯（混元） | LLM系统工程师 / 推理优化 | 正编 | 35–65K × 16 | 深圳/北京 |
| 百度（文心） | 大模型推理加速工程师 | 正编 | 30–60K × 16 | 北京 |
| 华为（昇腾/盘古） | AI推理框架工程师 | 正编 | 30–55K × 16–18 | 深圳/北京/上海 |
| 月之暗面（Moonshot） | Inference Infra 工程师 | 正编 | 50–100K × 16 | 北京 |
| 智谱AI | 大模型推理系统工程师 | 正编 | 35–70K × 15 | 北京 |
| 深度求索（DeepSeek） | 推理系统研发工程师 | 正编 | 40–80K × 16 | 杭州/北京 |
| MiniMax | Inference 工程师 | 正编 | 40–80K × 16 | 上海 |
| 商汤科技 | LLM推理优化工程师 | 正编 | 30–55K × 15 | 上海 |
| 步步高（vivo AI） | 大模型推理工程师 | 正编 | 25–45K × 15 | 深圳 |
| 科大讯飞 | 大模型推理优化研究员 | 正编 | 20–40K × 15 | 合肥/北京 |
| 字节跳动（实习） | LLM Inference Intern | 实习 | 300–400元/天 | 北京/上海 |
| 阿里云（实习） | 大模型推理优化实习生 | 实习 | 250–350元/天 | 杭州/北京 |
| 智谱AI（实习） | 推理系统实习生 | 实习 | 200–350元/天 | 北京 |
| 百度（实习） | 大模型推理实习生 | 实习 | 200–300元/天 | 北京 |

---

## 二、典型 JD 详解

### 1. 字节跳动 — LLM Inference 工程师（正编）

**薪资：** 40–80K/月 × 16薪（高级/专家级可谈）

**职责：**
- 负责大语言模型推理服务的系统设计与优化，提升吞吐量（Throughput）和降低延迟（Latency）
- 研究并实现 continuous batching、speculative decoding、PagedAttention / RadixAttention 等核心推理技术
- 优化 CUDA kernel（FlashAttention、GEMM、量化推理），包括 INT8/FP8/AWQ/GPTQ 等量化方案
- 参与模型并行（Tensor Parallelism / Pipeline Parallelism / Expert Parallelism）系统设计
- 与模型团队协作，针对 MoE 架构（如 DeepSeek-V3、Mixtral）进行推理专项优化

**要求：**
- 本科及以上，CS/EE/数学相关专业
- 熟悉 CUDA 编程，有 CUDA kernel 开发经验
- 深入理解 Transformer 架构，熟悉 attention 机制
- 有 vLLM / TensorRT-LLM / SGLang / TGI 等推理框架使用或开发经验者优先
- 熟悉 GPU 体系结构（A100/H100/H800 等）
- 有分布式系统、网络通信（NCCL、InfiniBand）经验优先

**加分项：**
- 有 SGLang / vLLM 开源社区贡献经验
- 熟悉 MoE / Speculative Decoding / Disaggregated Prefill（PD 分离）等前沿方向

---

### 2. 阿里巴巴通义 — 大模型推理优化工程师（正编）

**薪资：** 35–70K/月 × 15–16薪

**职责：**
- 负责通义系列大模型推理引擎的研发与优化
- 研究并落地 KV Cache 管理、Prefix Cache、Chunked Prefill 等优化技术
- 对模型进行量化（INT8/FP8/AWQ）和模型压缩
- 负责算子优化，包括 FlashAttention、RoPE、LayerNorm 等关键算子的 CUDA 实现
- 参与阿里云 PAI 推理平台建设

**要求：**
- 熟悉深度学习框架（PyTorch/JAX），具备 CUDA/C++ 开发能力
- 有 LLM 推理引擎（TensorRT、TensorRT-LLM、vLLM）研发经验
- 了解模型量化、蒸馏、剪枝等技术
- 本科及以上，3年以上相关经验（P7及以上）

---

### 3. 月之暗面（Moonshot AI）— Inference Infra 工程师（正编）

**薪资：** 50–100K/月 × 16薪（头部初创，薪资竞争力最强之一）

**职责：**
- 构建 Kimi 系列模型超长上下文（1M tokens）推理基础设施
- 研究 Disaggregated Prefill-Decode（PD 分离）架构，优化 TTFT 和 TPOT
- 开发高效 KV Cache 传输协议（类 mooncake/nixl 方向）
- 优化 MLA（Multi-head Latent Attention）等新型 attention 机制的推理实现
- 参与 GPU 集群调度与资源管理系统设计

**要求：**
- 深入理解 GPU 内存层次、PCIe/NVLink/RDMA 传输机制
- 熟悉分布式系统，有 NCCL/UCX/RDMA 开发经验优先
- 有大规模推理系统（>1000 GPU）工程经验优先

---

### 4. 深度求索（DeepSeek）— 推理系统研发工程师（正编）

**薪资：** 40–80K/月 × 16薪

**职责：**
- 负责 DeepSeek 系列模型（V2/V3/R1）推理引擎优化
- 研究 MoE 模型专项优化：Expert Parallelism、EP 负载均衡、All-to-All 通信优化
- 推进 SGLang / vLLM 等开源框架在 DeepSeek 模型上的适配与深度优化
- 研究 FP8 量化、投机采样（Speculative Decoding）等推理提效技术

**要求：**
- 熟悉 CUDA 开发，有高性能 GEMM/Attention kernel 优化经验
- 了解 DeepSeek 架构（MLA、MoE）特性
- 有 SGLang / vLLM 开源贡献者背景优先

---

### 5. 华为（昇腾/盘古）— AI推理框架工程师（正编）

**薪资：** 30–55K/月 × 16–18薪（含股票/年终丰厚）

**职责：**
- 负责 MindSpore / MindIE 推理框架在昇腾 NPU 上的算子优化与适配
- 大模型推理性能优化：量化（W8A8/W4A16）、图编译优化、算子融合
- 支持 Ascend 910B/920 硬件上的 LLM 推理部署
- 参与 Atlas 推理加速卡的 SDK 研发

**要求：**
- 熟悉昇腾/CUDA 并行编程，了解 NPU 体系结构优先
- 有深度学习编译器（TVM/XLA/MindSpore）经验优先
- 本科/硕士，3年以上经验

---

### 6. 智谱AI — 大模型推理系统工程师（正编）

**薪资：** 35–70K/月 × 15薪

**职责：**
- 负责 GLM 系列模型推理服务的性能优化
- 实现并优化 vLLM / SGLang 等框架适配 GLM 架构
- 研究 Long Context 推理优化（Streaming、Chunked Prefill）
- 构建高可用推理服务（多副本、负载均衡、熔断）

---

### 7. 实习岗位汇总

| 公司 | 方向 | 日薪 | 要求 |
|------|------|------|------|
| 字节跳动 | LLM Inference Intern | 350–400元/天 | 在读硕博，熟悉 CUDA/PyTorch，有开源贡献优先 |
| 阿里云通义 | 大模型推理优化实习 | 280–350元/天 | 在读硕博，熟悉 vLLM/TRT-LLM |
| 智谱AI | 推理系统实习 | 250–350元/天 | 在读本硕，有 SGLang/vLLM 使用经验 |
| 百度文心 | 大模型推理实习 | 220–300元/天 | 在读本硕博 |
| MiniMax | Inference Intern | 300–400元/天 | 在读硕博，有 CUDA 开发经验 |
| 月之暗面 | Inference Infra Intern | 350–450元/天 | 在读硕博，强 CS 背景 |
| DeepSeek | 推理优化实习 | 300–400元/天 | 在读硕博，有开源经历优先 |

---

## 三、薪资分布分析

```
月薪（K，不含股票/奖金）

初创/独角兽（月之暗面/MiniMax/DeepSeek/智谱）
    高级工程师：50–100K
    中级工程师：35–60K
    ████████████████████████

字节/阿里/腾讯
    高级/专家：45–80K
    中级：30–50K
    ██████████████████████

百度/商汤/华为
    高级/专家：35–60K
    中级：25–40K
    ████████████████

传统大厂（科大讯飞/步步高等）
    高级：20–40K
    中级：15–25K
    ██████████
```

### 薪资构成（典型offer结构）
- **字节/阿里/腾讯/百度**：月薪 × 15–16 + 年终奖（0–6月）
- **月之暗面/MiniMax/DeepSeek**：月薪 × 16 + 股票期权（价值可观）+ 年终
- **华为**：月薪 × 16–18 + 股票激励（TUP）+ 年终
- **实习**：日薪制，一般无股票

---

## 四、核心技能要求（综合各JD）

### 必备技能
1. **CUDA 编程**：熟悉 GPU 并行编程，有 kernel 优化经验（FlashAttention、GEMM）
2. **Transformer 架构**：深入理解 MHA/MQA/GQA/MLA/MoE 等变体
3. **推理框架**：vLLM / TensorRT-LLM / SGLang / TGI 使用或开发经验
4. **模型量化**：INT8 / FP8 / AWQ / GPTQ / SmoothQuant
5. **分布式推理**：TP / PP / EP 并行，NCCL 通信

### 加分技能
1. **开源贡献**：SGLang / vLLM / TensorRT-LLM PR记录
2. **PD分离/Disaggregated Inference**：KV cache 传输、RDMA/nixl
3. **Speculative Decoding**：Draft model、EAGLE、Medusa 等
4. **编译优化**：MLIR / Triton / TVM kernel 编写
5. **超长上下文**：1M+ token 推理优化，稀疏 attention

---

## 五、招聘渠道

| 渠道 | 说明 |
|------|------|
| [Boss直聘](https://www.zhipin.com) | 搜索"LLM推理优化"、"大模型推理加速" |
| [牛客网](https://www.nowcoder.com/jobs) | 有社区讨论，可看薪资爆料 |
| [各公司官网招聘页] | 字节/阿里/腾讯/华为官网投递更稳定 |
| [脉脉](https://maimai.cn) | 内推信息、薪资爆料较多 |
| GitHub | SGLang/vLLM 贡献者常被猎头直接联系 |
| 微信/知乎/朋友圈 | 初创公司（月之暗面/MiniMax）常走内推 |

---

## 六、总结建议

1. **薪资天花板**：月之暗面 > MiniMax ≈ DeepSeek > 字节 > 阿里 ≈ 腾讯 > 百度 ≈ 华为 > 商汤 > 讯飞
2. **成长性**：初创公司（DeepSeek/月之暗面）技术迭代快，股票收益空间大；大厂平台稳定
3. **进入门槛**：有 SGLang / vLLM 开源 PR 记录可大幅提升竞争力
4. **实习转正**：月之暗面/字节实习转正率较高，建议优先争取实习机会
5. **赛道热度**：PD分离（Disaggregated Inference）、MoE推理优化、FP8量化是2025年最热方向

---

*本文档基于公开信息整理，薪资数据为市场参考区间，以实际offer为准。建议结合脉脉薪资爆料和Boss直聘实时数据做校准。*
