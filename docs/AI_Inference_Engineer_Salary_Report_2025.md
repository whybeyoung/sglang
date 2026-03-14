# 国内 AI 推理加速/优化工程师 招聘市场调研报告（2025-2026）

> **说明**：薪酬数据综合 Boss 直聘、猎聘、脉脉、拉勾等平台公开信息及业内交流口径，仅供参考。实际薪酬因城市、公司、个人能力浮动。

---

## 一、正式岗位（正编/社招）

### 1. 薪酬区间总览

| 级别 | 年限 | 月薪区间（税前） | 年包估算（含年终） | 备注 |
|------|------|------------------|-------------------|------|
| 初级（P5/L6） | 1-3 年 | 25K-40K × 14-16 薪 | 40W-65W | 有 CUDA/推理框架经验即可 |
| 中级（P6/L7） | 3-5 年 | 40K-60K × 14-16 薪 | 65W-100W | 能独立做性能优化、问题定位 |
| 高级（P7/L8） | 5-8 年 | 60K-90K × 15-18 薪 | 100W-160W | 框架源码级能力、分布式架构 |
| 专家（P8/L9+） | 8+ 年 | 80K-130K+ × 15-18 薪 | 140W-250W+ | 技术 leader、系统架构设计 |

> **热门城市溢价**：北京/上海/深圳 为主要招聘地，杭州其次。北京/上海薪酬最高，深圳次之。
> **股票/期权**：头部公司（字节、快手、月之暗面、DeepSeek、MiniMax 等）高级别岗位通常有 RSU/期权，可额外增加 30%-100% 年包。

### 2. 主要招聘公司及薪酬参考

| 公司 | 岗位名称 | 级别 | 薪酬参考 |
|------|---------|------|---------|
| **字节跳动** | 推理优化工程师 | 2-1 / 2-2 | 40K-80K × 15薪 + RSU |
| **快手** | LLM 推理加速工程师 | L7-L8 | 45K-75K × 16薪 + 期权 |
| **阿里（通义/达摩院）** | 大模型推理优化 | P6-P7 | 40K-70K × 16薪 + RSU |
| **百度（文心）** | 推理引擎研发 | T5-T7 | 35K-65K × 14.6薪 |
| **腾讯（混元）** | AI 推理优化工程师 | T9-T11 | 40K-75K × 16-18薪 + RSU |
| **华为（昇腾/盘古）** | 推理框架开发工程师 | 16-18级 | 35K-60K × 16薪 + 分红 |
| **月之暗面（Moonshot）** | Inference Engineer | — | 50K-90K + 期权（给得多） |
| **DeepSeek（幻方）** | 推理系统工程师 | — | 50K-80K + 年终（据传 6-12 个月） |
| **MiniMax** | LLM Serving Engineer | — | 45K-80K + 期权 |
| **智谱 AI** | 推理优化工程师 | — | 40K-65K + 期权 |
| **科大讯飞** | 推理加速优化工程师 | — | 30K-55K × 14-16薪 |
| **商汤** | 推理引擎研发 | — | 35K-60K × 14薪 + RSU |
| **硅基流动（SiliconFlow）** | Inference Infra | — | 40K-70K + 期权 |
| **阶跃星辰（StepFun）** | 推理系统工程师 | — | 50K-85K + 期权 |

### 3. 典型 JD 内容（正编高阶）

以下为综合多家头部公司 JD 整理的典型要求：

#### 岗位职责
1. **推理框架深度优化**：基于 SGLang / vLLM / TensorRT-LLM 等主流框架进行二次开发与性能调优，支撑在线推理服务
2. **Kernel/算子优化**：使用 CUDA / Triton 编写或优化 attention、GEMM、sampling 等高频算子，提升推理吞吐与首包延迟
3. **分布式推理架构**：设计与实现 TP/PP/CP/EP 多卡并行策略，Prefill-Decode 分离（PD disaggregation）架构
4. **显存与调度优化**：KV Cache 管理（PagedAttention、Token Pooling）、Continuous Batching、动态调度策略
5. **模型接入与适配**：将新模型（DeepSeek-V3/R1、Qwen3、Llama4 等）高效接入推理框架，保证精度与性能
6. **Profiling 与瓶颈分析**：使用 Nsight Systems/Compute、PyTorch Profiler 等工具定位 kernel、通信、调度瓶颈
7. **量化与压缩**：FP8/INT8/INT4 量化、MoE expert 并行、投机解码（Speculative Decoding）等先进技术落地

#### 任职要求
- 计算机/数学/电子等相关专业本科及以上，3 年以上推理/高性能计算相关经验
- 精通 Python，熟练 C++/CUDA，有 Triton 编程经验优先
- 深入理解至少一个主流推理框架（SGLang/vLLM/TRT-LLM）源码，能独立定位问题
- 理解 Transformer / MoE 架构，了解 FlashAttention / FlashInfer / PagedAttention 原理
- 有 GPU 性能 profiling 经验，能分析 kernel 利用率、显存占用、通信开销
- 有分布式系统或大规模 GPU 集群经验优先
- 有开源贡献（SGLang/vLLM/FlashAttention/DeepSpeed 等）优先

#### 加分项
- 熟悉 NCCL / RDMA / NVLink 通信优化
- 有 Prefill-Decode 分离部署经验
- 有投机解码（Eagle/Medusa/MTP）实战经验
- 有 Context Parallelism（CP）/ Pipeline Parallelism（PP）实战经验
- 了解 MaaS 平台架构与 SLO 保障

### 4. 典型 JD 内容（正编初中级）

#### 岗位职责
1. 参与推理框架的日常维护与功能开发
2. 编写 benchmark 与性能测试脚本，协助定位性能问题
3. 协助高级工程师完成算子优化与模型适配
4. 维护推理服务的 CI/CD 流程与监控

#### 任职要求
- 计算机相关专业本科及以上，1 年以上相关经验
- 熟练 Python，了解 C++/CUDA 基础
- 了解 Transformer 模型推理流程，用过 vLLM 或 SGLang
- 有 GPU 编程基础，了解 CUDA 编程模型
- 有良好的代码习惯与协作意识

---

## 二、实习岗位

### 1. 薪酬区间总览

| 公司类型 | 日薪区间 | 月薪估算（22天） | 备注 |
|---------|---------|----------------|------|
| 头部大厂（字节/阿里/腾讯/快手） | 400-600 元/天 | 8.8K-13.2K | 部分给 500-700 |
| AI 创业公司（月之暗面/DeepSeek/MiniMax） | 500-800 元/天 | 11K-17.6K | AI 创业公司普遍给得高 |
| 中厂（科大讯飞/商汤/百川等） | 300-500 元/天 | 6.6K-11K | |
| 外企（英伟达/AMD/Intel） | 500-700 元/天 | 11K-15.4K | 有些按月薪发 |

> **转正优势**：推理加速方向实习转正率较高（50%-80%），因人才稀缺。转正后薪酬通常在对应级别下限+。

### 2. 主要实习招聘公司参考

| 公司 | 岗位名称 | 日薪参考 | 要求 |
|------|---------|---------|------|
| **字节跳动** | 推理优化实习生 | 500-600/天 | 硕士在读，了解 CUDA/推理框架 |
| **阿里达摩院** | LLM 推理实习生 | 450-550/天 | 了解 vLLM/SGLang，有 GPU 编程基础 |
| **月之暗面** | Inference 实习生 | 600-800/天 | 对推理框架有深度理解 |
| **DeepSeek** | 推理系统实习生 | 600-800/天 | 有 CUDA 经验，了解分布式推理 |
| **MiniMax** | 推理优化实习生 | 500-700/天 | 有推理框架使用经验 |
| **智谱 AI** | 推理引擎实习 | 400-500/天 | 了解 Transformer 推理流程 |
| **科大讯飞** | 推理加速实习生 | 300-400/天 | 有 Python/C++ 基础 |
| **硅基流动** | Serving 实习生 | 450-600/天 | 了解推理框架 |
| **英伟达 (中国)** | CUDA/TRT 实习生 | 500-700/天 | CUDA 编程能力强 |

### 3. 典型实习 JD

#### 岗位职责
1. 参与 LLM 推理框架（SGLang/vLLM）的功能开发与性能测试
2. 协助编写 CUDA/Triton kernel，优化推理性能
3. 搭建 benchmark 环境，进行端到端性能对比测试
4. 参与推理系统的文档编写与技术分享

#### 任职要求
- 计算机/数学/AI 相关专业硕士/博士在读，每周至少 4 天实习
- 熟练 Python，了解 C++
- 了解 Transformer 模型结构与推理流程
- 了解 GPU 编程基础（CUDA/Triton 至少接触过一种）
- 有 vLLM/SGLang/TensorRT-LLM 使用经验优先
- 有相关开源项目贡献经验优先

#### 加分项
- 有 FlashAttention / PagedAttention / Continuous Batching 相关理解
- 了解量化（FP8/INT8）、投机解码等优化技术
- 有竞赛或论文经验（MLSys/OSDI/SOSP/ATC 等系统会议）

---

## 三、市场趋势与洞察

### 1. 供需关系
- **严重供不应求**：有源码级推理框架经验的人极少，是当前 AI 领域最稀缺的工程人才之一
- **薪资涨幅明显**：2024-2025 年该方向薪酬同比上涨 20%-40%，尤其是有 SGLang/vLLM 贡献者背景的候选人
- **创业公司溢价**：月之暗面、DeepSeek、阶跃星辰等给的期权/薪酬激进，往往比大厂高 20%-50%

### 2. 技术热点（JD 高频关键词）
```
SGLang > vLLM > TensorRT-LLM（框架热度排名）
FlashAttention / FlashInfer（注意力优化）
PagedAttention / Continuous Batching（调度）
Prefill-Decode Disaggregation（PD 分离）
TP / PP / CP / EP（分布式并行）
CUDA / Triton（GPU 编程）
FP8 / INT8 量化
Speculative Decoding（投机解码）
DeepSeek-V3/R1 模型优化
MoE Expert Parallelism
```

### 3. 面试重点
- **系统理解**：推理框架端到端流程（tokenizer → scheduler → model_runner → detokenizer）
- **Kernel 层**：FlashAttention 原理、GEMM 优化、memory-bound vs compute-bound 分析
- **调度层**：Continuous Batching、Prefill/Decode 混合调度、KV Cache 管理
- **分布式**：TP/PP/CP 各自的通信模式与瓶颈、NCCL 原语
- **实战能力**：Profiling 方法论、性能瓶颈定位、实际优化案例

### 4. 求职建议
1. **贡献开源**：给 SGLang/vLLM 提 PR 是最有说服力的简历加分项
2. **深度 > 广度**：精通一个框架 > 浅尝多个框架
3. **会讲故事**：能清晰描述"发现什么瓶颈 → 怎么分析 → 怎么优化 → 效果如何"
4. **关注前沿**：PD 分离、CP、投机解码、MoE EP 是当前最热的优化方向

---

## 四、参考 JD 样本（本地存档）

参见：[大模型推理加速优化工程师（高阶）JD](./JD_LLM_Inference_Acceleration_Engineer_Senior.md)

---

*报告整理时间：2025-2026，数据来源为公开招聘平台及业内交流信息，仅供内部参考。*
