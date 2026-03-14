# SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR 自动化测试脚本

这个工具用于自动化测试 `SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR` 参数从 0.0 到 1.0（步长 0.05）的性能表现。

## 功能特性

- ✅ 自动在4台机器上启动分布式SGLang服务器
- ✅ 等待所有节点就绪后自动运行benchmark
- ✅ 自动收集每次测试结果并保存到文件
- ✅ 生成测试摘要文件
- ✅ 支持中断恢复（Ctrl+C安全退出）

## 前置要求

1. **安装 sshpass**（用于SSH密码认证）
   ```bash
   # macOS
   brew install hudochenkov/sshpass/sshpass
   
   # Linux (Ubuntu/Debian)
   apt-get install sshpass
   
   # Linux (CentOS/RHEL)
   yum install sshpass
   ```

2. **配置SSH连接信息**
   - 确保可以通过SSH连接到4台机器
   - 第一台机器IP应以243结尾（如 26.5.27.243）

3. **确保模型路径正确**
   - 所有机器上的模型路径应该一致
   - 默认路径: `/work/models/Qwen3-235B-A22B-FP8/`

## 使用方法

### 方法1: Python脚本（推荐）

1. **配置SSH信息**

   编辑 `scripts/bench_dynamic_chunking_factor.py`，修改 `SSH_CONFIGS` 部分：
   ```python
   SSH_CONFIGS = [
       {"host": "26.5.27.243", "port": 22, "password": "your_password_1", "user": "root"},
       {"host": "26.5.27.244", "port": 22, "password": "your_password_2", "user": "root"},
       {"host": "26.5.27.245", "port": 22, "password": "your_password_3", "user": "root"},
       {"host": "26.5.27.246", "port": 22, "password": "your_password_4", "user": "root"},
   ]
   ```

   或者使用JSON配置文件：
   ```bash
   # 1. 复制示例配置文件
   cp scripts/bench_dynamic_chunking_factor_config.json.example scripts/bench_dynamic_chunking_factor_config.json
   
   # 2. 编辑配置文件，填入SSH信息
   vim scripts/bench_dynamic_chunking_factor_config.json
   
   # 3. 运行脚本时指定配置文件
   python3 scripts/bench_dynamic_chunking_factor.py --config scripts/bench_dynamic_chunking_factor_config.json
   ```

2. **运行测试**

   ```bash
   # 使用默认配置（测试 0.0 到 1.0，步长 0.05）
   python3 scripts/bench_dynamic_chunking_factor.py
   
   # 自定义测试范围
   python3 scripts/bench_dynamic_chunking_factor.py --start-factor 0.0 --end-factor 0.5 --step 0.1
   
   # 使用JSON配置文件
   python3 scripts/bench_dynamic_chunking_factor.py --config scripts/bench_dynamic_chunking_factor_config.json
   ```

### 方法2: Shell脚本

1. **配置SSH信息**

   编辑 `scripts/bench_dynamic_chunking_factor.sh`，修改配置区域：
   ```bash
   NODE0="26.5.27.243:22:root:your_password_1"
   NODE1="26.5.27.244:22:root:your_password_2"
   NODE2="26.5.27.245:22:root:your_password_3"
   NODE3="26.5.27.246:22:root:your_password_4"
   ```

2. **运行测试**

   ```bash
   bash scripts/bench_dynamic_chunking_factor.sh
   ```

## 输出结果

测试结果会保存在 `./bench_results/` 目录下：

```
bench_results/
├── summary.txt                    # 测试摘要（包含所有factor的关键指标）
├── factor_0.00.txt               # factor=0.00 的完整benchmark输出
├── factor_0.05.txt               # factor=0.05 的完整benchmark输出
├── factor_0.10.txt               # ...
└── ...
```

### 结果文件格式

每个 `factor_XX.XX.txt` 文件包含：
- Factor值
- 完整的benchmark输出（包括吞吐量、延迟等指标）

`summary.txt` 文件包含：
- 所有测试点的关键指标摘要
- 便于快速对比不同factor值的性能

## 测试流程

对于每个factor值（0.0, 0.05, 0.10, ..., 0.95, 1.0），脚本会：

1. **停止之前的服务器**（如果有）
2. **在4台机器上启动服务器**
   - 节点0: `--node-rank 0`
   - 节点1: `--node-rank 1`
   - 节点2: `--node-rank 2`
   - 节点3: `--node-rank 3`
3. **等待所有服务器就绪**
   - 通过健康检查端点 `/health` 确认
   - 超时时间: 600秒
4. **在第一个节点运行benchmark**
   - 使用 `sglang.bench_serving` 模块
   - 配置见脚本中的 `BENCH_CONFIG`
5. **保存结果到文件**
6. **停止所有服务器**
7. **等待10秒后继续下一个测试**

## 自定义配置

### 修改服务器参数

编辑脚本中的 `SERVER_CONFIG`（Python）或相应变量（Shell）：

```python
SERVER_CONFIG = {
    "nnodes": 4,
    "port": 30000,
    "dist_init_addr": "26.5.27.243:62001",
    "tp": 4,
    "pp_size": 8,
    # ... 其他参数
}
```

### 修改Benchmark参数

编辑脚本中的 `BENCH_CONFIG`（Python）或相应变量（Shell）：

```python
BENCH_CONFIG = {
    "num_prompt": 10,
    "random_input": 131072,
    "random_output": 1,
    "max_concurrency": 1,
    # ... 其他参数
}
```

### 修改测试范围

```bash
# Python脚本
python3 scripts/bench_dynamic_chunking_factor.py \
    --start-factor 0.0 \
    --end-factor 0.5 \
    --step 0.1

# Shell脚本需要手动修改循环部分
```

## 故障排查

### 1. SSH连接失败

- 检查SSH配置是否正确（IP、端口、用户名、密码）
- 确认网络连通性：`ping <host>`
- 测试SSH连接：`sshpass -p <password> ssh -p <port> <user>@<host> "echo test"`

### 2. 服务器启动失败

- 检查模型路径是否存在：`ls /work/models/Qwen3-235B-A22B-FP8/`
- 查看服务器日志：`/tmp/sglang_node<rank>_factor<factor>.log`
- 确认端口未被占用：`netstat -tuln | grep 30000`

### 3. Benchmark失败

- 确认服务器已完全启动（等待足够时间）
- 检查benchmark数据集路径：`ls ./ShareGPT_V3_unfiltered_cleaned_split.json`
- 查看第一个节点的服务器日志

### 4. 服务器未能就绪

- 增加超时时间：修改 `SERVER_STARTUP_TIMEOUT`
- 检查网络连接和防火墙设置
- 确认所有节点的时间同步

## 注意事项

1. **测试时间**: 每个factor值大约需要10-20分钟（取决于服务器启动和benchmark执行时间），总共21个测试点，预计需要3.5-7小时

2. **资源占用**: 确保4台机器都有足够的GPU内存和计算资源

3. **网络稳定性**: 确保4台机器之间的网络连接稳定

4. **中断恢复**: 如果测试中断，可以手动指定起始factor值继续测试：
   ```bash
   python3 scripts/bench_dynamic_chunking_factor.py --start-factor 0.50
   ```

5. **日志文件**: 每个节点的服务器日志保存在 `/tmp/sglang_node<rank>_factor<factor>.log`，可用于调试

## 示例输出

```
将测试以下factor值: [0.0, 0.05, 0.10, ..., 0.95, 1.0]
总共 21 个测试点

==================================================================================
测试 1/21: factor = 0.0
==================================================================================

启动服务器 (factor=0.0)...
节点 0 (26.5.27.243): 启动中...
节点 1 (26.5.27.244): 启动中...
节点 2 (26.5.27.245): 启动中...
节点 3 (26.5.27.246): 启动中...

等待所有服务器就绪...
等待服务器就绪... (5s/600s)
等待服务器就绪... (10s/600s)
...
服务器已就绪 (耗时: 120s)

运行benchmark (factor=0.0)...
[benchmark输出...]

结果已保存到: ./bench_results/factor_0.00.txt

✓ Factor 0.0 测试完成
```

## 支持

如有问题，请检查：
1. 脚本日志输出
2. 各节点的服务器日志文件
3. SSH连接和网络状态



