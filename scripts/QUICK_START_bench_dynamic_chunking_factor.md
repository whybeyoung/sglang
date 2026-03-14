# 快速开始指南

## 1. 安装依赖

```bash
# macOS
brew install hudochenkov/sshpass/sshpass

# Linux
apt-get install sshpass  # Ubuntu/Debian
# 或
yum install sshpass      # CentOS/RHEL
```

## 2. 配置SSH信息

### 方法A: 直接编辑Python脚本

编辑 `scripts/bench_dynamic_chunking_factor.py`，找到 `SSH_CONFIGS` 部分并修改：

```python
SSH_CONFIGS = [
    {"host": "26.5.27.243", "port": 22, "password": "实际密码1", "user": "root"},
    {"host": "26.5.27.244", "port": 22, "password": "实际密码2", "user": "root"},
    {"host": "26.5.27.245", "port": 22, "password": "实际密码3", "user": "root"},
    {"host": "26.5.27.246", "port": 22, "password": "实际密码4", "user": "root"},
]
```

### 方法B: 使用JSON配置文件（推荐）

```bash
# 1. 复制示例配置
cp scripts/bench_dynamic_chunking_factor_config.json.example \
   scripts/bench_dynamic_chunking_factor_config.json

# 2. 编辑配置文件
vim scripts/bench_dynamic_chunking_factor_config.json

# 填入实际的SSH信息
```

## 3. 运行测试

```bash
# 使用Python脚本（推荐）
python3 scripts/bench_dynamic_chunking_factor.py

# 或使用JSON配置文件
python3 scripts/bench_dynamic_chunking_factor.py \
    --config scripts/bench_dynamic_chunking_factor_config.json

# 或使用Shell脚本
bash scripts/bench_dynamic_chunking_factor.sh
```

## 4. 查看结果

```bash
# 查看所有结果
ls -lh bench_results/

# 查看摘要
cat bench_results/summary.txt

# 查看特定factor的结果
cat bench_results/factor_0.50.txt
```

## 测试流程

脚本会自动：
1. ✅ 在4台机器上启动服务器（node-rank 0-3）
2. ✅ 等待所有服务器就绪（通过 `/health` 端点检查）
3. ✅ 在第一个节点运行benchmark
4. ✅ 保存结果到 `bench_results/factor_XX.XX.txt`
5. ✅ 停止服务器，继续下一个测试点

## 测试范围

默认测试：0.0, 0.05, 0.10, ..., 0.95, 1.0（共21个测试点）

自定义范围：
```bash
python3 scripts/bench_dynamic_chunking_factor.py \
    --start-factor 0.0 \
    --end-factor 0.5 \
    --step 0.1
```

## 注意事项

- ⏱️ 每个测试点预计需要10-20分钟
- 💾 确保模型路径在所有机器上一致
- 🌐 确保4台机器网络互通
- 📝 服务器日志保存在 `/tmp/sglang_node<rank>_factor<factor>.log`

## 故障排查

如果遇到问题：

1. **测试SSH连接**
   ```bash
   sshpass -p "密码" ssh -p 22 root@26.5.27.243 "echo test"
   ```

2. **查看服务器日志**
   ```bash
   sshpass -p "密码" ssh -p 22 root@26.5.27.243 "tail -100 /tmp/sglang_node0_factor0.00.log"
   ```

3. **检查服务器状态**
   ```bash
   sshpass -p "密码" ssh -p 22 root@26.5.27.243 "curl http://127.0.0.1:30000/health"
   ```

## 完整文档

更多详细信息请查看：`scripts/README_bench_dynamic_chunking_factor.md`



