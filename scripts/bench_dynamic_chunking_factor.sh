#!/bin/bash
# 自动化测试脚本：测试 SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR 从 0.0 到 1.0 (步长 0.05)
# 
# 使用方法:
#   bash scripts/bench_dynamic_chunking_factor.sh
#
# 需要先配置:
#   1. 设置SSH连接信息（见脚本中的配置区域）
#   2. 确保已安装 sshpass: brew install hudochenkov/sshpass/sshpass (macOS) 或 apt-get install sshpass (Linux)

set -euo pipefail

# ==================== 配置区域 ====================
# 请根据实际情况修改以下配置

# SSH连接信息 (格式: HOST:PORT:USER:PASSWORD)
# 四台机器使用相同的host，不同的SSH端口
NODE0="36.138.60.54:30243:root:Aipaasxylx1.t!@#"
NODE1="36.138.60.54:30241:root:Aipaasxylx1.t!@#"
NODE2="36.138.60.54:30242:root:Aipaasxylx1.t!@#"
NODE3="36.138.60.54:30239:root:Aipaasxylx1.t!@#"

# 模型路径
MODEL_PATH="/work/models"

# 服务器配置
NNODES=4
PORT=30000
DIST_INIT_ADDR="26.5.27.243:62001"
TP=8
PP_SIZE=4
HOST="0.0.0.0"

# Benchmark配置
BENCH_HOST="127.0.0.1"
BENCH_PORT=30000
DATASET_PATH="./ShareGPT_V3_unfiltered_cleaned_split.json"
NUM_PROMPT=10
RANDOM_INPUT=131072
RANDOM_OUTPUT=1
MAX_CONCURRENCY=1
WARMUP_REQUESTS=2
BACKEND="sglang"
DATASET_NAME="random"
RANDOM_RANGE_RATIO=1

# 服务器启动超时时间（秒）
SERVER_STARTUP_TIMEOUT=600
HEALTH_CHECK_INTERVAL=5

# 结果保存目录
RESULTS_DIR="./bench_results"

# ==================== 辅助函数 ====================

# 解析SSH配置
parse_ssh_config() {
    local config=$1
    IFS=':' read -r host port user password <<< "$config"
    echo "$host $port $user $password"
}

# SSH执行命令
ssh_execute() {
    local host=$1
    local port=$2
    local user=$3
    local password=$4
    local command=$5
    local background=${6:-false}
    
    if [ "$background" = "true" ]; then
        sshpass -p "$password" ssh -o StrictHostKeyChecking=no \
            -o UserKnownHostsFile=/dev/null \
            -p "$port" \
            "${user}@${host}" \
            "$command" > /dev/null 2>&1 &
        echo $!
    else
        sshpass -p "$password" ssh -o StrictHostKeyChecking=no \
            -o UserKnownHostsFile=/dev/null \
            -o ConnectTimeout=10 \
            -p "$port" \
            "${user}@${host}" \
            "$command"
    fi
}

# 等待服务器就绪
wait_for_server_ready() {
    local host=$1
    local port=$2
    local timeout=${3:-$SERVER_STARTUP_TIMEOUT}
    local start_time=$(date +%s)
    
    read -r node0_host node0_port node0_user node0_password <<< "$(parse_ssh_config "$NODE0")"
    
    while true; do
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        
        if [ $elapsed -ge $timeout ]; then
            echo "错误: 服务器在 ${timeout} 秒内未能就绪"
            return 1
        fi
        
        http_code=$(ssh_execute "$node0_host" "$node0_port" "$node0_user" "$node0_password" \
            "curl -s -o /dev/null -w '%{http_code}' http://${host}:${port}/health 2>/dev/null || echo '000'" 2>/dev/null | tr -d '\n')
        
        if [ "$http_code" = "200" ]; then
            echo "✓ 服务器已就绪 (耗时: ${elapsed}s)"
            return 0
        fi
        
        echo "等待服务器就绪... (${elapsed}s/${timeout}s)"
        sleep $HEALTH_CHECK_INTERVAL
    done
}

# 检查并停止所有SGLang服务器
kill_sglang_servers() {
    echo ""
    echo "检查并清理残留的SGLang进程..."
    
    for i in 0 1 2 3; do
        local var_name="NODE${i}"
        local config="${!var_name}"
        read -r host port user password <<< "$(parse_ssh_config "$config")"
        
        # 更精确地检查sglang服务器进程（排除grep和sshpass进程）
        # 只匹配真正的sglang.launch_server进程
        local processes=$(ssh_execute "$host" "$port" "$user" "$password" \
            "ps aux | grep -E 'python.*sglang\.launch_server|python.*-m sglang\.launch_server' | grep -v grep | grep -v 'sshpass\|ssh.*sglang' | wc -l" 2>/dev/null | tr -d '[:space:]' || echo "0")
        
        if [ "$processes" != "0" ] && [ "$processes" != "" ] && [ "$processes" -gt 0 ]; then
            echo "节点 $i (${host}): 发现 $processes 个残留进程，正在清理..."
            
            # 精确杀掉sglang.launch_server进程（排除grep和ssh相关进程）
            ssh_execute "$host" "$port" "$user" "$password" \
                "ps aux | grep -E 'python.*sglang\.launch_server|python.*-m sglang\.launch_server' | grep -v grep | grep -v 'sshpass\|ssh.*sglang' | awk '{print \$2}' | xargs -r kill -9 2>/dev/null || true" > /dev/null 2>&1 || true
            
            # 也清理bench_serving进程
            ssh_execute "$host" "$port" "$user" "$password" \
                "ps aux | grep -E 'bench_serving' | grep -v grep | grep -v 'sshpass\|ssh.*bench' | awk '{print \$2}' | xargs -r kill -9 2>/dev/null || true" > /dev/null 2>&1 || true
            
            # 等待进程完全退出
            sleep 2
            
            # 再次检查
            local remaining=$(ssh_execute "$host" "$port" "$user" "$password" \
                "ps aux | grep -E 'python.*sglang\.launch_server|python.*-m sglang\.launch_server' | grep -v grep | grep -v 'sshpass\|ssh.*sglang' | wc -l" 2>/dev/null | tr -d '[:space:]' || echo "0")
            
            if [ "$remaining" != "0" ] && [ "$remaining" != "" ] && [ "$remaining" -gt 0 ]; then
                echo "  警告: 节点 $i 仍有 $remaining 个进程未完全退出，尝试强制清理..."
                ssh_execute "$host" "$port" "$user" "$password" \
                    "ps aux | grep -E 'python.*sglang\.launch_server|python.*-m sglang\.launch_server' | grep -v grep | grep -v 'sshpass\|ssh.*sglang' | awk '{print \$2}' | xargs -r kill -9 2>/dev/null || true" > /dev/null 2>&1 || true
                sleep 2
            else
                echo "  节点 $i: 清理完成"
            fi
        else
            echo "节点 $i (${host}): 无残留进程"
        fi
    done
    
    echo "等待进程完全退出..."
    sleep 5
}

# 启动所有服务器
start_servers() {
    local factor=$1
    local pids=()
    
    echo ""
    echo "启动服务器 (factor=${factor})..."
    
    for i in 0 1 2 3; do
        local var_name="NODE${i}"
        local config="${!var_name}"
        read -r host port user password <<< "$(parse_ssh_config "$config")"
        
        local cmd="export SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR=${factor} && \
                   unset SGLANG_PP_LAYER_PARTITION && \
                   python3 -m sglang.launch_server \
                     --model-path ${MODEL_PATH} \
                     --nnodes ${NNODES} \
                     --port ${PORT} \
                     --dist-init-addr ${DIST_INIT_ADDR} \
                     --node-rank ${i} \
                     --tp ${TP} \
                     --pp-size ${PP_SIZE} \
                     --trust-remote-code \
                     --disable-radix-cache \
                     --mem-fraction-static 0.8 \
                     --max-running-requests 128 \
                     --chunked-prefill-size 12288 \
                     --attention-backend fa3 \
                     --watchdog-timeout 3600 \
                     --host ${HOST} \
                     --enable-dynamic-chunk \
                     > /tmp/sglang_node${i}_factor${factor}.log 2>&1"
        
        echo "节点 $i (${host}): 启动中..."
        ssh_execute "$host" "$port" "$user" "$password" "$cmd" true
        pids+=($!)
        sleep 2
    done
    
    echo "所有服务器启动命令已发送 (PIDs: ${pids[@]})"
}

# 运行benchmark
run_benchmark() {
    local factor=$1
    
    echo ""
    echo "运行benchmark (factor=${factor})..."
    
    read -r host port user password <<< "$(parse_ssh_config "$NODE0")"
    
    # 工作目录（第一个节点，243节点）
    local work_dir="/home/aiges"
    # 创建日志文件路径（使用绝对路径）
    local log_file="/home/aiges/bench_serving_factor${factor}.log"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "Benchmark开始时间: ${timestamp}"
    echo "执行节点: ${host}:${port}"
    echo "工作目录: ${work_dir}"
    echo "日志文件: ${log_file}"
    
    # 确保工作目录存在
    ssh_execute "$host" "$port" "$user" "$password" \
        "mkdir -p ${work_dir}" > /dev/null 2>&1 || true
    
    # 构建命令，将输出同时保存到日志文件和标准输出
    # 先写入日志头部（在工作目录下执行）
    ssh_execute "$host" "$port" "$user" "$password" \
        "cd ${work_dir} && \
         echo '=== Benchmark开始执行 ===' > ${log_file} && \
         echo '时间: ${timestamp}' >> ${log_file} && \
         echo 'Factor: ${factor}' >> ${log_file} && \
         echo '工作目录: ${work_dir}' >> ${log_file} && \
         echo '命令: python3 -m sglang.bench_serving --host ${BENCH_HOST} --port ${BENCH_PORT} --dataset-path ${DATASET_PATH} --num-prompt ${NUM_PROMPT} --random-input ${RANDOM_INPUT} --random-output ${RANDOM_OUTPUT} --max-concurrency ${MAX_CONCURRENCY} --warmup-requests ${WARMUP_REQUESTS} --backend ${BACKEND} --dataset-name ${DATASET_NAME} --random-range-ratio ${RANDOM_RANGE_RATIO} --tokenizer ${MODEL_PATH} --model ${MODEL_PATH}' >> ${log_file} && \
         echo '' >> ${log_file}" > /dev/null 2>&1 || true
    
    # 执行benchmark命令，在 /home/aiges 目录下执行，同时输出到日志和标准输出
    local cmd="cd ${work_dir} && \
               python3 -m sglang.bench_serving \
                 --host ${BENCH_HOST} \
                 --port ${BENCH_PORT} \
                 --dataset-path ${DATASET_PATH} \
                 --num-prompt ${NUM_PROMPT} \
                 --random-input ${RANDOM_INPUT} \
                 --random-output ${RANDOM_OUTPUT} \
                 --max-concurrency ${MAX_CONCURRENCY} \
                 --warmup-requests ${WARMUP_REQUESTS} \
                 --backend ${BACKEND} \
                 --dataset-name ${DATASET_NAME} \
                 --random-range-ratio ${RANDOM_RANGE_RATIO} \
                 --tokenizer ${MODEL_PATH} \
                 --model ${MODEL_PATH} 2>&1 | tee -a ${log_file}"
    
    # 执行命令并捕获输出
    echo "正在执行benchmark命令..."
    echo "（这可能需要几分钟时间，请耐心等待...）"
    local result=$(ssh_execute "$host" "$port" "$user" "$password" "$cmd" 2>&1)
    local exit_code=$?
    
    # 记录结束时间和退出码到日志（在工作目录下）
    ssh_execute "$host" "$port" "$user" "$password" \
        "cd ${work_dir} && \
         echo '' >> ${log_file} && \
         echo '=== Benchmark执行完成 ===' >> ${log_file} && \
         echo \"结束时间: \$(date '+%Y-%m-%d %H:%M:%S')\" >> ${log_file} && \
         echo \"退出码: ${exit_code}\" >> ${log_file}" > /dev/null 2>&1 || true
    
    # 检查执行状态
    echo ""
    if [ $exit_code -ne 0 ]; then
        echo "警告: benchmark命令执行返回非零退出码: ${exit_code}"
    else
        echo "✓ Benchmark命令执行完成"
    fi
    
    # 从远程服务器获取完整日志文件内容（从工作目录）
    echo "正在获取benchmark执行日志..."
    local log_content=$(ssh_execute "$host" "$port" "$user" "$password" \
        "cd ${work_dir} && cat ${log_file} 2>/dev/null || echo '=== 日志文件不存在或无法读取 ==='" 2>&1)
    
    # 保存日志到本地
    local local_log_file="${RESULTS_DIR}/bench_serving_factor${factor}.log"
    mkdir -p "$RESULTS_DIR"
    {
        echo "=== Benchmark执行日志 ==="
        echo "Factor: ${factor}"
        echo "开始时间: ${timestamp}"
        echo "日志文件: ${log_file}"
        echo "=========================================="
        echo ""
        echo "$log_content"
    } > "$local_log_file"
    
    echo "日志已保存到本地: ${local_log_file}"
    
    # 显示日志的最后几行，确认执行状态
    if [ -n "$log_content" ] && [ "$log_content" != "=== 日志文件不存在或无法读取 ===" ]; then
        echo ""
        echo "日志最后10行:"
        echo "----------------------------------------"
        echo "$log_content" | tail -10
        echo "----------------------------------------"
    else
        echo "警告: 无法获取日志文件内容"
    fi
    
    # 返回结果（优先使用日志内容，如果没有则使用直接输出）
    if [ -n "$log_content" ] && [ "$log_content" != "=== 日志文件不存在或无法读取 ===" ]; then
        echo "$log_content"
    else
        echo "$result"
    fi
}

# 提取TTFT mean值
extract_ttft_mean() {
    local result=$1
    # bench_serving输出格式: "Mean TTFT (ms): 123.45"
    local ttft=$(echo "$result" | grep -iE "Mean TTFT.*ms" | \
        grep -oE "[0-9]+\.[0-9]+" | head -1)
    
    # 如果没找到，尝试其他格式
    if [ -z "$ttft" ]; then
        ttft=$(echo "$result" | grep -iE "mean.*ttft|ttft.*mean" | \
            grep -oE "[0-9]+\.[0-9]+" | head -1)
    fi
    
    # 如果还是没找到，尝试查找"mean_ttft_ms"字段（JSON格式）
    if [ -z "$ttft" ]; then
        ttft=$(echo "$result" | grep -oE '"mean_ttft_ms"[[:space:]]*:[[:space:]]*[0-9]+\.[0-9]+' | \
            grep -oE "[0-9]+\.[0-9]+" | head -1)
    fi
    
    echo "$ttft"
}

# 保存结果
save_result() {
    local factor=$1
    local result=$2
    
    mkdir -p "$RESULTS_DIR"
    
    local filename="${RESULTS_DIR}/factor_${factor}.txt"
    {
        echo "Factor: ${factor}"
        echo "=================================================================================="
        echo "$result"
        echo ""
    } > "$filename"
    
    echo "结果已保存到: ${filename}"
}

# 保存TTFT mean到汇总文件
save_ttft_summary() {
    local factor=$1
    local ttft_mean=$2
    
    local ttft_file="${RESULTS_DIR}/ttft_summary.csv"
    
    # 如果文件不存在，创建CSV头部
    if [ ! -f "$ttft_file" ]; then
        echo "factor,ttft_mean" > "$ttft_file"
    fi
    
    # 追加数据
    echo "${factor},${ttft_mean}" >> "$ttft_file"
    echo "TTFT mean已记录: factor=${factor}, ttft_mean=${ttft_mean}"
}

# ==================== 主程序 ====================

main() {
    # 检查sshpass
    if ! command -v sshpass &> /dev/null; then
        echo "错误: 需要安装 sshpass"
        echo "安装方法:"
        echo "  macOS: brew install hudochenkov/sshpass/sshpass"
        echo "  Linux: apt-get install sshpass 或 yum install sshpass"
        exit 1
    fi
    
    # 检查SSH配置
    if [[ "$NODE0" == *"YOUR_PASSWORD"* ]] || [[ "$NODE1" == *"YOUR_PASSWORD"* ]]; then
        echo "错误: 请先配置SSH连接信息!"
        echo "请在脚本中修改 NODE0-NODE3 变量"
        exit 1
    fi
    
    # 创建结果目录
    mkdir -p "$RESULTS_DIR"
    
    # 生成factor值列表
    factors=()
    factor=0.0
    while (( $(echo "$factor <= 1.001" | bc -l) )); do
        factors+=($(printf "%.2f" $factor))
        factor=$(echo "$factor + 0.05" | bc -l)
    done
    
    echo "将测试以下factor值: ${factors[@]}"
    echo "总共 ${#factors[@]} 个测试点"
    echo ""
    
    # 测试摘要文件
    summary_file="${RESULTS_DIR}/summary.txt"
    {
        echo "SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR 测试摘要"
        echo "=================================================================================="
        echo ""
    } > "$summary_file"
    
    # 捕获中断信号，确保退出时清理进程
    trap 'echo ""; echo "用户中断测试，正在清理进程..."; kill_sglang_servers; exit 1' INT TERM
    
    # 脚本开始时也先清理一次
    echo "脚本启动，先检查并清理所有残留的SGLang进程..."
    kill_sglang_servers
    
    # 遍历所有factor值
    for idx in "${!factors[@]}"; do
        factor=${factors[$idx]}
        test_num=$((idx + 1))
        total=${#factors[@]}
        
        echo ""
        echo "=================================================================================="
        echo "测试 ${test_num}/${total}: factor = ${factor}"
        echo "=================================================================================="
        
        # 每次启动前都先检查并清理残留进程
        kill_sglang_servers
        
        # 启动所有服务器
        start_servers "$factor"
        
        # 等待服务器就绪
        echo ""
        echo "等待所有服务器就绪..."
        if ! wait_for_server_ready "$BENCH_HOST" "$BENCH_PORT"; then
            echo "错误: 服务器未能就绪，跳过此测试点"
            kill_sglang_servers
            continue
        fi
        
        echo "所有服务器已就绪!"
        
        # 运行benchmark
        echo ""
        echo "开始执行benchmark..."
        result=$(run_benchmark "$factor" 2>&1)
        benchmark_exit_code=$?
        
        # 检查benchmark是否成功执行
        echo ""
        if [ $benchmark_exit_code -ne 0 ]; then
            echo "警告: benchmark执行可能有问题，退出码: ${benchmark_exit_code}"
        fi
        
        # 检查结果是否为空
        if [ -z "$result" ] || [ ${#result} -lt 10 ]; then
            echo "错误: benchmark结果为空或过短，可能未正常执行"
            echo "结果长度: ${#result} 字符"
            echo "结果预览:"
            echo "$result" | head -20
        else
            echo "✓ Benchmark结果已获取，长度: ${#result} 字符"
        fi
        
        # 保存完整结果到单独文件
        save_result "$factor" "$result"
        
        # 提取TTFT mean
        ttft_mean=$(extract_ttft_mean "$result")
        if [ -n "$ttft_mean" ]; then
            save_ttft_summary "$factor" "$ttft_mean"
        else
            echo "警告: 未能提取到TTFT mean值"
            save_ttft_summary "$factor" "N/A"
        fi
        
        # 更新摘要
        {
            echo ""
            echo "Factor ${factor}:"
            echo "--------------------------------------------------------------------------------"
            echo "$result" | grep -iE "(throughput|latency|tokens|req/s|time|ttft)" || echo "未找到关键指标"
            if [ -n "$ttft_mean" ]; then
                echo "TTFT Mean: ${ttft_mean}"
            fi
            echo ""
        } >> "$summary_file"
        
        # 停止服务器
        kill_sglang_servers
        
        echo ""
        echo "✓ Factor ${factor} 测试完成"
        
        # 测试间隔
        if [ $test_num -lt $total ]; then
            echo ""
            echo "等待 10 秒后继续下一个测试..."
            sleep 10
        fi
    done
    
    echo ""
    echo "=================================================================================="
    echo "所有测试完成!"
    echo "结果保存在: ${RESULTS_DIR}"
    echo "摘要文件: ${summary_file}"
    echo "TTFT汇总文件: ${RESULTS_DIR}/ttft_summary.csv"
    echo "=================================================================================="
    
    # 显示TTFT汇总
    if [ -f "${RESULTS_DIR}/ttft_summary.csv" ]; then
        echo ""
        echo "TTFT Mean 汇总:"
        echo "--------------------------------------------------------------------------------"
        cat "${RESULTS_DIR}/ttft_summary.csv"
        echo ""
    fi
}

# 运行主程序
main "$@"

