#!/bin/bash
# 查看集群空闲 GPU 节点
# 用法: bash scripts/free_gpu.sh

JUMP="root@10.104.102.78"
JUMP_PORT=30022
MASTER="root@26.5.27.220"
SSH_OPTS="-o StrictHostKeyChecking=no -o LogLevel=ERROR"

# Python 脚本 base64 编码传到远端执行，避免嵌套引号转义问题
PY_SCRIPT=$(cat << 'PYSCRIPT' | base64
import json, subprocess

node_data = json.loads(subprocess.check_output(["kubectl","get","nodes","-o","json"]).decode())
pod_data = json.loads(subprocess.check_output(["kubectl","get","pods","--all-namespaces","-o","json"]).decode())

node_gpu_used = {}
for p in pod_data["items"]:
    if p["status"].get("phase") not in ("Running", "Pending"):
        continue
    node = p["spec"].get("nodeName", "")
    for c in p["spec"].get("containers", []):
        g = int(c.get("resources", {}).get("limits", {}).get("nvidia.com/gpu", 0))
        node_gpu_used[node] = node_gpu_used.get(node, 0) + g

fmt = "{:<35} {:<16} {:>8} {:>8} {:>8} {:>20} {}"
print(fmt.format("Node", "IP", "Total", "Used", "Free", "Sched", ""))
print("-" * 115)
total_all = used_all = 0
for n in sorted(node_data["items"], key=lambda x: x["metadata"]["name"]):
    name = n["metadata"]["name"]
    total = int(n["status"].get("allocatable", {}).get("nvidia.com/gpu", "0"))
    if total == 0:
        continue
    ip = ""
    for addr in n["status"].get("addresses", []):
        if addr.get("type") == "InternalIP":
            ip = addr["address"]
            break
    used = node_gpu_used.get(name, 0)
    free = total - used
    sched = "SchedulingDisabled" if n["spec"].get("unschedulable") else "OK"
    mark = " <<<" if free > 0 and sched == "OK" else ""
    print(fmt.format(name, ip, total, used, free, sched, mark))
    total_all += total
    used_all += used

print("-" * 115)
print(fmt.format("TOTAL", "", total_all, used_all, total_all - used_all, "", ""))
PYSCRIPT
)

ssh $SSH_OPTS -p $JUMP_PORT $JUMP \
  "ssh $SSH_OPTS $MASTER 'echo $PY_SCRIPT | base64 -d | python3'"
