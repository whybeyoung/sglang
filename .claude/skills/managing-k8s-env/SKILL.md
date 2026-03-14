---
name: managing-k8s-env
description: "Manages Kubernetes GPU cluster environments via kubectl. Use for checking free GPU nodes, pod status, resource usage, node allocation, and general k8s cluster operations."
---

# 测试环境 K8s 管理

通过 kubectl 管理 Kubernetes GPU 集群，提供常用运维操作的标准化流程。

## 集群连接信息

| 项目 | 值 |
|------|-----|
| 跳板机 | `10.104.102.78:30022`（已免密） |
| K8s Master | `26.5.27.220`（已免密） |
| API Server | `apiserver.cluster.local:6443`（内网，VIP `10.101.5.178`） |
| 默认命名空间 | `aiservice` |

### 执行 kubectl 命令（标准方式）

通过跳板机二跳到 master 执行 kubectl，**所有 kubectl 操作都用这个模式**：

```bash
ssh -o StrictHostKeyChecking=no -p 30022 root@10.104.102.78 \
  "ssh -o StrictHostKeyChecking=no root@26.5.27.220 'kubectl <命令>'"
```

示例：
```bash
# 查看 aiservice 命名空间 pod
ssh -o StrictHostKeyChecking=no -p 30022 root@10.104.102.78 \
  "ssh -o StrictHostKeyChecking=no root@26.5.27.220 'kubectl get pods -n aiservice -o wide'"
```

### SSH 登录 Master（交互式）

```bash
ssh -J root@10.104.102.78:30022 root@26.5.27.220
```

## 前置条件

- 本机 → 跳板机 `10.104.102.78:30022` 已免密
- 跳板机 → Master `26.5.27.220` 已免密
- Master 上已安装 kubectl 并配置好 kubeconfig

## 常用操作

### 1. 查看集群概览

```bash
# 所有节点状态
kubectl get nodes -o wide

# 节点标签（查看 GPU 型号等标签）
kubectl get nodes --show-labels | grep -i gpu
```

### 2. 查看空闲 GPU 节点（推荐）

运行内置脚本，一键输出每节点 GPU 总量/已用/空闲，`<<<` 标记可调度空闲节点：

```bash
bash .claude/skills/managing-k8s-env/scripts/free_gpu.sh
```

示例输出：
```
Node                                   Total     Used     Free                Sched
-----------------------------------------------------------------------------------------------
e01-cn-ioy4gcy9d0o                         8        0        8                   OK  <<<
e01-cn-to347zm0n09                         8        0        8                   OK  <<<
e01-cn-zp54h55j61o                         8        0        8                   OK  <<<
e01-cn-cfn4g5yxc08                         8        0        8   SchedulingDisabled
-----------------------------------------------------------------------------------------------
TOTAL                                    120       80       40
```

### 3. 查看 Pod 状态

```bash
# 所有命名空间的 pod 状态
kubectl get pods --all-namespaces -o wide

# 指定命名空间
kubectl get pods -n <namespace> -o wide

# 查看异常 pod
kubectl get pods --all-namespaces --field-selector status.phase!=Running,status.phase!=Succeeded

# 查看 pod 使用的 GPU 数
kubectl get pods --all-namespaces -o json | python3 -c "
import json, sys
pods = json.load(sys.stdin)['items']
print(f'{'Namespace':<25} {'Pod':<50} {'Node':<30} {'GPU':>5} {'Status':>10}')
print('-' * 125)
for p in pods:
    ns = p['metadata']['namespace']
    name = p['metadata']['name']
    node = p['spec'].get('nodeName', 'Pending')
    status = p['status']['phase']
    gpu = 0
    for c in p['spec'].get('containers', []):
        gpu += int(c.get('resources', {}).get('requests', {}).get('nvidia.com/gpu', 0))
        gpu += int(c.get('resources', {}).get('limits', {}).get('nvidia.com/gpu', 0))
    gpu = gpu // 2 if gpu > 0 else 0  # requests and limits often duplicate
    if gpu > 0:
        print(f'{ns:<25} {name:<50} {node:<30} {gpu:>5} {status:>10}')
"
```

### 4. 节点资源详情

```bash
# 查看单个节点详细资源分配
kubectl describe node <node-name>

# 查看所有节点的 CPU/Memory/GPU 使用情况
kubectl top nodes

# 查看某节点上的所有 pod
kubectl get pods --all-namespaces --field-selector spec.nodeName=<node-name> -o wide
```

### 5. GPU Pod 日志与调试

```bash
# 查看 pod 日志
kubectl logs -n <namespace> <pod-name> --tail=100

# 进入 pod 查看 GPU 状态
kubectl exec -it -n <namespace> <pod-name> -- nvidia-smi

# 查看 pod 事件（排查调度失败原因）
kubectl describe pod -n <namespace> <pod-name> | grep -A 20 "Events"
```

### 6. 常用 Pod 管理

```bash
# 删除 pod（触发重建）
kubectl delete pod -n <namespace> <pod-name>

# 强制删除卡住的 pod
kubectl delete pod -n <namespace> <pod-name> --grace-period=0 --force

# 查看 pod YAML
kubectl get pod -n <namespace> <pod-name> -o yaml
```

### 7. 节点管理

```bash
# 设置节点不可调度（维护时）
kubectl cordon <node-name>

# 恢复节点可调度
kubectl uncordon <node-name>

# 驱逐节点上的 pod（维护前）
kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data
```

### 8. 快速查找

```bash
# 查找特定模型推理服务的 pod
kubectl get pods --all-namespaces | grep -i <model-keyword>

# 查看所有 GPU 相关资源
kubectl get all --all-namespaces -l app=<label>

# 查看集群事件（最近 1 小时）
kubectl get events --all-namespaces --sort-by='.lastTimestamp' | tail -50
```

## 注意事项

- 执行写操作（delete/cordon/drain）前务必确认目标，避免误操作
- `kubectl drain` 会驱逐 pod，生产环境慎用
- GPU 数量统计以 `nvidia.com/gpu` 资源为准
- 不同集群可能使用不同的 GPU 资源名（如 `amd.com/gpu`），按实际情况调整
