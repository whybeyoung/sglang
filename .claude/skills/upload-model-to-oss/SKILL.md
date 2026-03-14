---
name: upload-model-to-oss
description: Upload model files from GPU server /data2 to Alibaba Cloud OSS using ossutil sync. Use when asked to upload, sync, or push model weights to OSS bucket maas-resource-bj.
---

# Upload Model to Alibaba Cloud OSS

Upload model directories from the GPU host machine to OSS bucket `maas-resource-bj`.

## Infrastructure

### Machine Topology

```
Local → Jump Host (10.104.102.78:30022) → GPU Host (26.5.27.241:22)
                                               │
                                          ┌────┴────────────────┐
                                          │  Host /data2/        │
                                          │  (shared with quant  │
                                          │   container via bind │
                                          │   mount)             │
                                          └─────────────────────┘
```

**CRITICAL DISTINCTION**:
- **Quant container** (port 5022): Used for quantization work, runs inside Docker
- **GPU Host machine** (port 22): The actual host, use this for ossutil uploads
- Both share the same `/data2` directory (bind mount), so model files are accessible from both

### SSH Commands

**To connect to host (port 22) for ossutil:**
```bash
sshpass -p 'Aipaasxylx1.t!@#' ssh \
  -o StrictHostKeyChecking=no \
  -o ConnectTimeout=20 \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=10 \
  -p 22 \
  -o ProxyJump=root@10.104.102.78:30022 \
  root@26.5.27.241 "<command>"
```

**To connect to quant container (port 5022):** (for quantization only, NOT ossutil)
```bash
sshpass -p 'Aipaasxylx1.t!@#' ssh -p 5022 -o ProxyJump=root@10.104.102.78:30022 root@26.5.27.241
```

## OSS Upload Command

```bash
ossutil sync --force --jobs=200 \
  /data2/<MODEL_DIR>/ \
  oss://maas-resource-bj/<OSS_PATH>/ \
  -e oss-cn-beijing-internal.aliyuncs.com
```

- `--jobs=200`: parallel upload threads
- `-e oss-cn-beijing-internal.aliyuncs.com`: internal endpoint (faster, no egress cost)
- Always use trailing slash on both source and destination

## Workflow

### Step 1: Check ossutil is available on host
```bash
sshpass -p 'Aipaasxylx1.t!@#' ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 \
  -p 22 -o ProxyJump=root@10.104.102.78:30022 root@26.5.27.241 \
  "which ossutil || which ossutil2 && ossutil version"
```

### Step 2: Verify source directory exists
```bash
sshpass -p 'Aipaasxylx1.t!@#' ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 \
  -p 22 -o ProxyJump=root@10.104.102.78:30022 root@26.5.27.241 \
  "du -sh /data2/<MODEL_DIR>/ && ls /data2/<MODEL_DIR>/ | wc -l"
```

### Step 3: Start background upload with nohup
```bash
sshpass -p 'Aipaasxylx1.t!@#' ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 \
  -p 22 -o ProxyJump=root@10.104.102.78:30022 root@26.5.27.241 \
  "nohup ossutil sync --force --jobs=200 \
    /data2/<MODEL_DIR>/ \
    oss://maas-resource-bj/<OSS_PATH>/ \
    -e oss-cn-beijing-internal.aliyuncs.com \
    > /data2/oss_upload_<MODEL_NAME>.log 2>&1 & echo PID=\$!"
```

### Step 4: Monitor progress
```bash
sshpass -p 'Aipaasxylx1.t!@#' ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 \
  -p 22 -o ProxyJump=root@10.104.102.78:30022 root@26.5.27.241 \
  "tail -20 /data2/oss_upload_<MODEL_NAME>.log && ps aux | grep ossutil | grep -v grep"
```

## Common Upload Examples

### Upload quantized FP8 model
```bash
ossutil sync --force --jobs=200 \
  /data2/Kimi-K25-FP8-new2/ \
  oss://maas-resource-bj/Kimi-K25-FP8-new2/ \
  -e oss-cn-beijing-internal.aliyuncs.com
```

### Upload BF16 model
```bash
ossutil sync --force --jobs=200 \
  /data2/Kimi-K2.5-BF16/ \
  oss://maas-resource-bj/Kimi-K2.5-BF16/ \
  -e oss-cn-beijing-internal.aliyuncs.com
```

## Notes

- ossutil must be configured with credentials (`~/.ossutilconfig`) on the host machine
- Internal endpoint `oss-cn-beijing-internal.aliyuncs.com` only works from within Alibaba Cloud VPC
- Large models (800GB+) take 30-60 minutes with `--jobs=200`
- If ossutil not found on host, check container port 5022 as fallback (container may also have ossutil)
- Log files go to `/data2/oss_upload_<name>.log` for easy monitoring
