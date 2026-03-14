#!/bin/bash
cd /usr/local/src/sglang

# Kill old processes
pkill -9 -f "sglang" 2>/dev/null
sleep 2

# Start node 0 (PP rank 0) with debugging flags
NCCL_DEBUG=WARN python3 -m sglang.launch_server \
  --model-path /work/models/ \
  --port 30000 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --served-model-name kimi-k2.5 \
  --disable-radix-cache \
  --tp 8 \
  --pp-size 2 \
  --mem-fraction-static 0.88 \
  --moe-dense-tp-size 1 \
  --nnodes 2 \
  --dist-init-addr 26.5.27.239:20102 \
  --node-rank 0 \
  --disable-cuda-graph \
  2>&1 | tee /tmp/sglang_debug0.log
