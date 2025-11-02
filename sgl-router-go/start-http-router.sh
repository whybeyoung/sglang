#!/bin/bash
# Start SGLang Router in HTTP mode
# Worker endpoint: http://10.109.185.20:8002

cd "$(dirname "$0")"

./bin/sglang-router \
  --host=0.0.0.0 \
  --port=30000 \
  --worker-urls="http://10.109.185.20:8001" \
  --policy=round_robin \
  --log-level=info

