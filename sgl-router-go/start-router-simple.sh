#!/bin/bash
# Quick start script for SGLang Router on 10.109.185.20:8002

./sglang-router \
  --host=10.109.185.20 \
  --port=8002 \
  --policy=round_robin \
  --log-level=info

