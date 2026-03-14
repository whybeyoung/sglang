#!/usr/bin/env python3
"""Check FP8 weights structure for shared experts and MoE experts."""
import safetensors.torch as st
import os, sys

model_dir = sys.argv[1] if len(sys.argv) > 1 else "/data2/Kimi-K25-FP8-new"

print("Checking shared_experts weights...")
for fname in sorted(os.listdir(model_dir)):
    if not fname.endswith(".safetensors"):
        continue
    data = st.load_file(os.path.join(model_dir, fname))
    shared_keys = [k for k in data.keys() if "shared_expert" in k]
    if shared_keys:
        print("=== " + fname + " ===")
        for k in sorted(shared_keys):
            t = data[k]
            print("  %s: dtype=%s shape=%s" % (k, t.dtype, list(t.shape)))
        break

print("\nChecking MoE expert 0 weights (first shard with experts)...")
for fname in sorted(os.listdir(model_dir)):
    if not fname.endswith(".safetensors"):
        continue
    data = st.load_file(os.path.join(model_dir, fname))
    expert_keys = [k for k in data.keys() if "experts.0." in k]
    if expert_keys:
        print("=== " + fname + " ===")
        for k in sorted(expert_keys):
            t = data[k]
            print("  %s: dtype=%s shape=%s" % (k, t.dtype, list(t.shape)))
        break

print("\nChecking dense MLP (layer 0) weights...")
for fname in sorted(os.listdir(model_dir)):
    if not fname.endswith(".safetensors"):
        continue
    data = st.load_file(os.path.join(model_dir, fname))
    dense_keys = [k for k in data.keys() if "layers.0.mlp." in k and "expert" not in k]
    if dense_keys:
        print("=== " + fname + " ===")
        for k in sorted(dense_keys):
            t = data[k]
            print("  %s: dtype=%s shape=%s" % (k, t.dtype, list(t.shape)))
        break

print("\nChecking layer 0 attention weights (should be BF16/unquantized)...")
for fname in sorted(os.listdir(model_dir)):
    if not fname.endswith(".safetensors"):
        continue
    data = st.load_file(os.path.join(model_dir, fname))
    attn_keys = [k for k in data.keys() if "layers.0.self_attn." in k]
    if attn_keys:
        print("=== " + fname + " ===")
        for k in sorted(attn_keys)[:10]:
            t = data[k]
            print("  %s: dtype=%s shape=%s" % (k, t.dtype, list(t.shape)))
        break
