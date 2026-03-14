#!/usr/bin/env python3
import json
# Get actual model config dimensions
cfg = json.load(open("/work/models/config.json"))
tc = cfg.get("text_config", {})
print("hidden_size:", tc.get("hidden_size"))
print("intermediate_size:", tc.get("intermediate_size"))
print("moe_intermediate_size:", tc.get("moe_intermediate_size"))
print("n_routed_experts:", tc.get("n_routed_experts"))
print("n_shared_experts:", tc.get("n_shared_experts"))
print("num_experts_per_tok:", tc.get("num_experts_per_tok"))
print("num_hidden_layers:", tc.get("num_hidden_layers"))

# Check actual weight shapes in a shard with experts  
import safetensors.torch as st
import os
model_dir = "/work/models"
for fname in sorted(os.listdir(model_dir)):
    if not fname.endswith(".safetensors"):
        continue
    keys = list(st.load_file(os.path.join(model_dir, fname), device="cpu").keys())
    has_expert_scale = any("experts.0.gate_proj.weight_scale_inv" in k for k in keys)
    if has_expert_scale:
        data = st.load_file(os.path.join(model_dir, fname), device="cpu")
        for k in sorted(keys):
            if "experts.0." in k:
                print("%s: %s %s" % (k, data[k].dtype, list(data[k].shape)))
        break
