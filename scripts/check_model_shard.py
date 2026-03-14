#!/usr/bin/env python3
import safetensors.torch as st
import os, sys

model_dir = sys.argv[1] if len(sys.argv) > 1 else "/work/models"
shard1 = os.path.join(model_dir, "model-00001-of-000064.safetensors")
data = st.load_file(shard1)

fp8_count = sum(1 for k in data.keys() if data[k].dtype == st.torch.float8_e4m3fn)
scale_inv_count = sum(1 for k in data.keys() if "scale_inv" in k)
print("FP8 weight count:", fp8_count)
print("scale_inv count:", scale_inv_count)
print("Total keys:", len(data.keys()))
print("Sample keys and dtypes:")
for k in sorted(data.keys())[:10]:
    t = data[k]
    print("  %s: %s %s" % (k, t.dtype, list(t.shape)))
