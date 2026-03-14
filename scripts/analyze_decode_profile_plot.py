#!/usr/bin/env python3
"""
Run decode profile analysis and draw operator time proportion pie chart.
Usage: python scripts/analyze_decode_profile_plot.py <profile_dir> [--rank 0] [--top-pie 14] [--out decode_operator_pie.png]
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

# Reuse filter from analyze_decode_profile_trace
def _is_gpu_or_comm(name: str) -> bool:
    n = name.lower()
    return (
        "void " in n or "cuda" in n or n.startswith("aten::")
        or "deep_gemm" in n or "deep_ep" in n or "flash_" in n or "nvjet" in n
        or n.startswith("nccl") or "fusedaddrmsnorm" in n or "elementwise" in n
        or "sbtopk" in n or "quant_" in n or "catarraybatched" in n
        or "_kernel" in n or "rmsnormkernel" in n
    )


def _short_label(name: str, max_len: int = 26) -> str:
    """Shorten for pie: readable op/kernel name."""
    # Known mappings for common hotspots
    if "aten::copy_" in name:
        return "aten::copy_"
    if "aten::to" in name and "_to_copy" not in name:
        return "aten::to"
    if "aten::_to_copy" in name:
        return "aten::_to_copy"
    if "nsa_backend" in name and "init" in name:
        return "nsa init_forward_meta"
    if "eagle_draft_extend_cuda_graph" in name:
        return "eagle_draft_extend"
    if "cuda_graph_runner" in name and "replay" in name:
        return "cuda_graph replay"
    if "cudaMemcpyAsync" in name:
        return "cudaMemcpyAsync"
    if "deep_ep::internode_ll::dispatch" in name:
        return "deep_ep::dispatch"
    if "deep_ep::internode_ll::combine" in name:
        return "deep_ep::combine"
    if "deep_gemm::sm90_fp8_gemm" in name:
        return "deep_gemm FP8 GEMM"
    if "cudaStreamSynchronize" in name:
        return "cudaStreamSync"
    if "nccl:_all_gather" in name:
        return "nccl AllGather"
    if "ncclDevKernel_AllGather" in name:
        return "nccl AllGather kernel"
    if "flash_fwd_splitkv_mla" in name:
        return "flash MLA decode"
    if "flash_fwd_mla_combine" in name:
        return "flash MLA combine"
    if "replay" in name and "graphs.py" in name:
        return "cuda graph replay"
    if "cudaGraphLaunch" in name:
        return "cudaGraphLaunch"
    if "sbtopk::gatherTopK" in name:
        return "sbtopk gatherTopK"
    if "per_token_group_quant_8bit" in name:
        return "quant_8bit kernel"
    if "nvjet_tst" in name:
        return "nvjet MoE"
    if "FusedAddRMSNorm" in name:
        return "FusedAddRMSNorm"
    if "RMSNormKernel" in name:
        return "RMSNorm"
    if "elementwise_kernel" in name:
        return "elementwise"
    if len(name) <= max_len:
        return name
    for sep in ("(", "::", ":"):
        if sep in name:
            parts = name.split(sep)
            tail = (parts[-1].strip(")") if sep == "(" else parts[-1])[:max_len]
            if tail:
                return tail
    return name[: max_len - 3] + "..."


def main() -> int:
    parser = argparse.ArgumentParser(description="Decode profile analysis + operator time pie chart")
    parser.add_argument("path", help="Directory with *.trace.json.gz")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--top-pie", type=int, default=14, help="Number of slices in pie (rest as Other)")
    parser.add_argument("--out", default="decode_operator_pie.png", help="Output image path")
    args = parser.parse_args()

    path = Path(args.path)
    trace_files = sorted(path.glob("*.trace.json.gz"))
    if not trace_files:
        print(f"No *.trace.json.gz in {path}", file=sys.stderr)
        return 1
    rank_str = f"TP-{args.rank}-DP-{args.rank}-EP-{args.rank}"
    chosen = [f for f in trace_files if rank_str in f.name]
    trace_file = chosen[0] if chosen else trace_files[args.rank % len(trace_files)]

    with gzip.open(trace_file, "rt", encoding="utf-8") as f:
        data = json.load(f)
    events = data.get("traceEvents", [])

    by_name = defaultdict(lambda: {"dur": 0, "count": 0})
    for e in events:
        if e.get("ph") != "X":
            continue
        name, dur = e.get("name", ""), e.get("dur", 0)
        if not name or dur <= 0:
            continue
        by_name[name]["dur"] += dur
        by_name[name]["count"] += 1
    items = [(n, v) for n, v in by_name.items() if _is_gpu_or_comm(n)]
    items.sort(key=lambda x: -x[1]["dur"])
    total_dur = sum(v["dur"] for _, v in items)

    # Build pie data: top N + Other
    n_pie = min(args.top_pie, len(items))
    sizes = []
    labels = []
    for i in range(n_pie):
        name, v = items[i]
        pct = 100 * v["dur"] / total_dur if total_dur else 0
        sizes.append(pct)
        labels.append(f"{_short_label(name)} ({pct:.1f}%)")
    if len(items) > n_pie:
        other_dur = sum(v["dur"] for _, v in items[n_pie:])
        other_pct = 100 * other_dur / total_dur if total_dur else 0
        sizes.append(other_pct)
        labels.append(f"Other ({other_pct:.1f}%)")

    # Print table
    print(f"# Trace: {trace_file.name}  total_dur: {total_dur/1e6:.2f}s (GPU/comm)")
    print(f"{'%':<8} {'dur(s)':<10} {'count':<10}  name")
    print("-" * 95)
    for name, v in items[: 30]:
        pct = 100 * v["dur"] / total_dur if total_dur else 0
        print(f"{pct:<8.1f} {v['dur']/1e6:<10.3f} {v['count']:<10}  {name[:75]}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib", file=sys.stderr)
        return 0

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.Set3.colors if hasattr(plt.cm.Set3, "colors") else None
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="", startangle=90, colors=colors
    )
    for t in texts:
        t.set_fontsize(8)
    plt.title(f"Decode operator time proportion (GPU/comm)\n{trace_file.name}", fontsize=11)
    plt.tight_layout()
    out_path = path / args.out if not Path(args.out).is_absolute() else Path(args.out)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nPie chart saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
