#!/usr/bin/env python3
"""
Aggregate PyTorch profiler trace (Chrome trace JSON) by event name for decode profile analysis.
Output includes each operator's time proportion (%% of total dur).
Usage:
  python scripts/analyze_decode_profile_trace.py <dir_or_single_file> [--rank 0] [--top 80] [--gpu-only]
  python scripts/analyze_decode_profile_trace.py <dir> --all-ranks [--top 100] [--gpu-only]  # sum over all ranks
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path


def _is_gpu_or_comm(name: str) -> bool:
    n = name.lower()
    return (
        "void " in n
        or "cuda" in n
        or n.startswith("aten::")
        or "deep_gemm" in n
        or "deep_ep" in n
        or "flash_" in n
        or "nvjet" in n
        or n.startswith("nccl")
        or "fusedaddrmsnorm" in n
        or "elementwise" in n
        or "sbtopk" in n
        or "quant_" in n
        or "catarraybatched" in n
        or "_kernel" in n
        or "rmsnormkernel" in n
    )


def aggregate_trace(events: list, gpu_only: bool = False) -> list[tuple[str, dict]]:
    by_name: dict[str, dict] = defaultdict(lambda: {"dur": 0, "count": 0})
    for e in events:
        if e.get("ph") != "X":
            continue
        name = e.get("name", "")
        dur = e.get("dur", 0)
        if not name or dur <= 0:
            continue
        by_name[name]["dur"] += dur
        by_name[name]["count"] += 1
    items = list(by_name.items())
    if gpu_only:
        items = [(n, v) for n, v in items if _is_gpu_or_comm(n)]
    items.sort(key=lambda x: -x[1]["dur"])
    return items


def _load_events(trace_file: Path) -> list:
    opener = gzip.open if str(trace_file).endswith(".gz") else open
    with opener(trace_file, "rt", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("traceEvents", [])


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate decode trace by event name (with operator time %%)")
    parser.add_argument("path", help="Directory with *.trace.json.gz or single .trace.json.gz file")
    parser.add_argument("--rank", type=int, default=0, help="Which rank file to use (e.g. 0 for TP-0-DP-0-EP-0)")
    parser.add_argument("--top", type=int, default=80, help="Number of top events to print")
    parser.add_argument("--gpu-only", action="store_true", help="Only show GPU/communication events")
    parser.add_argument(
        "--all-ranks",
        action="store_true",
        help="Process all *.trace.json.gz in dir and sum dur/count by name (global operator time proportion)",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if path.is_file():
        trace_files = [path]
    else:
        trace_files = sorted(path.glob("*.trace.json.gz"))
        if not trace_files:
            print(f"No *.trace.json.gz under {path}", file=sys.stderr)
            return 1
        if not args.all_ranks:
            rank_str = f"TP-{args.rank}-DP-{args.rank}-EP-{args.rank}"
            chosen = [f for f in trace_files if rank_str in f.name]
            trace_files = chosen if chosen else [trace_files[args.rank % len(trace_files)]]

    # Aggregate: single file or merge all
    if args.all_ranks and len(trace_files) > 1:
        by_name: dict[str, dict] = defaultdict(lambda: {"dur": 0, "count": 0})
        total_events = 0
        for trace_file in trace_files:
            events = _load_events(trace_file)
            total_events += len(events)
            for e in events:
                if e.get("ph") != "X":
                    continue
                name, dur = e.get("name", ""), e.get("dur", 0)
                if not name or dur <= 0:
                    continue
                by_name[name]["dur"] += dur
                by_name[name]["count"] += e.get("count", 1)
        items = list(by_name.items())
        if args.gpu_only:
            items = [(n, v) for n, v in items if _is_gpu_or_comm(n)]
        items.sort(key=lambda x: -x[1]["dur"])
        total_dur = sum(v["dur"] for _, v in items)
        print(f"# All-ranks: {len(trace_files)} files, {total_events} events, total_dur: {total_dur/1e6:.2f}s")
    else:
        trace_file = trace_files[0]
        events = _load_events(trace_file)
        items = aggregate_trace(events, gpu_only=args.gpu_only)
        total_dur = sum(v["dur"] for _, v in items)
        print(f"# Trace: {trace_file.name}  (events: {len(events)}, total_dur: {total_dur/1e6:.2f}s)")

    if args.gpu_only:
        print("# GPU/comm only")
    print("# 各算子时间占比 (operator time proportion)")
    print(f"{'dur(s)':<10} {'%':<8} {'count':<10}  name")
    print("-" * 105)
    for name, v in items[: args.top]:
        pct = 100 * v["dur"] / total_dur if total_dur else 0
        print(f"{v['dur']/1e6:<10.3f} {pct:<8.1f} {v['count']:<10}  {name[:82]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
