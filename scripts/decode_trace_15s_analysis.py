#!/usr/bin/env python3
"""
Find the ~17s all_gather block: when does each rank enter (ts) and duration.
Straggler = rank that enters last; others wait for it.
"""
import json
import gzip
from pathlib import Path

TRACE_DIR = Path("/Users/yangyanbo/Downloads/chrome_down/glm5/1772299504.5988448")
RANKS = 16

def get_long_all_gather_and_sync(trace_path):
    """Get events: (ts, dur) for nccl:_all_gather_base or cudaStreamSynchronize with dur > 5s in first 25s."""
    with gzip.open(trace_path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    events = data.get("traceEvents", [])
    min_ts = min(e["ts"] for e in events if "ts" in e)
    window = min_ts + 25e6  # first 25s
    result = []
    for e in events:
        if e.get("ph") != "X" or "ts" not in e:
            continue
        if e["ts"] > window:
            continue
        dur = e.get("dur", 0)
        if dur < 5e6:  # 5s
            continue
        name = e.get("name", "")
        if "nccl:_all_gather_base" in name or "cudaStreamSynchronize" in name or "all_gather" in name and "scheduler_dp_attn" in name:
            result.append((name[:60], e["ts"], dur))
    return min_ts, result

def main():
    print("First long (>5s) all_gather/sync per rank (relative to rank0 min_ts)\n")
    ref_min = None
    rank_data = []
    for r in range(RANKS):
        path = TRACE_DIR / f"1772299504.6059935-TP-{r}-DP-{r}-EP-{r}.trace.json.gz"
        if not path.exists():
            continue
        min_ts, long_evs = get_long_all_gather_and_sync(path)
        if ref_min is None:
            ref_min = min_ts
        rel = [(n, (ts - ref_min) / 1e6, dur / 1e6) for n, ts, dur in long_evs]
        rank_data.append((r, min_ts, long_evs, rel))
        if rel:
            for n, start_s, dur_s in rel[:3]:
                print(f"EP-{r:2d}: start={start_s:.3f}s  dur={dur_s:.2f}s  {n}")
        else:
            print(f"EP-{r:2d}: no long all_gather/sync in first 25s")
    # When does each rank ENTER the long all_gather (ts)?
    print("\n--- When each rank ENTERS the long all_gather (who is straggler?) ---")
    entries = []
    for r, min_ts, long_evs, rel in rank_data:
        for n, ts, dur in long_evs:
            if "nccl:_all_gather_base" in n or "scheduler_dp_attn" in n and "all_gather" in n:
                entries.append((r, ts, dur))
                break
    if entries:
        # Align by ref_min (rank0's min_ts)
        entries_rel = [(r, (ts - ref_min) / 1e6, dur / 1e6) for r, ts, dur in entries]
        entries_rel.sort(key=lambda x: x[1])
        for r, start_s, dur_s in entries_rel:
            print(f"  EP-{r:2d}  enters at {start_s:.3f}s   dur={dur_s:.2f}s")
        latest = max(entries_rel, key=lambda x: x[1])
        print(f"\nStraggler: EP-{latest[0]} enters at {latest[1]:.3f}s (others wait until then)")

if __name__ == "__main__":
    main()
