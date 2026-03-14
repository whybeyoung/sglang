#!/usr/bin/env python3
"""
Analyze decode EP16 Perfetto traces to find cause of head ~15s delay.
Compares first 20s across ranks: nccl all_gather, zmq recv, sync, etc.
"""
import json
import gzip
import os
from pathlib import Path

TRACE_DIR = Path("/Users/yangyanbo/Downloads/chrome_down/glm5/1772299504.5988448")
RANKS = 16

# Key event names (substring match)
KEY_EVENTS = [
    "nccl:_all_gather_base",
    "cudaStreamSynchronize",
    "recv_limit_reached",
    "zmq",
    "recv_pyobj",
    "_allocatable_tokens",
    "gloo:broadcast",
]

def load_events(trace_path):
    with gzip.open(trace_path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("traceEvents", []), data.get("baseTimeNanoseconds", 0)

def analyze_rank(rank):
    fname = f"1772299504.6059935-TP-{rank}-DP-{rank}-EP-{rank}.trace.json.gz"
    path = TRACE_DIR / fname
    if not path.exists():
        return None, None
    events, base_ns = load_events(path)
    # ts in trace is in microseconds (PyTorch profiler)
    t0 = min(e.get("ts", float("inf")) for e in events if "ts" in e)
    # Collect first occurrence and long durations in first 20s (20e6 us)
    window_us = 20_000_000
    first = {}
    long_dur = []  # (name, start_us, dur_us) for dur > 1s
    for e in events:
        if e.get("ph") != "X":
            continue
        ts = e.get("ts")
        dur = e.get("dur", 0)
        rel_us = ts - t0
        if rel_us > window_us:
            continue
        name = e.get("name", "")
        if name not in first:
            first[name] = (rel_us, dur)
        if dur > 1_000_000:  # > 1s
            long_dur.append((name, rel_us, dur))
    return t0, {"first": first, "long_dur": long_dur, "t0": t0}

def main():
    print("=== Decode EP16 trace analysis (first 20s per rank) ===\n")
    # Use EP-0 as time base to align ranks (same wall clock)
    _, ref = analyze_rank(0)
    if ref is None:
        print("Trace dir not found:", TRACE_DIR)
        return
    t0_ref = ref["t0"]

    # 1) First occurrence of nccl all_gather per rank
    print("1) First 'nccl:_all_gather_base' per rank (relative ms from rank0 first event)")
    first_all_gather = []
    for r in range(RANKS):
        _, info = analyze_rank(r)
        if info is None:
            continue
        first = info["first"].get("nccl:_all_gather_base")
        if first:
            rel_ms = (first[0]) / 1000.0
            first_all_gather.append((r, rel_ms, first[1] / 1000.0))
        else:
            first_all_gather.append((r, None, None))
    for r, start_ms, dur_ms in first_all_gather:
        if start_ms is not None:
            print(f"   EP-{r:2d}: first at {start_ms/1000:.2f}s, dur={dur_ms:.1f}ms")
        else:
            print(f"   EP-{r:2d}: no all_gather in first 20s")
    if first_all_gather:
        with_ag = [(r, s) for r, s, d in first_all_gather if s is not None]
        if with_ag:
            latest_rank = max(with_ag, key=lambda x: x[1])
            print(f"   -> Straggler (latest first all_gather): EP-{latest_rank[0]} at {latest_rank[1]/1000:.2f}s\n")

    # 2) Long blocking events (>1s) in first 20s
    print("2) Long blocking events (>1s) in first 20s (sample: EP-0, EP-1)")
    for r in [0, 1]:
        _, info = analyze_rank(r)
        if info is None:
            continue
        long_dur = sorted(info["long_dur"], key=lambda x: x[1])[:15]
        print(f"   EP-{r}:")
        for name, start_us, dur_us in long_dur:
            print(f"      at {start_us/1e6:.2f}s, dur={dur_us/1e6:.2f}s: {name[:70]}")
    print()

    # 3) First recv_limit_reached / zmq recv (decode waiting on prefill?)
    print("3) First 'recv_limit_reached' and ZMQ recv per rank (relative ms)")
    for r in range(RANKS):
        _, info = analyze_rank(r)
        if info is None:
            continue
        recv_limit = info["first"].get("recv_limit_reached")
        zmq_ev = None
        for k in info["first"]:
            if "zmq" in k or "recv_pyobj" in k or "recv_copy" in k:
                zmq_ev = (k, info["first"][k])
                break
        if recv_limit or zmq_ev:
            rl_ms = recv_limit[0] / 1000.0 if recv_limit else None
            zmq_ms = zmq_ev[1][0] / 1000.0 if zmq_ev else None
            print(f"   EP-{r:2d}: recv_limit_reached first at {rl_ms/1000:.2f}s" if rl_ms else f"   EP-{r:2d}: no recv_limit_reached", end="")
            if zmq_ms is not None:
                print(f", zmq first at {zmq_ms/1000:.2f}s")
            else:
                print()
    print()

    # 4) First cudaStreamSynchronize and its duration (often blocks after all_gather)
    print("4) First 'cudaStreamSynchronize' and duration per rank (first 20s)")
    for r in range(RANKS):
        _, info = analyze_rank(r)
        if info is None:
            continue
        sync = info["first"].get("cudaStreamSynchronize")
        if sync:
            print(f"   EP-{r:2d}: first at {sync[0]/1e6:.2f}s, dur={sync[1]/1000:.1f}ms")
    print()

    # 5) When does first all_gather *end* per rank (so we see sync barrier)
    print("5) First nccl all_gather start vs first cudaStreamSynchronize (EP-0 detail)")
    events_0, _ = load_events(TRACE_DIR / "1772299504.6059935-TP-0-DP-0-EP-0.trace.json.gz")
    t0_0 = min(e.get("ts", float("inf")) for e in events_0 if "ts" in e)
    ag_times = []
    sync_times = []
    for e in events_0:
        if e.get("ph") != "X":
            continue
        ts = e.get("ts") - t0_0
        if ts > 20e6:
            break
        name = e.get("name", "")
        if name == "nccl:_all_gather_base":
            ag_times.append((ts / 1e6, e.get("dur", 0) / 1000))
        if name == "cudaStreamSynchronize":
            sync_times.append((ts / 1e6, e.get("dur", 0) / 1000))
    print("   First 5 all_gather (start_s, dur_ms):", ag_times[:5])
    print("   First 5 cudaStreamSynchronize (start_s, dur_ms):", sync_times[:5])

if __name__ == "__main__":
    main()
