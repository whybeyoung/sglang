#!/usr/bin/env python3
"""Fast extraction: first nccl all_gather and first cudaStreamSynchronize per rank."""
import json
import gzip
from pathlib import Path

TRACE_DIR = Path("/Users/yangyanbo/Downloads/chrome_down/glm5/1772299504.5988448")
RANKS = 16

def extract_first_events(trace_path):
    """Stream through trace and get first ts in file, first nccl all_gather, first cudaStreamSynchronize."""
    t0 = None
    first_ag = None  # (ts, dur)
    first_sync = None
    # Read in chunks: PyTorch trace is one big JSON array of events
    with gzip.open(trace_path, "rt", encoding="utf-8") as f:
        content = f.read(2 * 1024 * 1024)  # first 2MB enough for early events
    # Parse just to find traceEvents start
    start = content.find('"traceEvents":')
    if start == -1:
        return None, None, None
    # Find first few complete events and parse manually for ts/name/dur
    import re
    # Get first ts from any event in first 50k chars
    for m in re.finditer(r'"ts":\s*([\d.]+)', content[:100000]):
        if t0 is None:
            t0 = float(m.group(1))
            break
    for m in re.finditer(r'"name":\s*"([^"]+)"[^}]*"ts":\s*([\d.]+)[^}]*"dur":\s*([\d.]+)', content, re.DOTALL):
        name, ts, dur = m.group(1), float(m.group(2)), float(m.group(3))
        if "nccl:_all_gather_base" in name and first_ag is None:
            first_ag = (ts, dur)
        if "cudaStreamSynchronize" in name and first_sync is None:
            first_sync = (ts, dur)
        if first_ag and first_sync:
            break
    # If regex failed, try event-by-event in first 100k
    if first_ag is None or first_sync is None:
        # Simpler: find line with "nccl:_all_gather_base" then next line has ts
        lines = content[:200000].split('\n')
        for i, line in enumerate(lines):
            if '"nccl:_all_gather_base"' in line and first_ag is None and i + 1 < len(lines):
                for j in range(i+1, min(i+3, len(lines))):
                    tm = re.search(r'"ts":\s*([\d.]+)', lines[j])
                    dm = re.search(r'"dur":\s*([\d.]+)', lines[j])
                    if tm and dm:
                        first_ag = (float(tm.group(1)), float(dm.group(1)))
                        break
            if '"cudaStreamSynchronize"' in line and first_sync is None and i + 1 < len(lines):
                for j in range(i+1, min(i+3, len(lines))):
                    tm = re.search(r'"ts":\s*([\d.]+)', lines[j])
                    dm = re.search(r'"dur":\s*([\d.]+)', lines[j])
                    if tm and dm:
                        first_sync = (float(tm.group(1)), float(dm.group(1)))
                        break
            if t0 is None and '"ts":' in line:
                tm = re.search(r'"ts":\s*([\d.]+)', line)
                if tm:
                    t0 = float(tm.group(1))
    return t0, first_ag, first_sync

def main():
    print("Rank  t0(us)        first_all_gather_ts(us)   first_sync_ts(us)   rel_ag(s)  rel_sync(s)")
    print("-" * 90)
    ref_t0 = None
    for r in range(RANKS):
        path = TRACE_DIR / f"1772299504.6059935-TP-{r}-DP-{r}-EP-{r}.trace.json.gz"
        if not path.exists():
            continue
        t0, ag, sync = extract_first_events(path)
        if ref_t0 is None and t0 is not None:
            ref_t0 = t0
        if t0 is None:
            t0 = ref_t0 or 0
        rel_ag = (ag[0] - ref_t0) / 1e6 if ag and ref_t0 else None
        rel_sync = (sync[0] - ref_t0) / 1e6 if sync and ref_t0 else None
        ag_ts = ag[0] if ag else ""
        sync_ts = sync[0] if sync else ""
        print(f"EP-{r:2d}  {t0}  {ag_ts}  {sync_ts}  {rel_ag:.2f}s" if rel_ag else f"EP-{r:2d}  {t0}  {ag_ts}  {sync_ts}  -", end="")
        print(f"  {rel_sync:.2f}s" if rel_sync else "  -")
    # Which rank has latest first all_gather?
    results = []
    for r in range(RANKS):
        path = TRACE_DIR / f"1772299504.6059935-TP-{r}-DP-{r}-EP-{r}.trace.json.gz"
        if not path.exists():
            continue
        t0, ag, sync = extract_first_events(path)
        if ref_t0 is None and t0 is not None:
            ref_t0 = t0
        if ag and ref_t0:
            results.append((r, (ag[0] - ref_t0) / 1e6))
    if results:
        latest = max(results, key=lambda x: x[1])
        print(f"\nStraggler (latest first all_gather): EP-{latest[0]} at {latest[1]:.2f}s relative to ref")

if __name__ == "__main__":
    main()
