#!/usr/bin/env python3
"""
Compute decode-phase throughput from bench_serving JSONL output (--output-details).

Usage:
  python scripts/pd_decode_throughput_from_bench.py benchmark_xxx.jsonl

Use with PD benchmark (e.g. --random-input-len 100000 --random-output-len 512)
to get decode-only tokens/s when prefill is done on a separate node.
"""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Compute decode-phase throughput from bench_serving JSONL (--output-details)."
    )
    parser.add_argument(
        "jsonl",
        type=str,
        help="Path to JSONL file from bench_serving with --output-details",
    )
    parser.add_argument(
        "--method",
        choices=("itl", "latency", "both"),
        default="both",
        help="itl = 1/mean(itl); latency = (output_len-1)/(latency-ttft); both = print both",
    )
    args = parser.parse_args()

    decode_by_itl = []
    decode_by_latency = []

    with open(args.jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            itls = d.get("itls", [])
            ttfts = d.get("ttfts", [])
            latencies = d.get("latencies", [])
            output_lens = d.get("output_lens", [])
            errors = d.get("errors", [])

            n = len(itls)
            if n != len(ttfts) or (latencies and n != len(latencies)) or n != len(output_lens):
                print("Warning: length mismatch in jsonl line, skipping.", file=sys.stderr)
                continue

            for i in range(n):
                if errors and i < len(errors) and errors[i]:
                    continue
                itl = itls[i]
                if itl:
                    decode_by_itl.append(len(itl) / sum(itl))
                if latencies and i < len(latencies) and i < len(ttfts) and i < len(output_lens):
                    ttft = ttfts[i]
                    lat = latencies[i]
                    out_len = output_lens[i]
                    decode_time = lat - ttft
                    if decode_time > 0 and out_len > 0:
                        decode_by_latency.append((out_len - 1) / decode_time)

    if not decode_by_itl and not decode_by_latency:
        print("No valid samples (empty itls or missing latencies).", file=sys.stderr)
        sys.exit(1)

    if args.method in ("itl", "both") and decode_by_itl:
        mean_itl = sum(decode_by_itl) / len(decode_by_itl)
        print(f"Decode throughput (from ITL):     {mean_itl:.2f} tokens/s  (samples: {len(decode_by_itl)})")
    if args.method in ("latency", "both") and decode_by_latency:
        mean_lat = sum(decode_by_latency) / len(decode_by_latency)
        print(f"Decode throughput (from latency): {mean_lat:.2f} tokens/s  (samples: {len(decode_by_latency)})")


if __name__ == "__main__":
    main()
