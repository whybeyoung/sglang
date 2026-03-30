#!/usr/bin/env python3
"""Run a mixed long/short pressure workload and detect PP HiCache mismatches.

Example:
python3 test/manual/hicache/pp_hicache_mixed_pressure.py \
  --base-url http://127.0.0.1:8001 \
  --log-file /tmp/sglang_pp2_mooncake_hosthit.log \
  --duration-seconds 600 \
  --parallel 32
"""

from __future__ import annotations

import argparse
import json
import random
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests


HICACHE_MATCH_RE = re.compile(
    r"\[HiCacheMatch\]\s+rid=(?P<rid>\S+).*?"
    r"device_hit=(?P<device_hit>\d+)\s+"
    r"host_hit=(?P<host_hit>\d+)\s+"
    r"(?:total_cached|totall_cached)=(?P<total_cached>\d+)\s+"
    r"page_size=(?P<page_size>\d+)\s+"
    r"pp=(?P<pp>\d+)\s+cp=(?P<cp>\d+)\s+tp=(?P<tp>\d+)\s+"
    r"last_device_node=(?P<last_device_node>-?\d+)\s+"
    r"last_host_node=(?P<last_host_node>-?\d+)"
)


@dataclass(frozen=True)
class MatchRecord:
    rid: str
    pp: int
    cp: int
    tp: int
    device_hit: int
    host_hit: int
    total_cached: int
    page_size: int
    last_device_node: int
    last_host_node: int


@dataclass(frozen=True)
class MismatchRecord:
    rid: str
    cp: int
    tp: int
    pair_index: int
    pp0: MatchRecord
    pp1: MatchRecord


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mixed long/short HiCache pressure workload with PP mismatch detection."
    )
    parser.add_argument("--base-url", required=True, help="Server base URL, e.g. http://127.0.0.1:8000")
    parser.add_argument("--log-file", required=True, help="Server log file to scan for HiCacheMatch lines")
    parser.add_argument("--duration-seconds", type=int, default=900, help="How long to run pressure traffic")
    parser.add_argument("--parallel", type=int, default=32, help="Number of concurrent request workers")
    parser.add_argument("--request-timeout", type=int, default=300, help="Per-request timeout seconds")
    parser.add_argument("--max-new-tokens", type=int, default=16, help="Generated tokens per request")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--long-hot-count", type=int, default=8, help="Reusable long prompt pool size")
    parser.add_argument("--short-hot-count", type=int, default=32, help="Reusable short prompt pool size")
    parser.add_argument("--long-token-len", type=int, default=4096, help="Approximate token count for long prompts")
    parser.add_argument("--short-token-len", type=int, default=128, help="Approximate token count for short prompts")
    parser.add_argument("--long-hot-weight", type=float, default=0.35, help="Traffic weight for reusable long prompts")
    parser.add_argument("--long-churn-weight", type=float, default=0.35, help="Traffic weight for unique-ish long prompts")
    parser.add_argument("--short-hot-weight", type=float, default=0.15, help="Traffic weight for reusable short prompts")
    parser.add_argument("--short-churn-weight", type=float, default=0.15, help="Traffic weight for unique-ish short prompts")
    parser.add_argument("--report-json", help="Optional output path for final JSON summary")
    parser.add_argument("--stop-on-request-error", action="store_true", help="Stop workers after the first request failure")
    parser.add_argument("--stop-on-mismatch", action="store_true", help="Exit non-zero if any PP mismatch is detected")
    return parser.parse_args()


def build_prompt(label: str, approx_tokens: int, salt: str) -> str:
    header = (
        "You are a careful math tutor. Read the context and answer briefly.\n"
        f"Scenario label: {label}\n"
        f"Scenario salt: {salt}\n"
        "Context begins below.\n"
    )
    marker_interval = 256
    body_parts: List[str] = []
    remaining = approx_tokens
    marker_idx = 0
    while remaining > 0:
        chunk = min(marker_interval, remaining)
        body_parts.append(" hi" * chunk)
        remaining -= chunk
        if remaining > 0:
            body_parts.append(f" marker_{label}_{salt}_{marker_idx}")
            marker_idx += 1
    body = "".join(body_parts)
    footer = (
        "\nQuestion: Summarize the repeated pattern in one sentence and report the salt.\n"
        "Answer:"
    )
    return header + body + footer


def build_hot_pools(args: argparse.Namespace) -> Dict[str, List[str]]:
    return {
        "long_hot": [
            build_prompt("long_hot", args.long_token_len, f"hot_{i}")
            for i in range(args.long_hot_count)
        ],
        "short_hot": [
            build_prompt("short_hot", args.short_token_len, f"hot_{i}")
            for i in range(args.short_hot_count)
        ],
    }


def choose_request_type(rng: random.Random, args: argparse.Namespace) -> str:
    choices = [
        ("long_hot", args.long_hot_weight),
        ("long_churn", args.long_churn_weight),
        ("short_hot", args.short_hot_weight),
        ("short_churn", args.short_churn_weight),
    ]
    labels, weights = zip(*choices)
    return rng.choices(labels, weights=weights, k=1)[0]


def make_payload(
    req_type: str,
    worker_id: int,
    iteration: int,
    pools: Dict[str, List[str]],
    rng: random.Random,
    args: argparse.Namespace,
) -> Dict:
    if req_type == "long_hot":
        prompt = rng.choice(pools["long_hot"])
    elif req_type == "short_hot":
        prompt = rng.choice(pools["short_hot"])
    elif req_type == "long_churn":
        prompt = build_prompt(
            "long_churn",
            args.long_token_len,
            f"worker_{worker_id}_iter_{iteration}_{rng.randint(0, 10**9)}",
        )
    else:
        prompt = build_prompt(
            "short_churn",
            args.short_token_len,
            f"worker_{worker_id}_iter_{iteration}_{rng.randint(0, 10**9)}",
        )

    return {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": args.max_new_tokens,
            "ignore_eos": True,
        },
    }


def healthcheck(base_url: str, timeout: int = 30) -> None:
    response = requests.get(f"{base_url}/health", timeout=timeout)
    response.raise_for_status()


def parse_match_records(log_file: Path, start_offset: int) -> List[MatchRecord]:
    if not log_file.exists():
        raise FileNotFoundError(f"log file not found: {log_file}")

    with log_file.open("rb") as f:
        f.seek(start_offset)
        data = f.read().decode("utf-8", errors="ignore")

    records: List[MatchRecord] = []
    for line in data.splitlines():
        match = HICACHE_MATCH_RE.search(line)
        if not match:
            continue
        groups = match.groupdict()
        records.append(
            MatchRecord(
                rid=groups["rid"],
                pp=int(groups["pp"]),
                cp=int(groups["cp"]),
                tp=int(groups["tp"]),
                device_hit=int(groups["device_hit"]),
                host_hit=int(groups["host_hit"]),
                total_cached=int(groups["total_cached"]),
                page_size=int(groups["page_size"]),
                last_device_node=int(groups["last_device_node"]),
                last_host_node=int(groups["last_host_node"]),
            )
        )
    return records


def find_mismatches(
    records: List[MatchRecord],
) -> Tuple[List[MismatchRecord], int, int]:
    by_key: Dict[Tuple[str, int, int], Dict[int, List[MatchRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        by_key[(record.rid, record.cp, record.tp)][record.pp].append(record)

    mismatches: List[MismatchRecord] = []
    paired = 0
    unpaired = 0
    for (rid, cp, tp), pp_map in by_key.items():
        pp0_records = pp_map.get(0, [])
        pp1_records = pp_map.get(1, [])
        pair_count = min(len(pp0_records), len(pp1_records))
        paired += pair_count
        unpaired += abs(len(pp0_records) - len(pp1_records))
        for pair_index in range(pair_count):
            pp0 = pp0_records[pair_index]
            pp1 = pp1_records[pair_index]
            if (
                pp0.device_hit != pp1.device_hit
                or pp0.host_hit != pp1.host_hit
                or pp0.total_cached != pp1.total_cached
                or pp0.last_host_node != pp1.last_host_node
            ):
                mismatches.append(
                    MismatchRecord(
                        rid=rid,
                        cp=cp,
                        tp=tp,
                        pair_index=pair_index,
                        pp0=pp0,
                        pp1=pp1,
                    )
                )
    return mismatches, paired, unpaired


def summarize_records(records: List[MatchRecord]) -> Dict[str, int]:
    host_hit_records = sum(1 for r in records if r.host_hit > 0)
    device_hit_records = sum(1 for r in records if r.device_hit > 0)
    mixed_hit_records = sum(1 for r in records if r.device_hit > 0 and r.host_hit > 0)
    return {
        "records": len(records),
        "host_hit_records": host_hit_records,
        "device_hit_records": device_hit_records,
        "mixed_hit_records": mixed_hit_records,
    }


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    base_url = args.base_url.rstrip("/")
    log_file = Path(args.log_file)

    healthcheck(base_url)
    start_offset = log_file.stat().st_size if log_file.exists() else 0

    pools = build_hot_pools(args)
    deadline = time.time() + args.duration_seconds
    stop_event = threading.Event()
    counters = Counter()
    latencies: List[float] = []
    counter_lock = threading.Lock()
    mismatch_preview: List[Dict] = []

    def worker(worker_id: int) -> None:
        session = requests.Session()
        local_rng = random.Random(args.seed + worker_id * 1000003)
        iteration = 0
        while time.time() < deadline and not stop_event.is_set():
            iteration += 1
            req_type = choose_request_type(local_rng, args)
            payload = make_payload(req_type, worker_id, iteration, pools, local_rng, args)
            start = time.time()
            try:
                response = session.post(
                    f"{base_url}/generate",
                    json=payload,
                    timeout=args.request_timeout,
                )
                latency = time.time() - start
                with counter_lock:
                    counters["requests_total"] += 1
                    counters[f"type_{req_type}"] += 1
                    latencies.append(latency)
                if response.status_code != 200:
                    with counter_lock:
                        counters["request_errors"] += 1
                        counters[f"status_{response.status_code}"] += 1
                    if args.stop_on_request_error:
                        stop_event.set()
                else:
                    with counter_lock:
                        counters["requests_ok"] += 1
            except Exception:
                with counter_lock:
                    counters["requests_total"] += 1
                    counters["request_errors"] += 1
                if args.stop_on_request_error:
                    stop_event.set()

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        for worker_id in range(args.parallel):
            executor.submit(worker, worker_id)

    records = parse_match_records(log_file, start_offset)
    mismatches, paired, unpaired = find_mismatches(records)
    if mismatches:
        mismatch_preview = [
            {
                "rid": item.rid,
                "cp": item.cp,
                "tp": item.tp,
                "pair_index": item.pair_index,
                "pp0": asdict(item.pp0),
                "pp1": asdict(item.pp1),
            }
            for item in mismatches[:10]
        ]

    latencies_sorted = sorted(latencies)
    p50_ms = int(latencies_sorted[int(len(latencies_sorted) * 0.50)] * 1000) if latencies_sorted else 0
    p95_ms = int(latencies_sorted[int(len(latencies_sorted) * 0.95)] * 1000) if latencies_sorted else 0

    summary = {
        "base_url": base_url,
        "log_file": str(log_file),
        "duration_seconds": args.duration_seconds,
        "parallel": args.parallel,
        "requests_total": counters["requests_total"],
        "requests_ok": counters["requests_ok"],
        "request_errors": counters["request_errors"],
        "p50_latency_ms": p50_ms,
        "p95_latency_ms": p95_ms,
        "paired_keys": paired,
        "unpaired_records": unpaired,
        "mismatch_count": len(mismatches),
        "match_summary": summarize_records(records),
        "request_mix": {
            "long_hot": counters["type_long_hot"],
            "long_churn": counters["type_long_churn"],
            "short_hot": counters["type_short_hot"],
            "short_churn": counters["type_short_churn"],
        },
        "mismatch_preview": mismatch_preview,
    }

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.report_json:
        Path(args.report_json).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if args.stop_on_mismatch and mismatches:
        return 2
    if counters["request_errors"] > 0 and args.stop_on_request_error:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
