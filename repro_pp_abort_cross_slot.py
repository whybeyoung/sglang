"""E2E repro for https://github.com/sgl-project/sglang/pull/29405

Bug: under pipeline parallelism, `Scheduler.abort_request` only scanned
`self.running_batch` (aliased to the *current* microbatch slot) plus the stale
`self.cur_batch` (the *previous* step's batch). Requests decoding in any other
microbatch slot (`running_mbs[]` / `mbs[]`) were never marked FINISH_ABORT and
kept generating to completion. Needs more than 2 microbatch slots, i.e.
pp_loop_size = pp_size + pp_async_batch_depth > 2.

Pure client script — launch the server yourself, e.g.:
    python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B \
        --pp-size 4 --pp-max-micro-batch-size 1 --port 8000
(2 GPUs also work: --pp-size 2 --pp-async-batch-depth 1)

Then:
    python repro_pp_abort_cross_slot.py --base-url http://127.0.0.1:8000

PD-disaggregation deployments: workers reject direct /generate ("Disaggregated
request received without bootstrap room id"), and the sgl-router has no
/abort_request route (mini_lb needs the /abort_request broadcast patch). Send
generate through the router; abort via the patched LB or directly to a worker.

If PP is on the DECODE side, the classic symptom shows (streams keep producing
tokens after abort):
    python repro_pp_abort_cross_slot.py \
        --base-url http://<router>:<port> --abort-url http://<decode-worker>:<port>

If PP is only on the PREFILL side, streams close normally (decode aborts fine)
— the bug instead shows as the prefill worker finishing the whole chunked
prefill + KV transfer of already-aborted requests. Watch its load:
    python repro_pp_abort_cross_slot.py \
        --base-url http://<router>:<port> \
        --abort-url http://<patched-lb-or-prefill-worker>:<port> \
        --watch-load-url http://<prefill-worker>:<port>
Prefill worker needs --pp-size >= 3 (or pp 2 + --pp-async-batch-depth 1) AND
--pp-max-micro-batch-size 1; if drain times look ambiguous, raise --input-len.

  RED (bug reproduces) — server built BEFORE the fix
  (2e6670739973533ac4bc2810613d4d4d3d564b1c~1): only ~2 of the requests end
  with finish_reason=abort; the others keep streaming long after abort_all.
  Script exits 1, prints "BUG REPRODUCED".

  GREEN (fix verified) — server on the fix commit or newer: all requests end
  with finish_reason=abort within the grace period. Script exits 0, prints
  "FIX VERIFIED".

Detection logic:
  1. Start one streaming /generate per microbatch slot (ignore_eos, large
     max_new_tokens) with known rids. With --pp-max-micro-batch-size 1, N
     concurrently-decoding requests necessarily occupy N distinct slots.
  2. Wait until every request is actively decoding (has produced tokens).
  3. POST /abort_request {"abort_all": true}.
  4. A request is "aborted OK" if its stream closes within the grace period
     with finish_reason abort. A request that keeps streaming past the grace
     period (or runs to finish_reason=length) leaked through the abort scan.
"""

import argparse
import json
import sys
import threading
import time
import uuid

import requests

PROMPT = "Count slowly from one to one thousand, spelling out every number:"
DECODE_TOKENS_BEFORE_ABORT = 8


class StreamWorker(threading.Thread):
    def __init__(
        self, base_url, rid, max_new_tokens, timeout_s, input_len=0, seq_offset=0
    ):
        super().__init__(daemon=True)
        self.base_url = base_url
        self.rid = rid
        self.max_new_tokens = max_new_tokens
        self.timeout_s = timeout_s
        self.input_len = input_len
        self.seq_offset = seq_offset
        self.tokens = 0
        self.finish_reason = None
        self.closed_at = None
        self.error = None
        self.stop_event = threading.Event()

    def run(self):
        payload = {
            "rid": self.rid,
            "stream": True,
            "sampling_params": {
                "max_new_tokens": self.max_new_tokens,
                "ignore_eos": True,
                "temperature": 0.0,
            },
        }
        if self.input_len > 0:
            # unique sequence per request so radix/prefix caching can't let
            # later requests skip their prefill
            payload["input_ids"] = [
                100 + ((self.seq_offset * 131 + i) % 997) for i in range(self.input_len)
            ]
        else:
            payload["text"] = PROMPT
        stray_lines = []
        try:
            with requests.post(
                f"{self.base_url}/generate",
                json=payload,
                stream=True,
                timeout=(10, self.timeout_s),
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines(decode_unicode=True):
                    if self.stop_event.is_set():
                        break
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        # mini_lb streams backend error bodies through as-is
                        # (it never checks the backend status code), so a 400
                        # from a worker shows up here instead of raising.
                        stray_lines.append(line)
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        # generated text may contain \r etc. that iter_lines
                        # splits on, truncating the SSE line; chunks are
                        # cumulative so skipping one loses nothing
                        continue
                    meta = chunk.get("meta_info", {})
                    self.tokens = meta.get("completion_tokens", self.tokens)
                    fr = meta.get("finish_reason")
                    if fr:
                        self.finish_reason = (
                            fr.get("type") if isinstance(fr, dict) else fr
                        )
        except Exception as e:  # noqa: BLE001 - report, don't crash the repro
            self.error = repr(e)
        finally:
            self.closed_at = time.monotonic()
            if (
                self.error is None
                and self.finish_reason is None
                and not self.stop_event.is_set()
            ):
                body = " | ".join(stray_lines)[:500] or "no data chunks received"
                self.error = f"stream ended without finish_reason: {body}"


def wait_http_ready(base_url, timeout_s):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            r = requests.get(f"{base_url}/health", timeout=5)
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise RuntimeError(f"server at {base_url} not ready within {timeout_s}s")


def fetch_server_info(url):
    """Best-effort: PD routers may not expose /get_server_info."""
    try:
        r = requests.get(f"{url}/get_server_info", timeout=5)
        if r.status_code == 200:
            info = r.json()
            # mini_lb returns {"prefill": [...], "decode": [...]}; the PP abort
            # bug lives in the DECODE workers' scheduler, so read their config.
            decode_infos = info.get("decode")
            if isinstance(decode_infos, list) and decode_infos:
                print(
                    f"[info] {url} is a PD LB; using decode worker config "
                    f"({len(decode_infos)} decode worker(s))"
                )
                return decode_infos[0]
            return info
    except (requests.RequestException, ValueError):
        pass
    return {}


def sample_load(url):
    """Sum num_reqs / num_tokens across dp ranks from /get_load; None on error."""
    try:
        r = requests.get(f"{url}/get_load", timeout=5)
        if r.status_code == 200:
            loads = r.json()
            return (
                sum(x["num_reqs"] for x in loads),
                sum(x["num_tokens"] for x in loads),
            )
    except (requests.RequestException, ValueError, KeyError, TypeError):
        pass
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="where /generate goes; in PD-disagg deployments this must be the router",
    )
    ap.add_argument(
        "--abort-url",
        default=None,
        help="where /abort_request goes (default: --base-url). In PD-disagg "
        "deployments the router has no /abort_request, so point this at "
        "the DECODE worker's own HTTP port",
    )
    ap.add_argument(
        "--num-reqs",
        type=int,
        default=0,
        help="how many concurrent requests to start (default: microbatch "
        "slot count from /get_server_info, else 4)",
    )
    ap.add_argument("--max-new-tokens", type=int, default=6000)
    ap.add_argument(
        "--grace-s",
        type=float,
        default=25.0,
        help="seconds after abort before declaring a leak",
    )
    ap.add_argument("--ready-timeout-s", type=float, default=60.0)
    ap.add_argument(
        "--watch-load-url",
        default=None,
        help="PP PREFILL worker URL to watch via /get_load. Use when only "
        "the prefill side runs PP (decode aborts fine, so streams close "
        "normally): the bug then shows as the prefill worker keeping "
        "aborted requests until their chunked prefill completes",
    )
    ap.add_argument(
        "--input-len",
        type=int,
        default=0,
        help="send input_ids of this length instead of a text prompt "
        "(default 32768 in --watch-load-url mode: prefill must be slow "
        "enough that the leak window is observable)",
    )
    ap.add_argument(
        "--leak-window-s",
        type=float,
        default=3.0,
        help="watch mode: prefill load must drain to 0 within this many "
        "seconds after abort, else it's a leak",
    )
    ap.add_argument(
        "--watch-min-reqs",
        type=int,
        default=0,
        help="watch mode: abort once the watched worker has at least this "
        "many in-flight reqs (default: its microbatch slot count, "
        "min 3). Prefill is transient, so requiring all --num-reqs "
        "at once is unrealistic",
    )
    args = ap.parse_args()
    abort_url = args.abort_url or args.base_url
    if args.watch_load_url and args.input_len == 0:
        args.input_len = 32768

    wait_http_ready(args.base_url, args.ready_timeout_s)
    info = (
        fetch_server_info(args.watch_load_url)
        if args.watch_load_url
        else (fetch_server_info(abort_url) or fetch_server_info(args.base_url))
    )
    n_slots = 0
    if info:
        pp_size = info.get("pp_size", 1)
        async_depth = info.get("pp_async_batch_depth", 0) or 0
        mb_size = info.get("pp_max_micro_batch_size")
        n_slots = pp_size + async_depth
        print(
            f"[info] pp_size={pp_size} pp_async_batch_depth={async_depth} "
            f"pp_max_micro_batch_size={mb_size} -> {n_slots} microbatch slots"
        )
        if n_slots <= 2:
            print(
                "[warn] bug needs > 2 microbatch slots; this config cannot reproduce it"
            )
        if mb_size != 1:
            print(
                "[warn] --pp-max-micro-batch-size 1 recommended so each request "
                "occupies its own slot; results may be inconclusive otherwise"
            )
    else:
        print(
            "[warn] /get_server_info unavailable (router?); cannot verify PP slot config"
        )

    n_reqs = args.num_reqs or max(n_slots, 4)
    run_tag = uuid.uuid4().hex[:8]
    stream_timeout = args.grace_s + args.max_new_tokens * 2  # generous read timeout
    workers = [
        StreamWorker(
            args.base_url,
            f"ppabort-{run_tag}-{i}",
            args.max_new_tokens,
            stream_timeout,
            input_len=args.input_len,
            seq_offset=i + 1,
        )
        for i in range(n_reqs)
    ]
    for w in workers:
        w.start()

    if args.watch_load_url:
        # Prefill-side mode: prefill is transient, so don't wait for all
        # requests at once — abort as soon as the watched worker holds enough
        # in-flight reqs to occupy more than 2 microbatch slots.
        watch_min = args.watch_min_reqs or max(n_slots, 3)
        print(
            f"[step] started {n_reqs} streaming requests (input_len={args.input_len}), "
            f"waiting until {args.watch_load_url} holds >= {watch_min} in-flight reqs ..."
        )
        deadline = time.monotonic() + 120
        last_report = 0.0
        while time.monotonic() < deadline:
            if any(w.error for w in workers):
                for w in workers:
                    if w.error:
                        raise RuntimeError(
                            f"request {w.rid} failed before abort: {w.error}"
                        )
            if all(w.closed_at is not None for w in workers):
                states = {w.rid: (w.finish_reason, w.tokens) for w in workers}
                raise RuntimeError(
                    f"all streams ended before the watched worker ever showed them "
                    f"prefilling — they never reached the PP prefill worker: {states}"
                )
            s = sample_load(args.watch_load_url)
            if s and s[0] >= watch_min and s[1] > 0:
                break
            now = time.monotonic()
            if now - last_report > 2:
                last_report = now
                n_active = sum(1 for w in workers if w.closed_at is None)
                n_decoding = sum(1 for w in workers if w.tokens > 0)
                print(
                    f"[wait] watched load={s} (need num_reqs>={watch_min}); "
                    f"streams: {n_active} open, {n_decoding} already decoding"
                )
                if s is not None and s[0] == 0 and n_active > n_decoding:
                    print(
                        "[warn] watched worker reports 0 reqs while streams are "
                        "queued/prefilling elsewhere — is --watch-load-url really "
                        "the prefill worker's HTTP port (not its bootstrap port, "
                        "not an idle replica)?"
                    )
            time.sleep(0.5)
        else:
            raise RuntimeError(
                f"watched worker never showed {watch_min} in-flight reqs within 120s; "
                f"last sample={sample_load(args.watch_load_url)!r}"
            )
        time.sleep(0.5)  # let chunked prefill spread across the microbatch slots
        load_at_abort = sample_load(args.watch_load_url)
        print(
            f"[step] watched load (num_reqs, num_tokens) = {load_at_abort}; sending abort_all ..."
        )
    else:
        print(
            f"[step] started {n_reqs} streaming requests, waiting until all are decoding ..."
        )
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if any(w.error for w in workers):
                for w in workers:
                    if w.error:
                        raise RuntimeError(
                            f"request {w.rid} failed before abort: {w.error}"
                        )
            if all(w.tokens >= DECODE_TOKENS_BEFORE_ABORT for w in workers):
                break
            time.sleep(0.2)
        else:
            state = {w.rid: w.tokens for w in workers}
            raise RuntimeError(f"not all requests reached decode within 120s: {state}")

    tokens_at_abort = {w.rid: w.tokens for w in workers}
    if not args.watch_load_url:
        print(
            f"[step] all decoding (tokens: {list(tokens_at_abort.values())}); sending abort_all ..."
        )
    abort_t0 = time.monotonic()
    r = requests.post(
        f"{abort_url}/abort_request", json={"rid": "", "abort_all": True}, timeout=10
    )
    r.raise_for_status()

    print(f"[step] abort_all accepted; grace period {args.grace_s}s ...")
    grace_deadline = abort_t0 + args.grace_s

    if args.watch_load_url:
        traj = []
        drain_at = None
        while time.monotonic() < grace_deadline:
            s = sample_load(args.watch_load_url)
            dt = time.monotonic() - abort_t0
            if s:
                traj.append((dt, s[0], s[1]))
                if s[0] == 0:
                    drain_at = dt
                    break
            time.sleep(0.3)

        print()
        print(f"{'t_after_abort':<14} {'num_reqs':<9} num_tokens")
        for dt, nr, nt in traj:
            print(f"{dt:>10.1f}s   {nr:<9} {nt}")
        print()
        if drain_at is not None and drain_at <= args.leak_window_s:
            print(
                f"FIX VERIFIED (prefill-side): watched worker drained to 0 reqs "
                f"{drain_at:.1f}s after abort_all (threshold {args.leak_window_s}s)."
            )
            rc = 0
        else:
            where = (
                f"drained only after {drain_at:.1f}s"
                if drain_at is not None
                else f"still {traj[-1][1] if traj else '?'} reqs at grace deadline"
            )
            print(
                f"BUG REPRODUCED (prefill-side): {where} — aborted requests kept "
                f"running their chunked prefill in microbatch slots the "
                f"pre-#29405 abort scan never visits."
            )
            rc = 1
        for w in workers:
            w.stop_event.set()
        return rc

    while time.monotonic() < grace_deadline:
        if all(w.closed_at is not None for w in workers):
            break
        time.sleep(0.5)

    leaked, aborted = [], []
    print()
    print(
        f"{'rid':<24} {'aborted?':<9} {'finish_reason':<14} {'tokens_after_abort':<19} closed_after_abort"
    )
    for w in workers:
        extra = w.tokens - tokens_at_abort[w.rid]
        closed = w.closed_at is not None and w.closed_at <= grace_deadline
        ok = closed and w.finish_reason == "abort"
        dur = f"{w.closed_at - abort_t0:6.1f}s" if w.closed_at else "  still streaming"
        print(
            f"{w.rid:<24} {'YES' if ok else 'NO':<9} {str(w.finish_reason):<14} {extra:<19} {dur}"
        )
        (aborted if ok else leaked).append(w)

    print()
    if leaked:
        print(
            f"BUG REPRODUCED: {len(leaked)}/{n_reqs} requests survived abort_all — "
            f"they sit in microbatch slots the pre-#29405 abort scan never visits."
        )
        rc = 1
    else:
        print(
            f"FIX VERIFIED: all {n_reqs}/{n_reqs} requests were aborted "
            f"(finish_reason=abort) within {args.grace_s}s."
        )
        rc = 0

    for w in workers:
        w.stop_event.set()
    return rc


if __name__ == "__main__":
    sys.exit(main())
