from __future__ import annotations

import dataclasses
import logging
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from sglang.srt.disaggregation.kv_events import EventPublisherFactory, KVEventBatch
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import (
    DisaggregationMetrics,
    GetLoadReqInput,
    GetLoadReqOutput,
    GetLoadsReqInput,
    GetLoadsReqOutput,
    LoRAMetrics,
    MemoryMetrics,
    QueueMetrics,
    SpeculativeMetrics,
)
from sglang.srt.managers.scheduler import ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.observability.metrics_collector import (
    DPCooperationInfo,
    QueueCount,
    SchedulerMetricsCollector,
    SchedulerStats,
    compute_routing_key_stats,
)
from sglang.srt.utils import get_bool_env_var
from sglang.srt.utils.device_timer import DeviceTimer, GapTimer
from sglang.srt.utils.scheduler_status_logger import SchedulerStatusLogger

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.schedule_policy import PrefillAdder
    from sglang.srt.managers.scheduler import EmbeddingBatchResult, Scheduler

logger = logging.getLogger(__name__)

RECORD_STEP_TIME = get_bool_env_var("SGLANG_RECORD_STEP_TIME")
LOG_FORWARD_ITERS = envs.SGLANG_LOG_FORWARD_ITERS.get()
ENABLE_METRICS_DEVICE_TIMER = envs.SGLANG_ENABLE_METRICS_DEVICE_TIMER.get()
DEBUG_HICACHE_STATE_GROWTH = (
    os.getenv("SGLANG_DEBUG_HICACHE_STATE_GROWTH", "1") == "1"
)
DEBUG_HICACHE_STATE_GROWTH_INTERVAL_SEC = 10.0


@dataclasses.dataclass
class PrefillStats:
    """Stats for logging prefill batch metrics."""

    log_input_tokens: int
    log_hit_tokens: int
    new_token_ratio: float
    num_running_reqs: QueueCount
    num_new_seqs: int  # len(can_run_list)

    @classmethod
    def from_adder(
        cls,
        adder: PrefillAdder,
        running_reqs: List[Req],
        enable_priority_scheduling: bool = False,
    ):
        return cls(
            log_input_tokens=adder.log_input_tokens,
            log_hit_tokens=adder.log_hit_tokens,
            new_token_ratio=adder.new_token_ratio,
            num_running_reqs=QueueCount.from_reqs(
                running_reqs, enable_priority_scheduling
            ),
            num_new_seqs=len(adder.can_run_list),
        )


class KvMetrics:
    def __init__(self):
        self.request_active_slots = None
        self.request_total_slots = None
        self.kv_active_blocks = None
        self.kv_total_blocks = None
        self.num_requests_waiting = None
        self.gpu_cache_usage_perc = None
        self.gpu_prefix_cache_hit_rate = None
        self.data_parallel_rank = None


class SchedulerMetricsMixin:
    def init_metrics(
        self: Scheduler, tp_rank: int, pp_rank: int, dp_rank: Optional[int]
    ):
        # Basic stats
        self.forward_ct_decode = 0
        self.num_generated_tokens = 0
        self.last_decode_stats_tic = time.perf_counter()
        self.last_prefill_stats_tic = time.perf_counter()
        self.last_prefill_tokens = 0
        self.last_gen_throughput: float = 0.0
        self.last_input_throughput: float = 0.0
        self.step_time_dict = defaultdict(list)  # Dict[batch size -> step time]
        self._hicache_write_backup_barrier_hits = 0
        self._last_hicache_write_backup_barrier_hits = 0
        self._prefill_pick_calls = 0
        self._prefill_pick_total_ms = 0.0
        self._prefill_pick_max_ms = 0.0
        self._prefill_run_calls = 0
        self._prefill_run_total_ms = 0.0
        self._prefill_run_max_ms = 0.0
        self._prefill_forward_calls = 0
        self._prefill_forward_total_ms = 0.0
        self._prefill_forward_max_ms = 0.0
        self._prefill_copy_wait_calls = 0
        self._prefill_copy_wait_total_ms = 0.0
        self._prefill_copy_wait_max_ms = 0.0
        self._prefill_post_calls = 0
        self._prefill_post_total_ms = 0.0
        self._prefill_post_max_ms = 0.0
        self._pp_comm_send_calls = 0
        self._pp_comm_send_total_ms = 0.0
        self._pp_comm_send_max_ms = 0.0
        self._pp_comm_recv_calls = 0
        self._pp_comm_recv_total_ms = 0.0
        self._pp_comm_recv_max_ms = 0.0
        self._pp_comm_tp_bcast_calls = 0
        self._pp_comm_tp_bcast_total_ms = 0.0
        self._pp_comm_tp_bcast_max_ms = 0.0
        self._pp_comm_cp_bcast_calls = 0
        self._pp_comm_cp_bcast_total_ms = 0.0
        self._pp_comm_cp_bcast_max_ms = 0.0
        self._pp_comm_wait_calls = 0
        self._pp_comm_wait_total_ms = 0.0
        self._pp_comm_wait_max_ms = 0.0
        self._pp_comm_wait_req_calls = 0
        self._pp_comm_wait_req_total_ms = 0.0
        self._pp_comm_wait_req_max_ms = 0.0
        self._pp_comm_wait_bootstrap_calls = 0
        self._pp_comm_wait_bootstrap_total_ms = 0.0
        self._pp_comm_wait_bootstrap_max_ms = 0.0
        self._pp_comm_wait_transfer_calls = 0
        self._pp_comm_wait_transfer_total_ms = 0.0
        self._pp_comm_wait_transfer_max_ms = 0.0
        self._pp_comm_wait_output_calls = 0
        self._pp_comm_wait_output_total_ms = 0.0
        self._pp_comm_wait_output_max_ms = 0.0
        self._pp_comm_wait_proxy_calls = 0
        self._pp_comm_wait_proxy_total_ms = 0.0
        self._pp_comm_wait_proxy_max_ms = 0.0
        self._pp_comm_wait_consensus_bootstrap_calls = 0
        self._pp_comm_wait_consensus_bootstrap_total_ms = 0.0
        self._pp_comm_wait_consensus_bootstrap_max_ms = 0.0
        self._pp_comm_wait_release_calls = 0
        self._pp_comm_wait_release_total_ms = 0.0
        self._pp_comm_wait_release_max_ms = 0.0
        self._prefill_pick_early_full_or_empty_count = 0
        self._prefill_pick_chunked_capacity_block_count = 0
        self._prefill_pick_test_retract_block_count = 0
        self._prefill_pick_locally_revoked_break_count = 0
        self._prefill_pick_running_full_break_count = 0
        self._prefill_pick_prefetch_break_count = 0
        self._prefill_pick_write_backup_break_count = 0
        self._prefill_pick_add_no_token_break_count = 0
        self._prefill_pick_add_other_break_count = 0
        self._prefill_pick_empty_result_count = 0
        self._pp_frontier_ack_recv_count = 0
        self._pp_frontier_ack_activate_count = 0
        self._pp_frontier_ack_consume_count = 0
        self._pp_frontier_ack_consume_miss_count = 0
        self._pp_frontier_ack_pending_same_mb_miss_count = 0
        self._pp_frontier_ack_chunked_mismatch_count = 0
        self._pp_frontier_ack_waiting_mismatch_count = 0
        self._pp_frontier_ack_exhausted_count = 0

        # The number of accepted tokens and forward ct for the recent `decode_log_interval` batches (for logging)
        self.spec_num_accepted_tokens = 0
        self.spec_num_forward_ct = 0
        # The total number of accepted tokens and forward ct for the whole server lifetime
        self.spec_total_num_accepted_tokens = 0
        self.spec_total_num_forward_ct = 0

        # For PD disaggregation
        self.kv_transfer_speed_gb_s: float = 0.0
        self.kv_transfer_latency_ms: float = 0.0

        self.stats = SchedulerStats()

        # Metrics
        self.enable_mfu_metrics = False
        self.enable_metrics = self.server_args.enable_metrics
        self.is_stats_logging_rank = self.attn_tp_rank == 0
        self.current_scheduler_metrics_enabled = self.enable_metrics and (
            self.attn_tp_rank == 0 or self.server_args.enable_metrics_for_all_schedulers
        )
        if self.enable_metrics:
            if self.server_args.disaggregation_mode == DisaggregationMode.PREFILL.value:
                engine_type = "prefill"
            elif (
                self.server_args.disaggregation_mode == DisaggregationMode.DECODE.value
            ):
                engine_type = "decode"
            else:
                engine_type = "unified"

            labels = {
                "model_name": self.server_args.served_model_name,
                "engine_type": engine_type,
                "tp_rank": tp_rank,
                "pp_rank": pp_rank,
                "moe_ep_rank": self.moe_ep_rank,
            }
            if self.enable_priority_scheduling:
                labels["priority"] = ""
            if dp_rank is not None:
                labels["dp_rank"] = dp_rank
            if self.server_args.extra_metric_labels:
                labels.update(self.server_args.extra_metric_labels)
            self.metrics_collector = SchedulerMetricsCollector(
                labels=labels,
                enable_lora=self.enable_lora,
                enable_hierarchical_cache=self.enable_hierarchical_cache,
                server_args=self.server_args,
            )
            self.enable_mfu_metrics = bool(self.server_args.enable_mfu_metrics)
            if self.enable_mfu_metrics:
                self._init_estimated_perf_constants()
                self._mfu_log_flops = 0.0
                self._mfu_log_read_bytes = 0.0
                self._mfu_log_write_bytes = 0.0

            if ENABLE_METRICS_DEVICE_TIMER:
                self.forward_pass_device_timer = DeviceTimer(
                    reporter=self.metrics_collector.increment_gpu_execution_seconds,
                )
                self.bubble_timer = GapTimer(
                    reporter=self.metrics_collector.increment_gpu_overlap_wait_seconds,
                )

        if self.enable_kv_cache_events:
            self.init_kv_events(self.server_args.kv_events_config)

        self.scheduler_status_logger = SchedulerStatusLogger.maybe_create(
            enable_metrics=self.enable_metrics
        )

    def init_kv_events(self: Scheduler, kv_events_config: Optional[str]):
        if self.enable_kv_cache_events:
            self.kv_event_publisher = EventPublisherFactory.create(
                kv_events_config, self.attn_dp_rank
            )

    def update_spec_metrics(self: Scheduler, bs: int, num_accepted_tokens: int):
        self.spec_num_accepted_tokens += num_accepted_tokens + bs
        self.spec_num_forward_ct += bs
        self.num_generated_tokens += num_accepted_tokens

    def _init_estimated_perf_constants(self: Scheduler) -> None:
        model_config = self.model_config
        hf_text_config = model_config.hf_text_config

        hidden_size = float(model_config.hidden_size)
        num_layers = float(getattr(model_config, "num_attention_layers", 0))
        head_dim = float(getattr(model_config, "head_dim", 0))
        num_attn_heads = float(model_config.get_num_attention_heads(self.tp_size))
        num_kv_heads = float(model_config.get_num_kv_heads(self.tp_size))
        intermediate_size = getattr(hf_text_config, "intermediate_size", None)
        if intermediate_size is None:
            intermediate_size = getattr(hf_text_config, "ffn_hidden_size", 0)
        intermediate_size = float(intermediate_size)

        dtype_num_bytes = getattr(model_config.dtype, "itemsize", None)
        if dtype_num_bytes is None:
            dtype_num_bytes = 2
        # Keep this estimator lightweight and consistent with current server dtype.
        # KV cache quantization-aware bytes can be added in a follow-up.
        act_bytes = float(dtype_num_bytes)
        w_bytes = float(dtype_num_bytes)
        cache_bytes = float(dtype_num_bytes)

        # Linear-layer FLOPs per token on one GPU.
        attn_linear_flops = (
            2.0 * hidden_size * head_dim * (num_attn_heads + 2.0 * num_kv_heads)
            + 2.0 * hidden_size * head_dim * num_attn_heads
        )
        mlp_flops = (
            6.0 * hidden_size * intermediate_size if intermediate_size > 0 else 0.0
        )
        self._linear_flops_per_token = max(
            0.0, (attn_linear_flops + mlp_flops) * num_layers
        )

        # Attention dot-product FLOPs coefficient to multiply token-context product.
        # attn_qk + attn_av = 4 * q * TC * d * L
        self._attn_dot_flops_coeff = 4.0 * num_attn_heads * head_dim * num_layers

        # KV cache bytes (write one K and one V vector per generated token).
        self._kv_cache_bytes_per_token = (
            2.0 * num_layers * num_kv_heads * head_dim * cache_bytes
        )

        # Weight read bytes per token.
        self._weight_read_bytes_per_token = (
            hidden_size
            * head_dim
            * (num_attn_heads + 2.0 * num_kv_heads)
            * w_bytes
            * num_layers
            + hidden_size * head_dim * num_attn_heads * w_bytes * num_layers
            + (
                3.0 * hidden_size * intermediate_size * w_bytes * num_layers
                if intermediate_size > 0
                else 0.0
            )
        )

        # Activation movement bytes per token (coarse approximation).
        self._qkv_act_bytes_per_token = (
            hidden_size * act_bytes * num_layers
            + (num_attn_heads + 2.0 * num_kv_heads) * head_dim * act_bytes * num_layers
            + head_dim * num_attn_heads * act_bytes * num_layers
            + hidden_size * act_bytes * num_layers
        )
        self._ffn_act_bytes_per_token = (
            3.0 * intermediate_size * act_bytes * num_layers
            if intermediate_size > 0
            else 0.0
        )

        # Prefill reads Q/K/V activations from on-device memory.
        self._prefill_attn_act_read_per_token = (
            (num_attn_heads + 2.0 * num_kv_heads) * head_dim * act_bytes * num_layers
        )

        # Decode reads Q from activation memory; K/V reads are from KV cache.
        self._decode_q_read_bytes_per_token = (
            num_attn_heads * head_dim * act_bytes * num_layers
        )

    def _estimate_prefill_perf(
        self: Scheduler, num_tokens: int
    ) -> Tuple[float, float, float]:
        tokens = max(0, int(num_tokens))
        if tokens == 0:
            return 0.0, 0.0, 0.0

        # Causal prefill token-context product.
        context_product = tokens * (tokens + 1) / 2.0
        flops = (
            tokens * self._linear_flops_per_token
            + self._attn_dot_flops_coeff * context_product
        )

        read_bytes = (
            tokens * self._weight_read_bytes_per_token
            + tokens * self._qkv_act_bytes_per_token
            + tokens * self._prefill_attn_act_read_per_token
        )
        write_bytes = (
            tokens * self._kv_cache_bytes_per_token
            + tokens * self._qkv_act_bytes_per_token
            + tokens * self._ffn_act_bytes_per_token
        )
        return flops, read_bytes, write_bytes

    def _estimate_decode_perf(
        self: Scheduler, batch: ScheduleBatch, num_tokens: int
    ) -> Tuple[float, float, float]:
        tokens = max(0, int(num_tokens))
        if tokens == 0:
            return 0.0, 0.0, 0.0

        total_context = float(batch.seq_lens_cpu.sum().item())
        flops = (
            tokens * self._linear_flops_per_token
            + self._attn_dot_flops_coeff * total_context
        )
        read_bytes = (
            tokens * self._weight_read_bytes_per_token
            + tokens * self._qkv_act_bytes_per_token
            + tokens * self._decode_q_read_bytes_per_token
            + total_context * self._kv_cache_bytes_per_token
        )
        write_bytes = (
            tokens * self._kv_cache_bytes_per_token
            + tokens * self._qkv_act_bytes_per_token
            + tokens * self._ffn_act_bytes_per_token
        )
        return flops, read_bytes, write_bytes

    def reset_metrics(self: Scheduler):
        self.forward_ct_decode = 0
        self.num_generated_tokens = 0
        self.spec_num_accepted_tokens = 0
        self.spec_num_forward_ct = 0
        self.spec_total_num_accepted_tokens = 0
        self.spec_total_num_forward_ct = 0

    def _record_prefill_pick_timing(self: Scheduler, elapsed_ms: float):
        self._prefill_pick_calls += 1
        self._prefill_pick_total_ms += elapsed_ms
        self._prefill_pick_max_ms = max(self._prefill_pick_max_ms, elapsed_ms)

    def _record_prefill_run_timing(self: Scheduler, elapsed_ms: float):
        self._prefill_run_calls += 1
        self._prefill_run_total_ms += elapsed_ms
        self._prefill_run_max_ms = max(self._prefill_run_max_ms, elapsed_ms)

    def _record_prefill_forward_timing(self: Scheduler, elapsed_ms: float):
        self._prefill_forward_calls += 1
        self._prefill_forward_total_ms += elapsed_ms
        self._prefill_forward_max_ms = max(self._prefill_forward_max_ms, elapsed_ms)

    def _record_prefill_copy_wait_timing(self: Scheduler, elapsed_ms: float):
        self._prefill_copy_wait_calls += 1
        self._prefill_copy_wait_total_ms += elapsed_ms
        self._prefill_copy_wait_max_ms = max(
            self._prefill_copy_wait_max_ms, elapsed_ms
        )

    def _record_prefill_post_timing(self: Scheduler, elapsed_ms: float):
        self._prefill_post_calls += 1
        self._prefill_post_total_ms += elapsed_ms
        self._prefill_post_max_ms = max(self._prefill_post_max_ms, elapsed_ms)

    def _record_pp_comm_send_timing(self: Scheduler, elapsed_ms: float):
        self._pp_comm_send_calls += 1
        self._pp_comm_send_total_ms += elapsed_ms
        self._pp_comm_send_max_ms = max(self._pp_comm_send_max_ms, elapsed_ms)

    def _record_pp_comm_recv_timing(self: Scheduler, elapsed_ms: float):
        self._pp_comm_recv_calls += 1
        self._pp_comm_recv_total_ms += elapsed_ms
        self._pp_comm_recv_max_ms = max(self._pp_comm_recv_max_ms, elapsed_ms)

    def _record_pp_comm_tp_bcast_timing(self: Scheduler, elapsed_ms: float):
        self._pp_comm_tp_bcast_calls += 1
        self._pp_comm_tp_bcast_total_ms += elapsed_ms
        self._pp_comm_tp_bcast_max_ms = max(
            self._pp_comm_tp_bcast_max_ms, elapsed_ms
        )

    def _record_pp_comm_cp_bcast_timing(self: Scheduler, elapsed_ms: float):
        self._pp_comm_cp_bcast_calls += 1
        self._pp_comm_cp_bcast_total_ms += elapsed_ms
        self._pp_comm_cp_bcast_max_ms = max(
            self._pp_comm_cp_bcast_max_ms, elapsed_ms
        )

    def _record_pp_comm_wait_timing(
        self: Scheduler, elapsed_ms: float, kind: Optional[str] = None
    ):
        self._pp_comm_wait_calls += 1
        self._pp_comm_wait_total_ms += elapsed_ms
        self._pp_comm_wait_max_ms = max(self._pp_comm_wait_max_ms, elapsed_ms)
        if kind == "req":
            self._pp_comm_wait_req_calls += 1
            self._pp_comm_wait_req_total_ms += elapsed_ms
            self._pp_comm_wait_req_max_ms = max(
                self._pp_comm_wait_req_max_ms, elapsed_ms
            )
        elif kind == "bootstrap":
            self._pp_comm_wait_bootstrap_calls += 1
            self._pp_comm_wait_bootstrap_total_ms += elapsed_ms
            self._pp_comm_wait_bootstrap_max_ms = max(
                self._pp_comm_wait_bootstrap_max_ms, elapsed_ms
            )
        elif kind == "transfer":
            self._pp_comm_wait_transfer_calls += 1
            self._pp_comm_wait_transfer_total_ms += elapsed_ms
            self._pp_comm_wait_transfer_max_ms = max(
                self._pp_comm_wait_transfer_max_ms, elapsed_ms
            )
        elif kind == "output":
            self._pp_comm_wait_output_calls += 1
            self._pp_comm_wait_output_total_ms += elapsed_ms
            self._pp_comm_wait_output_max_ms = max(
                self._pp_comm_wait_output_max_ms, elapsed_ms
            )
        elif kind == "proxy":
            self._pp_comm_wait_proxy_calls += 1
            self._pp_comm_wait_proxy_total_ms += elapsed_ms
            self._pp_comm_wait_proxy_max_ms = max(
                self._pp_comm_wait_proxy_max_ms, elapsed_ms
            )
        elif kind == "consensus_bootstrap":
            self._pp_comm_wait_consensus_bootstrap_calls += 1
            self._pp_comm_wait_consensus_bootstrap_total_ms += elapsed_ms
            self._pp_comm_wait_consensus_bootstrap_max_ms = max(
                self._pp_comm_wait_consensus_bootstrap_max_ms, elapsed_ms
            )
        elif kind == "release":
            self._pp_comm_wait_release_calls += 1
            self._pp_comm_wait_release_total_ms += elapsed_ms
            self._pp_comm_wait_release_max_ms = max(
                self._pp_comm_wait_release_max_ms, elapsed_ms
            )

    def _record_prefill_pick_reason(self: Scheduler, reason: str):
        if reason == "early_full_or_empty":
            self._prefill_pick_early_full_or_empty_count += 1
        elif reason == "chunked_capacity_block":
            self._prefill_pick_chunked_capacity_block_count += 1
        elif reason == "test_retract_block":
            self._prefill_pick_test_retract_block_count += 1
        elif reason == "locally_revoked_break":
            self._prefill_pick_locally_revoked_break_count += 1
        elif reason == "running_full_break":
            self._prefill_pick_running_full_break_count += 1
        elif reason == "prefetch_break":
            self._prefill_pick_prefetch_break_count += 1
        elif reason == "write_backup_break":
            self._prefill_pick_write_backup_break_count += 1
        elif reason == "add_no_token_break":
            self._prefill_pick_add_no_token_break_count += 1
        elif reason == "add_other_break":
            self._prefill_pick_add_other_break_count += 1
        elif reason == "empty_result":
            self._prefill_pick_empty_result_count += 1

    def _record_pp_frontier_ack_recv(self: Scheduler):
        self._pp_frontier_ack_recv_count += 1

    def _record_pp_frontier_ack_activate(self: Scheduler):
        self._pp_frontier_ack_activate_count += 1

    def _record_pp_frontier_ack_consume(self: Scheduler):
        self._pp_frontier_ack_consume_count += 1

    def _record_pp_frontier_ack_consume_miss(
        self: Scheduler, pending_same_mb: bool = False
    ):
        self._pp_frontier_ack_consume_miss_count += 1
        if pending_same_mb:
            self._pp_frontier_ack_pending_same_mb_miss_count += 1

    def _record_pp_frontier_ack_chunked_mismatch(self: Scheduler):
        self._pp_frontier_ack_chunked_mismatch_count += 1

    def _record_pp_frontier_ack_waiting_mismatch(self: Scheduler):
        self._pp_frontier_ack_waiting_mismatch_count += 1

    def _record_pp_frontier_ack_exhausted(self: Scheduler):
        self._pp_frontier_ack_exhausted_count += 1

    def consume_prefill_stage_perf_snapshot(self: Scheduler) -> dict[str, float]:
        def _avg(total: float, calls: int) -> float:
            return total / calls if calls else 0.0

        snapshot = {
            "pick_calls": self._prefill_pick_calls,
            "pick_avg_ms": _avg(self._prefill_pick_total_ms, self._prefill_pick_calls),
            "pick_max_ms": self._prefill_pick_max_ms,
            "pick_early_full_or_empty": self._prefill_pick_early_full_or_empty_count,
            "pick_chunked_capacity_block": self._prefill_pick_chunked_capacity_block_count,
            "pick_test_retract_block": self._prefill_pick_test_retract_block_count,
            "pick_locally_revoked_break": self._prefill_pick_locally_revoked_break_count,
            "pick_running_full_break": self._prefill_pick_running_full_break_count,
            "pick_prefetch_break": self._prefill_pick_prefetch_break_count,
            "pick_write_backup_break": self._prefill_pick_write_backup_break_count,
            "pick_add_no_token_break": self._prefill_pick_add_no_token_break_count,
            "pick_add_other_break": self._prefill_pick_add_other_break_count,
            "pick_empty_result": self._prefill_pick_empty_result_count,
            "run_calls": self._prefill_run_calls,
            "run_avg_ms": _avg(self._prefill_run_total_ms, self._prefill_run_calls),
            "run_max_ms": self._prefill_run_max_ms,
            "forward_calls": self._prefill_forward_calls,
            "forward_avg_ms": _avg(
                self._prefill_forward_total_ms, self._prefill_forward_calls
            ),
            "forward_max_ms": self._prefill_forward_max_ms,
            "copy_wait_calls": self._prefill_copy_wait_calls,
            "copy_wait_avg_ms": _avg(
                self._prefill_copy_wait_total_ms, self._prefill_copy_wait_calls
            ),
            "copy_wait_max_ms": self._prefill_copy_wait_max_ms,
            "post_calls": self._prefill_post_calls,
            "post_avg_ms": _avg(self._prefill_post_total_ms, self._prefill_post_calls),
            "post_max_ms": self._prefill_post_max_ms,
        }
        self._prefill_pick_calls = 0
        self._prefill_pick_total_ms = 0.0
        self._prefill_pick_max_ms = 0.0
        self._prefill_pick_early_full_or_empty_count = 0
        self._prefill_pick_chunked_capacity_block_count = 0
        self._prefill_pick_test_retract_block_count = 0
        self._prefill_pick_locally_revoked_break_count = 0
        self._prefill_pick_running_full_break_count = 0
        self._prefill_pick_prefetch_break_count = 0
        self._prefill_pick_write_backup_break_count = 0
        self._prefill_pick_add_no_token_break_count = 0
        self._prefill_pick_add_other_break_count = 0
        self._prefill_pick_empty_result_count = 0
        self._prefill_run_calls = 0
        self._prefill_run_total_ms = 0.0
        self._prefill_run_max_ms = 0.0
        self._prefill_forward_calls = 0
        self._prefill_forward_total_ms = 0.0
        self._prefill_forward_max_ms = 0.0
        self._prefill_copy_wait_calls = 0
        self._prefill_copy_wait_total_ms = 0.0
        self._prefill_copy_wait_max_ms = 0.0
        self._prefill_post_calls = 0
        self._prefill_post_total_ms = 0.0
        self._prefill_post_max_ms = 0.0
        return snapshot

    def consume_pp_comm_perf_snapshot(self: Scheduler) -> dict[str, float]:
        def _avg(total: float, calls: int) -> float:
            return total / calls if calls else 0.0

        snapshot = {
            "send_calls": self._pp_comm_send_calls,
            "send_avg_ms": _avg(self._pp_comm_send_total_ms, self._pp_comm_send_calls),
            "send_max_ms": self._pp_comm_send_max_ms,
            "recv_calls": self._pp_comm_recv_calls,
            "recv_avg_ms": _avg(self._pp_comm_recv_total_ms, self._pp_comm_recv_calls),
            "recv_max_ms": self._pp_comm_recv_max_ms,
            "tp_bcast_calls": self._pp_comm_tp_bcast_calls,
            "tp_bcast_avg_ms": _avg(
                self._pp_comm_tp_bcast_total_ms, self._pp_comm_tp_bcast_calls
            ),
            "tp_bcast_max_ms": self._pp_comm_tp_bcast_max_ms,
            "cp_bcast_calls": self._pp_comm_cp_bcast_calls,
            "cp_bcast_avg_ms": _avg(
                self._pp_comm_cp_bcast_total_ms, self._pp_comm_cp_bcast_calls
            ),
            "cp_bcast_max_ms": self._pp_comm_cp_bcast_max_ms,
            "wait_calls": self._pp_comm_wait_calls,
            "wait_avg_ms": _avg(self._pp_comm_wait_total_ms, self._pp_comm_wait_calls),
            "wait_max_ms": self._pp_comm_wait_max_ms,
            "wait_req_calls": self._pp_comm_wait_req_calls,
            "wait_req_avg_ms": _avg(
                self._pp_comm_wait_req_total_ms, self._pp_comm_wait_req_calls
            ),
            "wait_req_max_ms": self._pp_comm_wait_req_max_ms,
            "wait_bootstrap_calls": self._pp_comm_wait_bootstrap_calls,
            "wait_bootstrap_avg_ms": _avg(
                self._pp_comm_wait_bootstrap_total_ms,
                self._pp_comm_wait_bootstrap_calls,
            ),
            "wait_bootstrap_max_ms": self._pp_comm_wait_bootstrap_max_ms,
            "wait_transfer_calls": self._pp_comm_wait_transfer_calls,
            "wait_transfer_avg_ms": _avg(
                self._pp_comm_wait_transfer_total_ms,
                self._pp_comm_wait_transfer_calls,
            ),
            "wait_transfer_max_ms": self._pp_comm_wait_transfer_max_ms,
            "wait_output_calls": self._pp_comm_wait_output_calls,
            "wait_output_avg_ms": _avg(
                self._pp_comm_wait_output_total_ms, self._pp_comm_wait_output_calls
            ),
            "wait_output_max_ms": self._pp_comm_wait_output_max_ms,
            "wait_proxy_calls": self._pp_comm_wait_proxy_calls,
            "wait_proxy_avg_ms": _avg(
                self._pp_comm_wait_proxy_total_ms, self._pp_comm_wait_proxy_calls
            ),
            "wait_proxy_max_ms": self._pp_comm_wait_proxy_max_ms,
            "wait_consensus_bootstrap_calls": self._pp_comm_wait_consensus_bootstrap_calls,
            "wait_consensus_bootstrap_avg_ms": _avg(
                self._pp_comm_wait_consensus_bootstrap_total_ms,
                self._pp_comm_wait_consensus_bootstrap_calls,
            ),
            "wait_consensus_bootstrap_max_ms": self._pp_comm_wait_consensus_bootstrap_max_ms,
            "wait_release_calls": self._pp_comm_wait_release_calls,
            "wait_release_avg_ms": _avg(
                self._pp_comm_wait_release_total_ms, self._pp_comm_wait_release_calls
            ),
            "wait_release_max_ms": self._pp_comm_wait_release_max_ms,
        }
        self._pp_comm_send_calls = 0
        self._pp_comm_send_total_ms = 0.0
        self._pp_comm_send_max_ms = 0.0
        self._pp_comm_recv_calls = 0
        self._pp_comm_recv_total_ms = 0.0
        self._pp_comm_recv_max_ms = 0.0
        self._pp_comm_tp_bcast_calls = 0
        self._pp_comm_tp_bcast_total_ms = 0.0
        self._pp_comm_tp_bcast_max_ms = 0.0
        self._pp_comm_cp_bcast_calls = 0
        self._pp_comm_cp_bcast_total_ms = 0.0
        self._pp_comm_cp_bcast_max_ms = 0.0
        self._pp_comm_wait_calls = 0
        self._pp_comm_wait_total_ms = 0.0
        self._pp_comm_wait_max_ms = 0.0
        self._pp_comm_wait_req_calls = 0
        self._pp_comm_wait_req_total_ms = 0.0
        self._pp_comm_wait_req_max_ms = 0.0
        self._pp_comm_wait_bootstrap_calls = 0
        self._pp_comm_wait_bootstrap_total_ms = 0.0
        self._pp_comm_wait_bootstrap_max_ms = 0.0
        self._pp_comm_wait_transfer_calls = 0
        self._pp_comm_wait_transfer_total_ms = 0.0
        self._pp_comm_wait_transfer_max_ms = 0.0
        self._pp_comm_wait_output_calls = 0
        self._pp_comm_wait_output_total_ms = 0.0
        self._pp_comm_wait_output_max_ms = 0.0
        self._pp_comm_wait_proxy_calls = 0
        self._pp_comm_wait_proxy_total_ms = 0.0
        self._pp_comm_wait_proxy_max_ms = 0.0
        self._pp_comm_wait_consensus_bootstrap_calls = 0
        self._pp_comm_wait_consensus_bootstrap_total_ms = 0.0
        self._pp_comm_wait_consensus_bootstrap_max_ms = 0.0
        self._pp_comm_wait_release_calls = 0
        self._pp_comm_wait_release_total_ms = 0.0
        self._pp_comm_wait_release_max_ms = 0.0
        return snapshot

    def consume_pp_frontier_ack_snapshot(self: Scheduler) -> dict[str, int]:
        snapshot = {
            "recv": self._pp_frontier_ack_recv_count,
            "activate": self._pp_frontier_ack_activate_count,
            "consume": self._pp_frontier_ack_consume_count,
            "consume_miss": self._pp_frontier_ack_consume_miss_count,
            "pending_same_mb_miss": self._pp_frontier_ack_pending_same_mb_miss_count,
            "chunked_mismatch": self._pp_frontier_ack_chunked_mismatch_count,
            "waiting_mismatch": self._pp_frontier_ack_waiting_mismatch_count,
            "ack_exhausted": self._pp_frontier_ack_exhausted_count,
        }
        self._pp_frontier_ack_recv_count = 0
        self._pp_frontier_ack_activate_count = 0
        self._pp_frontier_ack_consume_count = 0
        self._pp_frontier_ack_consume_miss_count = 0
        self._pp_frontier_ack_pending_same_mb_miss_count = 0
        self._pp_frontier_ack_chunked_mismatch_count = 0
        self._pp_frontier_ack_waiting_mismatch_count = 0
        self._pp_frontier_ack_exhausted_count = 0
        return snapshot

    def report_prefill_stats(
        self: Scheduler,
        prefill_stats: PrefillStats,
        can_run_cuda_graph: bool,
        dp_cooperation_info: Optional[DPCooperationInfo] = None,
    ):
        if (
            not self.is_stats_logging_rank
            and not self.current_scheduler_metrics_enabled
        ):
            return

        gap_latency = time.perf_counter() - self.last_prefill_stats_tic
        self.last_prefill_stats_tic = time.perf_counter()
        self.last_input_throughput = self.last_prefill_tokens / gap_latency
        self.last_prefill_tokens = prefill_stats.log_input_tokens

        # TODO: generalize this for various memory pools
        msg_parts = []
        num_used = token_usage = full_token_usage = None

        if self.is_hybrid_swa:
            full_num_used, swa_num_used, full_tok, swa_token_usage, *_ = (
                self._get_swa_token_info()
            )
            num_used = max(full_num_used, swa_num_used)
            token_usage = max(full_tok, swa_token_usage)
            full_token_usage = full_tok
            msg_parts += [
                f"full token usage: {full_tok:.2f}",
                f"swa token usage: {swa_token_usage:.2f}",
            ]

        if self.is_hybrid_ssm:
            num_used_m, _, full_tok_m, mamba_usage, *_ = self._get_mamba_token_info()
            num_used = max(num_used, num_used_m) if num_used is not None else num_used_m
            token_usage = (
                max(token_usage, mamba_usage)
                if token_usage is not None
                else max(full_tok_m, mamba_usage)
            )
            if full_token_usage is None:
                full_token_usage = full_tok_m
                msg_parts.append(f"full token usage: {full_tok_m:.2f}")
            msg_parts.append(f"mamba usage: {mamba_usage:.2f}")

        if full_token_usage is None:
            num_used, tok, _, _ = self._get_token_info()
            full_token_usage = tok
            token_usage = tok
            msg_parts.append(f"token usage: {tok:.2f}")

        assert (
            num_used is not None
            and token_usage is not None
            and full_token_usage is not None
        )
        token_usage_msg = ", ".join(msg_parts) + ", "

        self.stats.new_token_ratio = prefill_stats.new_token_ratio
        iter_msg = f" [{self.forward_ct + 1}]" if LOG_FORWARD_ITERS else ""

        msg = (
            f"Prefill batch{iter_msg}, "
            f"#new-seq: {prefill_stats.num_new_seqs}, "
            f"#new-token: {prefill_stats.log_input_tokens}, "
            f"#cached-token: {prefill_stats.log_hit_tokens}, "
            f"{token_usage_msg}"
            f"#running-req: {prefill_stats.num_running_reqs.total}, "
            f"#queue-req: {len(self.waiting_queue)}, "
        )

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            msg += f"#prealloc-req: {len(self.disagg_prefill_bootstrap_queue.queue)}, "
            msg += f"#inflight-req: {len(self.disagg_prefill_inflight_queue)}, "

        if (
            self.server_args.language_only
            and self.server_args.encoder_transfer_backend == "zmq_to_scheduler"
        ):
            msg += f"waiting-image-req: {len(self.mm_receiver.waiting_list)}, "
        graph_backend = defaultdict(
            lambda: "cuda graph",
            {
                "cpu": "cpu graph",
                "npu": "npu graph",
            },
        )

        msg += f"{graph_backend[self.device]}: {can_run_cuda_graph}, "
        msg += f"input throughput (token/s): {self.last_input_throughput:.2f}"

        if self.enable_mfu_metrics and gap_latency > 0:
            flops, _, _ = self._estimate_prefill_perf(prefill_stats.log_input_tokens)
            tflops_per_s = flops / gap_latency / 1e12
            msg += f", est. prefill TFLOPS/s (per GPU): {tflops_per_s:.2f}"

        if self.is_stats_logging_rank:
            logger.info(msg)

        if self.current_scheduler_metrics_enabled:
            self.metrics_collector.increment_prefill_cuda_graph_pass(
                value=can_run_cuda_graph
            )
            self.metrics_collector.increment_realtime_tokens(
                prefill_compute_tokens=prefill_stats.log_input_tokens,
                prefill_cache_tokens=prefill_stats.log_hit_tokens,
                dp_cooperation_info=dp_cooperation_info,
            )
            if self.enable_mfu_metrics:
                flops, read_bytes, write_bytes = self._estimate_prefill_perf(
                    prefill_stats.log_input_tokens
                )
                self.metrics_collector.increment_estimated_perf(
                    num_flops_per_gpu=flops,
                    num_read_bytes_per_gpu=read_bytes,
                    num_write_bytes_per_gpu=write_bytes,
                )

            # Basics
            total_tokens = prefill_stats.log_input_tokens + prefill_stats.log_hit_tokens
            cache_hit_rate = (
                prefill_stats.log_hit_tokens / total_tokens if total_tokens > 0 else 0.0
            )

            self.stats.num_running_reqs = prefill_stats.num_running_reqs
            self.stats.num_running_reqs_offline_batch = 0
            self.stats.num_used_tokens = num_used
            self.stats.token_usage = token_usage
            self.stats.full_token_usage = full_token_usage
            if self.is_hybrid_swa:
                self.stats.swa_token_usage = swa_token_usage
            if self.is_hybrid_ssm:
                self.stats.mamba_usage = mamba_usage

            priority_enabled = self.enable_priority_scheduling
            self.stats.num_queue_reqs = QueueCount.from_reqs(
                self.waiting_queue, priority_enabled
            )
            self.stats.num_grammar_queue_reqs = len(self.grammar_manager)
            self.stats.cache_hit_rate = cache_hit_rate

            self.stats.max_total_num_tokens = self.max_total_num_tokens

            # Retract
            self.stats.num_retracted_reqs = self.num_retracted_reqs
            self.stats.num_paused_reqs = self.num_paused_reqs
            self.num_retracted_reqs = self.num_paused_reqs = 0

            # PD disaggregation
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                self.stats.num_prefill_prealloc_queue_reqs = QueueCount.from_reqs(
                    self.disagg_prefill_bootstrap_queue.queue, priority_enabled
                )
                self.stats.num_prefill_inflight_queue_reqs = QueueCount.from_reqs(
                    self.disagg_prefill_inflight_queue, priority_enabled
                )
                self.stats.kv_transfer_speed_gb_s = self.kv_transfer_speed_gb_s
                self.stats.kv_transfer_latency_ms = self.kv_transfer_latency_ms
            elif self.disaggregation_mode == DisaggregationMode.DECODE:
                self.stats.num_decode_prealloc_queue_reqs = QueueCount.from_reqs(
                    self.disagg_decode_prealloc_queue.queue, priority_enabled
                )
                self.stats.num_decode_transfer_queue_reqs = QueueCount.from_reqs(
                    self.disagg_decode_transfer_queue.queue, priority_enabled
                )

            # Others
            self.calculate_utilization()
            self.update_lora_metrics()
            self._log_hicache_stats()
            self.metrics_collector.log_stats(self.stats)
            self._emit_kv_metrics()
        self._publish_kv_events()

    def report_decode_stats(
        self: Scheduler,
        can_run_cuda_graph: bool,
        running_batch: ScheduleBatch = None,
        num_accepted_tokens: int = 0,
    ):
        batch = running_batch or self.running_batch

        # Every-iteration work: realtime token counting + status logger
        if self.current_scheduler_metrics_enabled:
            decode_tokens = batch.batch_size() + num_accepted_tokens
            self.metrics_collector.increment_realtime_tokens(
                # TODO unify this w/ the bumping logic in `Scheduler.num_generated_tokens` accumulator
                decode_tokens=decode_tokens,
                dp_cooperation_info=batch.dp_cooperation_info,
            )
            if self.enable_mfu_metrics:
                flops, read_bytes, write_bytes = self._estimate_decode_perf(
                    batch, decode_tokens
                )
                self.metrics_collector.increment_estimated_perf(
                    num_flops_per_gpu=flops,
                    num_read_bytes_per_gpu=read_bytes,
                    num_write_bytes_per_gpu=write_bytes,
                )
                self._mfu_log_flops += flops
                self._mfu_log_read_bytes += read_bytes
                self._mfu_log_write_bytes += write_bytes

            if x := self.scheduler_status_logger:
                x.maybe_dump(batch, self.waiting_queue)

        # Periodic work: log + heavy metrics at decode_log_interval
        if self.forward_ct_decode % self.server_args.decode_log_interval != 0:
            return
        if (
            not self.is_stats_logging_rank
            and not self.current_scheduler_metrics_enabled
        ):
            return

        gap_latency = time.perf_counter() - self.last_decode_stats_tic
        self.last_decode_stats_tic = time.perf_counter()
        self.last_gen_throughput = self.num_generated_tokens / gap_latency

        self.num_generated_tokens = 0
        num_running_reqs = len(batch.reqs)
        num_running_reqs_offline_batch = 0

        # TODO: generalize this for various memory pools
        msg_parts = []
        num_used = token_usage = full_token_usage = None

        if self.is_hybrid_swa:
            full_num_used, swa_num_used, full_tok, swa_token_usage, *_ = (
                self._get_swa_token_info()
            )
            num_used = max(full_num_used, swa_num_used)
            token_usage = max(full_tok, swa_token_usage)
            full_token_usage = full_tok
            msg_parts += [
                f"#full token: {full_num_used}",
                f"full token usage: {full_tok:.2f}",
                f"#swa token: {swa_num_used}",
                f"swa token usage: {swa_token_usage:.2f}",
            ]

        if self.is_hybrid_ssm:
            num_used_m, mamba_num, full_tok_m, mamba_usage, *_ = (
                self._get_mamba_token_info()
            )
            num_used = max(num_used, num_used_m) if num_used is not None else num_used_m
            token_usage = (
                max(token_usage, mamba_usage)
                if token_usage is not None
                else max(full_tok_m, mamba_usage)
            )
            if full_token_usage is None:
                full_token_usage = full_tok_m
                msg_parts += [
                    f"#full token: {num_used_m}",
                    f"full token usage: {full_tok_m:.2f}",
                ]
            msg_parts += [
                f"mamba num: {mamba_num}",
                f"mamba usage: {mamba_usage:.2f}",
            ]

        if full_token_usage is None:
            num_used, tok, _, _ = self._get_token_info()
            full_token_usage = tok
            token_usage = tok
            msg_parts.append(f"#token: {num_used}, token usage: {tok:.2f}")

        assert (
            num_used is not None
            and token_usage is not None
            and full_token_usage is not None
        )
        token_usage_msg = ", ".join(msg_parts) + ", "

        if RECORD_STEP_TIME:
            self.step_time_dict[num_running_reqs].append(
                gap_latency / self.server_args.decode_log_interval
            )

        iter_msg = f" [{self.forward_ct}]" if LOG_FORWARD_ITERS else ""
        msg = f"Decode batch{iter_msg}, #running-req: {num_running_reqs}, {token_usage_msg}"

        if self.spec_algorithm.is_none():
            spec_accept_length = 0
            spec_accept_rate = 0
        else:
            spec_accept_length = (
                self.spec_num_accepted_tokens / self.spec_num_forward_ct
            )
            # Calculate acceptance rate: accepted tokens / total draft tokens
            draft_tokens_fallback = (self.server_args.speculative_num_steps or 0) + 1
            num_draft_tokens = (
                self.server_args.speculative_num_draft_tokens or draft_tokens_fallback
            )
            total_draft_tokens = self.spec_num_forward_ct * num_draft_tokens

            spec_accept_rate = (
                self.spec_num_accepted_tokens / total_draft_tokens
                if total_draft_tokens > 0
                else 0
            )
            self.spec_total_num_accepted_tokens += self.spec_num_accepted_tokens
            self.spec_total_num_forward_ct += self.spec_num_forward_ct
            self.spec_num_accepted_tokens = self.spec_num_forward_ct = 0
            msg += f"accept len: {spec_accept_length:.2f}, accept rate: {spec_accept_rate:.2f}, "
        cache_hit_rate = 0.0

        if self.disaggregation_mode == DisaggregationMode.DECODE:
            msg += f"pre-allocated usage: {self.disagg_decode_prealloc_queue.num_tokens_pre_allocated / self.max_total_num_tokens:.2f}, "
            msg += f"#prealloc-req: {len(self.disagg_decode_prealloc_queue.queue)}, "
            msg += f"#transfer-req: {len(self.disagg_decode_transfer_queue.queue)}, "
            msg += f"#retracted-req: {len(self.disagg_decode_prealloc_queue.retracted_queue)}, "

        if (
            self.server_args.language_only
            and self.server_args.encoder_transfer_backend == "zmq_to_scheduler"
        ):
            msg += f"waiting-image-req: {len(self.mm_receiver.waiting_list)}, "

        graph_backend = defaultdict(
            lambda: "cuda graph",
            {
                "cpu": "cpu graph",
                "npu": "npu graph",
            },
        )
        msg += (
            f"{graph_backend[self.device]}: {can_run_cuda_graph}, "
            f"gen throughput (token/s): {self.last_gen_throughput:.2f}, "
            f"#queue-req: {len(self.waiting_queue)}"
        )

        if self.enable_mfu_metrics and gap_latency > 0:
            flops_per_s = self._mfu_log_flops / gap_latency
            read_bytes_per_s = self._mfu_log_read_bytes / gap_latency
            write_bytes_per_s = self._mfu_log_write_bytes / gap_latency
            tflops_per_s = flops_per_s / 1e12
            read_gb_per_s = read_bytes_per_s / 1e9
            write_gb_per_s = write_bytes_per_s / 1e9
            msg += (
                f", est. decode TFLOPS/s (per GPU): {tflops_per_s:.2f}, "
                f"est. read BW (GB/s per GPU): {read_gb_per_s:.2f}, "
                f"est. write BW (GB/s per GPU): {write_gb_per_s:.2f}"
            )
            self._mfu_log_flops = 0.0
            self._mfu_log_read_bytes = 0.0
            self._mfu_log_write_bytes = 0.0

        if self.is_stats_logging_rank:
            logger.info(msg)
        if self.current_scheduler_metrics_enabled:
            priority_enabled = self.enable_priority_scheduling
            # Basics
            self.stats.num_running_reqs = QueueCount.from_reqs(
                batch.reqs, priority_enabled
            )
            self.stats.num_running_reqs_offline_batch = num_running_reqs_offline_batch
            self.stats.num_used_tokens = num_used
            # maximum usage of all pools
            self.stats.token_usage = token_usage
            # usage of full attention
            self.stats.full_token_usage = full_token_usage
            if self.is_hybrid_swa:
                self.stats.swa_token_usage = swa_token_usage
            if self.is_hybrid_ssm:
                self.stats.mamba_usage = mamba_usage
            self.stats.decode_sum_seq_lens = batch.seq_lens_cpu.sum().item()
            self.stats.gen_throughput = self.last_gen_throughput
            self.stats.num_queue_reqs = QueueCount.from_reqs(
                self.waiting_queue, priority_enabled
            )
            self.stats.num_grammar_queue_reqs = len(self.grammar_manager)
            self.stats.cache_hit_rate = cache_hit_rate

            self.stats.max_total_num_tokens = self.max_total_num_tokens

            # Speculative decoding
            self.stats.spec_accept_rate = spec_accept_rate
            self.stats.spec_accept_length = spec_accept_length

            # Retract
            self.stats.num_retracted_reqs = self.num_retracted_reqs
            self.stats.num_paused_reqs = self.num_paused_reqs
            self.num_retracted_reqs = self.num_paused_reqs = 0

            # PD disaggregation
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                self.stats.num_prefill_prealloc_queue_reqs = QueueCount.from_reqs(
                    self.disagg_prefill_bootstrap_queue.queue, priority_enabled
                )
                self.stats.num_prefill_inflight_queue_reqs = QueueCount.from_reqs(
                    self.disagg_prefill_inflight_queue, priority_enabled
                )
            elif self.disaggregation_mode == DisaggregationMode.DECODE:
                self.stats.num_decode_prealloc_queue_reqs = QueueCount.from_reqs(
                    self.disagg_decode_prealloc_queue.queue, priority_enabled
                )
                self.stats.num_decode_transfer_queue_reqs = QueueCount.from_reqs(
                    self.disagg_decode_transfer_queue.queue, priority_enabled
                )
            running_routing_keys = [r.routing_key for r in batch.reqs]
            waiting_routing_keys = [r.routing_key for r in self.waiting_queue]
            (
                self.stats.num_unique_running_routing_keys,
                self.stats.routing_key_running_req_counts,
            ) = compute_routing_key_stats(running_routing_keys)
            _, self.stats.routing_key_all_req_counts = compute_routing_key_stats(
                running_routing_keys + waiting_routing_keys
            )

            # Others
            self.calculate_utilization()
            self.update_lora_metrics()
            self._log_hicache_stats()
            self.metrics_collector.log_stats(self.stats)
            self._emit_kv_metrics()
        self._publish_kv_events()

    def log_batch_result_stats(
        self: Scheduler,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
    ):
        self._maybe_log_hicache_state_growth()
        if not self.enable_metrics:
            return
        if not isinstance(result, GenerationBatchResult):
            return

        if (m := result.expert_distribution_metrics) is not None:
            self.metrics_collector.increment_eplb_balancedness(
                forward_mode=batch.forward_mode.name.lower(),
                balancedness=m.eplb_balancedness.item(),
            )

    def _maybe_log_hicache_state_growth(self: Scheduler):
        if not DEBUG_HICACHE_STATE_GROWTH or not self.enable_hierarchical_cache:
            return

        now = time.monotonic()
        last = getattr(self, "_last_hicache_state_growth_log_ts", 0.0)
        if now - last < DEBUG_HICACHE_STATE_GROWTH_INTERVAL_SEC:
            return
        self._last_hicache_state_growth_log_ts = now

        tc = self.tree_cache
        cc = getattr(tc, "cache_controller", None)
        match_perf = (
            tc.consume_match_perf_snapshot()
            if hasattr(tc, "consume_match_perf_snapshot")
            else {}
        )
        replay_perf = (
            tc.consume_write_backup_replay_snapshot()
            if hasattr(tc, "consume_write_backup_replay_snapshot")
            else {}
        )
        prefill_stage_perf = (
            self.consume_prefill_stage_perf_snapshot()
            if hasattr(self, "consume_prefill_stage_perf_snapshot")
            else {}
        )
        pp_comm_perf = (
            self.consume_pp_comm_perf_snapshot()
            if hasattr(self, "consume_pp_comm_perf_snapshot")
            else {}
        )
        pp_frontier_ack = (
            self.consume_pp_frontier_ack_snapshot()
            if hasattr(self, "consume_pp_frontier_ack_snapshot")
            else {}
        )
        tree_churn = (
            tc.consume_tree_churn_snapshot()
            if hasattr(tc, "consume_tree_churn_snapshot")
            and getattr(self, "attn_tp_rank", 0) == 0
            and getattr(self, "attn_cp_rank", 0) == 0
            else {}
        )
        tree_shape = (
            tc.get_tree_shape_snapshot()
            if hasattr(tc, "get_tree_shape_snapshot")
            and getattr(self, "attn_tp_rank", 0) == 0
            and getattr(self, "attn_cp_rank", 0) == 0
            else {}
        )
        barrier_hits = getattr(self, "_hicache_write_backup_barrier_hits", 0)
        last_barrier_hits = getattr(
            self, "_last_hicache_write_backup_barrier_hits", 0
        )
        barrier_hits_delta = barrier_hits - last_barrier_hits
        self._last_hicache_write_backup_barrier_hits = barrier_hits

        def _safe_len(name: str) -> int:
            value = getattr(tc, name, None)
            if value is None:
                return 0
            try:
                return len(value)
            except TypeError:
                return 0

        logger.warning(
            "[HiCacheStateGrowth] pp=%s cp=%s tp=%s waiting=%s bootstrap=%s inflight=%s "
            "ongoing_prefetch=%s loaded=%s zero_hit=%s retry=%s auth_revoke=%s "
            "soft_skip=%s staged_skip=%s deferred_revoke=%s local_revoke=%s "
            "local_revoke_q=%s pending_events=%s outgoing_events=%s finalize_ticket=%s "
            "finalize_barrier=%s prefetch_tokens_occupied=%s revoke_q=%s ack_backup_q=%s "
            "host_release_q=%s match_calls=%s match_avg_ms=%.3f match_max_ms=%.3f "
            "match_avg_walk=%.2f match_max_walk=%s match_avg_splits=%.2f "
            "match_avg_host_climb=%.2f match_avg_backup_climb=%.2f match_avg_segments=%.2f "
            "match_avg_key_len=%.2f match_avg_aligned_len=%.2f match_avg_device_hit=%.2f "
            "match_avg_host_hit=%.2f match_root_only_calls=%s match_split_calls=%s "
            "match_host_climb_calls=%s match_backup_climb_calls=%s "
            "match_multi_segment_calls=%s match_device_hit_calls=%s "
            "match_host_hit_calls=%s match_avg_path_fanout=%.2f match_max_path_fanout=%s "
            "prefill_pick_calls=%s prefill_pick_avg_ms=%.3f prefill_pick_max_ms=%.3f "
            "prefill_run_calls=%s prefill_run_avg_ms=%.3f prefill_run_max_ms=%.3f "
            "prefill_forward_calls=%s prefill_forward_avg_ms=%.3f prefill_forward_max_ms=%.3f "
            "prefill_copy_wait_calls=%s prefill_copy_wait_avg_ms=%.3f prefill_copy_wait_max_ms=%.3f "
            "prefill_post_calls=%s prefill_post_avg_ms=%.3f prefill_post_max_ms=%.3f "
            "pp_send_calls=%s pp_send_avg_ms=%.3f pp_send_max_ms=%.3f "
            "pp_recv_calls=%s pp_recv_avg_ms=%.3f pp_recv_max_ms=%.3f "
            "pp_tp_bcast_calls=%s pp_tp_bcast_avg_ms=%.3f pp_tp_bcast_max_ms=%.3f "
            "pp_cp_bcast_calls=%s pp_cp_bcast_avg_ms=%.3f pp_cp_bcast_max_ms=%.3f "
            "pp_wait_calls=%s pp_wait_avg_ms=%.3f pp_wait_max_ms=%.3f "
            "pp_wait_req_calls=%s pp_wait_req_avg_ms=%.3f pp_wait_req_max_ms=%.3f "
            "pp_wait_bootstrap_calls=%s pp_wait_bootstrap_avg_ms=%.3f pp_wait_bootstrap_max_ms=%.3f "
            "pp_wait_transfer_calls=%s pp_wait_transfer_avg_ms=%.3f pp_wait_transfer_max_ms=%.3f "
            "pp_wait_output_calls=%s pp_wait_output_avg_ms=%.3f pp_wait_output_max_ms=%.3f "
            "pp_wait_proxy_calls=%s pp_wait_proxy_avg_ms=%.3f pp_wait_proxy_max_ms=%.3f "
            "pp_wait_consensus_bootstrap_calls=%s pp_wait_consensus_bootstrap_avg_ms=%.3f "
            "pp_wait_consensus_bootstrap_max_ms=%.3f "
            "pp_wait_release_calls=%s pp_wait_release_avg_ms=%.3f pp_wait_release_max_ms=%.3f "
            "pick_early_full_or_empty=%s pick_chunked_capacity_block=%s pick_test_retract_block=%s "
            "pick_locally_revoked_break=%s pick_running_full_break=%s pick_prefetch_break=%s "
            "pick_write_backup_break=%s pick_add_no_token_break=%s pick_add_other_break=%s "
            "pick_empty_result=%s "
            "ack_recv=%s ack_activate=%s ack_consume=%s ack_consume_miss=%s "
            "ack_pending_same_mb_miss=%s ack_chunked_mismatch=%s "
            "ack_waiting_mismatch=%s ack_exhausted=%s ack_pending_slots=%s ack_active_slots=%s "
            "wb_barrier_hits=%s wb_replay_local=%s wb_replay_auth=%s wb_replay_miss=%s "
            "wb_pending=%s wb_commit_nodes=%s wb_event_count=%s wb_event_avg_nodes=%.2f "
            "wb_event_max_nodes=%s wb_event_est_bytes=%s wb_event_avg_est_bytes=%.2f "
            "wb_event_max_est_bytes=%s tree_nodes=%s tree_evicted=%s "
            "tree_backuped=%s tree_leaves=%s tree_max_depth=%s tree_max_fanout=%s "
            "tree_allocated_id=%s tree_alloc_delta=%s alloc_match_split=%s "
            "alloc_insert_split=%s alloc_host_insert_split=%s alloc_device_leaf=%s "
            "alloc_host_leaf=%s delete_regular_leaf=%s delete_host_leaf=%s",
            getattr(self, "pp_rank", None),
            getattr(self, "attn_cp_rank", None),
            getattr(self, "attn_tp_rank", None),
            len(getattr(self, "waiting_queue", [])),
            len(getattr(getattr(self, "disagg_prefill_bootstrap_queue", None), "queue", [])),
            len(getattr(self, "disagg_prefill_inflight_queue", [])),
            _safe_len("ongoing_prefetch"),
            _safe_len("prefetch_loaded_tokens_by_reqid"),
            _safe_len("zero_hit_prefetch_req_ids"),
            _safe_len("pp_retry_prefetch_req_ids"),
            _safe_len("pp_authoritative_revoked_req_ids"),
            _safe_len("pp_soft_skipped_req_ids"),
            _safe_len("pp_staged_prefetch_skip_req_ids"),
            _safe_len("pp_deferred_revoke_req_ids"),
            _safe_len("pp_locally_revoked_req_ids"),
            _safe_len("pp_locally_revoked_req_queue"),
            _safe_len("pp_pending_host_tree_events"),
            _safe_len("pp_outgoing_host_tree_events"),
            _safe_len("pp_finalize_ticket_req_ids"),
            getattr(tc, "pp_finalize_ticket_barrier_rid", None),
            getattr(cc, "prefetch_tokens_occupied", 0) if cc is not None else 0,
            cc.prefetch_revoke_queue.qsize() if cc is not None else 0,
            cc.ack_backup_queue.qsize() if cc is not None else 0,
            cc.host_mem_release_queue.qsize() if cc is not None else 0,
            match_perf.get("calls", 0),
            match_perf.get("avg_ms", 0.0),
            match_perf.get("max_ms", 0.0),
            match_perf.get("avg_walk", 0.0),
            match_perf.get("max_walk", 0),
            match_perf.get("avg_splits", 0.0),
            match_perf.get("avg_host_climb", 0.0),
            match_perf.get("avg_backup_climb", 0.0),
            match_perf.get("avg_segments", 0.0),
            match_perf.get("avg_key_len", 0.0),
            match_perf.get("avg_aligned_len", 0.0),
            match_perf.get("avg_device_hit", 0.0),
            match_perf.get("avg_host_hit", 0.0),
            match_perf.get("root_only_calls", 0),
            match_perf.get("split_calls", 0),
            match_perf.get("host_climb_calls", 0),
            match_perf.get("backup_climb_calls", 0),
            match_perf.get("multi_segment_calls", 0),
            match_perf.get("device_hit_calls", 0),
            match_perf.get("host_hit_calls", 0),
            match_perf.get("avg_path_fanout", 0.0),
            match_perf.get("max_path_fanout", 0),
            prefill_stage_perf.get("pick_calls", 0),
            prefill_stage_perf.get("pick_avg_ms", 0.0),
            prefill_stage_perf.get("pick_max_ms", 0.0),
            prefill_stage_perf.get("run_calls", 0),
            prefill_stage_perf.get("run_avg_ms", 0.0),
            prefill_stage_perf.get("run_max_ms", 0.0),
            prefill_stage_perf.get("forward_calls", 0),
            prefill_stage_perf.get("forward_avg_ms", 0.0),
            prefill_stage_perf.get("forward_max_ms", 0.0),
            prefill_stage_perf.get("copy_wait_calls", 0),
            prefill_stage_perf.get("copy_wait_avg_ms", 0.0),
            prefill_stage_perf.get("copy_wait_max_ms", 0.0),
            prefill_stage_perf.get("post_calls", 0),
            prefill_stage_perf.get("post_avg_ms", 0.0),
            prefill_stage_perf.get("post_max_ms", 0.0),
            pp_comm_perf.get("send_calls", 0),
            pp_comm_perf.get("send_avg_ms", 0.0),
            pp_comm_perf.get("send_max_ms", 0.0),
            pp_comm_perf.get("recv_calls", 0),
            pp_comm_perf.get("recv_avg_ms", 0.0),
            pp_comm_perf.get("recv_max_ms", 0.0),
            pp_comm_perf.get("tp_bcast_calls", 0),
            pp_comm_perf.get("tp_bcast_avg_ms", 0.0),
            pp_comm_perf.get("tp_bcast_max_ms", 0.0),
            pp_comm_perf.get("cp_bcast_calls", 0),
            pp_comm_perf.get("cp_bcast_avg_ms", 0.0),
            pp_comm_perf.get("cp_bcast_max_ms", 0.0),
            pp_comm_perf.get("wait_calls", 0),
            pp_comm_perf.get("wait_avg_ms", 0.0),
            pp_comm_perf.get("wait_max_ms", 0.0),
            pp_comm_perf.get("wait_req_calls", 0),
            pp_comm_perf.get("wait_req_avg_ms", 0.0),
            pp_comm_perf.get("wait_req_max_ms", 0.0),
            pp_comm_perf.get("wait_bootstrap_calls", 0),
            pp_comm_perf.get("wait_bootstrap_avg_ms", 0.0),
            pp_comm_perf.get("wait_bootstrap_max_ms", 0.0),
            pp_comm_perf.get("wait_transfer_calls", 0),
            pp_comm_perf.get("wait_transfer_avg_ms", 0.0),
            pp_comm_perf.get("wait_transfer_max_ms", 0.0),
            pp_comm_perf.get("wait_output_calls", 0),
            pp_comm_perf.get("wait_output_avg_ms", 0.0),
            pp_comm_perf.get("wait_output_max_ms", 0.0),
            pp_comm_perf.get("wait_proxy_calls", 0),
            pp_comm_perf.get("wait_proxy_avg_ms", 0.0),
            pp_comm_perf.get("wait_proxy_max_ms", 0.0),
            pp_comm_perf.get("wait_consensus_bootstrap_calls", 0),
            pp_comm_perf.get("wait_consensus_bootstrap_avg_ms", 0.0),
            pp_comm_perf.get("wait_consensus_bootstrap_max_ms", 0.0),
            pp_comm_perf.get("wait_release_calls", 0),
            pp_comm_perf.get("wait_release_avg_ms", 0.0),
            pp_comm_perf.get("wait_release_max_ms", 0.0),
            prefill_stage_perf.get("pick_early_full_or_empty", 0),
            prefill_stage_perf.get("pick_chunked_capacity_block", 0),
            prefill_stage_perf.get("pick_test_retract_block", 0),
            prefill_stage_perf.get("pick_locally_revoked_break", 0),
            prefill_stage_perf.get("pick_running_full_break", 0),
            prefill_stage_perf.get("pick_prefetch_break", 0),
            prefill_stage_perf.get("pick_write_backup_break", 0),
            prefill_stage_perf.get("pick_add_no_token_break", 0),
            prefill_stage_perf.get("pick_add_other_break", 0),
            prefill_stage_perf.get("pick_empty_result", 0),
            pp_frontier_ack.get("recv", 0),
            pp_frontier_ack.get("activate", 0),
            pp_frontier_ack.get("consume", 0),
            pp_frontier_ack.get("consume_miss", 0),
            pp_frontier_ack.get("pending_same_mb_miss", 0),
            pp_frontier_ack.get("chunked_mismatch", 0),
            pp_frontier_ack.get("waiting_mismatch", 0),
            pp_frontier_ack.get("ack_exhausted", 0),
            len(getattr(self, "pp_pending_launch_frontier_ack_by_mb", {})),
            len(getattr(self, "pp_launch_frontier_ack_by_mb", {})),
            barrier_hits_delta,
            replay_perf.get("apply_local_ack", 0),
            replay_perf.get("apply_authoritative", 0),
            replay_perf.get("miss", 0),
            replay_perf.get("pending_wb_events", 0),
            replay_perf.get("commit_nodes", 0),
            replay_perf.get("event_count", 0),
            replay_perf.get("event_avg_nodes", 0.0),
            replay_perf.get("event_max_nodes", 0),
            replay_perf.get("event_est_bytes", 0),
            replay_perf.get("event_avg_est_bytes", 0.0),
            replay_perf.get("event_max_est_bytes", 0),
            tree_shape.get("nodes", -1),
            tree_shape.get("evicted_nodes", -1),
            tree_shape.get("backuped_nodes", -1),
            tree_shape.get("leaf_nodes", -1),
            tree_shape.get("max_depth", -1),
            tree_shape.get("max_fanout", -1),
            tree_shape.get("allocated_node_id", -1),
            tree_churn.get("allocated_delta", -1),
            tree_churn.get("alloc_match_split", -1),
            tree_churn.get("alloc_insert_split", -1),
            tree_churn.get("alloc_host_insert_split", -1),
            tree_churn.get("alloc_device_leaf", -1),
            tree_churn.get("alloc_host_leaf", -1),
            tree_churn.get("delete_regular_leaf", -1),
            tree_churn.get("delete_host_leaf", -1),
        )

    def _emit_kv_metrics(self: Scheduler):
        if not self.enable_kv_cache_events:
            return

        kv_metrics = KvMetrics()
        kv_metrics.request_active_slots = self.stats.num_running_reqs.total
        kv_metrics.request_total_slots = self.max_running_requests
        kv_metrics.kv_active_blocks = int(
            self.stats.token_usage * self.max_total_num_tokens
        )
        kv_metrics.kv_total_blocks = self.max_total_num_tokens
        kv_metrics.num_requests_waiting = self.stats.num_queue_reqs.total
        kv_metrics.gpu_cache_usage_perc = self.stats.token_usage
        kv_metrics.gpu_prefix_cache_hit_rate = self.stats.cache_hit_rate
        kv_metrics.data_parallel_rank = self.dp_rank if self.dp_rank is not None else 0

        if not self.send_metrics_from_scheduler.closed:
            self.send_metrics_from_scheduler.send_pyobj(kv_metrics)

    def _publish_kv_events(self: Scheduler):
        if not self.enable_kv_cache_events:
            return

        events = self.tree_cache.take_events()
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

    def _log_hicache_stats(self: Scheduler):
        """Populate HiCache host-tier stats on self.stats.

        These are pushed to Prometheus by SchedulerMetricsCollector.log_stats().
        """
        if not self.enable_hierarchical_cache:
            return

        host_pool = getattr(self.tree_cache, "token_to_kv_pool_host", None) or getattr(
            self.tree_cache, "full_kv_pool_host", None
        )
        assert host_pool is not None, "Host pool not found"
        self.stats.hicache_host_used_tokens = (
            host_pool.size - host_pool.available_size()
        )
        self.stats.hicache_host_total_tokens = host_pool.size

    def update_lora_metrics(self: Scheduler):
        """Update LoRA pool metrics for monitoring and autoscaling."""
        if not self.enable_lora:
            return

        try:
            # Get LoRA memory pool stats
            lora_manager = self.tp_worker.model_runner.lora_manager
            if lora_manager is None or lora_manager.memory_pool is None:
                return

            mem_pool = lora_manager.memory_pool
            slots_total = mem_pool.max_loras_per_batch

            # Calculate active adapters from running batch
            # This gives a true measure of current load for autoscaling purposes
            active_lora_ids = set()

            # For PP mode, check all running micro batches
            if hasattr(self, "running_mbs") and self.running_mbs:
                for batch in self.running_mbs:
                    if batch and hasattr(batch, "reqs"):
                        for req in batch.reqs:
                            if hasattr(req, "lora_id") and req.lora_id is not None:
                                active_lora_ids.add(req.lora_id)
            # For normal mode, check running_batch
            elif hasattr(self, "running_batch") and self.running_batch:
                if hasattr(self.running_batch, "reqs"):
                    for req in self.running_batch.reqs:
                        if hasattr(req, "lora_id") and req.lora_id is not None:
                            active_lora_ids.add(req.lora_id)

            # Count active adapters (excluding None for base model)
            slots_used = len(active_lora_ids)
            utilization = slots_used / slots_total if slots_total > 0 else 0.0

            # Update stats
            self.stats.lora_pool_slots_used = slots_used
            self.stats.lora_pool_slots_total = slots_total
            self.stats.lora_pool_utilization = utilization

        except Exception as e:
            logger.warning(f"Failed to update LoRA metrics: {e}")

    def calculate_utilization(self: Scheduler):
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.stats.utilization = -1
        else:
            if (
                self.stats.max_running_requests_under_SLO is not None
                and self.stats.max_running_requests_under_SLO > 0
            ):
                self.stats.utilization = max(
                    self.stats.num_running_reqs.total
                    / self.stats.max_running_requests_under_SLO,
                    self.stats.token_usage / 0.9,
                )

    def get_load(self: Scheduler, _: GetLoadReqInput = None) -> GetLoadReqOutput:
        if self.is_hybrid_swa:
            full_num_used, swa_num_used, *_ = self._get_swa_token_info()
            num_tokens = max(full_num_used, swa_num_used)
        elif self.is_hybrid_ssm:
            num_tokens = self._get_mamba_token_info()[0]
        else:
            num_tokens = self._get_token_info()[0]

        # Tokens in waiting queue, bootstrap queue, prealloc queue
        waiting_queues = [self.waiting_queue]
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            waiting_queues.append(self.disagg_prefill_bootstrap_queue.queue)
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            waiting_queues.append(self.disagg_decode_prealloc_queue.queue)
            waiting_queues.append(self.disagg_decode_transfer_queue.queue)
            waiting_queues.append(self.disagg_decode_prealloc_queue.retracted_queue)

        num_tokens += sum(req.seqlen for queue in waiting_queues for req in queue)
        num_waiting_reqs = sum(len(queue) for queue in waiting_queues)

        return GetLoadReqOutput(
            dp_rank=self.dp_rank,
            num_reqs=len(self.running_batch.reqs) + num_waiting_reqs,
            num_waiting_reqs=num_waiting_reqs,
            num_tokens=num_tokens,
            ts_tic=time.perf_counter(),
        )

    def get_loads(self: Scheduler, req: GetLoadsReqInput = None) -> GetLoadsReqOutput:
        """
        Get comprehensive load metrics for /v1/loads endpoint.

        Args:
            req: Request containing include list and optional dp_rank filter

        Returns:
            GetLoadsReqOutput with core metrics and optional detailed sections
        """
        if req is None:
            req = GetLoadsReqInput()

        include = set(req.include) if req.include else {"core"}
        include_all = "all" in include

        num_running_reqs = len(self.running_batch.reqs)

        waiting_queues = [self.waiting_queue]
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            waiting_queues.append(self.disagg_prefill_bootstrap_queue.queue)
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            waiting_queues.append(self.disagg_decode_prealloc_queue.queue)
            waiting_queues.append(self.disagg_decode_transfer_queue.queue)
            waiting_queues.append(self.disagg_decode_prealloc_queue.retracted_queue)

        num_waiting_reqs = sum(len(queue) for queue in waiting_queues)

        if self.is_hybrid_swa:
            full_num_used, swa_num_used, *_ = self._get_swa_token_info()
            num_used_tokens = max(full_num_used, swa_num_used)
        elif self.is_hybrid_ssm:
            num_used_tokens = self._get_mamba_token_info()[0]
        else:
            num_used_tokens = self._get_token_info()[0]

        token_usage = (
            num_used_tokens / self.max_total_num_tokens
            if self.max_total_num_tokens > 0
            else 0.0
        )

        memory = None
        if include_all or "memory" in include:
            try:
                memory = MemoryMetrics(
                    weight_gb=round(
                        self.tp_worker.model_runner.weight_load_mem_usage, 3
                    ),
                    kv_cache_gb=round(
                        self.token_to_kv_pool_allocator.get_kvcache().mem_usage, 3
                    ),
                    graph_gb=round(self.tp_worker.model_runner.graph_mem_usage, 3),
                    token_capacity=int(self.max_total_num_tokens),
                )
            except AttributeError as e:
                logger.debug(f"Memory metrics not available: {e}")

        speculative = None
        if include_all or "spec" in include:
            if not self.spec_algorithm.is_none() and self.spec_total_num_forward_ct > 0:
                speculative = SpeculativeMetrics(
                    accept_length=(
                        self.spec_total_num_accepted_tokens
                        / self.spec_total_num_forward_ct
                    ),
                    accept_rate=self.stats.spec_accept_rate,
                )

        lora = None
        if include_all or "lora" in include:
            if hasattr(self, "lora_scheduler") and self.lora_scheduler is not None:
                lora = LoRAMetrics(
                    slots_used=self.stats.lora_pool_slots_used,
                    slots_total=self.stats.lora_pool_slots_total,
                    utilization=self.stats.lora_pool_utilization,
                )

        disaggregation = None
        if include_all or "disagg" in include:
            mode_str = "null"
            prefill_prealloc = 0
            prefill_inflight = 0
            decode_prealloc = 0
            decode_transfer = 0
            decode_retracted = 0

            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                mode_str = "prefill"
                prefill_prealloc = len(self.disagg_prefill_bootstrap_queue.queue)
                prefill_inflight = len(self.disagg_prefill_inflight_queue)
            elif self.disaggregation_mode == DisaggregationMode.DECODE:
                mode_str = "decode"
                decode_prealloc = len(self.disagg_decode_prealloc_queue.queue)
                decode_transfer = len(self.disagg_decode_transfer_queue.queue)
                decode_retracted = len(
                    self.disagg_decode_prealloc_queue.retracted_queue
                )

            disaggregation = DisaggregationMetrics(
                mode=mode_str,
                prefill_prealloc_queue_reqs=prefill_prealloc,
                prefill_inflight_queue_reqs=prefill_inflight,
                decode_prealloc_queue_reqs=decode_prealloc,
                decode_transfer_queue_reqs=decode_transfer,
                decode_retracted_queue_reqs=decode_retracted,
                kv_transfer_speed_gb_s=self.stats.kv_transfer_speed_gb_s,
                kv_transfer_latency_ms=self.stats.kv_transfer_latency_ms,
            )

        queues = None
        if include_all or "queues" in include:
            queues = QueueMetrics(
                waiting=len(self.waiting_queue),
                grammar=self.stats.num_grammar_queue_reqs,
                paused=self.stats.num_paused_reqs,
                retracted=self.stats.num_retracted_reqs,
            )

        return GetLoadsReqOutput(
            dp_rank=self.dp_rank,
            timestamp=time.time(),
            num_running_reqs=num_running_reqs,
            num_waiting_reqs=num_waiting_reqs,
            num_used_tokens=num_used_tokens,
            max_total_num_tokens=self.max_total_num_tokens,
            token_usage=round(token_usage, 4),
            gen_throughput=round(self.stats.gen_throughput, 2),
            cache_hit_rate=round(self.stats.cache_hit_rate, 4),
            utilization=round(self.stats.utilization, 4),
            max_running_requests=self.max_running_requests,
            memory=memory,
            speculative=speculative,
            lora=lora,
            disaggregation=disaggregation,
            queues=queues,
        )

    @contextmanager
    def record_forward_metrics(self: Scheduler, batch: ScheduleBatch):
        if not (self.enable_metrics and ENABLE_METRICS_DEVICE_TIMER):
            yield
            return

        category = "forward_" + batch.forward_mode.name.lower()
        with self.forward_pass_device_timer.wrap(
            metadata=dict(
                category=category,
                dp_cooperation_info=batch.dp_cooperation_info,
            ),
        ):
            yield

    @contextmanager
    def record_bubble_metrics(self: Scheduler, batch: ScheduleBatch):
        if not (self.enable_metrics and ENABLE_METRICS_DEVICE_TIMER):
            yield
            return

        category = "forward_" + batch.forward_mode.name.lower()
        with self.bubble_timer.wrap(
            metadata=dict(
                category=category,
                dp_cooperation_info=batch.dp_cooperation_info,
            ),
        ):
            yield

    def cancel_bubble_timer(self: Scheduler):
        if self.enable_metrics and ENABLE_METRICS_DEVICE_TIMER:
            self.bubble_timer.cancel()
