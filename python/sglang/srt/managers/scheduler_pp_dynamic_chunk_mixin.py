from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch.distributed

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.warmup import PP_WARMUP_RID_PREFIX

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


def calculate_all_chunk_sizes(
    total_seq_len: int,
    quadratic_coeff_a: float,
    linear_coeff_b: float,
    target_chunk_time: float,
    page_size: int,
    context_len: int,
) -> List[int]:
    """
    Calculate all chunk sizes for a request based on dynamic chunk function.
    
    Per formula: Solve f(L+x) - f(L) = T
    where f(L) = a*L^2 + b*L + c
    This expands to: ax^2 + (2aL+b)x - T = 0
    
    Args:
        total_seq_len: Total sequence length of the request
        quadratic_coeff_a: Quadratic coefficient a in f(l) = al^2 + bl + c
        linear_coeff_b: Linear coefficient b in f(l) = al^2 + bl + c
        target_chunk_time: Target chunk time T
        page_size: Page size for alignment
        context_len: Maximum context length
        
    Returns:
        List of chunk sizes
    """
    chunk_sizes = []
    current_seq_len = 0
    
    # Use page_size for alignment (at least 1)
    alignment_size = max(page_size, 1)
    
    while current_seq_len < total_seq_len:
        remaining = total_seq_len - current_seq_len
        
        # If remaining tokens are less than alignment_size, use all remaining
        if remaining < alignment_size:
            if remaining > 0:
                chunk_sizes.append(remaining)
            break
        
        # Calculate dynamic chunk size using the formula:
        # Solve f(L+x) - f(L) = T
        # where f(L) = a*L^2 + b*L + c
        # This expands to: ax^2 + (2aL+b)x - T = 0
        # A = a, B = 2aL + b, C = -T
        A = quadratic_coeff_a
        B = 2 * quadratic_coeff_a * current_seq_len + linear_coeff_b
        C = -target_chunk_time
        
        discriminant = B * B - 4 * A * C
        
        if discriminant < 0:
            raise ValueError(
                f"Discriminant is negative ({discriminant:.2e}). "
                f"No real solution for chunk size. "
                f"A={A:.2e}, B={B:.2e}, C={C:.2e}, L={current_seq_len}, T={target_chunk_time:.4f}s."
            )
        
        sqrt_discriminant = math.sqrt(discriminant)
        calculated_chunk_size_float = (-B + sqrt_discriminant) / (2 * A)
        
        if calculated_chunk_size_float <= 0:
            raise ValueError(
                f"Calculated chunk size is non-positive ({calculated_chunk_size_float:.2f}). "
                f"A={A:.2e}, B={B:.2e}, C={C:.2e}, L={current_seq_len}, T={target_chunk_time:.4f}s."
            )
        
        calculated_chunk_size = int(calculated_chunk_size_float)
        
        # Align to page_size (round down to nearest multiple)
        dynamic_chunk_size = (calculated_chunk_size // alignment_size) * alignment_size
        
        # Ensure aligned size is at least alignment_size
        if dynamic_chunk_size < alignment_size:
            dynamic_chunk_size = alignment_size
        
        # Ensure we don't exceed remaining tokens or context_len
        max_allowed_chunk = min(
            remaining,
            context_len - current_seq_len - 100  # Leave 100 tokens margin
        )
        dynamic_chunk_size = min(dynamic_chunk_size, max_allowed_chunk)
        
        # Align again after min operation
        dynamic_chunk_size = (dynamic_chunk_size // alignment_size) * alignment_size
        
        if dynamic_chunk_size < alignment_size:
            # If after alignment we have less than alignment_size, use remaining tokens
            dynamic_chunk_size = remaining
        
        chunk_sizes.append(dynamic_chunk_size)
        current_seq_len += dynamic_chunk_size
    
    return chunk_sizes


class SchedulerPPDynamicChunkMixin:
    """
    Mixin for PP mode dynamic chunk size adjustment.
    
    This mixin provides functionality to:
    1. Collect warmup data during PP chunk tuning warmup
    2. Fit quadratic function f(l) = al^2 + bl + c from collected data
    3. Calculate dynamic chunk sizes for requests based on fitted coefficients
    """

    def init_pp_dynamic_chunk_size(self: "Scheduler"):
        """Initialize PP dynamic chunk size coefficients (will be set during warmup)."""
        # Initialize PP dynamic chunk size coefficients (will be set during warmup)
        self.quadratic_coeff_a = 0.0
        self.linear_coeff_b = 0.0
        self.constant_coeff_c = 0.0
        self.target_chunk_time: Optional[float] = None
        self.warmup_complete = False
        # Warmup data collection: collect (sequence_length, total_latency) pairs per request
        self.warmup_request_data: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        # Track cumulative latency per warmup request
        self.warmup_request_cumulative_latency: Dict[str, float] = {}
        # Track which requests have recorded f(0) = 0
        self._recorded_f0_for_rid: set = set()

    def _is_warmup_request(self: "Scheduler", batch: ScheduleBatch) -> bool:
        """Check if the batch contains warmup requests (identified by rid prefix)."""
        return any(
            req.rid is not None and req.rid.startswith(PP_WARMUP_RID_PREFIX)
            for req in batch.reqs
        )

    def _record_warmup_chunk_time(self: "Scheduler", batch: ScheduleBatch):
        """Record chunk execution time for warmup requests."""
        if not self._is_warmup_request(batch) or self.warmup_complete:
            return
        
        if batch.forward_mode != ForwardMode.EXTEND:
            return
        
        for req in batch.reqs:
            if req.rid is None or not req.rid.startswith(PP_WARMUP_RID_PREFIX):
                continue
            
            # Calculate chunk time
            if (
                req.time_stats.prefill_start_time_host > 0
                and req.time_stats.prefill_end_time_host > 0
            ):
                chunk_time = (
                    req.time_stats.prefill_end_time_host
                    - req.time_stats.prefill_start_time_host
                )
            else:
                continue
            
            # Extract chunk information
            if hasattr(req, "extend_input_len") and req.extend_input_len > 0:
                chunk_size = req.extend_input_len
                seq_len_at_start = len(req.prefix_indices)
                current_seq_len = seq_len_at_start + chunk_size
            else:
                # Non-chunked prefill
                current_seq_len = req.seqlen
                chunk_size = current_seq_len
                seq_len_at_start = 0
            
            # For the first chunk of each request (seq_len_at_start=0), record f(0) = 0
            if seq_len_at_start == 0:
                if req.rid not in self._recorded_f0_for_rid:
                    if req.rid not in self.warmup_request_data:
                        self.warmup_request_data[req.rid] = []
                    # Record f(0) = 0 (before processing any chunks)
                    self.warmup_request_data[req.rid].append((0, 0.0))
                    self._recorded_f0_for_rid.add(req.rid)
                    # Initialize cumulative latency for this request
                    self.warmup_request_cumulative_latency[req.rid] = 0.0
            
            # Accumulate chunk time to get total latency
            cumulative_latency = self.warmup_request_cumulative_latency.get(req.rid, 0.0)
            cumulative_latency += chunk_time
            self.warmup_request_cumulative_latency[req.rid] = cumulative_latency
            
            # Collect (sequence_length, total_latency) pair
            self.warmup_request_data[req.rid].append((current_seq_len, cumulative_latency))
            
            logger.info(
                f"[PP Dynamic Chunk] [PP{self.pp_rank}] Warmup sample for request {req.rid}: "
                f"seq_len={current_seq_len}, total_latency={cumulative_latency:.4f}s "
                f"(chunk_time={chunk_time:.4f}s, chunk_size={chunk_size}, "
                f"seq_len_at_start={seq_len_at_start})"
            )
            
            # Check if we have enough samples for fitting
            total_samples = sum(len(data) for data in self.warmup_request_data.values())
            all_seq_lens = set()
            for data in self.warmup_request_data.values():
                all_seq_lens.update(seq_len for seq_len, _ in data)
            unique_seq_lens = len(all_seq_lens)
            
            # Warmup completion condition: at least 10 samples with at least 3 unique seq_lens
            warmup_samples_needed = 10
            if unique_seq_lens >= 3 and total_samples >= warmup_samples_needed:
                self._complete_warmup()

    def _complete_warmup(self: "Scheduler"):
        """Complete warmup phase by fitting f(l) = al^2 + bl + c.
        
        In PP mode, we collect data from all PP ranks and fit a unified set of coefficients
        on PP rank 0, then broadcast to all ranks.
        """
        if self.warmup_complete:
            return
        
        self.warmup_complete = True
        
        # Collect ALL data points from all requests on this rank
        local_warmup_seq_lens: List[int] = []
        local_warmup_total_latencies: List[float] = []
        
        for req_rid, data in self.warmup_request_data.items():
            for seq_len, total_latency in data:
                local_warmup_seq_lens.append(seq_len)
                local_warmup_total_latencies.append(total_latency)
        
        logger.info(
            f"[PP Dynamic Chunk] [PP{self.pp_rank}] Collected {len(local_warmup_seq_lens)} local data points "
            f"from {len(self.warmup_request_data)} requests."
        )
        
        # In PP mode, gather data from all ranks and fit on rank 0, then broadcast
        if self.pp_size > 1:
            local_data = list(zip(local_warmup_seq_lens, local_warmup_total_latencies))
            
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier(group=self.pp_group.cpu_group)
                
                gathered_data_list = [None] * self.pp_size
                torch.distributed.all_gather_object(
                    gathered_data_list, local_data, group=self.pp_group.cpu_group
                )
            else:
                gathered_data_list = [local_data]
            
            # On rank 0, aggregate all data and fit
            if self.pp_rank == 0:
                warmup_seq_lens: List[int] = []
                warmup_total_latencies: List[float] = []
                
                for rank_data in gathered_data_list:
                    if rank_data is not None:
                        for seq_len, total_latency in rank_data:
                            warmup_seq_lens.append(seq_len)
                            warmup_total_latencies.append(total_latency)
                
                logger.info(
                    f"[PP Dynamic Chunk] [PP0] Aggregated {len(warmup_seq_lens)} total data points "
                    f"from all {self.pp_size} PP ranks for unified fitting."
                )
                
                if len(warmup_seq_lens) < 3:
                    raise ValueError(
                        f"Not enough data points for fitting ({len(warmup_seq_lens)} < 3). "
                        "Need at least 3 samples with different sequence lengths."
                    )
                
                # Fit on rank 0
                fitted_a, fitted_b, fitted_c = self._fit_quadratic_coefficients(
                    warmup_seq_lens, warmup_total_latencies
                )
                
                # Broadcast coefficients to all ranks
                coeffs_to_broadcast = [fitted_a, fitted_b, fitted_c]
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.broadcast_object_list(
                        coeffs_to_broadcast, src=0, group=self.pp_group.cpu_group
                    )
                
                fitted_a, fitted_b, fitted_c = coeffs_to_broadcast
            else:
                # Other ranks: receive coefficients from rank 0
                coeffs_to_receive = [None, None, None]
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.broadcast_object_list(
                        coeffs_to_receive, src=0, group=self.pp_group.cpu_group
                    )
                else:
                    raise RuntimeError("Distributed not available but PP size > 1")
                
                fitted_a, fitted_b, fitted_c = coeffs_to_receive
        else:
            # Single rank mode: fit locally
            if len(local_warmup_seq_lens) < 3:
                raise ValueError(
                    f"Not enough data points for fitting ({len(local_warmup_seq_lens)} < 3). "
                    "Need at least 3 samples with different sequence lengths."
                )
            
            fitted_a, fitted_b, fitted_c = self._fit_quadratic_coefficients(
                local_warmup_seq_lens, local_warmup_total_latencies
            )
        
        # Store coefficients (same for all ranks in PP mode)
        self.quadratic_coeff_a = fitted_a
        self.linear_coeff_b = fitted_b
        self.constant_coeff_c = fitted_c
        
        # Calculate target chunk time: chunk_time = f(base_chunk_size) - f(0)
        def f(l: float) -> float:
            """Total latency function: f(l) = al^2 + bl + c"""
            return fitted_a * l * l + fitted_b * l + fitted_c
        
        self.target_chunk_time = f(self.chunked_prefill_size) - f(0.0)
        
        if self.target_chunk_time <= 0:
            raise ValueError(
                f"[PP{self.pp_rank}] Calculated target_chunk_time={self.target_chunk_time:.4f}s is not positive. "
                "Check warmup data quality."
            )
        
        logger.info(
            f"[PP Dynamic Chunk] [PP{self.pp_rank}] Warmup complete. "
            f"Target chunk time: {self.target_chunk_time:.4f}s, "
            f"Base chunk size: {self.chunked_prefill_size}, "
            f"Coefficients: a={self.quadratic_coeff_a:.2e}, "
            f"b={self.linear_coeff_b:.2e}, c={self.constant_coeff_c:.2e}"
        )

    def _fit_quadratic_coefficients(
        self: "Scheduler",
        warmup_seq_lens: List[int],
        warmup_total_latencies: List[float],
    ) -> Tuple[float, float, float]:
        """Fit quadratic coefficients f(l) = al^2 + bl + c from data points."""
        L = np.array(warmup_seq_lens, dtype=np.float64)
        T = np.array(warmup_total_latencies, dtype=np.float64)
        
        if len(L) < 3:
            raise ValueError(
                f"Not enough data points for fitting ({len(L)} < 3). "
                "Need at least 3 samples with different sequence lengths."
            )
        
        # Build design matrix for f(l) = al^2 + bl + c
        X = np.column_stack([L * L, L, np.ones_like(L)])  # [l^2, l, 1]
        
        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(X, T, rcond=None)
            if len(coeffs) >= 3:
                fitted_a = float(coeffs[0])  # quadratic coefficient
                fitted_b = float(coeffs[1])  # linear coefficient
                fitted_c = float(coeffs[2])  # constant coefficient
            else:
                raise ValueError("Failed to fit coefficients: insufficient rank")
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Failed to fit f(l) = al^2 + bl + c: {e}")
        
        # Validate coefficients
        if fitted_a <= 0:
            raise ValueError(
                f"[PP{self.pp_rank}] Fitted quadratic coefficient a={fitted_a:.2e} is not positive. "
                "Attention has O(n^2) complexity, so a must be positive. "
                "Check warmup data quality."
            )
        
        if fitted_b < 0:
            logger.warning(
                f"[PP Dynamic Chunk] [PP{self.pp_rank}] Fitted linear coefficient b={fitted_b:.2e} is negative. "
                f"Setting b=0."
            )
            fitted_b = 0.0
        
        return fitted_a, fitted_b, fitted_c

    def _calculate_dynamic_chunk_sizes_for_request(self: "Scheduler", req: Req):
        """
        Calculate all chunk sizes for a request and assign to req.chunked_prefill_sizes.
        
        This function:
        1. Checks if warmup is complete and coefficients are available
        2. Calculates all chunk sizes based on the request's total sequence length
        3. Assigns the calculated chunk sizes list directly to req.chunked_prefill_sizes
        
        Args:
            req: The request object to calculate chunk sizes for
        """
        if (
            self.chunked_prefill_size is not None
            and self.pp_size > 1
            and self.warmup_complete
            and self.quadratic_coeff_a > 0
            and self.target_chunk_time is not None
            and self.target_chunk_time > 0
        ):
            total_seq_len = len(req.origin_input_ids)
            if total_seq_len > self.chunked_prefill_size:
                # Calculate all chunk sizes using fitted coefficients
                chunk_sizes = calculate_all_chunk_sizes(
                    total_seq_len=total_seq_len,
                    quadratic_coeff_a=self.quadratic_coeff_a,
                    linear_coeff_b=self.linear_coeff_b,
                    target_chunk_time=self.target_chunk_time,
                    page_size=self.page_size,
                    context_len=self.model_config.context_len,
                )
                if chunk_sizes:
                    # Assign calculated chunk sizes directly to request
                    req.chunked_prefill_sizes = chunk_sizes

    def _get_dynamic_chunk_size_for_batch(self: "Scheduler") -> Optional[int]:
        """Get dynamic chunk size for current chunked request if available."""
        if (
            self.chunked_req is not None
            and self.chunked_req.chunked_prefill_sizes is not None
        ):
            # Handle both list and iterator cases
            chunk_sizes = self.chunked_req.chunked_prefill_sizes
            if isinstance(chunk_sizes, list):
                # Convert list to iterator and save it back
                self.chunked_req.chunked_prefill_sizes = iter(chunk_sizes)
                chunk_sizes = self.chunked_req.chunked_prefill_sizes
            
            # Use dynamic chunk size from pre-calculated list/iterator
            try:
                return next(chunk_sizes)
            except StopIteration:
                # Fallback to original chunk size if iterator is exhausted
                return self.chunked_prefill_size
        return None

