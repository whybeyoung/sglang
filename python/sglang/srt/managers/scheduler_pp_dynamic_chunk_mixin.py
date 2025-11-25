from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import torch
import torch.distributed

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.sampling_params import SamplingParams

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


class ChunkSizePredictor:
    """
    Predictor for dynamic chunk size based on quadratic latency model.
    
    Models latency as: f(l) = a*l^2 + b*l + c
    Predicts next chunk size x such that: f(L+x) - f(L) = target_latency
    """

    def __init__(self):
        self.quadratic_coeff_a = 0.0
        self.linear_coeff_b = 0.0
        self.constant_coeff_c = 0.0
        self.use_linear_model = False  # Flag to indicate if using linear model
        self.target_latency: Optional[float] = None
        self.is_ready = False

    def fit(self, seq_lens: List[int], latencies: List[float]):
        """Fit quadratic coefficients f(l) = al^2 + bl + c from data points."""
        L = np.array(seq_lens, dtype=np.float64)
        T = np.array(latencies, dtype=np.float64)

        if len(L) < 8:
            raise ValueError(
                f"Not enough data points for quadratic fitting ({len(L)} < 8). "
                "Need at least 8 samples with different sequence lengths."
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
                f"Fitted quadratic coefficient a={fitted_a:.2e} is not positive. "
                "Attention has O(n^2) complexity, so a must be positive. "
                "Check warmup data quality."
            )

        if fitted_b < 0:
            logger.warning(
                f"Fitted linear coefficient b={fitted_b:.2e} is negative. Setting b=0."
            )
            fitted_b = 0.0

        self.quadratic_coeff_a = fitted_a
        self.linear_coeff_b = fitted_b
        self.constant_coeff_c = fitted_c

        logger.info(
            f"[ChunkSizePredictor] Fitted coefficients: a={fitted_a:.2e}, "
            f"b={fitted_b:.2e}, c={fitted_c:.2e}"
        )
        self.use_linear_model = False

    def fit_linear(self, seq_lens: List[int], latencies: List[float]):
        """
        Fit linear model: f(l) = b*l + c
        
        This is a simpler model that assumes latency grows linearly with sequence length.
        """
        L = np.array(seq_lens, dtype=np.float64)
        T = np.array(latencies, dtype=np.float64)

        if len(L) < 8:
            raise ValueError(
                f"Not enough data points for linear fitting ({len(L)} < 8). "
                "Need at least 8 samples with different sequence lengths."
            )

        # Build design matrix for f(l) = bl + c
        X = np.column_stack([L, np.ones_like(L)])  # [l, 1]

        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(X, T, rcond=None)
            if len(coeffs) >= 2:
                fitted_b = float(coeffs[0])  # linear coefficient
                fitted_c = float(coeffs[1])  # constant coefficient
            else:
                raise ValueError("Failed to fit linear coefficients: insufficient rank")
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Failed to fit f(l) = bl + c: {e}")

        # Validate coefficients
        if fitted_b <= 0:
            raise ValueError(
                f"Fitted linear coefficient b={fitted_b:.2e} is not positive. "
                "Latency should increase with sequence length, so b must be positive. "
                "Check warmup data quality."
            )

        # Store coefficients (set quadratic coefficient to 0 for linear model)
        self.quadratic_coeff_a = 0.0
        self.linear_coeff_b = fitted_b
        self.constant_coeff_c = fitted_c
        self.use_linear_model = True

        logger.info(
            f"[ChunkSizePredictor] Fitted linear coefficients: "
            f"b={fitted_b:.2e}, c={fitted_c:.2e}"
        )

    def set_target_latency(self, base_chunk_size: int):
        """Set target latency based on base chunk size: target = f(base_chunk_size) - f(0)."""
        def f(l: float) -> float:
            """Total latency function: f(l) = al^2 + bl + c (or bl + c for linear)"""
            if self.use_linear_model:
                return self.linear_coeff_b * l + self.constant_coeff_c
            else:
                return self.quadratic_coeff_a * l * l + self.linear_coeff_b * l + self.constant_coeff_c

        self.target_latency = f(float(base_chunk_size)) - f(0.0)

        if self.target_latency <= 0:
            raise ValueError(
                f"Calculated target_latency={self.target_latency:.2f}ms is not positive. "
                "Check warmup data quality."
            )

        logger.info(
            f"[ChunkSizePredictor] Target latency: {self.target_latency:.2f}ms "
            f"(base_chunk_size={base_chunk_size})"
        )

    def predict_next_chunk_size(
        self,
        history_len: int,
        page_size: int,
        context_len: int,
        max_chunk_size: Optional[int] = None,
    ) -> Optional[int]:
        """
        Predict next chunk size x such that f(history_len + x) - f(history_len) = target_latency.
        
        Args:
            history_len: Current sequence length (L)
            page_size: Page size for alignment
            context_len: Maximum context length
            max_chunk_size: Maximum allowed chunk size (optional)
            
        Returns:
            Predicted chunk size, or None if prediction fails
        """
        if not self.is_ready or self.target_latency is None:
            return None

        # Handle linear model: f(l) = bl + c
        if self.use_linear_model:
            # For linear model: f(L+x) - f(L) = b*(L+x) + c - (b*L + c) = b*x = T
            # So x = T / b
            if self.linear_coeff_b <= 0:
                return None
            
            calculated_chunk_size_float = self.target_latency / self.linear_coeff_b
        else:
            # Handle quadratic model: f(l) = al^2 + bl + c
            if self.quadratic_coeff_a <= 0:
                return None

            # Solve f(L+x) - f(L) = T
            # where f(L) = a*L^2 + b*L + c
            # This expands to: ax^2 + (2aL+b)x - T = 0
            # A = a, B = 2aL + b, C = -T
            A = self.quadratic_coeff_a
            B = 2 * self.quadratic_coeff_a * history_len + self.linear_coeff_b
            C = -self.target_latency

            discriminant = B * B - 4 * A * C

            if discriminant < 0:
                logger.warning(
                    f"Discriminant is negative ({discriminant:.2e}). "
                    f"No real solution for chunk size. L={history_len}, T={self.target_latency:.2f}ms."
                )
                return None

            sqrt_discriminant = math.sqrt(discriminant)
            calculated_chunk_size_float = (-B + sqrt_discriminant) / (2 * A)

        if calculated_chunk_size_float <= 0:
            logger.warning(
                f"Calculated chunk size is non-positive ({calculated_chunk_size_float:.2f}). "
                f"L={history_len}, T={self.target_latency:.2f}ms."
            )
            return None

        calculated_chunk_size = int(calculated_chunk_size_float)

        # Align to page_size (round down to nearest multiple)
        alignment_size = max(page_size, 1)
        dynamic_chunk_size = (calculated_chunk_size // alignment_size) * alignment_size

        # Ensure aligned size is at least alignment_size
        if dynamic_chunk_size < alignment_size:
            dynamic_chunk_size = alignment_size

        # Apply constraints
        max_allowed = context_len - history_len - 100  # Leave 100 tokens margin
        if max_chunk_size is not None:
            max_allowed = min(max_allowed, max_chunk_size)
        dynamic_chunk_size = min(dynamic_chunk_size, max_allowed)

        # Align again after min operation
        dynamic_chunk_size = (dynamic_chunk_size // alignment_size) * alignment_size

        if dynamic_chunk_size < alignment_size:
            return None

        return dynamic_chunk_size


class SchedulerPPDynamicChunkMixin:
    """
    Mixin for PP mode dynamic chunk size adjustment.
    
    Per reference implementation:
    1. Profile prefill latency on PP0 (first rank)
    2. Broadcast profiling data to all ranks
    3. All ranks fit coefficients using ChunkSizePredictor
    4. Dynamically predict next chunk size at runtime
    """

    def init_pp_dynamic_chunk_size(self: "Scheduler", server_args):
        """Initialize PP dynamic chunk size predictor."""
        # Initialize attributes to default values
        # This ensures the attributes exist even when pp_size <= 1
        self.enable_dynamic_chunking = False
        self.length_predictor = None
        self.dynamic_chunking_model = server_args.dynamic_chunking_model
        
        if self.pp_size <= 1:
            return

        self.length_predictor = ChunkSizePredictor()
        # Enable dynamic chunking only if explicitly enabled via server_args
        # and chunked_prefill_size is set
        self.enable_dynamic_chunking = (
            server_args.enable_dynamic_chunking
            and self.chunked_prefill_size is not None
            and self.chunked_prefill_size > 0
        )
        # Store model type for fitting
        self.dynamic_chunking_model = getattr(server_args, "dynamic_chunking_model", "linear")

    def profile_pp_prefill_latency(self: "Scheduler"):
        """
        Profile prefill latency for dynamic chunk sizing.
        
        Only runs on PP0 (first rank), then broadcasts data to all ranks.
        All ranks fit coefficients using the same data.
        """
        # Early return if PP is not enabled or dynamic chunking is disabled
        if self.pp_size <= 1:
            return
        if not self.enable_dynamic_chunking:
            return

        seq_lens: List[int] = []
        latencies: List[float] = []

        if self.pp_group.is_first_rank:
            logger.info("Profiling prefill latency for dynamic chunk sizing...")

            # Create requests with different lengths: base_chunk_size // (2**i) for i in range(10)
            input_ids_list = []
            for i in range(10):
                chunk_size = self.chunked_prefill_size // (2 ** i)
                if chunk_size <= 0:
                    break
                input_ids = np.random.randint(
                    0, 10000, size=chunk_size, dtype=np.int64
                ).tolist()
                input_ids_list.append(input_ids)

            sampling_params = SamplingParams(
                temperature=0,
                max_new_tokens=1,
            )

            # Create and profile requests
            for i, input_ids in enumerate(input_ids_list):
                req = Req(
                    rid=str(i),
                    origin_input_text="",
                    origin_input_ids=input_ids,
                    sampling_params=sampling_params,
                )
                req.fill_ids = req.origin_input_ids
                req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
                req.logprob_start_len = len(req.origin_input_ids) - 1

                # Prepare batch
                batch = ScheduleBatch.init_new(
                    [req],
                    self.req_to_token_pool,
                    self.token_to_kv_pool_allocator,
                    self.tree_cache,
                    self.model_config,
                    False,
                    self.spec_algorithm,
                )

                current_seq_len = len(req.fill_ids)
                proxy_tensors = {
                    "hidden_states": torch.zeros(
                        (current_seq_len, self.tp_worker.model_runner.model_config.hidden_size),
                        dtype=self.tp_worker.model_runner.model_config.dtype,
                        device="cuda",
                    ),
                    "residual": torch.zeros(
                        (current_seq_len, self.tp_worker.model_runner.model_config.hidden_size),
                        dtype=self.tp_worker.model_runner.model_config.dtype,
                        device="cuda",
                    ),
                }
                from sglang.srt.managers.scheduler_pp_mixin import PPProxyTensors

                pp_proxy = PPProxyTensors(proxy_tensors)

                # Measure latency with CUDA synchronization for accurate timing
                # Synchronize before starting timing to ensure clean measurement
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                batch.prepare_for_extend()
                model_worker_batch = batch.get_model_worker_batch()
                from sglang.srt.model_executor.forward_batch_info import ForwardBatch

                forward_batch = ForwardBatch.init_new(
                    model_worker_batch, self.tp_worker.model_runner
                )
                _, _ = self.tp_worker.model_runner.forward(
                    forward_batch=forward_batch, pp_proxy_tensors=pp_proxy
                )
                
                # Synchronize after forward to ensure GPU operations complete
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                latency_seconds = time.perf_counter() - start
                latency_ms = latency_seconds * 1e3  # Convert to milliseconds
                seq_lens.append(len(input_ids))
                latencies.append(latency_ms)

                # Release KV cache
                from sglang.srt.mem_cache.common import release_kv_cache

                release_kv_cache(req, self.tree_cache)

            logger.info(
                f"[PP Dynamic Chunk] [PP0] Profiled {len(seq_lens)} samples: "
                f"seq_lens={seq_lens}, latencies_ms={latencies}"
            )

        # Broadcast data to all ranks
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            data_to_sync = [seq_lens, latencies]
            self.pp_group.broadcast_object_list(data_to_sync, src=0)
            seq_lens, latencies = data_to_sync

        # All ranks fit coefficients using the same data
        # Use model type specified by server_args
        # Both models require at least 8 data points
        if len(seq_lens) < 8:
            logger.warning(
                f"[PP Dynamic Chunk] [PP{self.pp_rank}] Not enough profiling data "
                f"({len(seq_lens)} < 8). Both quadratic and linear models require at least 8 samples. "
                f"Dynamic chunking disabled."
            )
            return
        
        if self.dynamic_chunking_model == "linear":
            # Linear model: f(l) = bl + c
            self.length_predictor.fit_linear(seq_lens, latencies)
            self.length_predictor.set_target_latency(self.chunked_prefill_size)
            self.length_predictor.is_ready = True
            logger.info(
                f"[PP Dynamic Chunk] [PP{self.pp_rank}] Predictor ready (linear). "
                f"Target latency: {self.length_predictor.target_latency:.2f}ms"
            )
        else:
            # Quadratic model: f(l) = al^2 + bl + c
            self.length_predictor.fit(seq_lens, latencies)
            self.length_predictor.set_target_latency(self.chunked_prefill_size)
            self.length_predictor.is_ready = True
            logger.info(
                f"[PP Dynamic Chunk] [PP{self.pp_rank}] Predictor ready (quadratic). "
                f"Target latency: {self.length_predictor.target_latency:.2f}ms"
            )

    def predict_next_chunk_size(self: "Scheduler", history_len: int) -> Optional[int]:
        """
        Predict next chunk size dynamically based on current history length.
        
        Args:
            history_len: Current sequence length
            
        Returns:
            Predicted chunk size, or None to use default chunked_prefill_size
        """
        if not self.enable_dynamic_chunking or self.length_predictor is None or not self.length_predictor.is_ready:
            return None

        max_chunk_size = getattr(self, "max_prefill_tokens", None)
        predicted_size = self.length_predictor.predict_next_chunk_size(
            history_len=history_len,
            page_size=self.page_size,
            context_len=self.model_config.context_len,
            max_chunk_size=max_chunk_size,
        )

        if predicted_size is not None:
            logger.debug(
                f"[PP Dynamic Chunk] [PP{self.pp_rank}] Predicted chunk size: "
                f"{predicted_size} (history_len={history_len})"
            )

        return predicted_size
