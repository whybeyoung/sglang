from __future__ import annotations

import dataclasses
import logging
from collections import deque
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class AuthoritativeTreeOp:
    """A deterministic tree-state mutation replayed across PP stages.

    The first integration only wires up the transport and sequencing layer so
    HiCache can start converging on a single committed view. Concrete tree
    mutations are applied by the cache implementation via `apply_fn`.
    """

    op_type: str
    payload: dict[str, Any]
    op_seq: Optional[int] = None


@dataclasses.dataclass
class AuthoritativePrefetchReadySummary:
    req_id: str
    prefix_len: int
    host_hit_length: int
    storage_hit_length: int
    input_len: Optional[int] = None
    last_host_node_ref: Optional[dict[str, Any]] = None


class AuthoritativeTreeCoordinator:
    """Replicate committed HiCache tree mutations across PP stages.

    PP rank 0 is treated as the authoritative sequencer. Every stage enters
    `sync()` at the same scheduler barrier and receives the same ordered op log.
    """

    def __init__(self, *, enabled: bool, src_rank: int = 0):
        self.enabled = enabled
        self.src_rank = src_rank
        self._pending_local: deque[AuthoritativeTreeOp] = deque()
        self._committed_seq = 0

    @property
    def committed_seq(self) -> int:
        return self._committed_seq

    def queue_local(self, op_type: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self._pending_local.append(
            AuthoritativeTreeOp(op_type=op_type, payload=dict(payload))
        )

    def sync(self, apply_fn, transform_fn=None) -> int:
        """Broadcast authoritative ops and apply them in order on every PP rank."""
        if not self.enabled:
            return self._committed_seq

        try:
            from sglang.srt.distributed.parallel_state import get_pp_group

            pp_group = get_pp_group()
        except Exception:
            pp_group = None
        if pp_group is None:
            self._drain_local_without_broadcast(apply_fn)
            return self._committed_seq
        if pp_group.world_size <= 1:
            self._drain_local_without_broadcast(apply_fn)
            return self._committed_seq

        ops: list[AuthoritativeTreeOp] | None = None
        if pp_group.rank_in_group == self.src_rank:
            staged_ops = list(self._pending_local)
            self._pending_local.clear()
            if transform_fn is not None:
                staged_ops = list(transform_fn(staged_ops))
            ops = []
            for op in staged_ops:
                self._committed_seq += 1
                op.op_seq = self._committed_seq
                ops.append(op)
        else:
            self._pending_local.clear()

        data = [ops]
        pp_group.broadcast_object_list(data, src=self.src_rank)
        recv_ops = data[0] or []
        for op in recv_ops:
            if op.op_seq is None:
                raise ValueError(f"Authoritative op missing sequence: {op}")
            self._committed_seq = max(self._committed_seq, op.op_seq)
            apply_fn(op)
        return self._committed_seq

    def _drain_local_without_broadcast(self, apply_fn) -> None:
        while self._pending_local:
            op = self._pending_local.popleft()
            self._committed_seq += 1
            op.op_seq = self._committed_seq
            apply_fn(op)

    def pending_ops(self) -> Iterable[AuthoritativeTreeOp]:
        return tuple(self._pending_local)
