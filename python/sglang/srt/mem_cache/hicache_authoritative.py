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
        """Apply queued ops in order on the local PP rank.

        A previous version tried to broadcast these ops from PP0 inside
        HiCache's scheduler-side event loop. Under `pp-size > 1` that created a
        deadlock: PP0 could enter the extra collective while PP1 was blocked in
        the normal pipeline recv path. Keep the authoritative replay logic
        local-only until it is piggybacked on an existing PP synchronization
        point.
        """
        if not self.enabled:
            return self._committed_seq
        staged_ops = list(self._pending_local)
        self._pending_local.clear()
        if transform_fn is not None:
            staged_ops = list(transform_fn(staged_ops))
        for op in staged_ops:
            self._committed_seq += 1
            op.op_seq = self._committed_seq
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
