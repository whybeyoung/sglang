from __future__ import annotations

import atexit
import dataclasses
import heapq
import json
import logging
import os
import threading
import time
from queue import Empty
from typing import TYPE_CHECKING, Dict, List, Optional

import torch

from sglang.srt.managers.cache_controller import HiCacheController, PrefetchOperation
from sglang.srt.mem_cache.hicache_authoritative import (
    AuthoritativePrefetchReadySummary,
    AuthoritativeTreeCoordinator,
    AuthoritativeTreeOp,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    EvictResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    NSATokenToKVPool,
)
from sglang.srt.mem_cache.memory_pool_host import (
    HostPoolGroup,
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
    NSAIndexerHostPool,
    NSATokenToKVPoolHost,
    PoolEntry,
)
from sglang.srt.mem_cache.radix_cache import (
    RadixCache,
    RadixKey,
    TreeNode,
    compute_node_hash_values,
    split_node_hash_value,
)
from sglang.srt.observability.metrics_collector import StorageMetricsCollector
from sglang.srt.utils import bind_to_closest_numa_node_cuda

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class LatchedPrefetchReadyResult:
    match_result: MatchResult
    storage_hit_length: int
    input_len: Optional[int] = None


class HiRadixCache(RadixCache):

    def __init__(self, params: CacheInitParams, server_args: ServerArgs):
        self._enable_metrics_flag = params.enable_metrics
        if server_args.hicache_io_backend == "direct":
            # FIXME: move this logic into server_args parsing
            if server_args.hicache_mem_layout == "page_first":
                server_args.hicache_mem_layout = "page_first_direct"
                logger.warning(
                    "Page first layout is not supported with direct IO backend, switching to page first direct layout"
                )

        if not server_args.disable_hicache_numa_detect:
            bind_to_closest_numa_node_cuda()

        self.page_size = params.page_size
        self.kv_cache = params.token_to_kv_pool_allocator.get_kvcache()
        self.use_nsa_pool_controller = isinstance(self.kv_cache, NSATokenToKVPool)

        if (
            self.use_nsa_pool_controller
            and server_args.hicache_storage_backend == "mooncake"
            and server_args.hicache_mem_layout
            not in ["page_first", "page_first_direct"]
        ):
            server_args.hicache_mem_layout = (
                "page_first_direct"
                if server_args.hicache_io_backend == "direct"
                else "page_first"
            )
            logger.warning(
                "Mooncake storage backend with NSA requires page_first layout, "
                f"switching to {server_args.hicache_mem_layout}."
            )

        if isinstance(self.kv_cache, MHATokenToKVPool):
            self.token_to_kv_pool_host = MHATokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
            )
        elif isinstance(self.kv_cache, NSATokenToKVPool):
            self.token_to_kv_pool_host = NSATokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
            )
        elif isinstance(self.kv_cache, MLATokenToKVPool):
            self.token_to_kv_pool_host = MLATokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
                allocator_type=server_args.hicache_storage_backend,
            )
        else:
            raise ValueError(f"HiRadixCache only supports MHA and MLA yet")

        self.tp_group = params.tp_cache_group
        self.attn_cp_group = params.attn_cp_cache_group
        self.attn_tp_group = params.attn_tp_cache_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)
        self.pp_rank = params.pp_rank
        self.pp_size = params.pp_size
        self.attn_cp_rank = params.attn_cp_rank
        self.attn_cp_size = params.attn_cp_size
        self.enable_storage = server_args.hicache_storage_backend is not None
        self.enable_storage_metrics = self.enable_storage and params.enable_metrics
        self.extra_metric_labels = server_args.extra_metric_labels
        # The authoritative PP replay path is still experimental. Keep it opt-in so
        # default HiCache service startup does not introduce extra PP collectives in
        # the scheduler event loop.
        self.authoritative_tree = AuthoritativeTreeCoordinator(
            enabled=(
                self.pp_size > 1
                and os.getenv("SGLANG_ENABLE_HICACHE_AUTHORITATIVE_PP", "0") == "1"
            )
        )
        # Until PP authoritative ready/contract replay is fully wired into the
        # main scheduler path, default PP HiCache reads to a conservative
        # device-only view. This avoids stage-local host-hit divergence such as
        # PP0 host_hit>0 while PP1 host_hit=0, which can later surface as batch
        # shape mismatches in model forward.
        self.pp_device_only_match_fallback = (
            self.pp_size > 1
            and not self.authoritative_tree.enabled
            and os.getenv("SGLANG_DISABLE_PP_HICACHE_DEVICE_ONLY_FALLBACK", "0")
            != "1"
        )

        (
            extra_config,
            prefetch_threshold,
            prefetch_timeout_base,
            prefetch_timeout_per_ki_token,
            hicache_storage_pass_prefix_keys,
        ) = self._parse_storage_backend_extra_config(
            server_args.hicache_storage_backend_extra_config
        )
        # TODO: support more timeout check functions
        self.is_prefetch_timeout = self._prefetch_timeout_check_linear_func
        self.prefetch_stop_policy = server_args.hicache_storage_prefetch_policy

        self.load_cache_event = threading.Event()
        if self.use_nsa_pool_controller:
            if server_args.hicache_storage_backend not in (None, "file", "mooncake"):
                raise ValueError(
                    "NSA pool-based HiCache only supports file and mooncake storage backends."
                )
            self.nsa_indexer_host_pool = NSAIndexerHostPool(self.token_to_kv_pool_host)
            self.host_pool_group = HostPoolGroup(
                [
                    PoolEntry(
                        name=PoolName.KV,
                        host_pool=self.token_to_kv_pool_host,
                        device_pool=self.kv_cache,
                        layer_mapper=lambda layer_id: layer_id,
                        is_primary_index_anchor=True,
                    ),
                    PoolEntry(
                        name=PoolName.NSA,
                        host_pool=self.nsa_indexer_host_pool,
                        device_pool=self.kv_cache,
                        layer_mapper=lambda layer_id: layer_id,
                    ),
                ]
            )
            self.cache_controller = HybridCacheController(
                params.token_to_kv_pool_allocator,
                self.host_pool_group,
                self.page_size,
                self.tp_group,
                load_cache_event=self.load_cache_event,
                write_policy=server_args.hicache_write_policy,
                io_backend=server_args.hicache_io_backend,
                storage_backend=server_args.hicache_storage_backend,
                prefetch_threshold=prefetch_threshold,
                model_name=server_args.served_model_name,
                storage_backend_extra_config=extra_config,
                pp_rank=self.pp_rank,
                pp_size=self.pp_size,
                attn_cp_rank=self.attn_cp_rank,
                attn_cp_size=self.attn_cp_size,
                enable_storage_metrics=self.enable_storage_metrics,
            )
        else:
            self.cache_controller = HiCacheController(
                token_to_kv_pool_allocator=params.token_to_kv_pool_allocator,
                mem_pool_host=self.token_to_kv_pool_host,
                page_size=self.page_size,
                tp_group=self.tp_group,
                load_cache_event=self.load_cache_event,
                attn_cp_group=self.attn_cp_group,
                attn_tp_group=self.attn_tp_group,
                write_policy=server_args.hicache_write_policy,
                io_backend=server_args.hicache_io_backend,
                storage_backend=server_args.hicache_storage_backend,
                prefetch_threshold=prefetch_threshold,
                model_name=server_args.served_model_name,
                storage_backend_extra_config=extra_config,
                pp_rank=self.pp_rank,
                pp_size=self.pp_size,
                attn_cp_rank=self.attn_cp_rank,
                attn_cp_size=self.attn_cp_size,
                enable_storage_metrics=self.enable_storage_metrics,
            )
        self._apply_storage_runtime_config(
            storage_backend=server_args.hicache_storage_backend,
            prefetch_threshold=prefetch_threshold,
            prefetch_timeout_base=prefetch_timeout_base,
            prefetch_timeout_per_ki_token=prefetch_timeout_per_ki_token,
            hicache_storage_pass_prefix_keys=hicache_storage_pass_prefix_keys,
            enable_storage=self.enable_storage,
            enable_storage_metrics=self.enable_storage_metrics,
            extra_metric_labels=self.extra_metric_labels,
        )

        # record the nodes with ongoing write through
        self.ongoing_write_through = {}
        # record the node segments with ongoing load back
        self.ongoing_load_back = {}
        # record the ongoing prefetch requests
        self.ongoing_prefetch = {}
        self.ongoing_backup = {}
        # track per-request tokens loaded from storage (L3 hits)
        # key: request_id, value: number of tokens actually loaded from storage
        self.prefetch_loaded_tokens_by_reqid: dict[str, int] = {}
        # track finalized prefetch match results for stable scheduler consumption
        self.prefetch_ready_results_by_reqid: dict[
            str, LatchedPrefetchReadyResult
        ] = {}
        self.authoritative_prefetch_ready_by_reqid: dict[
            str, AuthoritativePrefetchReadySummary
        ] = {}
        self.authoritative_prefetch_loaded_tokens_by_reqid: dict[str, int] = {}
        self.authoritative_host_insert_rebuild_by_reqid: dict[str, dict[str, object]] = {}
        self.authoritative_host_insert_skeleton_by_reqid: dict[
            str, list[dict[str, object]]
        ] = {}
        self.authoritative_host_insert_missing_by_reqid: dict[
            str, list[dict[str, object]]
        ] = {}
        self.authoritative_pending_backup_refs: dict[str, dict[str, object]] = {}
        self.authoritative_pending_backup_node_ids: set[int] = set()
        self.authoritative_pending_backup_reasons: dict[str, str] = {}
        self.authoritative_backuped_node_ids: set[int] = set()
        self.authoritative_host_visible_node_ids: set[int] = set()
        self.authoritative_resolution_stats: dict[str, int] = {}
        self._last_authoritative_resolution_log_ts = 0.0
        self._authoritative_node_by_id: dict[int, TreeNode] = {}
        self._authoritative_node_by_last_hash: dict[str, TreeNode] = {}
        self._authoritative_last_hash_by_node_id: dict[int, str] = {}
        # track requests whose prefetch was skipped (alloc failure, threshold, rate limit)
        self.prefetch_skipped_rids: set[str] = set()
        # todo: dynamically adjust the threshold
        self.write_through_threshold = (
            1 if server_args.hicache_write_policy == "write_through" else 2
        )
        self.load_back_threshold = 10

        # Detach storage backend automatically on process shutdown
        atexit.register(self.shutdown)

        self.evictable_host_leaves = set()

        super().__init__(params=params)

    def _all_reduce_attn_groups(self, tensor: torch.Tensor, op):
        reduced = False
        for group in (self.attn_cp_group, self.attn_tp_group):
            if group is not None and torch.distributed.get_world_size(group=group) > 1:
                torch.distributed.all_reduce(tensor, op=op, group=group)
                reduced = True
        if not reduced and self.tp_world_size > 1:
            torch.distributed.all_reduce(tensor, op=op, group=self.tp_group)

    def _barrier_attn_groups(self):
        waited = False
        for group in (self.attn_cp_group, self.attn_tp_group):
            if group is not None and torch.distributed.get_world_size(group=group) > 1:
                torch.distributed.barrier(group=group)
                waited = True
        if not waited and self.tp_world_size > 1:
            torch.distributed.barrier(group=self.tp_group)

    def queue_authoritative_tree_op(self, op_type: str, **payload) -> None:
        if not getattr(self.authoritative_tree, "enabled", False):
            return
        self.authoritative_tree.queue_local(op_type, payload)

    def apply_authoritative_tree_op(self, op: AuthoritativeTreeOp) -> None:
        """Apply a committed tree op.

        The initial integration wires up sequencing and barrier placement first.
        Concrete mutations are gradually migrated from local async paths onto this
        replay layer; unknown ops are ignored so partial rollout stays compatible.
        """
        if op.op_type == "BARRIER":
            return
        if op.op_type == "PREFETCH_READY_SUMMARY":
            req_id = op.payload["req_id"]
            if not self._is_authoritative_ready_stable(req_id):
                self._record_authoritative_resolution("prefetch_ready.defer_unstable")
                return
            self.authoritative_prefetch_ready_by_reqid[req_id] = (
                AuthoritativePrefetchReadySummary(
                    req_id=req_id,
                    prefix_len=op.payload["prefix_len"],
                    host_hit_length=op.payload["host_hit_length"],
                    storage_hit_length=op.payload["storage_hit_length"],
                    input_len=op.payload.get("input_len"),
                    last_host_node_ref=op.payload.get("last_host_node_ref"),
                )
            )
            return
        if op.op_type == "HOST_BACKUP_COMMIT":
            node = self._resolve_authoritative_node_ref(
                node_id=op.payload.get("node_id"),
                last_hash=op.payload.get("last_hash"),
            )
            if node is not None:
                self._record_authoritative_resolution("host_backup_commit.node_ref")
            if node is None:
                node = self._recover_backup_commit_node_from_payload(op.payload)
                if node is not None:
                    self._record_authoritative_resolution(
                        "host_backup_commit.payload_recover"
                    )
            if node is not None:
                backup_ref = self._make_authoritative_node_ref(node)
                backup_key = self._make_pending_backup_key(
                    node_ref=backup_ref, node=node
                )
                self.authoritative_pending_backup_refs[backup_key] = backup_ref
                self.authoritative_pending_backup_node_ids.add(node.id)
                self.authoritative_pending_backup_reasons[backup_key] = "await_stable"
                self._record_authoritative_resolution("host_backup_commit.defer_visible")
            else:
                repaired = self._repair_backup_commit_subtree_from_payload(op.payload)
                if repaired:
                    self._record_authoritative_resolution(
                        "host_backup_commit.payload_repair"
                    )
                    if os.getenv("SGLANG_DEBUG_HICACHE_MATCH_CHAIN", "0") == "1":
                        logger.warning(
                            "[HiCacheAuthoritative] repaired HOST_BACKUP_COMMIT subtree: "
                            "node_id=%s last_hash=%s pp=%s cp=%s tp=%s",
                            op.payload.get("node_id"),
                            op.payload.get("last_hash"),
                            self.pp_rank,
                            self.attn_cp_rank,
                            getattr(self.cache_controller, "tp_rank", None),
                        )
                else:
                    self._record_authoritative_resolution("host_backup_commit.unresolved")
                    if os.getenv("SGLANG_DEBUG_HICACHE_MATCH_CHAIN", "0") == "1":
                        logger.warning(
                            "[HiCacheAuthoritative] unresolved HOST_BACKUP_COMMIT node: "
                            "node_id=%s last_hash=%s pp=%s cp=%s tp=%s",
                            op.payload.get("node_id"),
                            op.payload.get("last_hash"),
                            self.pp_rank,
                            self.attn_cp_rank,
                            getattr(self.cache_controller, "tp_rank", None),
                        )
            return
        if op.op_type == "HOST_INSERT_FROM_STORAGE":
            req_id = op.payload.get("req_id")
            if req_id is not None:
                self.authoritative_prefetch_loaded_tokens_by_reqid[req_id] = int(
                    op.payload.get("loaded_from_storage", 0)
                )
                self._clear_host_insert_rebuild_blueprint(req_id)
            resolved_nodes: list[TreeNode] = []
            node_refs = op.payload.get("node_refs")
            if node_refs is None:
                node_refs = [
                    {"node_id": node_id, "last_hash": None}
                    for node_id in op.payload.get("node_ids", [])
                ]
            for node_ref in node_refs:
                node = self._resolve_authoritative_node_ref(node_ref=node_ref)
                if node is not None:
                    self._record_authoritative_resolution("host_insert.node_ref")
                    resolved_nodes.append(node)
                elif os.getenv("SGLANG_DEBUG_HICACHE_MATCH_CHAIN", "0") == "1":
                    logger.warning(
                        "[HiCacheAuthoritative] unresolved HOST_INSERT node_ref=%s "
                        "rid=%s pp=%s cp=%s tp=%s",
                        node_ref,
                        req_id,
                        self.pp_rank,
                        self.attn_cp_rank,
                        getattr(self.cache_controller, "tp_rank", None),
                    )
            recovered_nodes: list[TreeNode] = []
            if resolved_nodes and self._validate_host_insert_nodes_from_payload(
                resolved_nodes, op.payload
            ):
                for node in resolved_nodes:
                    self.authoritative_host_visible_node_ids.add(node.id)
            else:
                if resolved_nodes:
                    self._record_authoritative_resolution("host_insert.partial_mismatch")
                recovered_nodes = self._recover_host_insert_visible_nodes_from_payload(
                    op.payload
                )
                if recovered_nodes:
                    self._record_authoritative_resolution(
                        "host_insert.payload_recover"
                    )
                    if req_id is not None:
                        self._clear_host_insert_rebuild_blueprint(req_id)
                else:
                    repaired = self._repair_host_insert_subtree_from_payload(
                        op.payload
                    )
                    if repaired:
                        self._record_authoritative_resolution(
                            "host_insert.payload_repair"
                        )
                        if os.getenv("SGLANG_DEBUG_HICACHE_MATCH_CHAIN", "0") == "1":
                            logger.warning(
                                "[HiCacheAuthoritative] repaired HOST_INSERT subtree: "
                                "rid=%s pp=%s cp=%s tp=%s",
                                req_id,
                                self.pp_rank,
                                self.attn_cp_rank,
                                getattr(self.cache_controller, "tp_rank", None),
                            )
                        self._stage_host_insert_rebuild_blueprint(op.payload)
                    else:
                        self._record_authoritative_resolution(
                            "host_insert.unresolved"
                        )
                for node in recovered_nodes:
                    self.authoritative_host_visible_node_ids.add(node.id)
            return
        if op.op_type == "HOST_EVICT":
            node_refs = op.payload.get("node_refs")
            if node_refs is None:
                node_refs = [
                    {"node_id": node_id, "last_hash": None}
                    for node_id in op.payload.get("node_ids", [])
                ]
            for node_ref in node_refs:
                self._apply_authoritative_host_evict(node_ref=node_ref)
            return
        if op.op_type == "DEVICE_EVICT":
            node_refs = op.payload.get("node_refs")
            if node_refs is None:
                node_refs = [
                    {"node_id": node_id, "last_hash": None}
                    for node_id in op.payload.get("node_ids", [])
                ]
            for node_ref in node_refs:
                self._apply_authoritative_device_evict(node_ref=node_ref)
            return
        logger.debug(
            "Apply authoritative HiCache op seq=%s type=%s payload=%s",
            op.op_seq,
            op.op_type,
            op.payload,
        )

    def sync_authoritative_state(self):
        if not getattr(self.authoritative_tree, "enabled", False):
            return None
        return self.authoritative_tree.sync(
            self.apply_authoritative_tree_op,
            transform_fn=self._materialize_authoritative_tree_ops,
        )

    def _materialize_authoritative_tree_ops(
        self, pending_ops: list[AuthoritativeTreeOp]
    ) -> list[AuthoritativeTreeOp]:
        materialized: list[AuthoritativeTreeOp] = []
        for op in pending_ops:
            if op.op_type == "HOST_EVICT_REQUEST":
                node_ids = self._select_authoritative_host_evict_node_ids(
                    op.payload["num_tokens"]
                )
                materialized.append(
                    AuthoritativeTreeOp(
                        op_type="HOST_EVICT",
                        payload={
                            "node_refs": [
                                self._make_authoritative_node_ref(
                                    self._find_node_by_id(node_id)
                                )
                                for node_id in node_ids
                            ]
                        },
                    )
                )
                continue
            if op.op_type == "DEVICE_EVICT_REQUEST":
                node_ids = self._select_authoritative_device_evict_node_ids(
                    op.payload["num_tokens"]
                )
                materialized.append(
                    AuthoritativeTreeOp(
                        op_type="DEVICE_EVICT",
                        payload={
                            "node_refs": [
                                self._make_authoritative_node_ref(
                                    self._find_node_by_id(node_id)
                                )
                                for node_id in node_ids
                            ]
                        },
                    )
                )
                continue
            materialized.append(op)
        return materialized

    def _iter_nodes(self):
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(list(node.children.values())))

    def _register_authoritative_node(self, node: Optional[TreeNode]) -> None:
        if node is None:
            return
        node_id = getattr(node, "id", None)
        if node_id is None:
            return
        self._authoritative_node_by_id[node_id] = node
        last_hash = node.get_last_hash_value()
        if last_hash:
            self._authoritative_node_by_last_hash[last_hash] = node
            self._authoritative_last_hash_by_node_id[node_id] = last_hash

    def _refresh_authoritative_node_index(self, node: Optional[TreeNode]) -> None:
        if node is None:
            return
        node_id = getattr(node, "id", None)
        if node_id is None:
            return
        self._authoritative_node_by_id[node_id] = node
        old_last_hash = self._authoritative_last_hash_by_node_id.pop(node_id, None)
        if old_last_hash is not None:
            indexed = self._authoritative_node_by_last_hash.get(old_last_hash)
            if indexed is node:
                self._authoritative_node_by_last_hash.pop(old_last_hash, None)
        new_last_hash = node.get_last_hash_value()
        if new_last_hash:
            self._authoritative_node_by_last_hash[new_last_hash] = node
            self._authoritative_last_hash_by_node_id[node_id] = new_last_hash

    def _unregister_authoritative_node(self, node: Optional[TreeNode]) -> None:
        if node is None:
            return
        node_id = getattr(node, "id", None)
        if node_id is None:
            return
        indexed = self._authoritative_node_by_id.get(node_id)
        if indexed is node:
            self._authoritative_node_by_id.pop(node_id, None)
        old_last_hash = self._authoritative_last_hash_by_node_id.pop(node_id, None)
        if old_last_hash is not None:
            indexed = self._authoritative_node_by_last_hash.get(old_last_hash)
            if indexed is node:
                self._authoritative_node_by_last_hash.pop(old_last_hash, None)

    def _find_node_by_id(self, node_id: int) -> Optional[TreeNode]:
        node = self._authoritative_node_by_id.get(node_id)
        if node is not None:
            return node
        for node in self._iter_nodes():
            if getattr(node, "id", None) == node_id:
                self._register_authoritative_node(node)
                return node
        return None

    def _find_node_by_last_hash(self, last_hash: Optional[str]) -> Optional[TreeNode]:
        if not last_hash:
            return None
        node = self._authoritative_node_by_last_hash.get(last_hash)
        if node is not None:
            return node
        for node in self._iter_nodes():
            if node.get_last_hash_value() == last_hash:
                self._register_authoritative_node(node)
                return node
        return None

    def _make_authoritative_node_ref(
        self, node: Optional[TreeNode]
    ) -> dict[str, Optional[object]]:
        if node is None:
            return {"node_id": None, "last_hash": None}
        return {
            "node_id": getattr(node, "id", None),
            "last_hash": node.get_last_hash_value(),
        }

    def _make_pending_backup_key(
        self,
        node_ref: Optional[dict[str, object]] = None,
        *,
        node: Optional[TreeNode] = None,
    ) -> str:
        if node_ref is None and node is not None:
            node_ref = self._make_authoritative_node_ref(node)
        node_ref = node_ref or {}
        last_hash = node_ref.get("last_hash")
        node_id = node_ref.get("node_id")
        if last_hash:
            return f"h:{last_hash}"
        return f"i:{node_id}"

    def _resolve_authoritative_node_ref(
        self,
        node_ref: Optional[dict[str, object]] = None,
        *,
        node_id: Optional[int] = None,
        last_hash: Optional[str] = None,
    ) -> Optional[TreeNode]:
        if node_ref is not None:
            node_id = node_ref.get("node_id")  # type: ignore[assignment]
            last_hash = node_ref.get("last_hash")  # type: ignore[assignment]
        node = self._find_node_by_id(node_id) if node_id is not None else None
        if node is not None:
            return node
        return self._find_node_by_last_hash(last_hash)

    def _find_exact_host_path_nodes(
        self, anchor_node: Optional[TreeNode], token_ids: List[int]
    ) -> list[TreeNode]:
        if anchor_node is None:
            return []
        if len(token_ids) == 0:
            return []

        key = RadixKey(token_ids=list(token_ids), extra_key=anchor_node.key.extra_key)
        nodes: list[TreeNode] = []
        node = anchor_node
        child_key = self.get_child_key_fn(key)
        while len(key) > 0 and child_key in node.children:
            child = node.children[child_key]
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                break
            nodes.append(child)
            node = child
            key = key[prefix_len:]
            if len(key) > 0:
                child_key = self.get_child_key_fn(key)
        return nodes

    def _recover_host_insert_visible_nodes_from_payload(
        self, payload: dict[str, object]
    ) -> list[TreeNode]:
        anchor_node = self._resolve_authoritative_node_ref(
            node_ref=payload.get("anchor_node_ref")
        )
        if anchor_node is None:
            return []

        fetched_token_ids = list(payload.get("fetched_token_ids") or [])
        matched_length = int(payload.get("matched_length", 0))
        committed_tokens = int(payload.get("committed_tokens", 0))
        if committed_tokens <= matched_length:
            return []

        suffix_tokens = fetched_token_ids[matched_length:committed_tokens]
        path_nodes = self._find_exact_host_path_nodes(anchor_node, suffix_tokens)
        candidate_nodes = [
            node
            for node in path_nodes
            if node.evicted and node.backuped and len(node.host_value) > 0
        ]
        expected_hashes = list(payload.get("fetched_hash_value") or [])
        if not expected_hashes:
            return candidate_nodes

        matched_pages = matched_length // self.page_size
        expected_suffix_hashes = expected_hashes[matched_pages:]
        candidate_hashes: list[str] = []
        for node in candidate_nodes:
            if node.hash_value:
                candidate_hashes.extend(node.hash_value)
        if candidate_hashes[: len(expected_suffix_hashes)] != expected_suffix_hashes:
            self._record_authoritative_resolution("host_insert.hash_mismatch")
            if os.getenv("SGLANG_DEBUG_HICACHE_MATCH_CHAIN", "0") == "1":
                logger.warning(
                    "[HiCacheAuthoritative] HOST_INSERT hash mismatch: "
                    "expected=%s actual=%s pp=%s cp=%s tp=%s",
                    expected_suffix_hashes,
                    candidate_hashes,
                    self.pp_rank,
                    self.attn_cp_rank,
                    getattr(self.cache_controller, "tp_rank", None),
                )
            return []
        return candidate_nodes

    def _validate_host_insert_nodes_from_payload(
        self, nodes: list[TreeNode], payload: dict[str, object]
    ) -> bool:
        matched_length = int(payload.get("matched_length", 0))
        committed_tokens = int(payload.get("committed_tokens", 0))
        expected_tokens = committed_tokens - matched_length
        if expected_tokens <= 0:
            return True

        candidate_nodes = [
            node
            for node in nodes
            if node.evicted and node.backuped and len(node.host_value) > 0
        ]
        if sum(len(node.host_value) for node in candidate_nodes) < expected_tokens:
            return False

        expected_hashes = list(payload.get("fetched_hash_value") or [])
        if not expected_hashes:
            return True

        matched_pages = matched_length // self.page_size
        expected_suffix_hashes = expected_hashes[matched_pages:]
        candidate_hashes: list[str] = []
        for node in candidate_nodes:
            if node.hash_value:
                candidate_hashes.extend(node.hash_value)
        return candidate_hashes[: len(expected_suffix_hashes)] == expected_suffix_hashes

    def _can_repair_authoritative_host_subtree(self, node: Optional[TreeNode]) -> bool:
        if node is None or node == self.root_node:
            return False

        stack = [node]
        while stack:
            current = stack.pop()
            if (
                not current.evicted
                or not current.backuped
                or getattr(current, "lock_ref", 0) > 0
                or current.host_ref_counter > 0
            ):
                return False
            stack.extend(current.children.values())
        return True

    def _prune_authoritative_host_subtree(self, node: Optional[TreeNode]) -> bool:
        if not self._can_repair_authoritative_host_subtree(node):
            return False

        assert node is not None
        stack = [node]
        while stack:
            current = stack.pop()
            stack.extend(current.children.values())
            if current.backuped and len(current.host_value) > 0:
                self.cache_controller.evict_host(current.host_value)
                current.host_value = None
            self._discard_authoritative_visibility(current)
            if current in self.evictable_host_leaves:
                self.evictable_host_leaves.remove(current)

        parent = node.parent
        if parent is None:
            return False
        key = self.get_child_key_fn(node.key)
        popped = parent.children.pop(key, None)
        if popped is not node:
            return False
        self._update_host_leaf_status(parent)
        self._update_leaf_status(parent)
        return True

    def _repair_host_insert_subtree_from_payload(
        self, payload: dict[str, object]
    ) -> bool:
        anchor_node = self._resolve_authoritative_node_ref(
            node_ref=payload.get("anchor_node_ref")
        )
        if anchor_node is None:
            return False

        fetched_token_ids = list(payload.get("fetched_token_ids") or [])
        matched_length = int(payload.get("matched_length", 0))
        committed_tokens = int(payload.get("committed_tokens", 0))
        if committed_tokens <= matched_length:
            return False

        expected_hashes = list(payload.get("fetched_hash_value") or [])
        matched_pages = matched_length // self.page_size
        remaining_hashes = expected_hashes[matched_pages:]
        remaining_tokens = fetched_token_ids[matched_length:committed_tokens]

        parent = anchor_node
        key = RadixKey(
            token_ids=list(remaining_tokens), extra_key=anchor_node.key.extra_key
        )
        while len(key) > 0:
            child_key = self.get_child_key_fn(key)
            child = parent.children.get(child_key)
            if child is None:
                return False

            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                return self._prune_authoritative_host_subtree(child)

            consumed_pages = len(child.key) // self.page_size
            expected_child_hashes = remaining_hashes[:consumed_pages]
            actual_child_hashes = list(child.hash_value or [])[:consumed_pages]
            if expected_child_hashes and actual_child_hashes != expected_child_hashes:
                return self._prune_authoritative_host_subtree(child)

            key = key[prefix_len:]
            remaining_hashes = remaining_hashes[consumed_pages:]
            parent = child

        return False

    def _build_host_backup_commit_payload(self, node: TreeNode) -> dict[str, object]:
        return {
            "node_id": getattr(node, "id", None),
            "last_hash": node.get_last_hash_value(),
            "node_ref": self._make_authoritative_node_ref(node),
            "parent_node_ref": self._make_authoritative_node_ref(node.parent),
            "node_key_tokens": list(getattr(node.key, "token_ids", [])),
            "node_hash_value": list(node.hash_value or []),
            "key_len": len(getattr(node, "key", [])),
        }

    def _build_host_insert_rebuild_blueprint(
        self, payload: dict[str, object]
    ) -> Optional[dict[str, object]]:
        req_id = payload.get("req_id")
        anchor_node_ref = payload.get("anchor_node_ref")
        fetched_token_ids = list(payload.get("fetched_token_ids") or [])
        fetched_hash_value = list(payload.get("fetched_hash_value") or [])
        matched_length = int(payload.get("matched_length", 0))
        committed_tokens = int(payload.get("committed_tokens", 0))
        if req_id is None or anchor_node_ref is None or committed_tokens <= matched_length:
            return None
        segment_plan = self._build_host_insert_segment_plan(
            fetched_token_ids=fetched_token_ids,
            fetched_hash_value=fetched_hash_value,
            matched_length=matched_length,
            committed_tokens=committed_tokens,
        )
        return {
            "req_id": req_id,
            "anchor_node_ref": anchor_node_ref,
            "fetched_token_ids": fetched_token_ids,
            "fetched_hash_value": fetched_hash_value,
            "matched_length": matched_length,
            "committed_tokens": committed_tokens,
            "segment_plan": segment_plan,
        }

    def _build_host_insert_segment_plan(
        self,
        *,
        fetched_token_ids: list[int],
        fetched_hash_value: list[str],
        matched_length: int,
        committed_tokens: int,
    ) -> list[dict[str, object]]:
        suffix_tokens = fetched_token_ids[matched_length:committed_tokens]
        matched_pages = matched_length // self.page_size
        suffix_hashes = fetched_hash_value[matched_pages:]
        segments: list[dict[str, object]] = []
        if not suffix_tokens:
            return segments
        for page_idx, start in enumerate(range(0, len(suffix_tokens), self.page_size)):
            token_slice = suffix_tokens[start : start + self.page_size]
            hash_slice = suffix_hashes[page_idx : page_idx + 1]
            segments.append(
                {
                    "token_ids": list(token_slice),
                    "hash_value": list(hash_slice),
                    "page_index": matched_pages + page_idx,
                }
            )
        return segments

    def _stage_host_insert_rebuild_blueprint(
        self, payload: dict[str, object]
    ) -> bool:
        blueprint = self._build_host_insert_rebuild_blueprint(payload)
        if blueprint is None:
            return False
        req_id = blueprint["req_id"]
        self.authoritative_host_insert_rebuild_by_reqid[req_id] = blueprint
        self.authoritative_host_insert_skeleton_by_reqid[req_id] = list(
            blueprint.get("segment_plan", [])
        )
        self.authoritative_host_insert_missing_by_reqid[req_id] = []
        self._record_authoritative_resolution("host_insert.blueprint_stage")
        return True

    def _clear_host_insert_rebuild_blueprint(self, req_id: Optional[str]) -> None:
        if req_id is None:
            return
        self.authoritative_host_insert_rebuild_by_reqid.pop(req_id, None)
        self.authoritative_host_insert_skeleton_by_reqid.pop(req_id, None)
        self.authoritative_host_insert_missing_by_reqid.pop(req_id, None)

    def _try_apply_host_insert_rebuild_blueprint(
        self, req_id: str, blueprint: dict[str, object]
    ) -> bool:
        focus_blueprint = self._build_host_insert_focus_blueprint(req_id, blueprint)
        if focus_blueprint is not blueprint:
            self._record_authoritative_resolution("host_insert.blueprint_focus")
        normalized = self._normalize_host_insert_subtree_from_payload(focus_blueprint)
        if normalized:
            self._record_authoritative_resolution("host_insert.blueprint_normalize")
        recovered_nodes = self._recover_host_insert_visible_nodes_from_payload(
            focus_blueprint
        )
        if not recovered_nodes:
            return False
        for node in recovered_nodes:
            self.authoritative_host_visible_node_ids.add(node.id)
        self._clear_host_insert_rebuild_blueprint(req_id)
        self._record_authoritative_resolution("host_insert.blueprint_apply")
        return True

    def _replay_pending_host_insert_blueprints(self) -> None:
        if not self.authoritative_host_insert_rebuild_by_reqid:
            return
        pending_items = list(self.authoritative_host_insert_rebuild_by_reqid.items())
        for req_id, blueprint in pending_items:
            self._advance_host_insert_skeleton(req_id, blueprint)
            self._try_apply_host_insert_rebuild_blueprint(req_id, blueprint)

    def _is_backup_node_stable(self, node: Optional[TreeNode]) -> bool:
        if node is None:
            return False
        return (
            node.backuped
            and len(node.host_value) > 0
            and getattr(node, "host_ref_counter", 0) == 0
        )

    def _promote_stable_backup_visibility(self) -> None:
        if not self.authoritative_pending_backup_refs:
            self.authoritative_pending_backup_node_ids = set()
            return
        current_pending_ids: set[int] = set()
        pending_items = list(self.authoritative_pending_backup_refs.items())
        for backup_key, node_ref in pending_items:
            node = self._resolve_authoritative_node_ref(node_ref=node_ref)
            if not self._is_backup_node_stable(node):
                self.authoritative_pending_backup_reasons[backup_key] = (
                    self._describe_pending_backup_reason(node)
                )
                if node is not None:
                    current_pending_ids.add(node.id)
                continue
            self.authoritative_pending_backup_refs.pop(backup_key, None)
            self.authoritative_pending_backup_reasons.pop(backup_key, None)
            if node is None:
                continue
            self.authoritative_backuped_node_ids.add(node.id)
            self.authoritative_host_visible_node_ids.add(node.id)
            self._record_authoritative_resolution("host_backup_commit.promote_visible")
        self.authoritative_pending_backup_node_ids = current_pending_ids

    def _maybe_commit_stable_prefetch_ready_summaries(self) -> None:
        if not getattr(self.authoritative_tree, "enabled", False):
            return
        if not self.prefetch_ready_results_by_reqid:
            return
        for req_id, ready_result in list(self.prefetch_ready_results_by_reqid.items()):
            if not self._is_authoritative_ready_stable(req_id):
                continue
            if req_id in self.authoritative_prefetch_ready_by_reqid:
                continue
            self.queue_authoritative_tree_op(
                "PREFETCH_READY_SUMMARY",
                req_id=req_id,
                prefix_len=len(ready_result.match_result.device_indices)
                + ready_result.match_result.host_hit_length
                + ready_result.storage_hit_length,
                host_hit_length=ready_result.match_result.host_hit_length,
                storage_hit_length=ready_result.storage_hit_length,
                input_len=ready_result.input_len,
                last_host_node_ref=self._make_authoritative_node_ref(
                    ready_result.match_result.last_host_node
                ),
            )
            self._record_authoritative_resolution("prefetch_ready.requeue_stable")

    def _build_host_insert_focus_blueprint(
        self, req_id: str, blueprint: dict[str, object]
    ) -> dict[str, object]:
        focus_state = self._build_host_insert_focus_state(req_id, blueprint)
        if focus_state is None:
            return blueprint

        first_missing, remaining_segments = focus_state
        fetched_token_ids: list[int] = []
        fetched_hash_value: list[str] = []
        for segment in remaining_segments:
            fetched_token_ids.extend(list(segment.get("token_ids") or []))
            fetched_hash_value.extend(list(segment.get("hash_value") or []))

        if not fetched_token_ids:
            return blueprint

        return {
            "req_id": req_id,
            "anchor_node_ref": first_missing.get(
                "parent_node_ref", blueprint.get("anchor_node_ref")
            ),
            "fetched_token_ids": fetched_token_ids,
            "fetched_hash_value": fetched_hash_value,
            "matched_length": 0,
            "committed_tokens": len(fetched_token_ids),
        }

    def _build_host_insert_focus_state(
        self, req_id: str, blueprint: dict[str, object]
    ) -> Optional[tuple[dict[str, object], list[dict[str, object]]]]:
        missing_segments = self.authoritative_host_insert_missing_by_reqid.get(req_id)
        remaining_segments = self.authoritative_host_insert_skeleton_by_reqid.get(req_id)
        if not missing_segments or not remaining_segments:
            return None
        sorted_missing = self._sort_host_insert_missing_segments(missing_segments)
        first_missing = sorted_missing[0]
        missing_page = first_missing.get("page_index")
        focused_segments = [
            segment
            for segment in remaining_segments
            if segment.get("page_index") is None
            or missing_page is None
            or segment.get("page_index") >= missing_page
        ]
        return first_missing, focused_segments

    def _host_insert_missing_priority(self, reason: Optional[str]) -> int:
        priorities = {
            "key_mismatch": 0,
            "hash_mismatch": 1,
            "non_host_node": 2,
            "missing_child": 3,
        }
        return priorities.get(reason or "", 99)

    def _sort_host_insert_missing_segments(
        self, missing_segments: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        return sorted(
            missing_segments,
            key=lambda segment: (
                segment.get("page_index")
                if segment.get("page_index") is not None
                else 1 << 30,
                self._host_insert_missing_priority(segment.get("reason")),
                tuple(segment.get("token_ids") or []),
            ),
        )

    def _describe_host_insert_missing_segment(
        self,
        *,
        reason: str,
        req_id: str,
        segment: dict[str, object],
        parent: Optional[TreeNode],
    ) -> dict[str, object]:
        return {
            "reason": reason,
            "req_id": req_id,
            "parent_node_ref": self._make_authoritative_node_ref(parent),
            "token_ids": list(segment.get("token_ids") or []),
            "hash_value": list(segment.get("hash_value") or []),
            "page_index": segment.get("page_index"),
        }

    def _advance_host_insert_skeleton(
        self, req_id: str, blueprint: dict[str, object]
    ) -> bool:
        focus_state = self._build_host_insert_focus_state(req_id, blueprint)
        if focus_state is not None:
            first_missing, segment_plan = focus_state
            anchor_ref = first_missing.get(
                "parent_node_ref", blueprint.get("anchor_node_ref")
            )
            self._record_authoritative_resolution("host_insert.skeleton_focus")
        else:
            segment_plan = list(
                self.authoritative_host_insert_skeleton_by_reqid.get(
                    req_id, blueprint.get("segment_plan", [])
                )
            )
            anchor_ref = blueprint.get("anchor_node_ref")

        anchor_node = self._resolve_authoritative_node_ref(node_ref=anchor_ref)
        if anchor_node is None:
            return False

        if not segment_plan:
            return False

        parent = anchor_node
        advanced = False
        remaining_segments: list[dict[str, object]] = []
        missing_segments: list[dict[str, object]] = []
        for idx, segment in enumerate(segment_plan):
            token_ids = list(segment.get("token_ids") or [])
            if not token_ids:
                continue
            key = RadixKey(token_ids=token_ids, extra_key=parent.key.extra_key)
            child_key = self.get_child_key_fn(key)
            child = parent.children.get(child_key)
            if child is None:
                missing_segments.append(
                    self._describe_host_insert_missing_segment(
                        reason="missing_child",
                        req_id=req_id,
                        segment=segment,
                        parent=parent,
                    )
                )
                remaining_segments.append(segment)
                remaining_segments.extend(segment_plan[idx + 1 :])
                break

            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len <= 0:
                missing_segments.append(
                    self._describe_host_insert_missing_segment(
                        reason="key_mismatch",
                        req_id=req_id,
                        segment=segment,
                        parent=parent,
                    )
                )
                remaining_segments.append(segment)
                remaining_segments.extend(segment_plan[idx + 1 :])
                break
            if prefix_len < len(child.key):
                if not child.evicted or not child.backuped:
                    missing_segments.append(
                        self._describe_host_insert_missing_segment(
                            reason="non_host_node",
                            req_id=req_id,
                            segment=segment,
                            parent=parent,
                        )
                    )
                    remaining_segments.append(segment)
                    remaining_segments.extend(segment_plan[idx + 1 :])
                    break
                child = self._split_node(child.key, child, prefix_len)
                advanced = True

            expected_hash = list(segment.get("hash_value") or [])
            actual_hash = list(child.hash_value or [])[: len(expected_hash)]
            if expected_hash and actual_hash != expected_hash:
                missing_segments.append(
                    self._describe_host_insert_missing_segment(
                        reason="hash_mismatch",
                        req_id=req_id,
                        segment=segment,
                        parent=parent,
                    )
                )
                remaining_segments.append(segment)
                remaining_segments.extend(segment_plan[idx + 1 :])
                break
            parent = child
        else:
            remaining_segments = []

        self.authoritative_host_insert_skeleton_by_reqid[req_id] = remaining_segments
        self.authoritative_host_insert_missing_by_reqid[req_id] = (
            self._sort_host_insert_missing_segments(missing_segments)
        )
        if advanced:
            self._record_authoritative_resolution("host_insert.skeleton_advance")
        if missing_segments:
            self._record_authoritative_resolution("host_insert.skeleton_missing")
        if not remaining_segments:
            self._record_authoritative_resolution("host_insert.skeleton_complete")
        return advanced

    def _normalize_host_insert_subtree_from_payload(
        self, payload: dict[str, object]
    ) -> bool:
        anchor_node = self._resolve_authoritative_node_ref(
            node_ref=payload.get("anchor_node_ref")
        )
        if anchor_node is None:
            return False

        fetched_token_ids = list(payload.get("fetched_token_ids") or [])
        matched_length = int(payload.get("matched_length", 0))
        committed_tokens = int(payload.get("committed_tokens", 0))
        if committed_tokens <= matched_length:
            return False

        expected_hashes = list(payload.get("fetched_hash_value") or [])
        matched_pages = matched_length // self.page_size
        remaining_hashes = expected_hashes[matched_pages:]
        remaining_tokens = fetched_token_ids[matched_length:committed_tokens]
        key = RadixKey(
            token_ids=list(remaining_tokens), extra_key=anchor_node.key.extra_key
        )
        parent = anchor_node
        normalized = False

        while len(key) > 0:
            child_key = self.get_child_key_fn(key)
            child = parent.children.get(child_key)
            if child is None:
                return normalized

            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len <= 0:
                return normalized
            if prefix_len < len(child.key):
                if not child.evicted or not child.backuped:
                    return normalized
                child = self._split_node(child.key, child, prefix_len)
                normalized = True

            consumed_pages = len(child.key) // self.page_size
            expected_child_hashes = remaining_hashes[:consumed_pages]
            actual_child_hashes = list(child.hash_value or [])[:consumed_pages]
            if expected_child_hashes and actual_child_hashes != expected_child_hashes:
                return normalized

            key = key[prefix_len:]
            remaining_hashes = remaining_hashes[consumed_pages:]
            parent = child

        return normalized

    def _recover_backup_commit_node_from_payload(
        self, payload: dict[str, object]
    ) -> Optional[TreeNode]:
        parent_node = self._resolve_authoritative_node_ref(
            node_ref=payload.get("parent_node_ref")
        )
        node_key_tokens = list(payload.get("node_key_tokens") or [])
        if parent_node is None or not node_key_tokens:
            return None

        candidates = self._find_exact_host_path_nodes(parent_node, node_key_tokens)
        if len(candidates) != 1:
            return None

        candidate = candidates[0]
        expected_hash_value = list(payload.get("node_hash_value") or [])
        if expected_hash_value and list(candidate.hash_value or []) != expected_hash_value:
            return None
        if not candidate.backuped:
            return None
        return candidate

    def _repair_backup_commit_subtree_from_payload(
        self, payload: dict[str, object]
    ) -> bool:
        parent_node = self._resolve_authoritative_node_ref(
            node_ref=payload.get("parent_node_ref")
        )
        node_key_tokens = list(payload.get("node_key_tokens") or [])
        if parent_node is None or not node_key_tokens:
            return False

        key = RadixKey(
            token_ids=node_key_tokens, extra_key=parent_node.key.extra_key
        )
        child_key = self.get_child_key_fn(key)
        child = parent_node.children.get(child_key)
        if child is None:
            return False

        prefix_len = self.key_match_fn(child.key, key)
        if prefix_len < len(child.key):
            return self._prune_authoritative_host_subtree(child)

        expected_hash_value = list(payload.get("node_hash_value") or [])
        actual_hash_value = list(child.hash_value or [])
        if expected_hash_value and actual_hash_value != expected_hash_value:
            return self._prune_authoritative_host_subtree(child)

        return False

    def _select_authoritative_host_evict_node_ids(self, num_tokens: int) -> list[int]:
        leaves = list(self.evictable_host_leaves)
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)
        selected: list[int] = []
        num_evicted = 0
        while num_evicted < num_tokens and eviction_heap:
            _priority, node = heapq.heappop(eviction_heap)
            if node == self.root_node or not node.evicted or node.host_ref_counter > 0:
                continue
            selected.append(node.id)
            num_evicted += len(node.host_value)
        return selected

    def _select_authoritative_device_evict_node_ids(
        self, num_tokens: int
    ) -> list[int]:
        leaves = list(self.evictable_leaves)
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)
        selected: list[int] = []
        num_evicted = 0
        while num_evicted < num_tokens and eviction_heap:
            _priority, node = heapq.heappop(eviction_heap)
            if node.lock_ref > 0 or node.evicted:
                continue
            selected.append(node.id)
            num_evicted += len(node.value)
        return selected

    def _apply_authoritative_host_evict(
        self,
        node_id: Optional[int] = None,
        *,
        node_ref: Optional[dict[str, object]] = None,
    ) -> None:
        node = self._resolve_authoritative_node_ref(node_ref=node_ref, node_id=node_id)
        if node is None:
            if os.getenv("SGLANG_DEBUG_HICACHE_MATCH_CHAIN", "0") == "1":
                logger.warning(
                    "[HiCacheAuthoritative] unresolved HOST_EVICT node_ref=%s "
                    "node_id=%s pp=%s cp=%s tp=%s",
                    node_ref,
                    node_id,
                    self.pp_rank,
                    self.attn_cp_rank,
                    getattr(self.cache_controller, "tp_rank", None),
                )
            return
        if node == self.root_node or not node.evicted:
            return
        if node.host_ref_counter > 0:
            return
        self.cache_controller.evict_host(node.host_value)
        self._discard_authoritative_visibility(node)
        key = self.get_child_key_fn(node.key)
        popped = node.parent.children.pop(key, None)
        if popped is not node:
            return
        self._unregister_authoritative_node(node)
        if node in self.evictable_host_leaves:
            self.evictable_host_leaves.remove(node)
        self._update_host_leaf_status(node.parent)

    def _apply_authoritative_device_evict(
        self,
        node_id: Optional[int] = None,
        *,
        node_ref: Optional[dict[str, object]] = None,
    ) -> None:
        node = self._resolve_authoritative_node_ref(node_ref=node_ref, node_id=node_id)
        if node is None:
            if os.getenv("SGLANG_DEBUG_HICACHE_MATCH_CHAIN", "0") == "1":
                logger.warning(
                    "[HiCacheAuthoritative] unresolved DEVICE_EVICT node_ref=%s "
                    "node_id=%s pp=%s cp=%s tp=%s",
                    node_ref,
                    node_id,
                    self.pp_rank,
                    self.attn_cp_rank,
                    getattr(self.cache_controller, "tp_rank", None),
                )
            return
        if node.evicted or node.lock_ref > 0:
            return
        if node.backuped:
            self._evict_backuped(node)
        else:
            self._evict_regular(node)

    def _node_backup_visible(self, node: TreeNode) -> bool:
        if not getattr(self.authoritative_tree, "enabled", False):
            return node.backuped
        return (
            node.id in self.authoritative_backuped_node_ids
            or node.id in self.authoritative_host_visible_node_ids
        )

    def _discard_authoritative_visibility(self, node: Optional[TreeNode]) -> None:
        if node is None:
            return
        node_id = getattr(node, "id", None)
        if node_id is None:
            return
        self.authoritative_pending_backup_node_ids.discard(node_id)
        pending_keys_to_remove = []
        for backup_key, node_ref in self.authoritative_pending_backup_refs.items():
            if (
                node_ref.get("node_id") == node_id
                or node_ref.get("last_hash") == node.get_last_hash_value()
            ):
                pending_keys_to_remove.append(backup_key)
        for backup_key in pending_keys_to_remove:
            self.authoritative_pending_backup_refs.pop(backup_key, None)
            self.authoritative_pending_backup_reasons.pop(backup_key, None)
        self.authoritative_backuped_node_ids.discard(node_id)
        self.authoritative_host_visible_node_ids.discard(node_id)

    def _record_authoritative_resolution(self, name: str) -> None:
        self.authoritative_resolution_stats[name] = (
            self.authoritative_resolution_stats.get(name, 0) + 1
        )

    def _clear_prefetch_ready_for_host_node(self, host_node: Optional[TreeNode]) -> None:
        if host_node is None:
            return
        host_node_id = getattr(host_node, "id", None)
        host_node_hash = (
            host_node.get_last_hash_value()
            if hasattr(host_node, "get_last_hash_value")
            else None
        )
        if host_node_id is None and host_node_hash is None:
            return

        req_ids_to_clear: set[str] = set()
        for req_id, ready_result in self.prefetch_ready_results_by_reqid.items():
            last_host_node = getattr(ready_result.match_result, "last_host_node", None)
            if last_host_node is None:
                continue
            if getattr(last_host_node, "id", None) == host_node_id:
                req_ids_to_clear.add(req_id)
                continue
            if (
                host_node_hash is not None
                and hasattr(last_host_node, "get_last_hash_value")
                and last_host_node.get_last_hash_value() == host_node_hash
            ):
                req_ids_to_clear.add(req_id)

        for req_id, summary in self.authoritative_prefetch_ready_by_reqid.items():
            node_ref = getattr(summary, "last_host_node_ref", None)
            if node_ref is None:
                continue
            if node_ref.get("node_id") == host_node_id or (
                host_node_hash is not None and node_ref.get("last_hash") == host_node_hash
            ):
                req_ids_to_clear.add(req_id)

        for req_id in req_ids_to_clear:
            self.prefetch_ready_results_by_reqid.pop(req_id, None)
            self.authoritative_prefetch_ready_by_reqid.pop(req_id, None)
            self.prefetch_loaded_tokens_by_reqid.pop(req_id, None)
            self.authoritative_prefetch_loaded_tokens_by_reqid.pop(req_id, None)

    def get_authoritative_resolution_stats(self) -> dict[str, int]:
        return dict(self.authoritative_resolution_stats)

    def get_host_insert_missing_segments(
        self, req_id: Optional[str] = None
    ) -> dict[str, list[dict[str, object]]] | list[dict[str, object]]:
        if req_id is None:
            return {
                rid: list(segments)
                for rid, segments in self.authoritative_host_insert_missing_by_reqid.items()
            }
        return list(self.authoritative_host_insert_missing_by_reqid.get(req_id, []))

    def get_pending_backup_node_ids(self) -> list[int]:
        return sorted(self.authoritative_pending_backup_node_ids)

    def get_pending_backup_reasons(self) -> dict[str, str]:
        return dict(self.authoritative_pending_backup_reasons)

    def _iter_pending_backup_nodes(self) -> list[TreeNode]:
        nodes: list[TreeNode] = []
        for node_ref in self.authoritative_pending_backup_refs.values():
            node = self._resolve_authoritative_node_ref(node_ref=node_ref)
            if node is not None:
                nodes.append(node)
        return nodes

    def _is_ancestor_node(
        self, ancestor: Optional[TreeNode], node: Optional[TreeNode]
    ) -> bool:
        while node is not None and node != self.root_node:
            if node == ancestor:
                return True
            node = node.parent
        return ancestor == self.root_node and ancestor is not None

    def _has_relevant_pending_backup(
        self,
        req_id: Optional[str] = None,
        *,
        host_node: Optional[TreeNode] = None,
    ) -> bool:
        if not self.authoritative_pending_backup_refs:
            return False
        if host_node is None and req_id is not None:
            summary = self.authoritative_prefetch_ready_by_reqid.get(req_id)
            if summary is not None and summary.last_host_node_ref is not None:
                host_node = self._resolve_authoritative_node_ref(
                    node_ref=summary.last_host_node_ref
                )
        if host_node is None and req_id is not None:
            ready_result = self.prefetch_ready_results_by_reqid.get(req_id)
            if ready_result is not None:
                host_node = ready_result.match_result.last_host_node
        if host_node is None or host_node == self.root_node:
            return False
        for pending_node in self._iter_pending_backup_nodes():
            if self._is_ancestor_node(pending_node, host_node):
                return True
        return False

    def _describe_pending_backup_reason(self, node: Optional[TreeNode]) -> str:
        if node is None:
            return "node_missing"
        if not node.backuped or len(node.host_value) == 0:
            return "missing_host_value"
        if getattr(node, "host_ref_counter", 0) > 0:
            return "host_ref"
        return "await_stable"

    def _is_authoritative_ready_stable(self, req_id: Optional[str]) -> bool:
        if req_id is None:
            return True
        has_blueprint = req_id in self.authoritative_host_insert_rebuild_by_reqid
        has_skeleton = bool(self.authoritative_host_insert_skeleton_by_reqid.get(req_id))
        has_missing = bool(self.authoritative_host_insert_missing_by_reqid.get(req_id))
        has_pending_backup = self._has_relevant_pending_backup(req_id)
        return not (has_blueprint or has_skeleton or has_missing or has_pending_backup)

    def _format_authoritative_resolution_stats(self, limit: int = 6) -> str:
        stats = self.get_authoritative_resolution_stats()
        if not stats:
            return "none"
        items = sorted(stats.items(), key=lambda item: (-item[1], item[0]))
        return ",".join(f"{key}={value}" for key, value in items[:limit])

    def _format_host_insert_missing_segments(
        self, req_id: Optional[str], limit: int = 2
    ) -> str:
        if req_id is None:
            return "none"
        segments = self.get_host_insert_missing_segments(req_id)
        if not segments:
            return "none"
        parts = []
        for segment in segments[:limit]:
            parts.append(
                f"{segment.get('reason')}@p{segment.get('page_index')}:{segment.get('token_ids')}"
            )
        return ";".join(parts)

    def _format_pending_backup_nodes(self, limit: int = 4) -> str:
        if not self.authoritative_pending_backup_refs:
            return "none"
        pending = sorted(self.authoritative_pending_backup_refs.items())
        preview = ",".join(
            (
                f"{(self._resolve_authoritative_node_ref(node_ref=node_ref).id if self._resolve_authoritative_node_ref(node_ref=node_ref) is not None else backup_key)}:"
                f"{self.authoritative_pending_backup_reasons.get(backup_key, 'unknown')}"
            )
            for backup_key, node_ref in pending[:limit]
        )
        if len(pending) > limit:
            preview += ",..."
        return preview

    def _format_match_authoritative_gates(
        self,
        req_id: Optional[str],
        *,
        pre_clamp_device_hit: Optional[int] = None,
        pre_clamp_host_hit: Optional[int] = None,
        match_result: Optional[MatchResult] = None,
    ) -> str:
        gates: list[str] = []
        if req_id is not None and not self._is_authoritative_ready_stable(req_id):
            gates.append("unstable_ready")
        if req_id is not None and self.authoritative_host_insert_missing_by_reqid.get(req_id):
            gates.append("missing_gap")
        if (
            match_result is not None
            and self._has_relevant_pending_backup(
                req_id, host_node=match_result.last_host_node
            )
        ):
            gates.append("pending_backup")
        if (
            match_result is not None
            and pre_clamp_device_hit is not None
            and pre_clamp_host_hit is not None
            and (
                pre_clamp_device_hit != len(match_result.device_indices)
                or pre_clamp_host_hit != match_result.host_hit_length
            )
        ):
            gates.append("summary_clamp")
        return ",".join(gates) if gates else "none"

    def _format_match_clamp_delta(
        self,
        *,
        pre_clamp_device_hit: int,
        pre_clamp_host_hit: int,
        match_result: MatchResult,
    ) -> str:
        post_device_hit = len(match_result.device_indices)
        post_host_hit = match_result.host_hit_length
        if (
            pre_clamp_device_hit == post_device_hit
            and pre_clamp_host_hit == post_host_hit
        ):
            return "none"
        return (
            f"device:{pre_clamp_device_hit}->{post_device_hit},"
            f"host:{pre_clamp_host_hit}->{post_host_hit}"
        )

    def _maybe_log_authoritative_resolution_stats(self) -> None:
        interval_s = float(
            os.getenv("SGLANG_DEBUG_HICACHE_AUTHORITATIVE_STATS_INTERVAL", "0")
        )
        if interval_s <= 0:
            return
        now = time.monotonic()
        if now - self._last_authoritative_resolution_log_ts < interval_s:
            return
        self._last_authoritative_resolution_log_ts = now
        logger.warning(
            "[HiCacheAuthoritativeStats] stats=%s pp=%s cp=%s tp=%s",
            self._format_authoritative_resolution_stats(limit=12),
            self.pp_rank,
            self.attn_cp_rank,
            getattr(self.cache_controller, "tp_rank", None),
        )

    def _find_last_visible_host_ancestor(self, node: Optional[TreeNode]) -> TreeNode:
        while node is not None and node != self.root_node:
            if self._node_backup_visible(node):
                return node
            node = node.parent
        return self.root_node

    def _select_last_host_node_for_hit_length(
        self, deepest_visible_host_node: Optional[TreeNode], host_hit_length: int
    ) -> TreeNode:
        if (
            deepest_visible_host_node is None
            or deepest_visible_host_node == self.root_node
            or host_hit_length <= 0
        ):
            return self.root_node

        chain: list[TreeNode] = []
        cursor = deepest_visible_host_node
        while (
            cursor is not None
            and cursor != self.root_node
            and cursor.evicted
            and self._node_backup_visible(cursor)
        ):
            chain.append(cursor)
            cursor = cursor.parent

        if not chain:
            return self.root_node

        accumulated = 0
        selected = self.root_node
        for node in reversed(chain):
            accumulated += len(node.host_value)
            selected = node
            if accumulated >= host_hit_length:
                break
        return selected

    def _build_host_insert_from_storage_payload(
        self,
        *,
        req_id: str,
        anchor_node: TreeNode,
        fetched_token_ids: List[int],
        fetched_hash_value: List[str],
        inserted_nodes: list[TreeNode],
        loaded_from_storage: int,
        matched_length: int,
        committed_tokens: int,
    ) -> dict[str, object]:
        return {
            "req_id": req_id,
            "anchor_node_ref": self._make_authoritative_node_ref(anchor_node),
            "fetched_token_ids": list(fetched_token_ids),
            "fetched_hash_value": list(fetched_hash_value),
            "loaded_from_storage": loaded_from_storage,
            "matched_length": matched_length,
            "committed_tokens": committed_tokens,
            "node_ids": [getattr(node, "id", None) for node in inserted_nodes],
            "node_refs": [
                self._make_authoritative_node_ref(node) for node in inserted_nodes
            ],
        }

    def _clamp_ready_result_to_authoritative_summary(
        self,
        req_id: str,
        ready_result: Optional[LatchedPrefetchReadyResult],
    ) -> Optional[LatchedPrefetchReadyResult]:
        if ready_result is None:
            return None
        summary = self.authoritative_prefetch_ready_by_reqid.get(req_id)
        if summary is None:
            return ready_result
        if (
            ready_result.input_len is not None
            and summary.input_len is not None
            and summary.input_len != ready_result.input_len
        ):
            return ready_result
        return LatchedPrefetchReadyResult(
            match_result=self._clamp_match_result_to_authoritative_summary(
                req_id, ready_result.match_result, input_len=ready_result.input_len
            ),
            storage_hit_length=summary.storage_hit_length,
            input_len=ready_result.input_len,
        )

    def _clamp_match_result_to_authoritative_summary(
        self,
        req_id: Optional[str],
        match_result: MatchResult,
        *,
        input_len: Optional[int] = None,
    ) -> MatchResult:
        if req_id is None:
            return match_result
        summary = self.authoritative_prefetch_ready_by_reqid.get(req_id)
        if summary is None:
            return match_result
        if (
            input_len is not None
            and summary.input_len is not None
            and summary.input_len != input_len
        ):
            return match_result

        device_indices = match_result.device_indices
        if len(device_indices) > summary.prefix_len:
            device_indices = device_indices[: summary.prefix_len]

        host_hit_length = min(match_result.host_hit_length, summary.host_hit_length)
        last_host_node = self._select_last_host_node_for_hit_length(
            match_result.last_host_node, host_hit_length
        )
        return MatchResult(
            device_indices=device_indices,
            last_device_node=match_result.last_device_node,
            last_host_node=last_host_node,
            host_hit_length=host_hit_length,
            mamba_branching_seqlen=match_result.mamba_branching_seqlen,
        )

    def shutdown(self):
        """Best-effort auto-detach of storage backend on process shutdown.

        This keeps startup and runtime behavior consistent: if a backend was attached
        (either via CLI args or via admin API), we attempt to detach it on exit.
        """
        try:
            if self.enable_storage:
                self.detach_storage_backend()
        except Exception:
            logger.exception("Failed to detach storage backend on process shutdown.")

    def _apply_storage_runtime_config(
        self,
        *,
        storage_backend: Optional[str],
        prefetch_threshold: int,
        prefetch_timeout_base: float,
        prefetch_timeout_per_ki_token: float,
        hicache_storage_pass_prefix_keys: bool,
        enable_storage: bool,
        enable_storage_metrics: bool,
        extra_metric_labels: Optional[Dict[str, str]],
    ) -> None:
        prefetch_timeout_per_page = (
            self.page_size / 1024 * prefetch_timeout_per_ki_token
        )

        self.enable_storage = enable_storage
        self.prefetch_threshold = prefetch_threshold
        self.prefetch_timeout_base = prefetch_timeout_base
        self.prefetch_timeout_per_page = prefetch_timeout_per_page
        self.hicache_storage_pass_prefix_keys = hicache_storage_pass_prefix_keys
        self.enable_storage_metrics = enable_storage_metrics

        if self.enable_storage_metrics:
            labels = {
                "storage_backend": storage_backend,
                "tp_rank": self.cache_controller.tp_rank,
                "dp_rank": self.cache_controller.dp_rank,
                "pp_rank": self.cache_controller.pp_rank,
                "pp_size": self.cache_controller.pp_size,
                "attn_cp_rank": self.cache_controller.attn_cp_rank,
                "attn_cp_size": self.cache_controller.attn_cp_size,
            }
            if extra_metric_labels:
                labels.update(extra_metric_labels)
            existing_collector = getattr(self, "storage_metrics_collector", None)
            if existing_collector is None:
                self.storage_metrics_collector = StorageMetricsCollector(labels=labels)
            elif set(existing_collector.labels.keys()) == set(labels.keys()):
                existing_collector.labels = labels
            else:
                logger.warning(
                    "Storage metrics labels changed (%s -> %s). Keep existing labels to "
                    "avoid duplicate metric registration.",
                    sorted(existing_collector.labels.keys()),
                    sorted(labels.keys()),
                )

    def attach_storage_backend(
        self,
        storage_backend: str,
        storage_backend_extra_config_json: Optional[str] = None,
        served_model_name: Optional[str] = None,
        hicache_storage_prefetch_policy: Optional[str] = None,
        hicache_write_policy: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Attach (enable) storage backend at runtime.

        This will start storage threads inside `HiCacheController` and enable
        prefetch/backup paths. Caller must ensure there are no running/queued
        requests to avoid races.
        """
        if self.use_nsa_pool_controller and storage_backend not in ("file", "mooncake"):
            return (
                False,
                "NSA pool-based HiCache only supports file and mooncake storage backends.",
            )
        # Validate inputs first (no side effects).
        if hicache_storage_prefetch_policy is not None:
            allowed = ["best_effort", "wait_complete", "timeout"]
            if hicache_storage_prefetch_policy not in allowed:
                return (
                    False,
                    f"Invalid hicache_storage_prefetch_policy: {hicache_storage_prefetch_policy!r}. "
                    f"Expected one of {allowed}.",
                )

        if hicache_write_policy is not None:
            allowed = ["write_back", "write_through", "write_through_selective"]
            if hicache_write_policy not in allowed:
                return (
                    False,
                    f"Invalid hicache_write_policy: {hicache_write_policy!r}. "
                    f"Expected one of {allowed}.",
                )

        # If already enabled:
        # - backend unchanged: treat as success, update policies only.
        # - backend changed: treat as failure, do NOT update policies.
        if self.enable_storage:
            current_backend = self.cache_controller.storage_backend_type

            if current_backend == storage_backend:
                if hicache_storage_prefetch_policy is not None:
                    self.prefetch_stop_policy = hicache_storage_prefetch_policy
                    logger.info(
                        f"Set hicache_storage_prefetch_policy to {hicache_storage_prefetch_policy}"
                    )
                if hicache_write_policy is not None:
                    self.cache_controller.write_policy = hicache_write_policy
                    self.write_through_threshold = (
                        1 if hicache_write_policy == "write_through" else 2
                    )
                    logger.info(f"Set hicache_write_policy to {hicache_write_policy}")
                return (
                    True,
                    "HiCache storage backend already enabled with same backend; policies updated.",
                )

            return (
                False,
                f"HiCache storage backend is already enabled with backend '{current_backend}'. "
                f"Cannot attach different backend '{storage_backend}'. Detach first.",
            )

        # Not enabled: update policies before controller attach so storage threads observe new values.
        if hicache_storage_prefetch_policy is not None:
            self.prefetch_stop_policy = hicache_storage_prefetch_policy
            logger.info(
                f"Set hicache_storage_prefetch_policy to {hicache_storage_prefetch_policy}"
            )

        if hicache_write_policy is not None:
            self.cache_controller.write_policy = hicache_write_policy
            self.write_through_threshold = (
                1 if hicache_write_policy == "write_through" else 2
            )
            logger.info(f"Set hicache_write_policy to {hicache_write_policy}")

        logger.info(f"Attaching HiCache storage backend: {storage_backend}")
        try:
            (
                extra_config,
                prefetch_threshold,
                prefetch_timeout_base,
                prefetch_timeout_per_ki_token,
                hicache_storage_pass_prefix_keys,
            ) = self._parse_storage_backend_extra_config(
                storage_backend_extra_config_json
            )
        except Exception as e:
            logger.exception(f"Failed to parse storage_backend_extra_config_json: {e}")
            return (
                False,
                f"Failed to parse storage_backend_extra_config_json '{storage_backend_extra_config_json}': {e}",
            )

        try:
            attach_kwargs = dict(
                storage_backend=storage_backend,
                prefetch_threshold=prefetch_threshold,
                model_name=served_model_name,
                storage_backend_extra_config=extra_config,
            )
            if self.use_nsa_pool_controller:
                attach_kwargs["host_pools"] = self.host_pool_group.entries
            self.cache_controller.attach_storage_backend(**attach_kwargs)
        except Exception as e:
            logger.exception(
                f"Failed to attach storage backend '{storage_backend}': {e}"
            )
            return False, f"Failed to attach storage backend '{storage_backend}': {e}"

        self._apply_storage_runtime_config(
            storage_backend=storage_backend,
            prefetch_threshold=prefetch_threshold,
            prefetch_timeout_base=prefetch_timeout_base,
            prefetch_timeout_per_ki_token=prefetch_timeout_per_ki_token,
            hicache_storage_pass_prefix_keys=hicache_storage_pass_prefix_keys,
            enable_storage=True,
            enable_storage_metrics=self._enable_metrics_flag,
            extra_metric_labels=self.extra_metric_labels,
        )
        return True, "Attached HiCache storage backend successfully."

    def detach_storage_backend(self) -> tuple[bool, str]:
        """Detach (disable) storage backend at runtime.

        Caller must ensure there are no running/queued requests to avoid races.
        """
        try:
            # Drain any pending control queues before tearing down storage threads/backend.
            # IMPORTANT: this must happen before we clear `ongoing_*`, otherwise acks/releases
            # cannot be matched to nodes and may leak host pages / locks.
            self._drain_storage_control_queues_local()
            # Idempotent detach: always ask controller to best-effort cleanup, even if
            # `self.enable_storage` is already False (may be leftover state from a
            # previous partial detach).
            self.cache_controller.detach_storage_backend()
        except Exception as e:
            logger.exception("Failed to detach storage backend.")
            # Do NOT crash the server for admin operations. Return failure with detail.
            return False, f"Failed to detach HiCache storage backend: {e}"

        # Best-effort cleanup of any leftover bookkeeping.
        self._drain_storage_control_queues_local()
        # After controller threads are fully stopped, it's safe to force-release any
        # leftover pending ops (e.g., async prefetch/backup that didn't get a revoke/ack).
        self._force_release_pending_storage_ops()

        self.enable_storage = False
        self.enable_storage_metrics = False
        return True, "Detached HiCache storage backend successfully."

    def _force_release_pending_storage_ops(self):
        """Force release any leftover pending prefetch/backup bookkeeping.

        This is a safety net for detach/shutdown paths. It assumes storage threads
        have been stopped already (via controller.detach), so no concurrent access
        to these structures should happen.
        """
        cc = self.cache_controller

        # Force release leftover prefetch ops: free pre-allocated host pages and
        # drop the host protection on the matched prefix node.
        try:
            for req_id, info in list(self.ongoing_prefetch.items()):
                try:
                    last_host_node, token_ids, host_indices, _operation = info
                except Exception:
                    # Unexpected shape; just drop it.
                    self.ongoing_prefetch.pop(req_id, None)
                    continue

                try:
                    if host_indices is not None:
                        cc.mem_pool_host.free(host_indices)
                except Exception:
                    logger.exception(
                        "Failed to free host indices for prefetch %s", req_id
                    )

                try:
                    last_host_node.release_host()
                except Exception:
                    logger.exception(
                        "Failed to release host protection for prefetch %s", req_id
                    )

                try:
                    cc.prefetch_tokens_occupied -= len(token_ids)
                    if cc.prefetch_tokens_occupied < 0:
                        cc.prefetch_tokens_occupied = 0
                except Exception:
                    pass

                self.ongoing_prefetch.pop(req_id, None)
                self.prefetch_loaded_tokens_by_reqid.pop(req_id, None)
                self.prefetch_ready_results_by_reqid.pop(req_id, None)
                self.authoritative_prefetch_ready_by_reqid.pop(req_id, None)
                self.authoritative_prefetch_loaded_tokens_by_reqid.pop(req_id, None)
                self._clear_host_insert_rebuild_blueprint(req_id)
                self.prefetch_skipped_rids.discard(req_id)
        except Exception:
            logger.exception("Force release pending prefetch ops failed.")

        # Force release leftover backup ops: drop host protection on nodes.
        try:
            for ack_id, node in list(self.ongoing_backup.items()):
                try:
                    node.release_host()
                except Exception:
                    logger.exception(
                        "Failed to release host protection for backup op %s", ack_id
                    )
                self.ongoing_backup.pop(ack_id, None)
        except Exception:
            logger.exception("Force release pending backup ops failed.")

    def _drain_storage_control_queues_local(self):
        """Drain storage control queues without TP synchronization.

        This is intended for shutdown/detach paths where we want to make best-effort
        cleanup even if queue sizes temporarily differ across ranks.
        """
        self._drain_storage_control_queues_impl(
            n_revoke=None,
            n_backup=None,
            n_release=None,
            log_metrics=False,
        )

    def _drain_storage_control_queues_impl(
        self,
        n_revoke: Optional[int],
        n_backup: Optional[int],
        n_release: Optional[int],
        log_metrics: bool,
    ):
        cc = self.cache_controller

        def _drain_queue(q, limit: Optional[int]):
            drained = 0
            while limit is None or drained < limit:
                try:
                    item = q.get_nowait()
                except Empty:
                    break
                drained += 1
                yield item

        def _drain_revoke():
            for req_id in _drain_queue(cc.prefetch_revoke_queue, n_revoke):
                info = self.ongoing_prefetch.pop(req_id, None)
                self.prefetch_loaded_tokens_by_reqid.pop(req_id, None)
                self.prefetch_ready_results_by_reqid.pop(req_id, None)
                self.authoritative_prefetch_ready_by_reqid.pop(req_id, None)
                self.authoritative_prefetch_loaded_tokens_by_reqid.pop(req_id, None)
                self._clear_host_insert_rebuild_blueprint(req_id)
                self.prefetch_skipped_rids.discard(req_id)
                if info is not None:
                    last_host_node, token_ids, _, _ = info
                    if os.getenv("SGLANG_DEBUG_HICACHE_MATCH_CHAIN", "0") == "1":
                        logger.warning(
                            "[HiCacheMatchChain] revoke cleanup: rid=%s host_node=%s "
                            "token_count=%s pp=%s cp=%s tp=%s",
                            req_id,
                            getattr(last_host_node, "id", None),
                            len(token_ids),
                            self.pp_rank,
                            self.attn_cp_rank,
                            getattr(self.cache_controller, "tp_rank", None),
                        )
                    last_host_node.release_host()
                    cc.prefetch_tokens_occupied -= len(token_ids)
                    if cc.prefetch_tokens_occupied < 0:
                        cc.prefetch_tokens_occupied = 0

        def _drain_backup():
            for operation in _drain_queue(cc.ack_backup_queue, n_backup):
                ack_id = operation.id
                entry = self.ongoing_backup.pop(ack_id, None)
                if entry is not None:
                    entry.release_host()
                if log_metrics and self.enable_storage_metrics:
                    self.storage_metrics_collector.log_backuped_tokens(
                        operation.completed_tokens
                    )

        def _drain_release():
            host_indices_list = []
            for host_indices in _drain_queue(cc.host_mem_release_queue, n_release):
                host_indices_list.append(host_indices)
            if host_indices_list:
                host_indices = torch.cat(host_indices_list, dim=0)
                cc.mem_pool_host.free(host_indices)

        _drain_revoke()
        _drain_backup()
        _drain_release()

    def _parse_storage_backend_extra_config(
        self, storage_backend_extra_config: Optional[str]
    ):
        """
        Parse storage backend extra config JSON and extract specific parameters.

        Args:
            storage_backend_extra_config: JSON string containing extra configuration

        Returns:
            tuple: (extra_config_dict, prefetch_threshold, prefetch_timeout_base, prefetch_timeout_per_ki_token, hicache_storage_pass_prefix_keys)
        """
        # Parse extra config if provided. Extra config can be a JSON string or a json/toml/yaml file path prefixed with "@".
        extra_config = {}
        if storage_backend_extra_config:
            try:
                if storage_backend_extra_config.startswith("@"):
                    # Read config from a json/toml/yaml file
                    path = storage_backend_extra_config[1:]
                    ext = os.path.splitext(path)[1].lower()
                    with open(path, "rb" if ext == ".toml" else "r") as f:
                        if ext == ".json":
                            extra_config = json.load(f)
                        elif ext == ".toml":
                            import tomllib

                            extra_config = tomllib.load(f)
                        elif ext in (".yaml", ".yml"):
                            import yaml

                            extra_config = yaml.safe_load(f)
                        else:
                            raise ValueError(
                                f"Unsupported config file {path} (config format: {ext})"
                            )
                else:
                    # read config from JSON string
                    extra_config = json.loads(storage_backend_extra_config)
            except Exception as e:
                logger.error(f"Invalid backend extra config JSON: {e}")
                raise e

        prefetch_threshold = extra_config.pop("prefetch_threshold", 256)  # tokens
        prefetch_timeout_base = extra_config.pop("prefetch_timeout_base", 1)  # seconds
        prefetch_timeout_per_ki_token = extra_config.pop(
            "prefetch_timeout_per_ki_token", 0.25
        )  # seconds per 1024 tokens
        hicache_storage_pass_prefix_keys = extra_config.pop(
            "hicache_storage_pass_prefix_keys", False
        )

        if not isinstance(prefetch_threshold, int):
            raise ValueError(
                f"prefetch_threshold must be int, got {type(prefetch_threshold).__name__}"
            )
        if not isinstance(prefetch_timeout_base, (int, float)):
            raise ValueError(
                f"prefetch_timeout_base must be number, got {type(prefetch_timeout_base).__name__}"
            )
        if not isinstance(prefetch_timeout_per_ki_token, (int, float)):
            raise ValueError(
                f"prefetch_timeout_per_ki_token must be number, got {type(prefetch_timeout_per_ki_token).__name__}"
            )
        if not isinstance(hicache_storage_pass_prefix_keys, bool):
            raise ValueError(
                "hicache_storage_pass_prefix_keys must be bool, got "
                f"{type(hicache_storage_pass_prefix_keys).__name__}"
            )

        return (
            extra_config,
            prefetch_threshold,
            float(prefetch_timeout_base),
            float(prefetch_timeout_per_ki_token),
            hicache_storage_pass_prefix_keys,
        )

    def reset(self):
        TreeNode.counter = 0
        self.cache_controller.reset()
        self.token_to_kv_pool_host.clear()
        # Clear per-request tracking dicts
        self.prefetch_loaded_tokens_by_reqid.clear()
        self.prefetch_ready_results_by_reqid.clear()
        self.authoritative_prefetch_ready_by_reqid.clear()
        self.authoritative_prefetch_loaded_tokens_by_reqid.clear()
        self.authoritative_host_insert_rebuild_by_reqid.clear()
        self.authoritative_host_insert_skeleton_by_reqid.clear()
        self.authoritative_host_insert_missing_by_reqid.clear()
        self.authoritative_pending_backup_node_ids.clear()
        self.authoritative_pending_backup_reasons.clear()
        self.authoritative_backuped_node_ids.clear()
        self.authoritative_host_visible_node_ids.clear()
        self.authoritative_resolution_stats.clear()
        self._last_authoritative_resolution_log_ts = 0.0
        self._authoritative_node_by_id.clear()
        self._authoritative_node_by_last_hash.clear()
        self._authoritative_last_hash_by_node_id.clear()
        self.evictable_host_leaves.clear()
        super().reset()
        self._register_authoritative_node(self.root_node)

    def _build_latched_prefetch_ready_result(
        self, req: Req, storage_hit_length: int
    ) -> LatchedPrefetchReadyResult:
        input_len = len(req.fill_ids)
        max_prefix_len = input_len - 1
        if req.return_logprob and req.logprob_start_len >= 0:
            max_prefix_len = min(max_prefix_len, req.logprob_start_len)
        max_prefix_len = max(max_prefix_len, 0)
        token_ids = req.fill_ids[:max_prefix_len]
        match_result = self.match_prefix(
            MatchPrefixParams(
                key=RadixKey(token_ids=token_ids, extra_key=req.extra_key),
                req=req,
                cow_mamba=self.supports_mamba(),
            )
        )
        return LatchedPrefetchReadyResult(
            match_result=match_result,
            storage_hit_length=storage_hit_length,
            input_len=len(req.fill_ids),
        )

    def _recover_prefetch_committed_host_nodes(
        self,
        *,
        anchor_node: TreeNode,
        fetched_token_ids: List[int],
        fetched_hash_value: List[str],
        committed_tokens: int,
    ) -> list[TreeNode]:
        if committed_tokens <= 0:
            return []
        path_nodes = self._find_exact_host_path_nodes(
            anchor_node, fetched_token_ids[:committed_tokens]
        )
        if not path_nodes:
            return []

        selected: list[TreeNode] = []
        covered_tokens = 0
        candidate_hashes: list[str] = []
        for node in path_nodes:
            if not node.evicted or len(node.host_value) == 0:
                break
            selected.append(node)
            covered_tokens += len(node.host_value)
            if node.hash_value:
                candidate_hashes.extend(node.hash_value)
            if covered_tokens >= committed_tokens:
                break

        if covered_tokens < committed_tokens:
            return []

        expected_hashes = fetched_hash_value[: committed_tokens // self.page_size]
        if expected_hashes and candidate_hashes[: len(expected_hashes)] != expected_hashes:
            return []
        return selected

    def _build_authoritative_ready_result_from_prefetch_finalize(
        self,
        req: Req,
        *,
        anchor_node: TreeNode,
        fetched_token_ids: List[int],
        fetched_hash_value: List[str],
        committed_tokens: int,
        storage_hit_length: int,
    ) -> LatchedPrefetchReadyResult:
        base_ready = self._build_latched_prefetch_ready_result(req, storage_hit_length)
        host_nodes = self._recover_prefetch_committed_host_nodes(
            anchor_node=anchor_node,
            fetched_token_ids=fetched_token_ids,
            fetched_hash_value=fetched_hash_value,
            committed_tokens=committed_tokens,
        )
        host_hit_length = min(
            committed_tokens, sum(len(node.host_value) for node in host_nodes)
        )
        for node in host_nodes:
            self.authoritative_host_visible_node_ids.add(node.id)
        last_host_node = host_nodes[-1] if host_nodes else self.root_node
        return LatchedPrefetchReadyResult(
            match_result=MatchResult(
                device_indices=base_ready.match_result.device_indices,
                last_device_node=base_ready.match_result.last_device_node,
                last_host_node=last_host_node,
                host_hit_length=host_hit_length,
                mamba_branching_seqlen=base_ready.match_result.mamba_branching_seqlen,
            ),
            storage_hit_length=storage_hit_length,
            input_len=base_ready.input_len,
        )

    def get_height(self, node: TreeNode):
        height = 0
        while node != self.root_node:
            node = node.parent
            height += 1
        return height

    def clear_storage_backend(self) -> bool:
        if self.enable_storage:
            try:
                # Check if the storage backend has a clear method (for nixl backends)
                if hasattr(self.cache_controller.storage_backend, "clear"):
                    self.cache_controller.storage_backend.clear()
                    logger.info(
                        "Hierarchical cache storage backend cleared successfully!"
                    )
                    return True
                else:
                    logger.warning(
                        f"Storage backend {type(self.cache_controller.storage_backend).__name__} does not support clear operation."
                    )
                    return False
            except Exception as e:
                logger.error(f"Failed to clear hierarchical cache storage backend: {e}")
                return False
        else:
            logger.warning("Hierarchical cache storage backend is not enabled.")
            return False

    def write_backup(self, node: TreeNode, write_back=False):
        host_indices = self.cache_controller.write(
            device_indices=node.value,
            node_id=node.id,
        )
        if host_indices is None:
            self.evict_host(len(node.value))
            host_indices = self.cache_controller.write(
                device_indices=node.value,
                node_id=node.id,
            )
        if host_indices is not None:
            node.host_value = host_indices
            assert len(node.host_value) > 0
            self.ongoing_write_through[node.id] = node
            if not write_back:
                # no need to lock nodes if write back
                self.inc_lock_ref(node)
        else:
            return 0

        return len(host_indices)

    def write_backup_storage(self, node: TreeNode):
        prefix_keys = (
            node.get_prefix_hash_values(node.parent)
            if self.hicache_storage_pass_prefix_keys
            else None
        )
        archive_transfers = self.nsa_archive_transfers(node)

        if archive_transfers:
            operation_id = self.cache_controller.write_storage(
                node.host_value,
                node.key,
                node.hash_value,
                prefix_keys,
                extra_pools=archive_transfers,
            )
        else:
            operation_id = self.cache_controller.write_storage(
                node.host_value,
                node.key,
                node.hash_value,
                prefix_keys,
            )
        self.ongoing_backup[operation_id] = node
        node.protect_host()

    def _inc_hit_count(self, node: TreeNode, chunked=False):
        # skip the hit count update for chunked requests
        if self.cache_controller.write_policy == "write_back" or chunked:
            return
        node.hit_count += 1

        if not node.backuped:
            if node.hit_count >= self.write_through_threshold:
                # write to host if the node is not backuped
                self.write_backup(node)

    def writing_check(self, write_back=False):
        if write_back:
            # blocking till all write back complete
            while len(self.ongoing_write_through) > 0:
                for _, finish_event, ack_list in self.cache_controller.ack_write_queue:
                    finish_event.synchronize()
                    for ack_id in ack_list:
                        backuped_node = self.ongoing_write_through.pop(ack_id)
                        if self.enable_storage:
                            self.write_backup_storage(backuped_node)
                self.cache_controller.ack_write_queue.clear()
                assert len(self.ongoing_write_through) == 0
            return

        # NOTE: all ranks has the same ongoing_write_through, can skip sync if empty
        if len(self.ongoing_write_through) == 0:
            return

        finish_count = 0
        for _, finish_event, ack_list in self.cache_controller.ack_write_queue:
            if not finish_event.query():
                break
            finish_count += 1
        queue_size = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        # Keep cache state transitions identical across CPxTP participants.
        self._all_reduce_attn_groups(queue_size, torch.distributed.ReduceOp.MIN)

        finish_count = int(queue_size.item())
        while finish_count > 0:
            _, finish_event, ack_list = self.cache_controller.ack_write_queue.pop(0)
            finish_event.synchronize()
            for ack_id in ack_list:
                backuped_node = self.ongoing_write_through.pop(ack_id)
                self.queue_authoritative_tree_op(
                    "HOST_BACKUP_COMMIT",
                    **self._build_host_backup_commit_payload(backuped_node),
                )
                self.dec_lock_ref(backuped_node)
                if self.enable_storage:
                    self.write_backup_storage(backuped_node)
            finish_count -= 1

    def loading_check(self):
        finish_count = 0
        for _, finish_event, ack_list in self.cache_controller.ack_load_queue:
            if not finish_event.query():
                # the KV cache loading is still ongoing
                break
            finish_count += 1
            # no need to sync across TP workers as batch forwarding is synced
            for ack_id in ack_list:
                end_node = self.ongoing_load_back.pop(ack_id)
                self.dec_lock_ref(end_node)

        # ACK until all events are processed
        del self.cache_controller.ack_load_queue[:finish_count]

    def evictable_size(self):
        return self.evictable_size_

    def inc_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.key)
                self.protected_size_ += len(node.key)
                delta -= len(node.key)
            node.lock_ref += 1
            self._update_leaf_status(node)
            self._update_host_leaf_status(node)
            node = node.parent
        return delta

    def dec_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.key)
                self.protected_size_ -= len(node.key)
                delta += len(node.key)
            node.lock_ref -= 1
            self._update_leaf_status(node)
            self._update_host_leaf_status(node)
            if node.parent is None:
                assert (
                    node is self.root_node
                ), f"This request holds the node from another tree"
            node = node.parent
        return delta

    def _update_host_leaf_status(self, node: TreeNode):
        if not node.evicted or node.lock_ref > 0:
            if node in self.evictable_host_leaves:
                self.evictable_host_leaves.remove(node)
            return

        for child in node.children.values():
            if child.evicted:
                if node in self.evictable_host_leaves:
                    self.evictable_host_leaves.remove(node)
                return

        if node not in self.evictable_host_leaves:
            self.evictable_host_leaves.add(node)

    def _delete_leaf(self, node):
        self._discard_authoritative_visibility(node)
        self._unregister_authoritative_node(node)
        super()._delete_leaf(node)

    def evict(self, params: EvictParams) -> EvictResult:
        if getattr(self.authoritative_tree, "enabled", False):
            self.queue_authoritative_tree_op(
                "DEVICE_EVICT_REQUEST",
                num_tokens=params.num_tokens,
            )
        start_time = time.perf_counter()
        num_tokens = params.num_tokens
        leaves = list(self.evictable_leaves)
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        write_back_nodes = []
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)

            if x.lock_ref > 0:
                continue

            if not x.backuped:
                if self.cache_controller.write_policy == "write_back":
                    # write to host if the node is not backuped
                    num_evicted += self.write_backup(x, write_back=True)
                    write_back_nodes.append(x)
                else:
                    num_evicted += self._evict_regular(x)
            else:
                num_evicted += self._evict_backuped(x)

            for child in x.parent.children.values():
                if child in write_back_nodes:
                    continue
                if not child.evicted:
                    break
            else:
                # all children are evicted or no children
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

        if self.cache_controller.write_policy == "write_back":
            self.writing_check(write_back=True)
            for node in write_back_nodes:
                assert node.backuped
                self._evict_backuped(node)

        self.update_eviction_metrics(num_evicted, start_time)
        return EvictResult(num_tokens_evicted=num_evicted)

    def _evict_backuped(self, node: TreeNode):
        # evict a node already written to host
        num_evicted = self.cache_controller.evict_device(node.value)
        assert num_evicted > 0
        self.evictable_size_ -= num_evicted
        node.value = None
        self._update_leaf_status(node)
        self._update_host_leaf_status(node)
        # update leaf status for the parent because the node is evicted
        self._update_leaf_status(node.parent)
        return num_evicted

    def _evict_regular(self, node: TreeNode):
        # evict a node not initiated write to host
        self.cache_controller.mem_pool_device_allocator.free(node.value)
        num_evicted = len(node.value)
        self._delete_leaf(node)
        return num_evicted

    def evict_host(self, num_tokens: int):
        if getattr(self.authoritative_tree, "enabled", False):
            self.queue_authoritative_tree_op(
                "HOST_EVICT_REQUEST",
                num_tokens=num_tokens,
            )
        leaves = list(self.evictable_host_leaves)
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)
            if x == self.root_node:
                break
            # only evict the host value of evicted nodes
            if not x.evicted:
                continue

            # node is protected from eviction as it has ongoing prefetch or backup to storage
            if x.host_ref_counter > 0:
                continue

            num_evicted += self.cache_controller.evict_host(x.host_value)
            self._discard_authoritative_visibility(x)

            key = self.get_child_key_fn(x.key)
            v = x.parent.children.pop(key, None)
            assert v == x, f"parent does not have child key, {key}"
            if x in self.evictable_host_leaves:
                self.evictable_host_leaves.remove(x)
            self._update_host_leaf_status(x.parent)

            if len(x.parent.children) == 0 and x.parent.evicted:
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

    def load_back(
        self, node: TreeNode, mem_quota: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        # todo: more loading policies

        start_time = time.perf_counter()
        last_hit_node = node
        nodes_to_load = []
        while node.evicted:
            assert (
                self._node_backup_visible(node)
            ), "No backup available on evicted nodes, should not happen"
            nodes_to_load.insert(0, node)
            node = node.parent
        else:
            ancester_node = node

        # protect the ancestor nodes from eviction
        delta = self.inc_lock_ref(ancester_node)

        # load it all or not at all
        host_indices = torch.cat([n.host_value for n in nodes_to_load])
        if os.getenv("SGLANG_DEBUG_HICACHE_MATCH_CHAIN", "0") == "1":
            logger.warning(
                "[HiCacheMatchChain] load_back begin: last_hit_node=%s nodes_to_load=%s "
                "host_indices_len=%s mem_quota=%s delta=%s pp=%s cp=%s tp=%s",
                getattr(last_hit_node, "id", None),
                [getattr(n, "id", None) for n in nodes_to_load],
                len(host_indices),
                mem_quota,
                delta,
                self.pp_rank,
                self.attn_cp_rank,
                getattr(self.cache_controller, "tp_rank", None),
            )
        if len(host_indices) < self.load_back_threshold or (
            len(host_indices) > mem_quota + delta if mem_quota is not None else False
        ):
            # skip loading back if the total size is too small or exceeding the memory quota
            if os.getenv("SGLANG_DEBUG_HICACHE_MATCH_CHAIN", "0") == "1":
                logger.warning(
                    "[HiCacheMatchChain] load_back skipped: last_hit_node=%s "
                    "host_indices_len=%s threshold=%s mem_quota=%s delta=%s",
                    getattr(last_hit_node, "id", None),
                    len(host_indices),
                    self.load_back_threshold,
                    mem_quota,
                    delta,
                )
            self.dec_lock_ref(ancester_node)
            return None

        device_indices = self.cache_controller.load(
            host_indices=host_indices, node_id=last_hit_node.id
        )
        if device_indices is None:
            self.evict(EvictParams(num_tokens=len(host_indices)))
            device_indices = self.cache_controller.load(
                host_indices=host_indices, node_id=last_hit_node.id
            )
        self.dec_lock_ref(ancester_node)
        if device_indices is None:
            # no sufficient GPU memory to load back KV caches
            if os.getenv("SGLANG_DEBUG_HICACHE_MATCH_CHAIN", "0") == "1":
                logger.warning(
                    "[HiCacheMatchChain] load_back failed: last_hit_node=%s host_indices_len=%s",
                    getattr(last_hit_node, "id", None),
                    len(host_indices),
                )
            return None

        self.ongoing_load_back[last_hit_node.id] = last_hit_node
        offset = 0
        for node in nodes_to_load:
            node.value = device_indices[offset : offset + len(node.host_value)]
            offset += len(node.host_value)
        self.evictable_size_ += len(device_indices)
        self.inc_lock_ref(last_hit_node)
        if self.authoritative_tree.enabled:
            for loaded_node in nodes_to_load:
                self.authoritative_host_visible_node_ids.discard(loaded_node.id)
            self._clear_prefetch_ready_for_host_node(last_hit_node)

        if self.metrics_collector is not None:
            self.metrics_collector.observe_load_back_duration(
                time.perf_counter() - start_time
            )
            self.metrics_collector.increment_load_back_num_tokens(len(device_indices))

        if os.getenv("SGLANG_DEBUG_HICACHE_MATCH_CHAIN", "0") == "1":
            logger.warning(
                "[HiCacheMatchChain] load_back success: last_hit_node=%s device_indices_len=%s "
                "pp=%s cp=%s tp=%s",
                getattr(last_hit_node, "id", None),
                len(device_indices),
                self.pp_rank,
                self.attn_cp_rank,
                getattr(self.cache_controller, "tp_rank", None),
            )

        return device_indices

    def init_load_back(
        self,
        last_node: TreeNode,
        host_hit_length: int,
        mem_quota: Optional[int] = None,
    ):
        _ = host_hit_length  # unused, but kept for compatibility
        if last_node.evicted:
            loading_values = self.load_back(last_node, mem_quota)
            if loading_values is not None:
                logger.debug(
                    f"loading back {len(loading_values)} tokens for node {last_node.id}"
                )
                return loading_values, last_node

            while last_node.evicted:
                last_node = last_node.parent

        return (
            torch.empty((0,), dtype=torch.int64, device=self.device),
            last_node,
        )

    def ready_to_load_host_cache(self) -> int:
        """
        Notify the cache controller to start the KV cache loading.
        Return the consumer index for the schedule batch manager to track.
        """
        return self.cache_controller.start_loading()

    def is_load_ready(self, consumer_index: int) -> bool:
        if consumer_index < 0:
            return True
        return self.cache_controller.layer_done_counter.events[
            consumer_index
        ].finish_event.query()

    def check_hicache_events(self):
        self.writing_check()
        self.loading_check()
        if self.enable_storage:
            self.drain_storage_control_queues()
        # Keep authoritative visibility reconciliation inside the HiCache event
        # loop instead of introducing extra PP scheduler barriers.
        self.sync_authoritative_state()
        if self.authoritative_pending_backup_refs:
            self._promote_stable_backup_visibility()
        if self.authoritative_host_insert_rebuild_by_reqid:
            self._replay_pending_host_insert_blueprints()
        if self.prefetch_ready_results_by_reqid:
            self._maybe_commit_stable_prefetch_ready_summaries()
        self._maybe_log_authoritative_resolution_stats()
        if self.enable_storage_metrics:
            self.storage_metrics_collector.log_storage_metrics(
                self.cache_controller.storage_backend.get_stats()
            )

    def drain_storage_control_queues(self):
        """
        Combine prefetch revoke, backup ack, and host mem release checks
        to minimize TP synchronization and Python overhead.
        """
        cc = self.cache_controller

        qsizes = torch.tensor(
            [
                cc.prefetch_revoke_queue.qsize(),
                cc.ack_backup_queue.qsize(),
                cc.host_mem_release_queue.qsize(),
            ],
            dtype=torch.int,
        )
        self._all_reduce_attn_groups(qsizes, torch.distributed.ReduceOp.MIN)

        n_revoke, n_backup, n_release = map(int, qsizes.tolist())
        self._drain_storage_control_queues_impl(
            n_revoke=n_revoke,
            n_backup=n_backup,
            n_release=n_release,
            log_metrics=True,
        )

    # Timeout is linearly increasing with the number of pages
    def _prefetch_timeout_check_linear_func(self, operation: PrefetchOperation):
        # If hash_value has not been computed in timeout_base seconds, terminate it.
        return (
            time.monotonic() - operation.start_time
            > self.prefetch_timeout_base
            + len(operation.hash_value) * self.prefetch_timeout_per_page
        )

    def can_terminate_prefetch(self, operation: PrefetchOperation):
        can_terminate = True

        if self.prefetch_stop_policy == "best_effort":
            return can_terminate

        if len(operation.hash_value) == 0:
            completed = False
        else:
            completed = (
                operation.completed_tokens == len(operation.hash_value) * self.page_size
            )

        if self.prefetch_stop_policy == "wait_complete":
            if hasattr(operation, "has_ready_extra_pool_results"):
                can_terminate = (
                    completed and operation.has_ready_extra_pool_results()
                )
            else:
                can_terminate = completed
        elif self.prefetch_stop_policy == "timeout":
            can_terminate = completed or self.is_prefetch_timeout(operation)
        else:
            # unknown prefetch stop policy, just return True
            return True

        operation_terminated = operation.is_terminated()
        states = torch.tensor(
            [1 - int(can_terminate), int(operation_terminated)],
            dtype=torch.int,
        )
        self._all_reduce_attn_groups(states, torch.distributed.ReduceOp.MAX)
        can_terminate = states[0].item() == 0
        operation_terminated = states[1].item() == 1
        # the operation should be terminated if it is already terminated on any TP worker
        # or it meets the termination condition on all TP workers
        can_terminate = can_terminate or operation_terminated
        return can_terminate

    def check_prefetch_progress(self, req_or_id: Req | str) -> bool:
        req = req_or_id if not isinstance(req_or_id, str) else None
        req_id = req.rid if req is not None else req_or_id
        if req_id not in self.ongoing_prefetch:
            # there is no ongoing prefetch for this request or it has been revoked
            return True

        # todo: more policies for prefetch progress such as timeout
        # the current policy is to prefetch with best effort and terminate when queuing is over
        last_host_node, token_ids, host_indices, operation = self.ongoing_prefetch[
            req_id
        ]

        if operation.host_indices is None:
            # prefetch has not been issued due to insufficient host memory
            return True

        if not self.can_terminate_prefetch(operation):
            return False

        completed_tokens, hash_value = self.cache_controller.terminate_prefetch(
            operation
        )
        logger.debug(f"Prefetch {req_id} completed with {completed_tokens} tokens")

        min_completed_tokens = completed_tokens
        if self.use_nsa_pool_controller:
            min_completed_tokens = (
                self.cache_controller.get_usable_prefetch_token_count(operation)
            )
        # Synchronize workers before mutating host cache tree state.
        completed_tokens_tensor = torch.tensor(min_completed_tokens, dtype=torch.int)
        self._all_reduce_attn_groups(
            completed_tokens_tensor, torch.distributed.ReduceOp.MIN
        )
        min_completed_tokens = completed_tokens_tensor.item()
        fetched_token_ids = token_ids[:min_completed_tokens]
        written_indices = host_indices[:min_completed_tokens]
        inserted_nodes: list[TreeNode] = []
        matched_length = self._insert_helper_host(
            last_host_node,
            RadixKey(
                token_ids=fetched_token_ids, extra_key=last_host_node.key.extra_key
            ),
            written_indices,
            hash_value[: min_completed_tokens // self.page_size],
            inserted_nodes=inserted_nodes,
        )

        self.cache_controller.mem_pool_host.free(host_indices[:matched_length])
        self.cache_controller.append_host_mem_release(
            host_indices[min_completed_tokens:completed_tokens]
        )
        last_host_node.release_host()
        del self.ongoing_prefetch[req_id]
        self.cache_controller.prefetch_tokens_occupied -= len(token_ids)

        # Track tokens actually loaded from storage for this request (L3 hits)
        loaded_from_storage = min_completed_tokens - matched_length
        ready_storage_hit_length = (
            min_completed_tokens if self.authoritative_tree.enabled else loaded_from_storage
        )
        self.queue_authoritative_tree_op(
            "HOST_INSERT_FROM_STORAGE",
            **self._build_host_insert_from_storage_payload(
                req_id=req_id,
                anchor_node=last_host_node,
                fetched_token_ids=fetched_token_ids,
                fetched_hash_value=hash_value[: min_completed_tokens // self.page_size],
                inserted_nodes=inserted_nodes,
                loaded_from_storage=loaded_from_storage,
                matched_length=matched_length,
                committed_tokens=min_completed_tokens,
            ),
        )
        self.prefetch_loaded_tokens_by_reqid[req_id] = ready_storage_hit_length
        if req is not None:
            if self.authoritative_tree.enabled:
                ready_result = self._build_authoritative_ready_result_from_prefetch_finalize(
                    req,
                    anchor_node=last_host_node,
                    fetched_token_ids=fetched_token_ids,
                    fetched_hash_value=hash_value[: min_completed_tokens // self.page_size],
                    committed_tokens=min_completed_tokens,
                    storage_hit_length=ready_storage_hit_length,
                )
            else:
                ready_result = self._build_latched_prefetch_ready_result(
                    req, loaded_from_storage
                )
            self.queue_authoritative_tree_op(
                "PREFETCH_READY_SUMMARY",
                req_id=req_id,
                prefix_len=len(ready_result.match_result.device_indices)
                + ready_result.match_result.host_hit_length
                + ready_result.storage_hit_length,
                host_hit_length=ready_result.match_result.host_hit_length,
                storage_hit_length=ready_result.storage_hit_length,
                input_len=ready_result.input_len,
                last_host_node_ref=self._make_authoritative_node_ref(
                    ready_result.match_result.last_host_node
                ),
            )
            self.prefetch_ready_results_by_reqid[req_id] = ready_result
            if os.getenv("SGLANG_DEBUG_HICACHE_MATCH_CHAIN", "0") == "1":
                logger.warning(
                    "[HiCacheMatchChain] latched ready result: rid=%s "
                    "prefix_len=%s host_hit=%s storage_hit=%s "
                    "last_device_node=%s last_host_node=%s pp=%s cp=%s tp=%s",
                    req_id,
                    len(ready_result.match_result.device_indices),
                    ready_result.match_result.host_hit_length,
                    ready_result.storage_hit_length,
                    getattr(ready_result.match_result.last_device_node, "id", None),
                    getattr(ready_result.match_result.last_host_node, "id", None),
                    self.pp_rank,
                    self.attn_cp_rank,
                    getattr(self.cache_controller, "tp_rank", None),
                )

        if os.getenv("SGLANG_DEBUG_HICACHE_MATCH_CHAIN", "0") == "1":
            logger.warning(
                "[HiCacheMatchChain] prefetch finalize: rid=%s completed_tokens=%s "
                "min_completed_tokens=%s matched_length=%s loaded_from_storage=%s "
                "requested_tokens=%s hash_pages=%s auth_stats=%s missing_segments=%s "
                "pending_backups=%s pp=%s cp=%s tp=%s",
                req_id,
                completed_tokens,
                min_completed_tokens,
                matched_length,
                loaded_from_storage,
                len(token_ids),
                len(hash_value),
                self._format_authoritative_resolution_stats(),
                self._format_host_insert_missing_segments(req_id),
                self._format_pending_backup_nodes(),
                self.pp_rank,
                self.attn_cp_rank,
                getattr(self.cache_controller, "tp_rank", None),
            )

        if self.enable_storage_metrics:
            self.storage_metrics_collector.log_prefetched_tokens(loaded_from_storage)

        return True

    def terminate_prefetch(self, req_id: str):
        if req_id not in self.ongoing_prefetch:
            return

        _, _, _, operation = self.ongoing_prefetch[req_id]
        if operation.host_indices is None:
            return
        operation.mark_terminate()

    def pop_prefetch_loaded_tokens(self, req_id: str) -> int:
        """
        Pop and return the number of tokens loaded from storage for a request.
        Returns 0 if no prefetch was done or was revoked.
        This should be called after check_prefetch_progress() returns True.
        """
        local_loaded = self.prefetch_loaded_tokens_by_reqid.pop(req_id, 0)
        return self.authoritative_prefetch_loaded_tokens_by_reqid.pop(
            req_id, local_loaded
        )

    def pop_prefetch_ready_result(
        self, req_id: str, req: Optional[Req] = None
    ) -> Optional[LatchedPrefetchReadyResult]:
        if self.pp_device_only_match_fallback:
            self.prefetch_loaded_tokens_by_reqid.pop(req_id, None)
            self.prefetch_ready_results_by_reqid.pop(req_id, None)
            self.authoritative_prefetch_ready_by_reqid.pop(req_id, None)
            return None
        self.prefetch_loaded_tokens_by_reqid.pop(req_id, None)
        ready_result: Optional[LatchedPrefetchReadyResult] = None
        if self.authoritative_tree.enabled and req is not None:
            # Keep the deterministic prefetch-finalize result available for every
            # PP retry of the same request. Popping here can force only one rank
            # to rebuild from local live match, which reintroduces PP divergence.
            ready_result = self.prefetch_ready_results_by_reqid.get(req_id, None)
            ready_result = self._clamp_ready_result_to_authoritative_summary(
                req_id, ready_result
            )
            if ready_result is None:
                ready_result = self.build_authoritative_prefetch_ready_result(req)
        else:
            ready_result = self.prefetch_ready_results_by_reqid.pop(req_id, None)
            ready_result = self._clamp_ready_result_to_authoritative_summary(
                req_id, ready_result
            )
        if (
            ready_result is not None
            and os.getenv("SGLANG_DEBUG_HICACHE_MATCH_CHAIN", "0") == "1"
        ):
            logger.warning(
                "[HiCacheMatchChain] consume latched ready result: rid=%s "
                "prefix_len=%s host_hit=%s storage_hit=%s "
                "last_device_node=%s last_host_node=%s auth_stats=%s missing_segments=%s "
                "pending_backups=%s pp=%s cp=%s tp=%s",
                req_id,
                len(ready_result.match_result.device_indices),
                ready_result.match_result.host_hit_length,
                ready_result.storage_hit_length,
                getattr(ready_result.match_result.last_device_node, "id", None),
                getattr(ready_result.match_result.last_host_node, "id", None),
                self._format_authoritative_resolution_stats(),
                self._format_host_insert_missing_segments(req_id),
                self._format_pending_backup_nodes(),
                self.pp_rank,
                self.attn_cp_rank,
                getattr(self.cache_controller, "tp_rank", None),
            )
        return ready_result

    def is_prefetch_ready_result_usable(
        self, ready_result: Optional[LatchedPrefetchReadyResult]
    ) -> bool:
        if ready_result is None:
            return False
        last_host_node = getattr(ready_result.match_result, "last_host_node", None)
        if last_host_node is None:
            return True
        if not getattr(last_host_node, "evicted", False):
            return False
        if not self._node_backup_visible(last_host_node):
            return False
        return True

    def get_authoritative_prefetch_ready_summary(
        self, req_id: str
    ) -> Optional[AuthoritativePrefetchReadySummary]:
        return self.authoritative_prefetch_ready_by_reqid.get(req_id)

    def build_authoritative_prefetch_ready_result(
        self, req: Req
    ) -> Optional[LatchedPrefetchReadyResult]:
        """Reconcile a local live match with the PP-authoritative ready summary.

        The full authoritative tree replay is still being migrated. Until then we
        consume the committed ready summary and conservatively clamp the local
        match result so every PP stage shapes the next batch against the same
        ready-prefix upper bound.
        """
        summary = self.authoritative_prefetch_ready_by_reqid.get(req.rid)
        if summary is None:
            return None
        if not self._is_authoritative_ready_stable(req.rid):
            self._record_authoritative_resolution("prefetch_ready.skip_unstable")
            return None
        req_input_len = len(req.fill_ids)
        if summary.input_len is not None and summary.input_len != req_input_len:
            self.authoritative_prefetch_ready_by_reqid.pop(req.rid, None)
            self._record_authoritative_resolution("prefetch_ready.skip_stale")
            return None

        local_ready = self._build_latched_prefetch_ready_result(
            req, summary.storage_hit_length
        )
        return LatchedPrefetchReadyResult(
            match_result=self._clamp_match_result_to_authoritative_summary(
                req.rid, local_ready.match_result, input_len=local_ready.input_len
            ),
            storage_hit_length=summary.storage_hit_length,
            input_len=local_ready.input_len,
        )

    def was_prefetch_skipped(self, req_id: str) -> bool:
        """Return True if prefetch was skipped for this request
        (host alloc failure, below threshold, or rate-limited)."""
        return req_id in self.prefetch_skipped_rids

    def match_prefix(self, params: MatchPrefixParams):
        key = params.key
        req_id = params.req.rid if params.req is not None else None
        original_key_len = len(key)
        empty_value = torch.empty((0,), dtype=torch.int64, device=self.device)
        key, _ = self.maybe_bigram_convert(key)
        if self.disable or len(key) == 0:
            return MatchResult(
                device_indices=empty_value,
                last_device_node=self.root_node,
                last_host_node=self.root_node,
                host_hit_length=0,
            )

        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]
        else:
            page_aligned_len = len(key)

        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            value = torch.cat(value)
        else:
            value = empty_value

        host_hit_length = 0
        last_host_node = last_node
        while last_node.evicted:
            if not self._node_backup_visible(last_node):
                break
            host_hit_length += len(last_node.host_value)
            last_node = last_node.parent
        last_host_node = self._find_last_visible_host_ancestor(last_host_node)

        match_result = MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_host_node,
            host_hit_length=host_hit_length,
        )
        pre_clamp_device_hit = len(match_result.device_indices)
        pre_clamp_host_hit = match_result.host_hit_length
        match_result = self._clamp_match_result_to_authoritative_summary(
            req_id,
            match_result,
            input_len=(len(params.req.fill_ids) if params.req is not None else None),
        )
        if self.pp_device_only_match_fallback:
            if match_result.host_hit_length > 0:
                logger.warning_once(
                    "PP HiCache authoritative replay is disabled; "
                    "falling back to device-only prefix matching to keep PP "
                    "stages shape-consistent."
                )
            match_result = MatchResult(
                device_indices=match_result.device_indices,
                last_device_node=match_result.last_device_node,
                last_host_node=match_result.last_device_node,
                host_hit_length=0,
                mamba_branching_seqlen=match_result.mamba_branching_seqlen,
            )

        if (
            os.getenv("SGLANG_DEBUG_HICACHE_MATCH", "0") == "1"
            and self.pp_size > 1
        ):
            tp_rank = getattr(self.cache_controller, "tp_rank", None)
            logger.warning(
                "[HiCacheMatch] rid=%s key_len=%s aligned_len=%s device_hit=%s "
                "host_hit=%s total_cached=%s page_size=%s pp=%s cp=%s tp=%s "
                "last_device_node=%s last_host_node=%s auth_stats=%s missing_segments=%s "
                "pending_backups=%s gates=%s clamp_delta=%s",
                req_id,
                original_key_len,
                page_aligned_len,
                len(match_result.device_indices),
                match_result.host_hit_length,
                len(match_result.device_indices) + match_result.host_hit_length,
                self.page_size,
                self.pp_rank,
                self.attn_cp_rank,
                tp_rank,
                getattr(match_result.last_device_node, "id", None),
                getattr(match_result.last_host_node, "id", None),
                self._format_authoritative_resolution_stats(),
                self._format_host_insert_missing_segments(req_id),
                self._format_pending_backup_nodes(),
                self._format_match_authoritative_gates(
                    req_id,
                    pre_clamp_device_hit=pre_clamp_device_hit,
                    pre_clamp_host_hit=pre_clamp_host_hit,
                    match_result=match_result,
                ),
                self._format_match_clamp_delta(
                    pre_clamp_device_hit=pre_clamp_device_hit,
                    pre_clamp_host_hit=pre_clamp_host_hit,
                    match_result=match_result,
                ),
            )

        return match_result

    def prefetch_from_storage(
        self,
        req_id: str,
        last_host_node: TreeNode,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
    ):
        # align the number of fetching tokens to the page size
        prefetch_length = len(new_input_tokens) - (
            len(new_input_tokens) % self.page_size
        )
        new_input_tokens = new_input_tokens[:prefetch_length]
        if (
            not self.enable_storage
            or prefetch_length < self.prefetch_threshold
            or self.cache_controller.prefetch_rate_limited()
        ):
            self.prefetch_skipped_rids.add(req_id)
            return

        last_host_node.protect_host()
        host_indices = self.cache_controller.mem_pool_host.alloc(prefetch_length)
        if host_indices is None:
            self.evict_host(prefetch_length)
            host_indices = self.cache_controller.mem_pool_host.alloc(prefetch_length)
        if host_indices is None:
            last_host_node.release_host()
            self.prefetch_skipped_rids.add(req_id)
            return
        self.prefetch_skipped_rids.discard(req_id)
        prefetch_transfers = self.nsa_prefetch_transfers()
        if prefetch_transfers:
            operation = self.cache_controller.prefetch(
                req_id,
                host_indices,
                new_input_tokens,
                last_hash,
                prefix_keys,
                extra_pools=prefetch_transfers,
            )
        else:
            operation = self.cache_controller.prefetch(
                req_id,
                host_indices,
                new_input_tokens,
                last_hash,
                prefix_keys,
            )
        self.ongoing_prefetch[req_id] = (
            last_host_node,
            new_input_tokens,
            host_indices,
            operation,
        )
        self.cache_controller.prefetch_tokens_occupied += len(new_input_tokens)

    def nsa_backup_transfers(self) -> Optional[list[PoolTransfer]]:
        if not self.use_nsa_pool_controller:
            return None
        return [
            PoolTransfer(
                name=PoolName.NSA,
                hit_policy=PoolHitPolicy.ALL_PAGES,
                use_anchor_host_indices=True,
                use_anchor_device_indices=True,
            )
        ]

    def nsa_archive_transfers(self, node: TreeNode) -> Optional[list[PoolTransfer]]:
        if not self.use_nsa_pool_controller or not node.hash_value:
            return None
        return [
            PoolTransfer(
                name=PoolName.NSA,
                keys=node.hash_value,
                hit_policy=PoolHitPolicy.ALL_PAGES,
                use_anchor_host_indices=True,
                use_anchor_device_indices=True,
            )
        ]

    def nsa_prefetch_transfers(self) -> Optional[list[PoolTransfer]]:
        if not self.use_nsa_pool_controller:
            return None
        return [
            PoolTransfer(
                name=PoolName.NSA,
                hit_policy=PoolHitPolicy.ALL_PAGES,
                use_anchor_host_indices=True,
                use_anchor_device_indices=True,
            )
        ]

    def nsa_restore_transfers(self) -> Optional[list[PoolTransfer]]:
        if not self.use_nsa_pool_controller:
            return None
        return [
            PoolTransfer(
                name=PoolName.NSA,
                hit_policy=PoolHitPolicy.ALL_PAGES,
                use_anchor_host_indices=True,
                use_anchor_device_indices=True,
            )
        ]

    def _insert_helper_host(
        self,
        node: TreeNode,
        key: RadixKey,
        host_value,
        hash_value,
        inserted_nodes: Optional[list[TreeNode]] = None,
    ):
        node.last_access_time = time.monotonic()
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)

        matched_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(node.key, key)
            key = key[prefix_len:]
            host_value = host_value[prefix_len:]
            hash_value = hash_value[prefix_len // self.page_size :]
            matched_length += prefix_len

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                if (
                    inserted_nodes is not None
                    and new_node.evicted
                    and new_node.backuped
                ):
                    inserted_nodes.append(new_node)
                node = new_node

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode(priority=node.priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = None
            new_node.host_value = host_value.clone()
            new_node.hash_value = hash_value
            node.children[child_key] = new_node
            self._refresh_authoritative_node_index(new_node)
            if inserted_nodes is not None:
                inserted_nodes.append(new_node)
            self._update_host_leaf_status(new_node)
            self._update_leaf_status(node)
            self._update_host_leaf_status(node)

        return matched_length

    def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
        node.last_access_time = time.monotonic()
        child_key = self.get_child_key_fn(key)
        value = []

        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            if child.evicted and not self._node_backup_visible(child):
                break
            child.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                if not new_node.evicted:
                    value.append(new_node.value)
                node = new_node
                break
            else:
                if not child.evicted:
                    value.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = self.get_child_key_fn(key)

        return value, node

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
        # child node split into new_node -> child
        child_backup_visible = getattr(child, "id", None) in getattr(
            self, "authoritative_backuped_node_ids", set()
        )
        child_host_visible = getattr(child, "id", None) in getattr(
            self, "authoritative_host_visible_node_ids", set()
        )
        new_node = TreeNode(priority=child.priority)
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.hit_count = child.hit_count

        # split value and host value if exists
        if child.evicted:
            new_node.value = None
        else:
            new_node.value = child.value[:split_len].clone()
            child.value = child.value[split_len:].clone()
        if child.backuped:
            new_node.host_value = child.host_value[:split_len].clone()
            child.host_value = child.host_value[split_len:].clone()

        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )
        child.parent = new_node
        child.key = child.key[split_len:]
        new_node.parent.children[self.get_child_key_fn(key)] = new_node
        self._refresh_authoritative_node_index(new_node)
        self._refresh_authoritative_node_index(child)
        if child_backup_visible:
            self.authoritative_backuped_node_ids.add(new_node.id)
        if child_host_visible:
            self.authoritative_host_visible_node_ids.add(new_node.id)
        return new_node

    def insert(self, params: InsertParams) -> InsertResult:
        key = params.key
        value = params.value
        chunked = params.chunked
        priority = params.priority

        if priority is None:
            priority = 0
        key, value = self.maybe_bigram_convert(key, value)

        if len(key) == 0:
            return InsertResult(prefix_len=0)

        if self.is_eagle and value is not None:
            # Make sure the value len equal to the EAGLE bigram key len
            value = value[: len(key)]

        node = self.root_node
        child_key = self.get_child_key_fn(key)
        total_prefix_length = 0

        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            node.priority = max(node.priority, priority)
            prefix_len = self.key_match_fn(node.key, key)

            if prefix_len == len(node.key):
                if node.evicted:
                    # change the reference if the node is evicted
                    # this often happens in the case of KV cache recomputation
                    node.value = value[:prefix_len]
                    self.evictable_size_ += len(node.value)
                    self._update_leaf_status(node)
                    self._update_host_leaf_status(node)
                    # update parent status as a new leaf is added into device
                    self._update_leaf_status(node.parent)
                else:
                    self._inc_hit_count(node, chunked)
                    total_prefix_length += prefix_len
            else:
                # partial match, split the node
                new_node = self._split_node(node.key, node, prefix_len)
                # shared-prefix node should also reflect max priority
                new_node.priority = max(new_node.priority, priority)
                if new_node.evicted:
                    new_node.value = value[:prefix_len].clone()
                    self.evictable_size_ += len(new_node.value)
                    self._update_leaf_status(new_node)
                    self._update_host_leaf_status(new_node)
                    # update parent status as a new leaf is added into device
                    self._update_leaf_status(new_node.parent)
                else:
                    self._inc_hit_count(new_node, chunked)
                    total_prefix_length += prefix_len
                node = new_node

            key = key[prefix_len:]
            value = value[prefix_len:]

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode(priority=priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = value.clone()
            node.children[child_key] = new_node
            self.evictable_size_ += len(value)
            self._update_leaf_status(node)
            self._update_leaf_status(new_node)

            # Compute hash_value if storage is enabled
            if self.enable_storage:
                new_node.hash_value = compute_node_hash_values(new_node, self.page_size)
            self._refresh_authoritative_node_index(new_node)

            if self.cache_controller.write_policy != "write_back":
                self._inc_hit_count(new_node, chunked)
        return InsertResult(prefix_len=total_prefix_length)

    def release_aborted_request(self, rid: str):
        # Clean up storage hit tracking for aborted request
        self.prefetch_loaded_tokens_by_reqid.pop(rid, None)
        self.prefetch_ready_results_by_reqid.pop(rid, None)
        self.authoritative_prefetch_ready_by_reqid.pop(rid, None)
        self.authoritative_prefetch_loaded_tokens_by_reqid.pop(rid, None)
        self._clear_host_insert_rebuild_blueprint(rid)
        self.prefetch_skipped_rids.discard(rid)

        if rid not in self.ongoing_prefetch:
            return

        last_host_node, token_ids, host_indices, operation = self.ongoing_prefetch[rid]
        if operation.host_indices is None:
            return

        completed_tokens, _ = self.cache_controller.terminate_prefetch(operation)
        self._barrier_attn_groups()
        if os.getenv("SGLANG_DEBUG_HICACHE_MATCH_CHAIN", "0") == "1":
            logger.warning(
                "[HiCacheMatchChain] abort cleanup: rid=%s completed_tokens=%s "
                "host_node=%s token_count=%s pp=%s cp=%s tp=%s",
                rid,
                completed_tokens,
                getattr(last_host_node, "id", None),
                len(token_ids),
                self.pp_rank,
                self.attn_cp_rank,
                getattr(self.cache_controller, "tp_rank", None),
            )
        last_host_node.release_host()
        del self.ongoing_prefetch[rid]
        self.cache_controller.append_host_mem_release(host_indices[:completed_tokens])
        self.cache_controller.prefetch_tokens_occupied -= len(token_ids)
