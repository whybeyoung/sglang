import unittest
import sys
import types

if "IPython" not in sys.modules:
    ipython_module = types.ModuleType("IPython")
    ipython_display = types.ModuleType("IPython.display")
    ipython_display.HTML = lambda *args, **kwargs: None
    ipython_display.display = lambda *args, **kwargs: None
    ipython_module.display = ipython_display
    sys.modules["IPython"] = ipython_module
    sys.modules["IPython.display"] = ipython_display

from sglang.srt.mem_cache.hicache_authoritative import (
    AuthoritativePrefetchReadySummary,
    AuthoritativeTreeCoordinator,
)
from sglang.srt.mem_cache.base_prefix_cache import MatchResult
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache, LatchedPrefetchReadyResult
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.managers.schedule_batch import Req
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase
import torch

register_cuda_ci(est_time=5, suite="stage-b-test-small-1-gpu")


class TestHiCacheAuthoritative(CustomTestCase):
    def test_single_rank_sync_applies_local_ops_in_order(self):
        coordinator = AuthoritativeTreeCoordinator(enabled=True)
        applied = []

        coordinator.queue_local("HOST_BACKUP_COMMIT", {"node_id": 7})
        coordinator.queue_local("DEVICE_EVICT_REQUEST", {"num_tokens": 64})

        committed_seq = coordinator.sync(lambda op: applied.append((op.op_seq, op.op_type)))

        self.assertEqual(committed_seq, 2)
        self.assertEqual(
            applied,
            [(1, "HOST_BACKUP_COMMIT"), (2, "DEVICE_EVICT_REQUEST")],
        )

    def test_authoritative_ready_result_clamps_local_match(self):
        req = Req(
            rid="rid-1",
            origin_input_text="",
            origin_input_ids=[1, 2, 3, 4],
            sampling_params=None,
        )
        req.fill_ids = [1, 2, 3, 4]
        req.output_ids = []
        req.return_logprob = False
        req.logprob_start_len = -1

        class DummyTreeCache:
            def supports_mamba(self):
                return False

            def pop_prefetch_ready_result(self, _rid, req=None):
                assert req is not None
                return LatchedPrefetchReadyResult(
                    match_result=MatchResult(
                        device_indices=torch.arange(2, dtype=torch.int64),
                        last_device_node=None,
                        last_host_node=None,
                        host_hit_length=1,
                    ),
                    storage_hit_length=1,
                )

            def match_prefix(self, params):
                raise AssertionError("authoritative ready should win before live match")

        req.init_next_round_input(DummyTreeCache())

        self.assertEqual(len(req.prefix_indices), 2)
        self.assertEqual(req.host_hit_length, 1)
        self.assertEqual(req.storage_hit_length, 1)

    def test_pop_prefetch_ready_result_falls_back_to_authoritative_ready(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.prefetch_loaded_tokens_by_reqid = {}
        cache.prefetch_ready_results_by_reqid = {}
        cache.authoritative_prefetch_ready_by_reqid = {}
        cache.build_authoritative_prefetch_ready_result = lambda req: LatchedPrefetchReadyResult(
            match_result=MatchResult(
                device_indices=torch.arange(2, dtype=torch.int64),
                last_device_node=None,
                last_host_node=None,
                host_hit_length=1,
            ),
            storage_hit_length=4,
            input_len=len(req.fill_ids),
        )
        cache._clamp_ready_result_to_authoritative_summary = (
            lambda req_id, ready_result: ready_result
        )

        req = types.SimpleNamespace(rid="rid-pop", fill_ids=[1, 2, 3, 4])

        ready_result = cache.pop_prefetch_ready_result("rid-pop", req=req)

        self.assertIsNotNone(ready_result)
        self.assertEqual(len(ready_result.match_result.device_indices), 2)
        self.assertEqual(ready_result.storage_hit_length, 4)

    def test_hicache_clamps_match_result_with_authoritative_summary(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.authoritative_prefetch_ready_by_reqid = {
            "rid-2": AuthoritativePrefetchReadySummary(
                req_id="rid-2",
                prefix_len=2,
                host_hit_length=1,
                storage_hit_length=3,
            )
        }
        local_match = MatchResult(
            device_indices=torch.arange(4, dtype=torch.int64),
            last_device_node=None,
            last_host_node=None,
            host_hit_length=2,
        )

        clamped = cache._clamp_match_result_to_authoritative_summary(
            "rid-2", local_match
        )

        self.assertEqual(len(clamped.device_indices), 2)
        self.assertEqual(clamped.host_hit_length, 1)

    def test_is_authoritative_ready_stable_requires_no_pending_host_insert_state(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.authoritative_host_insert_rebuild_by_reqid = {"rid-s": {}}
        cache.authoritative_host_insert_skeleton_by_reqid = {}
        cache.authoritative_host_insert_missing_by_reqid = {}
        cache.authoritative_pending_backup_refs = {}

        self.assertFalse(cache._is_authoritative_ready_stable("rid-s"))

    def test_is_authoritative_ready_stable_ignores_irrelevant_pending_backup_refs(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.root_node = object()
        cache.authoritative_host_insert_rebuild_by_reqid = {}
        cache.authoritative_host_insert_skeleton_by_reqid = {}
        cache.authoritative_host_insert_missing_by_reqid = {}
        cache.authoritative_pending_backup_refs = {
            "hash-44": {"node_id": 44, "last_hash": "hash-44"}
        }
        cache.prefetch_ready_results_by_reqid = {
            "rid-s": LatchedPrefetchReadyResult(
                match_result=MatchResult(
                    device_indices=torch.empty((0,), dtype=torch.int64),
                    last_device_node=cache.root_node,
                    last_host_node=type(
                        "Node", (), {"id": 50, "parent": cache.root_node}
                    )(),
                    host_hit_length=1,
                ),
                storage_hit_length=0,
                input_len=1,
            )
        }
        pending = type("Node", (), {"id": 44, "parent": cache.root_node})()
        cache._resolve_authoritative_node_ref = (
            lambda node_ref=None, node_id=None, last_hash=None: pending
        )
        cache._is_ancestor_node = HiRadixCache._is_ancestor_node.__get__(cache, HiRadixCache)

        self.assertTrue(cache._is_authoritative_ready_stable("rid-s"))

    def test_is_authoritative_ready_stable_blocks_relevant_pending_backup(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        root = object()
        cache.root_node = root
        cache.authoritative_host_insert_rebuild_by_reqid = {}
        cache.authoritative_host_insert_skeleton_by_reqid = {}
        cache.authoritative_host_insert_missing_by_reqid = {}
        cache.authoritative_pending_backup_refs = {
            "hash-44": {"node_id": 44, "last_hash": "hash-44"}
        }

        pending = type("Node", (), {"id": 44, "parent": root})()
        host = type("Node", (), {"id": 55, "parent": pending})()
        cache.prefetch_ready_results_by_reqid = {
            "rid-s": LatchedPrefetchReadyResult(
                match_result=MatchResult(
                    device_indices=torch.empty((0,), dtype=torch.int64),
                    last_device_node=root,
                    last_host_node=host,
                    host_hit_length=1,
                ),
                storage_hit_length=0,
                input_len=1,
            )
        }
        cache._resolve_authoritative_node_ref = (
            lambda node_ref=None, node_id=None, last_hash=None: pending
        )

        self.assertFalse(cache._is_authoritative_ready_stable("rid-s"))

    def test_is_authoritative_ready_stable_uses_authoritative_summary_host_ref(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        root = object()
        cache.root_node = root
        cache.authoritative_host_insert_rebuild_by_reqid = {}
        cache.authoritative_host_insert_skeleton_by_reqid = {}
        cache.authoritative_host_insert_missing_by_reqid = {}
        cache.authoritative_pending_backup_refs = {
            "hash-44": {"node_id": 44, "last_hash": "hash-44"}
        }
        pending = type("Node", (), {"id": 44, "parent": root})()
        host = type("Node", (), {"id": 55, "parent": pending})()
        cache.authoritative_prefetch_ready_by_reqid = {
            "rid-s": AuthoritativePrefetchReadySummary(
                req_id="rid-s",
                prefix_len=0,
                host_hit_length=1,
                storage_hit_length=0,
                last_host_node_ref={"node_id": 55, "last_hash": "hash-55"},
            )
        }
        cache.prefetch_ready_results_by_reqid = {}

        def resolve(node_ref=None, node_id=None, last_hash=None):
            ref = node_ref or {"node_id": node_id, "last_hash": last_hash}
            if ref.get("last_hash") == "hash-44":
                return pending
            if ref.get("last_hash") == "hash-55":
                return host
            return None

        cache._resolve_authoritative_node_ref = resolve

        self.assertFalse(cache._is_authoritative_ready_stable("rid-s"))

    def test_hicache_prefetch_loaded_tokens_use_authoritative_value(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.prefetch_loaded_tokens_by_reqid = {"rid-3": 5}
        cache.authoritative_prefetch_loaded_tokens_by_reqid = {"rid-3": 3}

        loaded = cache.pop_prefetch_loaded_tokens("rid-3")

        self.assertEqual(loaded, 3)
        self.assertNotIn("rid-3", cache.prefetch_loaded_tokens_by_reqid)
        self.assertNotIn("rid-3", cache.authoritative_prefetch_loaded_tokens_by_reqid)

    def test_host_insert_from_storage_marks_nodes_visible(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.authoritative_prefetch_loaded_tokens_by_reqid = {}
        cache.authoritative_pending_backup_node_ids = set()
        cache.authoritative_backuped_node_ids = set()
        cache.authoritative_host_visible_node_ids = set()

        class Op:
            op_type = "HOST_INSERT_FROM_STORAGE"
            payload = {"req_id": "rid-4", "loaded_from_storage": 4, "node_ids": [11, 12]}
            op_seq = 1

        cache.apply_authoritative_tree_op(Op())

        self.assertEqual(
            cache.authoritative_prefetch_loaded_tokens_by_reqid["rid-4"], 4
        )
        self.assertEqual(cache.authoritative_backuped_node_ids, set())
        self.assertEqual(cache.authoritative_host_visible_node_ids, {11, 12})

    def test_host_backup_commit_stages_pending_visibility_first(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.authoritative_resolution_stats = {}
        cache.authoritative_pending_backup_node_ids = set()
        cache.authoritative_backuped_node_ids = set()
        cache.authoritative_host_visible_node_ids = set()
        cache._resolve_authoritative_node_ref = lambda *args, **kwargs: types.SimpleNamespace(id=21)
        cache.pp_rank = 0
        cache.attn_cp_rank = 0
        cache.cache_controller = types.SimpleNamespace(tp_rank=0)

        class Op:
            op_type = "HOST_BACKUP_COMMIT"
            payload = {"node_id": 21, "last_hash": "h21"}
            op_seq = 1

        cache.apply_authoritative_tree_op(Op())

        self.assertEqual(cache.authoritative_pending_backup_node_ids, {21})
        self.assertEqual(cache.authoritative_backuped_node_ids, set())
        self.assertEqual(cache.authoritative_host_visible_node_ids, set())

    def test_resolve_authoritative_node_ref_falls_back_to_last_hash(self):
        cache = HiRadixCache.__new__(HiRadixCache)

        class Node:
            def __init__(self, node_id, last_hash):
                self.id = node_id
                self._last_hash = last_hash

            def get_last_hash_value(self):
                return self._last_hash

        target = Node(21, "hash-21")
        cache._iter_nodes = lambda: iter([target])
        cache._find_node_by_id = lambda node_id: None

        resolved = cache._resolve_authoritative_node_ref(
            node_ref={"node_id": 999, "last_hash": "hash-21"}
        )

        self.assertIs(resolved, target)

    def test_node_backup_visible_checks_both_authoritative_sets(self):
        cache = HiRadixCache.__new__(HiRadixCache)

        class Tree:
            enabled = True

        class Node:
            def __init__(self, node_id):
                self.id = node_id
                self.host_value = [1]

            @property
            def backuped(self):
                return True

        cache.authoritative_tree = Tree()
        cache.authoritative_pending_backup_node_ids = {30}
        cache.authoritative_backuped_node_ids = {31}
        cache.authoritative_host_visible_node_ids = {32}

        self.assertFalse(cache._node_backup_visible(Node(30)))
        self.assertTrue(cache._node_backup_visible(Node(31)))
        self.assertTrue(cache._node_backup_visible(Node(32)))
        self.assertFalse(cache._node_backup_visible(Node(33)))

    def test_make_authoritative_node_ref_carries_last_hash(self):
        cache = HiRadixCache.__new__(HiRadixCache)

        class Node:
            id = 41

            def get_last_hash_value(self):
                return "hash-41"

        ref = cache._make_authoritative_node_ref(Node())

        self.assertEqual(ref["node_id"], 41)
        self.assertEqual(ref["last_hash"], "hash-41")

    def test_promote_stable_backup_visibility_moves_pending_node_to_visible(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.authoritative_resolution_stats = {}
        cache.authoritative_pending_backup_refs = {
            "hash-44": {"node_id": 44, "last_hash": "hash-44"}
        }
        cache.authoritative_pending_backup_node_ids = {44}
        cache.authoritative_pending_backup_reasons = {"hash-44": "await_stable"}
        cache.authoritative_backuped_node_ids = set()
        cache.authoritative_host_visible_node_ids = set()

        class Node:
            id = 44
            host_value = [1]
            host_ref_counter = 0

            @property
            def backuped(self):
                return True

            def get_last_hash_value(self):
                return "hash-44"

        cache._resolve_authoritative_node_ref = (
            lambda node_ref=None, node_id=None, last_hash=None: Node()
            if (node_ref and node_ref.get("last_hash") == "hash-44")
            or node_id == 44
            or last_hash == "hash-44"
            else None
        )

        cache._promote_stable_backup_visibility()

        self.assertEqual(cache.authoritative_pending_backup_refs, {})
        self.assertEqual(cache.authoritative_pending_backup_node_ids, set())
        self.assertEqual(cache.authoritative_pending_backup_reasons, {})
        self.assertEqual(cache.authoritative_backuped_node_ids, {44})
        self.assertEqual(cache.authoritative_host_visible_node_ids, {44})
        self.assertEqual(
            cache.authoritative_resolution_stats["host_backup_commit.promote_visible"],
            1,
        )

    def test_find_last_visible_host_ancestor_returns_nearest_visible_node(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        root = object()
        cache.root_node = root

        class Node:
            def __init__(self, node_id, parent):
                self.id = node_id
                self.parent = parent

        visible = Node(51, root)
        leaf = Node(52, visible)
        cache._node_backup_visible = lambda node: node is visible

        self.assertIs(cache._find_last_visible_host_ancestor(leaf), visible)

    def test_find_last_visible_host_ancestor_falls_back_to_root(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        root = object()
        cache.root_node = root

        class Node:
            def __init__(self, parent):
                self.parent = parent

        leaf = Node(root)
        cache._node_backup_visible = lambda node: False

        self.assertIs(cache._find_last_visible_host_ancestor(leaf), root)

    def test_split_node_preserves_authoritative_host_visibility(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.get_child_key_fn = lambda key: key.token_ids[0] if len(key.token_ids) else 0
        cache.page_size = 1
        cache.authoritative_backuped_node_ids = {61}
        cache.authoritative_host_visible_node_ids = {61}

        class Parent:
            def __init__(self):
                self.children = {}

        parent = Parent()

        class Child:
            def __init__(self):
                self.id = 61
                self.priority = 0
                self.parent = parent
                self.lock_ref = 0
                self.key = RadixKey([1, 2], None)
                self.hit_count = 0
                self.children = {}
                self.value = None
                self.host_value = torch.tensor([10, 11], dtype=torch.int64)
                self.hash_value = ["h1", "h2"]

            @property
            def evicted(self):
                return True

            @property
            def backuped(self):
                return True

        child = Child()
        parent.children[1] = child

        new_node = cache._split_node(child.key, child, 1)

        self.assertIn(new_node.id, cache.authoritative_backuped_node_ids)
        self.assertIn(new_node.id, cache.authoritative_host_visible_node_ids)

    def test_discard_authoritative_visibility(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.authoritative_pending_backup_node_ids = {71}
        cache.authoritative_backuped_node_ids = {71}
        cache.authoritative_host_visible_node_ids = {71}

        class Node:
            id = 71

        cache._discard_authoritative_visibility(Node())

        self.assertEqual(cache.authoritative_pending_backup_node_ids, set())
        self.assertEqual(cache.authoritative_backuped_node_ids, set())
        self.assertEqual(cache.authoritative_host_visible_node_ids, set())

    def test_apply_authoritative_host_evict_clears_both_visibility_sets(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.authoritative_pending_backup_node_ids = {72}
        cache.authoritative_backuped_node_ids = {72}
        cache.authoritative_host_visible_node_ids = {72}
        cache.evictable_host_leaves = set()
        cache.get_child_key_fn = lambda key: key

        class Parent:
            def __init__(self):
                self.children = {}

        parent = Parent()

        class Node:
            def __init__(self):
                self.id = 72
                self.parent = parent
                self.key = "k"
                self.host_value = [1]
                self.host_ref_counter = 0

            @property
            def evicted(self):
                return True

        node = Node()
        parent.children["k"] = node
        cache.root_node = object()
        cache.cache_controller = types.SimpleNamespace(evict_host=lambda host_value: len(host_value))
        cache._find_node_by_id = lambda node_id: node if node_id == 72 else None
        cache._update_host_leaf_status = lambda parent: None

        cache._apply_authoritative_host_evict(node_id=72)

        self.assertEqual(cache.authoritative_pending_backup_node_ids, set())
        self.assertEqual(cache.authoritative_backuped_node_ids, set())
        self.assertEqual(cache.authoritative_host_visible_node_ids, set())

    def test_build_host_insert_from_storage_payload_carries_structure_anchor(self):
        cache = HiRadixCache.__new__(HiRadixCache)

        class Node:
            def __init__(self, node_id, last_hash):
                self.id = node_id
                self._last_hash = last_hash

            def get_last_hash_value(self):
                return self._last_hash

        anchor = Node(111, "anchor-hash")
        inserted = [Node(112, "child-hash")]

        payload = cache._build_host_insert_from_storage_payload(
            req_id="rid-7",
            anchor_node=anchor,
            fetched_token_ids=[1, 2, 3, 4],
            fetched_hash_value=["h1", "h2"],
            inserted_nodes=inserted,
            loaded_from_storage=64,
            matched_length=0,
            committed_tokens=64,
        )

        self.assertEqual(payload["req_id"], "rid-7")
        self.assertEqual(payload["anchor_node_ref"]["node_id"], 111)
        self.assertEqual(payload["anchor_node_ref"]["last_hash"], "anchor-hash")
        self.assertEqual(payload["fetched_hash_value"], ["h1", "h2"])
        self.assertEqual(payload["node_refs"][0]["last_hash"], "child-hash")

    def test_build_host_insert_rebuild_blueprint_carries_replay_shape(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 2

        blueprint = cache._build_host_insert_rebuild_blueprint(
            {
                "req_id": "rid-blueprint",
                "anchor_node_ref": {"node_id": 1, "last_hash": "anchor"},
                "fetched_token_ids": [1, 2, 3, 4],
                "fetched_hash_value": ["h1", "h2"],
                "matched_length": 2,
                "committed_tokens": 4,
            }
        )

        self.assertEqual(blueprint["req_id"], "rid-blueprint")
        self.assertEqual(blueprint["anchor_node_ref"]["node_id"], 1)
        self.assertEqual(blueprint["matched_length"], 2)
        self.assertEqual(blueprint["committed_tokens"], 4)
        self.assertEqual(
            blueprint["segment_plan"],
            [
                {"token_ids": [3, 4], "hash_value": ["h2"], "page_index": 1},
            ],
        )

    def test_build_host_backup_commit_payload_carries_parent_and_hash(self):
        cache = HiRadixCache.__new__(HiRadixCache)

        class Parent:
            id = 141

            def get_last_hash_value(self):
                return "parent-hash"

        class Node:
            id = 142
            parent = Parent()
            key = RadixKey([1, 2], None)
            hash_value = ["node-h1", "node-h2"]

            def get_last_hash_value(self):
                return "node-last-hash"

        payload = cache._build_host_backup_commit_payload(Node())

        self.assertEqual(payload["node_ref"]["node_id"], 142)
        self.assertEqual(payload["parent_node_ref"]["node_id"], 141)
        self.assertEqual(payload["node_key_tokens"], [1, 2])
        self.assertEqual(payload["node_hash_value"], ["node-h1", "node-h2"])

    def test_recover_host_insert_visible_nodes_from_payload(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.get_child_key_fn = lambda key: key.token_ids[0]
        cache.key_match_fn = lambda key0, key1: min(len(key0), len(key1))
        cache.page_size = 1

        class Node:
            def __init__(self, node_id, key_tokens, parent=None, last_hash=None):
                self.id = node_id
                self.key = RadixKey(key_tokens, None)
                self.parent = parent
                self.children = {}
                self.host_value = [1] * len(key_tokens)
                self._last_hash = last_hash
                self.value = None

            def get_last_hash_value(self):
                return self._last_hash

            @property
            def evicted(self):
                return True

            @property
            def backuped(self):
                return True

        root = Node(120, [], None, None)
        anchor = Node(121, [10], root, "anchor-hash")
        child = Node(122, [11, 12], anchor, "child-hash")
        root.children[10] = anchor
        anchor.children[11] = child
        cache._resolve_authoritative_node_ref = (
            lambda node_ref=None, **kwargs: anchor
            if node_ref == {"node_id": 121, "last_hash": "anchor-hash"}
            else None
        )

        nodes = cache._recover_host_insert_visible_nodes_from_payload(
            {
                "anchor_node_ref": {"node_id": 121, "last_hash": "anchor-hash"},
                "fetched_token_ids": [11, 12],
                "fetched_hash_value": ["child-hash"],
                "matched_length": 0,
                "committed_tokens": 2,
            }
        )

        self.assertEqual(nodes, [child])

    def test_recover_host_insert_visible_nodes_from_payload_rejects_hash_mismatch(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.get_child_key_fn = lambda key: key.token_ids[0]
        cache.key_match_fn = lambda key0, key1: min(len(key0), len(key1))
        cache.page_size = 1

        class Node:
            def __init__(self, node_id, key_tokens, parent=None, last_hash=None):
                self.id = node_id
                self.key = RadixKey(key_tokens, None)
                self.parent = parent
                self.children = {}
                self.host_value = [1] * len(key_tokens)
                self._last_hash = last_hash
                self.value = None
                self.hash_value = [last_hash] if last_hash is not None else []

            def get_last_hash_value(self):
                return self._last_hash

            @property
            def evicted(self):
                return True

            @property
            def backuped(self):
                return True

        root = Node(130, [], None, None)
        anchor = Node(131, [10], root, "anchor-hash")
        child = Node(132, [11, 12], anchor, "actual-hash")
        root.children[10] = anchor
        anchor.children[11] = child
        cache._resolve_authoritative_node_ref = (
            lambda node_ref=None, **kwargs: anchor
            if node_ref == {"node_id": 131, "last_hash": "anchor-hash"}
            else None
        )

        nodes = cache._recover_host_insert_visible_nodes_from_payload(
            {
                "anchor_node_ref": {"node_id": 131, "last_hash": "anchor-hash"},
                "fetched_token_ids": [11, 12],
                "fetched_hash_value": ["expected-hash"],
                "matched_length": 0,
                "committed_tokens": 2,
            }
        )

        self.assertEqual(nodes, [])

    def test_validate_host_insert_nodes_from_payload_requires_full_coverage(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 1

        class Node:
            def __init__(self, node_id, host_len, hash_value):
                self.id = node_id
                self.host_value = [1] * host_len
                self.hash_value = [hash_value]
                self.value = None

            @property
            def evicted(self):
                return True

            @property
            def backuped(self):
                return True

        valid = cache._validate_host_insert_nodes_from_payload(
            [Node(1, 1, "h1")],
            {
                "matched_length": 0,
                "committed_tokens": 2,
                "fetched_hash_value": ["h1", "h2"],
            },
        )

        self.assertFalse(valid)

    def test_repair_host_insert_subtree_from_payload_prunes_mismatched_host_only_branch(
        self,
    ):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.get_child_key_fn = lambda key: key.token_ids[0]
        cache.key_match_fn = lambda key0, key1: min(len(key0), len(key1))
        cache.page_size = 1
        cache.authoritative_backuped_node_ids = {161, 162}
        cache.authoritative_host_visible_node_ids = {161, 162}
        cache.evictable_host_leaves = set()
        freed = []

        class Node:
            def __init__(self, node_id, key_tokens, parent=None, last_hash=None):
                self.id = node_id
                self.key = RadixKey(key_tokens, None)
                self.parent = parent
                self.children = {}
                self._last_hash = last_hash
                self.hash_value = [last_hash] if last_hash is not None else []
                self.host_value = [1] * max(1, len(key_tokens))
                self.value = None
                self.lock_ref = 0
                self.host_ref_counter = 0

            def get_last_hash_value(self):
                return self._last_hash

            @property
            def evicted(self):
                return True

            @property
            def backuped(self):
                return self.host_value is not None

        root = Node(160, [], None, None)
        anchor = Node(161, [10], root, "anchor-hash")
        child = Node(162, [11, 12], anchor, "actual-hash")
        root.children[10] = anchor
        anchor.children[11] = child
        cache.root_node = root
        cache.cache_controller = types.SimpleNamespace(
            evict_host=lambda host_value: freed.append(list(host_value)) or len(host_value)
        )
        cache._resolve_authoritative_node_ref = (
            lambda node_ref=None, **kwargs: anchor
            if node_ref == {"node_id": 161, "last_hash": "anchor-hash"}
            else None
        )
        cache._update_host_leaf_status = lambda node: None
        cache._update_leaf_status = lambda node: None

        repaired = cache._repair_host_insert_subtree_from_payload(
            {
                "anchor_node_ref": {"node_id": 161, "last_hash": "anchor-hash"},
                "fetched_token_ids": [11, 12],
                "fetched_hash_value": ["expected-hash"],
                "matched_length": 0,
                "committed_tokens": 2,
            }
        )

        self.assertTrue(repaired)
        self.assertEqual(freed, [[1, 1]])
        self.assertEqual(anchor.children, {})
        self.assertEqual(cache.authoritative_host_visible_node_ids, {161})
        self.assertEqual(cache.authoritative_backuped_node_ids, {161})

    def test_repair_host_insert_subtree_skips_protected_nodes(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.get_child_key_fn = lambda key: key.token_ids[0]
        cache.key_match_fn = lambda key0, key1: min(len(key0), len(key1))
        cache.page_size = 1
        cache.authoritative_backuped_node_ids = {171, 172}
        cache.authoritative_host_visible_node_ids = {171, 172}
        cache.evictable_host_leaves = set()
        freed = []

        class Node:
            def __init__(self, node_id, key_tokens, parent=None, last_hash=None):
                self.id = node_id
                self.key = RadixKey(key_tokens, None)
                self.parent = parent
                self.children = {}
                self._last_hash = last_hash
                self.hash_value = [last_hash] if last_hash is not None else []
                self.host_value = [1] * max(1, len(key_tokens))
                self.value = None
                self.lock_ref = 0
                self.host_ref_counter = 0

            def get_last_hash_value(self):
                return self._last_hash

            @property
            def evicted(self):
                return True

            @property
            def backuped(self):
                return self.host_value is not None

        root = Node(170, [], None, None)
        anchor = Node(171, [10], root, "anchor-hash")
        child = Node(172, [11, 12], anchor, "actual-hash")
        child.host_ref_counter = 1
        root.children[10] = anchor
        anchor.children[11] = child
        cache.root_node = root
        cache.cache_controller = types.SimpleNamespace(
            evict_host=lambda host_value: freed.append(list(host_value)) or len(host_value)
        )
        cache._resolve_authoritative_node_ref = (
            lambda node_ref=None, **kwargs: anchor
            if node_ref == {"node_id": 171, "last_hash": "anchor-hash"}
            else None
        )
        cache._update_host_leaf_status = lambda node: None
        cache._update_leaf_status = lambda node: None

        repaired = cache._repair_host_insert_subtree_from_payload(
            {
                "anchor_node_ref": {"node_id": 171, "last_hash": "anchor-hash"},
                "fetched_token_ids": [11, 12],
                "fetched_hash_value": ["expected-hash"],
                "matched_length": 0,
                "committed_tokens": 2,
            }
        )

        self.assertFalse(repaired)
        self.assertEqual(freed, [])
        self.assertIs(anchor.children[11], child)

    def test_stage_and_apply_host_insert_rebuild_blueprint(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.authoritative_host_insert_rebuild_by_reqid = {}
        cache.authoritative_host_visible_node_ids = set()
        cache.authoritative_resolution_stats = {}

        class VisibleNode:
            def __init__(self, node_id):
                self.id = node_id

        staged = cache._stage_host_insert_rebuild_blueprint(
            {
                "req_id": "rid-stage",
                "anchor_node_ref": {"node_id": 1, "last_hash": "anchor"},
                "fetched_token_ids": [11, 12],
                "fetched_hash_value": ["h11"],
                "matched_length": 0,
                "committed_tokens": 2,
            }
        )

        self.assertTrue(staged)
        cache._recover_host_insert_visible_nodes_from_payload = (
            lambda payload: [VisibleNode(211)]
        )

        applied = cache._try_apply_host_insert_rebuild_blueprint(
            "rid-stage",
            cache.authoritative_host_insert_rebuild_by_reqid["rid-stage"],
        )

        self.assertTrue(applied)
        self.assertIn(211, cache.authoritative_host_visible_node_ids)
        self.assertNotIn("rid-stage", cache.authoritative_host_insert_rebuild_by_reqid)
        self.assertEqual(
            cache.authoritative_resolution_stats["host_insert.blueprint_stage"], 1
        )
        self.assertEqual(
            cache.authoritative_resolution_stats["host_insert.blueprint_apply"], 1
        )

    def test_normalize_host_insert_rebuild_blueprint_splits_long_host_node(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.get_child_key_fn = lambda key: key.token_ids[0]
        cache.key_match_fn = lambda key0, key1: min(len(key0), len(key1))
        cache.page_size = 1
        cache.authoritative_backuped_node_ids = set()
        cache.authoritative_host_visible_node_ids = set()

        class Node:
            def __init__(self, node_id, key_tokens, parent=None, last_hash=None):
                self.id = node_id
                self.key = RadixKey(key_tokens, None)
                self.parent = parent
                self.children = {}
                self._last_hash = last_hash
                self.hash_value = (
                    [f"{last_hash}-{i}" for i in range(len(key_tokens))]
                    if last_hash is not None
                    else []
                )
                self.host_value = torch.tensor(list(range(len(key_tokens))), dtype=torch.int64)
                self.value = None
                self.lock_ref = 0
                self.hit_count = 0
                self.priority = 0

            def get_last_hash_value(self):
                return self.hash_value[-1] if self.hash_value else None

            @property
            def evicted(self):
                return True

            @property
            def backuped(self):
                return True

        root = Node(220, [])
        root.value = []
        root.host_value = []
        anchor = Node(221, [10], root, "anchor")
        child = Node(222, [11, 12, 13, 14], anchor, "child")
        root.children[10] = anchor
        anchor.children[11] = child
        cache.root_node = root
        cache._resolve_authoritative_node_ref = (
            lambda node_ref=None, **kwargs: anchor
            if node_ref == {"node_id": 221, "last_hash": anchor.get_last_hash_value()}
            else None
        )

        normalized = cache._normalize_host_insert_subtree_from_payload(
            {
                "anchor_node_ref": {
                    "node_id": 221,
                    "last_hash": anchor.get_last_hash_value(),
                },
                "fetched_token_ids": [11, 12],
                "fetched_hash_value": child.hash_value[:2],
                "matched_length": 0,
                "committed_tokens": 2,
            }
        )

        self.assertTrue(normalized)
        new_child = anchor.children[11]
        self.assertEqual(new_child.key.token_ids, [11, 12])
        self.assertEqual(child.key.token_ids, [13, 14])

    def test_advance_host_insert_skeleton_shrinks_remaining_segments(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.get_child_key_fn = lambda key: key.token_ids[0]
        cache.key_match_fn = lambda key0, key1: min(len(key0), len(key1))
        cache.page_size = 1
        cache.authoritative_backuped_node_ids = set()
        cache.authoritative_host_visible_node_ids = set()
        cache.authoritative_host_insert_missing_by_reqid = {}
        cache.authoritative_resolution_stats = {}

        class Node:
            def __init__(self, node_id, key_tokens, parent=None, base_hash="h"):
                self.id = node_id
                self.key = RadixKey(key_tokens, None)
                self.parent = parent
                self.children = {}
                self.hash_value = [
                    f"{base_hash}{token}" for token in key_tokens
                ]
                self.host_value = torch.tensor(list(range(len(key_tokens))), dtype=torch.int64)
                self.value = None
                self.lock_ref = 0
                self.hit_count = 0
                self.priority = 0

            def get_last_hash_value(self):
                return self.hash_value[-1] if self.hash_value else None

            @property
            def evicted(self):
                return True

            @property
            def backuped(self):
                return True

        root = Node(230, [])
        root.value = []
        root.host_value = []
        anchor = Node(231, [10], root, "a")
        child = Node(232, [11, 12, 13], anchor, "h")
        root.children[10] = anchor
        anchor.children[11] = child
        cache.root_node = root
        cache.authoritative_host_insert_skeleton_by_reqid = {
            "rid-skel": [
                {"token_ids": [11], "hash_value": ["h11"], "page_index": 0},
                {"token_ids": [12], "hash_value": ["h12"], "page_index": 1},
                {"token_ids": [13], "hash_value": ["h13"], "page_index": 2},
            ]
        }
        blueprint = {
            "req_id": "rid-skel",
            "anchor_node_ref": {
                "node_id": 231,
                "last_hash": anchor.get_last_hash_value(),
            },
            "segment_plan": cache.authoritative_host_insert_skeleton_by_reqid["rid-skel"],
        }
        cache._resolve_authoritative_node_ref = (
            lambda node_ref=None, **kwargs: anchor
            if node_ref == blueprint["anchor_node_ref"]
            else None
        )

        advanced = cache._advance_host_insert_skeleton("rid-skel", blueprint)

        self.assertTrue(advanced)
        self.assertEqual(
            cache.authoritative_resolution_stats["host_insert.skeleton_advance"], 1
        )
        self.assertEqual(cache.authoritative_host_insert_skeleton_by_reqid["rid-skel"], [])
        self.assertEqual(cache.authoritative_host_insert_missing_by_reqid["rid-skel"], [])

    def test_advance_host_insert_skeleton_records_missing_child(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.get_child_key_fn = lambda key: key.token_ids[0]
        cache.key_match_fn = lambda key0, key1: min(len(key0), len(key1))
        cache.page_size = 1
        cache.authoritative_resolution_stats = {}
        cache.authoritative_host_insert_missing_by_reqid = {}

        class Node:
            def __init__(self, node_id, key_tokens, parent=None, base_hash="h"):
                self.id = node_id
                self.key = RadixKey(key_tokens, None)
                self.parent = parent
                self.children = {}
                self.hash_value = [f"{base_hash}{token}" for token in key_tokens]
                self.host_value = []
                self.value = [] if node_id == 240 else None

            def get_last_hash_value(self):
                return self.hash_value[-1] if self.hash_value else None

            @property
            def evicted(self):
                return self.value is None

            @property
            def backuped(self):
                return True

        root = Node(240, [])
        anchor = Node(241, [10], root, "a")
        root.children[10] = anchor
        cache.root_node = root
        cache.authoritative_host_insert_skeleton_by_reqid = {
            "rid-gap": [
                {"token_ids": [11], "hash_value": ["h11"], "page_index": 0},
            ]
        }
        blueprint = {
            "req_id": "rid-gap",
            "anchor_node_ref": {
                "node_id": 241,
                "last_hash": anchor.get_last_hash_value(),
            },
            "segment_plan": cache.authoritative_host_insert_skeleton_by_reqid["rid-gap"],
        }
        cache._resolve_authoritative_node_ref = (
            lambda node_ref=None, **kwargs: anchor
            if node_ref == blueprint["anchor_node_ref"]
            else None
        )

        advanced = cache._advance_host_insert_skeleton("rid-gap", blueprint)

        self.assertFalse(advanced)
        self.assertEqual(
            cache.authoritative_resolution_stats["host_insert.skeleton_missing"], 1
        )
        self.assertEqual(
            cache.authoritative_host_insert_missing_by_reqid["rid-gap"][0]["reason"],
            "missing_child",
        )

    def test_recover_backup_commit_node_from_payload(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.get_child_key_fn = lambda key: key.token_ids[0]
        cache.key_match_fn = lambda key0, key1: min(len(key0), len(key1))

        class Node:
            def __init__(self, node_id, key_tokens, parent=None, last_hash=None):
                self.id = node_id
                self.key = RadixKey(key_tokens, None)
                self.parent = parent
                self.children = {}
                self._last_hash = last_hash
                self.hash_value = [last_hash] if last_hash is not None else []
                self.host_value = [1] * max(1, len(key_tokens))

            def get_last_hash_value(self):
                return self._last_hash

            @property
            def backuped(self):
                return True

        root = Node(150, [], None, None)
        parent = Node(151, [10], root, "parent-hash")
        child = Node(152, [11, 12], parent, "child-hash")
        root.children[10] = parent
        parent.children[11] = child

        cache._resolve_authoritative_node_ref = (
            lambda node_ref=None, **kwargs: parent
            if node_ref == {"node_id": 151, "last_hash": "parent-hash"}
            else None
        )

        recovered = cache._recover_backup_commit_node_from_payload(
            {
                "parent_node_ref": {"node_id": 151, "last_hash": "parent-hash"},
                "node_key_tokens": [11, 12],
                "node_hash_value": ["child-hash"],
            }
        )

        self.assertIs(recovered, child)

    def test_apply_authoritative_host_insert_uses_payload_recover_after_partial_mismatch(
        self,
    ):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.authoritative_resolution_stats = {}
        cache.authoritative_prefetch_loaded_tokens_by_reqid = {}
        cache.authoritative_host_insert_rebuild_by_reqid = {}
        cache.authoritative_host_visible_node_ids = set()
        cache._resolve_authoritative_node_ref = (
            lambda node_ref=None, **kwargs: object()
            if node_ref == {"node_id": 1, "last_hash": "h1"}
            else None
        )
        cache._validate_host_insert_nodes_from_payload = lambda nodes, payload: False

        class VisibleNode:
            def __init__(self, node_id):
                self.id = node_id

        cache._recover_host_insert_visible_nodes_from_payload = (
            lambda payload: [VisibleNode(201)]
        )
        cache._repair_host_insert_subtree_from_payload = lambda payload: False
        cache.pp_rank = 0
        cache.attn_cp_rank = 0
        cache.cache_controller = types.SimpleNamespace(tp_rank=0)

        class Op:
            op_type = "HOST_INSERT_FROM_STORAGE"
            payload = {
                "req_id": "rid-x",
                "loaded_from_storage": 64,
                "node_refs": [{"node_id": 1, "last_hash": "h1"}],
            }
            op_seq = 1

        cache.apply_authoritative_tree_op(Op())

        self.assertIn(201, cache.authoritative_host_visible_node_ids)
        self.assertEqual(
            cache.authoritative_resolution_stats["host_insert.partial_mismatch"], 1
        )
        self.assertEqual(
            cache.authoritative_resolution_stats["host_insert.payload_recover"], 1
        )

    def test_repair_backup_commit_subtree_from_payload_prunes_mismatched_branch(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.get_child_key_fn = lambda key: key.token_ids[0]
        cache.key_match_fn = lambda key0, key1: min(len(key0), len(key1))
        cache.page_size = 1
        cache.authoritative_backuped_node_ids = {181, 182}
        cache.authoritative_host_visible_node_ids = {181, 182}
        cache.evictable_host_leaves = set()
        freed = []

        class Node:
            def __init__(self, node_id, key_tokens, parent=None, last_hash=None):
                self.id = node_id
                self.key = RadixKey(key_tokens, None)
                self.parent = parent
                self.children = {}
                self._last_hash = last_hash
                self.hash_value = [last_hash] if last_hash is not None else []
                self.host_value = [1] * max(1, len(key_tokens))
                self.value = None
                self.lock_ref = 0
                self.host_ref_counter = 0

            def get_last_hash_value(self):
                return self._last_hash

            @property
            def evicted(self):
                return True

            @property
            def backuped(self):
                return self.host_value is not None

        root = Node(180, [], None, None)
        parent = Node(181, [10], root, "parent-hash")
        child = Node(182, [11, 12], parent, "actual-hash")
        root.children[10] = parent
        parent.children[11] = child
        cache.root_node = root
        cache.cache_controller = types.SimpleNamespace(
            evict_host=lambda host_value: freed.append(list(host_value)) or len(host_value)
        )
        cache._resolve_authoritative_node_ref = (
            lambda node_ref=None, **kwargs: parent
            if node_ref == {"node_id": 181, "last_hash": "parent-hash"}
            else None
        )
        cache._update_host_leaf_status = lambda node: None
        cache._update_leaf_status = lambda node: None

        repaired = cache._repair_backup_commit_subtree_from_payload(
            {
                "parent_node_ref": {"node_id": 181, "last_hash": "parent-hash"},
                "node_key_tokens": [11, 12],
                "node_hash_value": ["expected-hash"],
            }
        )

        self.assertTrue(repaired)
        self.assertEqual(freed, [[1, 1]])
        self.assertEqual(parent.children, {})
        self.assertEqual(cache.authoritative_host_visible_node_ids, {181})
        self.assertEqual(cache.authoritative_backuped_node_ids, {181})

    def test_repair_backup_commit_subtree_skips_protected_branch(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.get_child_key_fn = lambda key: key.token_ids[0]
        cache.key_match_fn = lambda key0, key1: min(len(key0), len(key1))
        cache.page_size = 1
        cache.authoritative_backuped_node_ids = {191, 192}
        cache.authoritative_host_visible_node_ids = {191, 192}
        cache.evictable_host_leaves = set()
        freed = []

        class Node:
            def __init__(self, node_id, key_tokens, parent=None, last_hash=None):
                self.id = node_id
                self.key = RadixKey(key_tokens, None)
                self.parent = parent
                self.children = {}
                self._last_hash = last_hash
                self.hash_value = [last_hash] if last_hash is not None else []
                self.host_value = [1] * max(1, len(key_tokens))
                self.value = None
                self.lock_ref = 0
                self.host_ref_counter = 0

            def get_last_hash_value(self):
                return self._last_hash

            @property
            def evicted(self):
                return True

            @property
            def backuped(self):
                return self.host_value is not None

        root = Node(190, [], None, None)
        parent = Node(191, [10], root, "parent-hash")
        child = Node(192, [11, 12], parent, "actual-hash")
        child.lock_ref = 1
        root.children[10] = parent
        parent.children[11] = child
        cache.root_node = root
        cache.cache_controller = types.SimpleNamespace(
            evict_host=lambda host_value: freed.append(list(host_value)) or len(host_value)
        )
        cache._resolve_authoritative_node_ref = (
            lambda node_ref=None, **kwargs: parent
            if node_ref == {"node_id": 191, "last_hash": "parent-hash"}
            else None
        )
        cache._update_host_leaf_status = lambda node: None
        cache._update_leaf_status = lambda node: None

        repaired = cache._repair_backup_commit_subtree_from_payload(
            {
                "parent_node_ref": {"node_id": 191, "last_hash": "parent-hash"},
                "node_key_tokens": [11, 12],
                "node_hash_value": ["expected-hash"],
            }
        )

        self.assertFalse(repaired)
        self.assertEqual(freed, [])
        self.assertIs(parent.children[11], child)

    def test_authoritative_resolution_stats_are_counted(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.authoritative_resolution_stats = {}

        cache._record_authoritative_resolution("host_insert.node_ref")
        cache._record_authoritative_resolution("host_insert.node_ref")
        cache._record_authoritative_resolution("host_insert.payload_recover")

        self.assertEqual(
            cache.get_authoritative_resolution_stats(),
            {
                "host_insert.node_ref": 2,
                "host_insert.payload_recover": 1,
            },
        )

    def test_apply_authoritative_host_backup_commit_records_payload_repair(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.authoritative_resolution_stats = {}
        cache.authoritative_backuped_node_ids = set()
        cache.authoritative_host_visible_node_ids = set()
        cache._resolve_authoritative_node_ref = lambda *args, **kwargs: None
        cache._recover_backup_commit_node_from_payload = lambda payload: None
        cache._repair_backup_commit_subtree_from_payload = lambda payload: True
        cache.pp_rank = 0
        cache.attn_cp_rank = 0
        cache.cache_controller = types.SimpleNamespace(tp_rank=0)

        class Op:
            op_type = "HOST_BACKUP_COMMIT"
            payload = {"node_id": 1, "last_hash": "h1"}
            op_seq = 1

        cache.apply_authoritative_tree_op(Op())

        self.assertEqual(
            cache.authoritative_resolution_stats["host_backup_commit.payload_repair"],
            1,
        )

    def test_apply_authoritative_host_backup_commit_uses_recovered_node_ref(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.authoritative_resolution_stats = {}
        cache.authoritative_pending_backup_refs = {}
        cache.authoritative_pending_backup_node_ids = set()
        cache.authoritative_pending_backup_reasons = {}
        cache.authoritative_backuped_node_ids = set()
        cache.authoritative_host_visible_node_ids = set()
        cache._resolve_authoritative_node_ref = lambda *args, **kwargs: None

        class Node:
            id = 91

            @staticmethod
            def get_last_hash_value():
                return "canon-91"

        cache._recover_backup_commit_node_from_payload = lambda payload: Node()
        cache._repair_backup_commit_subtree_from_payload = lambda payload: False

        class Op:
            op_type = "HOST_BACKUP_COMMIT"
            payload = {
                "node_id": 1,
                "last_hash": "stale-1",
                "node_ref": {"node_id": 1, "last_hash": "stale-1"},
            }
            op_seq = 1

        cache.apply_authoritative_tree_op(Op())

        self.assertEqual(
            cache.authoritative_pending_backup_refs["canon-91"],
            {"node_id": 91, "last_hash": "canon-91"},
        )
        self.assertEqual(cache.authoritative_pending_backup_node_ids, {91})
        self.assertEqual(
            cache.authoritative_pending_backup_reasons["canon-91"], "await_stable"
        )

    def test_build_authoritative_prefetch_ready_result_skips_unstable_summary(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.authoritative_prefetch_ready_by_reqid = {
            "rid-u": AuthoritativePrefetchReadySummary(
                req_id="rid-u",
                prefix_len=2,
                host_hit_length=1,
                storage_hit_length=0,
                input_len=3,
            )
        }
        cache.authoritative_host_insert_rebuild_by_reqid = {"rid-u": {}}
        cache.authoritative_host_insert_skeleton_by_reqid = {}
        cache.authoritative_host_insert_missing_by_reqid = {}
        cache.authoritative_pending_backup_refs = {}
        cache.authoritative_resolution_stats = {}

        class ReqStub:
            rid = "rid-u"

        ready = cache.build_authoritative_prefetch_ready_result(ReqStub())

        self.assertIsNone(ready)
        self.assertEqual(
            cache.authoritative_resolution_stats["prefetch_ready.skip_unstable"], 1
        )

    def test_maybe_commit_stable_prefetch_ready_summaries_requeues_summary(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.authoritative_prefetch_ready_by_reqid = {}
        cache.authoritative_host_insert_rebuild_by_reqid = {}
        cache.authoritative_host_insert_skeleton_by_reqid = {}
        cache.authoritative_host_insert_missing_by_reqid = {}
        cache.authoritative_pending_backup_refs = {}
        cache.authoritative_resolution_stats = {}
        cache.prefetch_ready_results_by_reqid = {
            "rid-r": LatchedPrefetchReadyResult(
                match_result=MatchResult(
                    device_indices=torch.arange(2, dtype=torch.int64),
                    last_device_node=None,
                    last_host_node=None,
                    host_hit_length=1,
                ),
                storage_hit_length=3,
                input_len=11,
            )
        }

        class Tree:
            enabled = True

        queued = []
        cache.authoritative_tree = Tree()
        cache.queue_authoritative_tree_op = (
            lambda op_type, **payload: queued.append((op_type, payload))
        )

        cache._maybe_commit_stable_prefetch_ready_summaries()

        self.assertEqual(queued[0][0], "PREFETCH_READY_SUMMARY")
        self.assertEqual(queued[0][1]["req_id"], "rid-r")
        self.assertEqual(queued[0][1]["input_len"], 11)
        self.assertEqual(
            queued[0][1]["last_host_node_ref"], {"node_id": None, "last_hash": None}
        )
        self.assertEqual(
            cache.authoritative_resolution_stats["prefetch_ready.requeue_stable"], 1
        )

    def test_build_authoritative_prefetch_ready_result_skips_stale_summary_input_len(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.authoritative_prefetch_ready_by_reqid = {
            "rid-stale": AuthoritativePrefetchReadySummary(
                req_id="rid-stale",
                prefix_len=2,
                host_hit_length=1,
                storage_hit_length=0,
                input_len=4,
            )
        }
        cache.authoritative_host_insert_rebuild_by_reqid = {}
        cache.authoritative_host_insert_skeleton_by_reqid = {}
        cache.authoritative_host_insert_missing_by_reqid = {}
        cache.authoritative_pending_backup_refs = {}
        cache.authoritative_resolution_stats = {}

        class ReqStub:
            rid = "rid-stale"
            fill_ids = [1, 2, 3]

        ready = cache.build_authoritative_prefetch_ready_result(ReqStub())

        self.assertIsNone(ready)
        self.assertNotIn("rid-stale", cache.authoritative_prefetch_ready_by_reqid)
        self.assertEqual(
            cache.authoritative_resolution_stats["prefetch_ready.skip_stale"], 1
        )

    def test_clamp_match_result_ignores_stale_summary_input_len(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.authoritative_prefetch_ready_by_reqid = {
            "rid-live": AuthoritativePrefetchReadySummary(
                req_id="rid-live",
                prefix_len=1,
                host_hit_length=0,
                storage_hit_length=0,
                input_len=8,
            )
        }

        original = MatchResult(
            device_indices=torch.arange(3, dtype=torch.int64),
            last_device_node=None,
            last_host_node=None,
            host_hit_length=2,
        )

        clamped = cache._clamp_match_result_to_authoritative_summary(
            "rid-live",
            original,
            input_len=6,
        )

        self.assertEqual(len(clamped.device_indices), 3)
        self.assertEqual(clamped.host_hit_length, 2)

    def test_get_authoritative_resolution_stats_returns_copy(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.authoritative_resolution_stats = {"host_insert.node_ref": 1}

        stats = cache.get_authoritative_resolution_stats()
        stats["host_insert.node_ref"] = 99

        self.assertEqual(cache.authoritative_resolution_stats["host_insert.node_ref"], 1)

    def test_get_host_insert_missing_segments_returns_copy(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.authoritative_host_insert_missing_by_reqid = {
            "rid-missing": [
                {"reason": "missing_child", "page_index": 1, "token_ids": [11]}
            ]
        }

        segments = cache.get_host_insert_missing_segments("rid-missing")
        segments[0]["reason"] = "mutated"

        self.assertEqual(
            cache.authoritative_host_insert_missing_by_reqid["rid-missing"][0]["reason"],
            "missing_child",
        )

    def test_get_pending_backup_node_ids_returns_sorted_copy(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.authoritative_pending_backup_node_ids = {5, 2, 9}

        pending = cache.get_pending_backup_node_ids()
        pending.append(99)

        self.assertEqual(cache.get_pending_backup_node_ids(), [2, 5, 9])

    def test_get_pending_backup_reasons_returns_copy(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.authoritative_pending_backup_reasons = {"hash-2": "host_ref"}

        reasons = cache.get_pending_backup_reasons()
        reasons["hash-2"] = "mutated"

        self.assertEqual(
            cache.authoritative_pending_backup_reasons["hash-2"], "host_ref"
        )

    def test_format_authoritative_resolution_stats_sorts_by_count(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.authoritative_resolution_stats = {
            "b": 1,
            "a": 3,
            "c": 2,
        }

        self.assertEqual(
            cache._format_authoritative_resolution_stats(),
            "a=3,c=2,b=1",
        )

    def test_format_host_insert_missing_segments(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.authoritative_host_insert_missing_by_reqid = {
            "rid-missing": [
                {"reason": "missing_child", "page_index": 1, "token_ids": [11]},
                {"reason": "hash_mismatch", "page_index": 2, "token_ids": [12]},
            ]
        }

        formatted = cache._format_host_insert_missing_segments("rid-missing")

        self.assertEqual(formatted, "missing_child@p1:[11];hash_mismatch@p2:[12]")

    def test_format_pending_backup_nodes(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.authoritative_pending_backup_refs = {
            "hash-3": {"node_id": 3, "last_hash": "hash-3"},
            "hash-5": {"node_id": 5, "last_hash": "hash-5"},
            "hash-7": {"node_id": 7, "last_hash": "hash-7"},
        }
        cache.authoritative_pending_backup_reasons = {
            "hash-3": "host_ref",
            "hash-5": "await_stable",
            "hash-7": "missing_host_value",
        }
        cache._resolve_authoritative_node_ref = (
            lambda node_ref=None, node_id=None, last_hash=None: type(
                "Node", (), {"id": node_ref["node_id"]}
            )()
            if node_ref is not None
            else None
        )

        formatted = cache._format_pending_backup_nodes()

        self.assertEqual(formatted, "3:host_ref,5:await_stable,7:missing_host_value")

    def test_has_relevant_pending_backup_uses_host_chain(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        root = object()
        cache.root_node = root
        cache.authoritative_pending_backup_refs = {
            "hash-3": {"node_id": 3, "last_hash": "hash-3"}
        }
        pending = type("Node", (), {"id": 3, "parent": root})()
        host = type("Node", (), {"id": 9, "parent": pending})()
        cache._resolve_authoritative_node_ref = (
            lambda node_ref=None, node_id=None, last_hash=None: pending
        )

        self.assertTrue(cache._has_relevant_pending_backup(host_node=host))
        self.assertFalse(cache._has_relevant_pending_backup(host_node=root))

    def test_describe_pending_backup_reason(self):
        cache = HiRadixCache.__new__(HiRadixCache)

        class Node:
            host_value = [1]
            host_ref_counter = 1

            @property
            def backuped(self):
                return True

        self.assertEqual(cache._describe_pending_backup_reason(Node()), "host_ref")

    def test_format_match_authoritative_gates_reports_sources(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        root = object()
        cache.root_node = root
        cache.authoritative_host_insert_rebuild_by_reqid = {"rid-g": {}}
        cache.authoritative_host_insert_skeleton_by_reqid = {}
        cache.authoritative_host_insert_missing_by_reqid = {
            "rid-g": [{"reason": "missing_child"}]
        }
        cache.authoritative_pending_backup_refs = {
            "hash-1": {"node_id": 1, "last_hash": "hash-1"}
        }
        pending = type("Node", (), {"id": 1, "parent": root})()
        host = type("Node", (), {"id": 2, "parent": pending})()
        cache._resolve_authoritative_node_ref = (
            lambda node_ref=None, node_id=None, last_hash=None: pending
        )

        gates = cache._format_match_authoritative_gates(
            "rid-g",
            pre_clamp_device_hit=4,
            pre_clamp_host_hit=2,
            match_result=MatchResult(
                device_indices=torch.arange(2, dtype=torch.int64),
                last_device_node=None,
                last_host_node=host,
                host_hit_length=1,
            ),
        )

        self.assertEqual(
            gates,
            "unstable_ready,missing_gap,pending_backup,summary_clamp",
        )

    def test_format_match_authoritative_gates_is_none_when_no_gates(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.root_node = object()
        cache.authoritative_host_insert_rebuild_by_reqid = {}
        cache.authoritative_host_insert_skeleton_by_reqid = {}
        cache.authoritative_host_insert_missing_by_reqid = {}
        cache.authoritative_pending_backup_refs = {}

        gates = cache._format_match_authoritative_gates(
            "rid-clean",
            pre_clamp_device_hit=2,
            pre_clamp_host_hit=1,
            match_result=MatchResult(
                device_indices=torch.arange(2, dtype=torch.int64),
                last_device_node=None,
                last_host_node=None,
                host_hit_length=1,
            ),
        )

        self.assertEqual(gates, "none")

    def test_format_match_clamp_delta(self):
        cache = HiRadixCache.__new__(HiRadixCache)

        delta = cache._format_match_clamp_delta(
            pre_clamp_device_hit=4,
            pre_clamp_host_hit=3,
            match_result=MatchResult(
                device_indices=torch.arange(2, dtype=torch.int64),
                last_device_node=None,
                last_host_node=None,
                host_hit_length=1,
            ),
        )

        self.assertEqual(delta, "device:4->2,host:3->1")

    def test_build_host_insert_focus_blueprint_uses_missing_parent_anchor(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.authoritative_host_insert_missing_by_reqid = {
            "rid-focus": [
                {
                    "reason": "missing_child",
                    "page_index": 1,
                    "token_ids": [12],
                    "parent_node_ref": {"node_id": 301, "last_hash": "parent-hash"},
                }
            ]
        }
        cache.authoritative_host_insert_skeleton_by_reqid = {
            "rid-focus": [
                {"token_ids": [12], "hash_value": ["h12"], "page_index": 1},
                {"token_ids": [13], "hash_value": ["h13"], "page_index": 2},
            ]
        }

        focused = cache._build_host_insert_focus_blueprint(
            "rid-focus",
            {
                "req_id": "rid-focus",
                "anchor_node_ref": {"node_id": 300, "last_hash": "root-anchor"},
                "fetched_token_ids": [11, 12, 13],
                "fetched_hash_value": ["h11", "h12", "h13"],
                "matched_length": 0,
                "committed_tokens": 3,
            },
        )

        self.assertEqual(focused["anchor_node_ref"]["node_id"], 301)
        self.assertEqual(focused["fetched_token_ids"], [12, 13])
        self.assertEqual(focused["fetched_hash_value"], ["h12", "h13"])

    def test_sort_host_insert_missing_segments_prefers_earlier_page_then_reason(self):
        cache = HiRadixCache.__new__(HiRadixCache)

        sorted_segments = cache._sort_host_insert_missing_segments(
            [
                {"reason": "missing_child", "page_index": 2, "token_ids": [13]},
                {"reason": "hash_mismatch", "page_index": 1, "token_ids": [12]},
                {"reason": "key_mismatch", "page_index": 1, "token_ids": [11]},
            ]
        )

        self.assertEqual(
            [segment["reason"] for segment in sorted_segments],
            ["key_mismatch", "hash_mismatch", "missing_child"],
        )

    def test_build_host_insert_focus_state_slices_from_first_missing_page(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.authoritative_host_insert_missing_by_reqid = {
            "rid-focus": [
                {
                    "reason": "missing_child",
                    "page_index": 2,
                    "token_ids": [13],
                    "parent_node_ref": {"node_id": 301, "last_hash": "parent-hash"},
                }
            ]
        }
        cache.authoritative_host_insert_skeleton_by_reqid = {
            "rid-focus": [
                {"token_ids": [11], "hash_value": ["h11"], "page_index": 0},
                {"token_ids": [12], "hash_value": ["h12"], "page_index": 1},
                {"token_ids": [13], "hash_value": ["h13"], "page_index": 2},
                {"token_ids": [14], "hash_value": ["h14"], "page_index": 3},
            ]
        }

        missing, focused_segments = cache._build_host_insert_focus_state(
            "rid-focus",
            {"req_id": "rid-focus"},
        )

        self.assertEqual(missing["page_index"], 2)
        self.assertEqual(
            [segment["page_index"] for segment in focused_segments],
            [2, 3],
        )

    def test_advance_host_insert_skeleton_uses_missing_parent_anchor(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.get_child_key_fn = lambda key: key.token_ids[0]
        cache.key_match_fn = lambda key0, key1: min(len(key0), len(key1))
        cache.page_size = 1
        cache.authoritative_resolution_stats = {}
        cache.authoritative_host_insert_skeleton_by_reqid = {
            "rid-focus": [
                {"token_ids": [12], "hash_value": ["h12"], "page_index": 1},
            ]
        }
        cache.authoritative_host_insert_missing_by_reqid = {
            "rid-focus": [
                {
                    "reason": "missing_child",
                    "page_index": 1,
                    "token_ids": [12],
                    "parent_node_ref": {"node_id": 311, "last_hash": "parent-hash"},
                }
            ]
        }

        class Node:
            def __init__(self, node_id, key_tokens, parent=None, base_hash="h"):
                self.id = node_id
                self.key = RadixKey(key_tokens, None)
                self.parent = parent
                self.children = {}
                self.hash_value = [f"{base_hash}{token}" for token in key_tokens]
                self.host_value = []
                self.value = [] if node_id == 310 else None

            def get_last_hash_value(self):
                return self.hash_value[-1] if self.hash_value else None

            @property
            def evicted(self):
                return self.value is None

            @property
            def backuped(self):
                return True

        root = Node(310, [])
        focus_parent = Node(311, [11], root, "p")
        child = Node(312, [12], focus_parent, "h")
        root.children[11] = focus_parent
        focus_parent.children[12] = child
        cache.root_node = root
        blueprint = {
            "req_id": "rid-focus",
            "anchor_node_ref": {"node_id": 300, "last_hash": "old-anchor"},
            "segment_plan": cache.authoritative_host_insert_skeleton_by_reqid["rid-focus"],
        }
        cache._resolve_authoritative_node_ref = (
            lambda node_ref=None, **kwargs: focus_parent
            if node_ref == {"node_id": 311, "last_hash": "parent-hash"}
            else None
        )

        advanced = cache._advance_host_insert_skeleton("rid-focus", blueprint)

        self.assertFalse(advanced)
        self.assertEqual(
            cache.authoritative_resolution_stats["host_insert.skeleton_focus"], 1
        )
        self.assertEqual(
            cache.authoritative_host_insert_missing_by_reqid["rid-focus"], []
        )

    def test_select_last_host_node_for_hit_length_uses_shallower_boundary(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        root = object()
        cache.root_node = root
        cache._node_backup_visible = lambda node: True

        class Node:
            def __init__(self, node_id, host_len, parent):
                self.id = node_id
                self.host_value = [0] * host_len
                self.parent = parent

            @property
            def evicted(self):
                return True

        n1 = Node(81, 64, root)
        n2 = Node(82, 64, n1)
        n3 = Node(83, 64, n2)

        selected = cache._select_last_host_node_for_hit_length(n3, 128)

        self.assertIs(selected, n2)

    def test_clamp_match_result_updates_last_host_node_with_host_hit(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        root = object()
        cache.root_node = root

        class Node:
            def __init__(self, node_id, host_len, parent):
                self.id = node_id
                self.host_value = [0] * host_len
                self.parent = parent

            @property
            def evicted(self):
                return True

        n1 = Node(91, 64, root)
        n2 = Node(92, 64, n1)
        n3 = Node(93, 64, n2)
        cache.authoritative_prefetch_ready_by_reqid = {
            "rid-5": AuthoritativePrefetchReadySummary(
                req_id="rid-5",
                prefix_len=0,
                host_hit_length=128,
                storage_hit_length=0,
            )
        }
        cache._node_backup_visible = lambda node: True

        clamped = cache._clamp_match_result_to_authoritative_summary(
            "rid-5",
            MatchResult(
                device_indices=torch.empty((0,), dtype=torch.int64),
                last_device_node=root,
                last_host_node=n3,
                host_hit_length=192,
            ),
        )

        self.assertEqual(clamped.host_hit_length, 128)
        self.assertIs(clamped.last_host_node, n2)

    def test_clamp_ready_result_updates_last_host_node_with_host_hit(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        root = object()
        cache.root_node = root

        class Node:
            def __init__(self, node_id, host_len, parent):
                self.id = node_id
                self.host_value = [0] * host_len
                self.parent = parent

            @property
            def evicted(self):
                return True

        n1 = Node(101, 64, root)
        n2 = Node(102, 64, n1)
        n3 = Node(103, 64, n2)
        cache.authoritative_prefetch_ready_by_reqid = {
            "rid-6": AuthoritativePrefetchReadySummary(
                req_id="rid-6",
                prefix_len=0,
                host_hit_length=128,
                storage_hit_length=7,
            )
        }
        cache._node_backup_visible = lambda node: True

        clamped = cache._clamp_ready_result_to_authoritative_summary(
            "rid-6",
            LatchedPrefetchReadyResult(
                match_result=MatchResult(
                    device_indices=torch.empty((0,), dtype=torch.int64),
                    last_device_node=root,
                    last_host_node=n3,
                    host_hit_length=192,
                ),
                storage_hit_length=9,
            ),
        )

        self.assertEqual(clamped.storage_hit_length, 7)
        self.assertEqual(clamped.match_result.host_hit_length, 128)
        self.assertIs(clamped.match_result.last_host_node, n2)

    def test_match_prefix_helper_stops_at_invisible_host_only_child(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.get_child_key_fn = lambda key: key.token_ids[0]
        cache.key_match_fn = lambda key0, key1: min(len(key0), len(key1))
        cache._node_backup_visible = lambda node: False

        class Node:
            def __init__(self, node_id, key_tokens, value=None, parent=None):
                self.id = node_id
                self.key = RadixKey(token_ids=key_tokens, extra_key=None)
                self.value = value
                self.parent = parent
                self.children = {}
                self.host_value = [1]
                self.last_access_time = 0

            @property
            def evicted(self):
                return self.value is None

        root = Node(1, [], value=[])
        child = Node(2, [10, 11], value=None, parent=root)
        root.children[10] = child

        value, last_node = cache._match_prefix_helper(root, RadixKey([10, 11], None))

        self.assertEqual(value, [])
        self.assertIs(last_node, root)


if __name__ == "__main__":
    unittest.main()
