import sys
import types
import unittest
from types import SimpleNamespace

import torch

if "IPython" not in sys.modules:
    ipython_module = types.ModuleType("IPython")
    ipython_display = types.ModuleType("IPython.display")
    ipython_display.HTML = lambda *args, **kwargs: None
    ipython_display.display = lambda *args, **kwargs: None
    ipython_module.display = ipython_display
    sys.modules["IPython"] = ipython_module
    sys.modules["IPython.display"] = ipython_display

from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_pp_mixin import (
    PPPrefillBatchContract,
    PPPrefillReadyView,
    SchedulerPPMixin,
    _ordered_intersection_prefill_ready_views,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, suite="stage-b-test-small-1-gpu")


class DummyScheduler(SchedulerPPMixin):
    def __init__(self, *, is_first_rank: bool):
        self.pp_group = SimpleNamespace(
            is_first_rank=is_first_rank, is_last_rank=not is_first_rank
        )
        self.pp_rank = 0 if is_first_rank else 1
        self.pp_size = 2
        self.attn_cp_size = 1
        self.attn_tp_size = 1
        self.attn_cp_rank = 0
        self.attn_tp_rank = 0
        self.waiting_queue = []
        self.chunked_req = None
        self.mbs = []
        self.running_mbs = []
        self.last_mbs = []
        self.disagg_prefill_inflight_queue = []
        self._sgl_pp_authoritative_prefill_retry_counts = {}
        self._sgl_pp_authoritative_prefill_contract_retry_counts = {}
        self._sgl_pp_upstream_prefill_batch_contract = None
        self._sgl_pp_scheduled_chunked_rid = None
        self.enable_hicache_storage = False
        self.enable_hierarchical_cache = False


class DummyPrefillScheduler(SchedulerDisaggregationPrefillMixin):
    def __init__(self):
        self.chunked_req = None
        self.waiting_queue = []
        self.last_batch = None
        self.enable_overlap = False
        self.running_batch = SimpleNamespace(batch_is_full=True)
        self.tree_cache = SimpleNamespace(
            cache_unfinished_req=lambda req, chunked=False: None
        )


def make_req(rid: str, *, extend_batch_idx: int = 0, is_chunked: int = 0):
    return SimpleNamespace(
        rid=rid,
        origin_input_ids=[1] * 16,
        output_ids=[],
        fill_ids=[1] * 16,
        extend_input_len=16,
        extend_batch_idx=extend_batch_idx,
        is_chunked=is_chunked,
        prefix_indices=torch.empty((0,), dtype=torch.int64),
        host_hit_length=0,
        storage_hit_length=0,
        return_logprob=False,
        logprob_start_len=0,
        extra_key=None,
        is_dllm=lambda: False,
    )


class TestPPAuthoritativePrefillRetry(CustomTestCase):
    def test_attn_cp_phase_ready_passthrough_single_rank(self):
        scheduler = DummyScheduler(is_first_rank=False)

        self.assertTrue(scheduler._pp_attn_cp_phase_ready(True))
        self.assertFalse(scheduler._pp_attn_cp_phase_ready(False))

    def test_ready_view_intersection_uses_downstream_upper_bound(self):
        views = _ordered_intersection_prefill_ready_views(
            [
                PPPrefillReadyView("rid-a", 16576, 1),
                PPPrefillReadyView("rid-b", 192, 1),
            ],
            [
                PPPrefillReadyView("rid-a", 192, 1),
                PPPrefillReadyView("rid-b", 16576, 1),
            ],
        )

        self.assertEqual(
            views,
            [PPPrefillReadyView("rid-a", 192, 1), PPPrefillReadyView("rid-b", 192, 1)],
        )

    def test_ready_view_intersection_requires_same_next_extend_batch_idx(self):
        views = _ordered_intersection_prefill_ready_views(
            [PPPrefillReadyView("rid-a", 16576, 2)],
            [PPPrefillReadyView("rid-a", 16576, 1)],
        )

        self.assertEqual(views, [])

    def test_empty_local_batch_becomes_retry_decision(self):
        scheduler = DummyScheduler(is_first_rank=False)
        scheduler.chunked_req = make_req("rid-chunk")

        authoritative_contract = [
            PPPrefillBatchContract(
                rid="rid-chunk",
                fill_len=16384,
                prefix_len=0,
                extend_len=16384,
                extend_batch_idx=1,
                is_chunked=1,
            )
        ]

        decision = scheduler._pp_partition_authoritative_prefill_batch_contract(
            batch=None,
            authoritative_contract=authoritative_contract,
        )

        self.assertEqual(decision.accepted_contract, [])
        self.assertEqual(decision.unexpected_local_entries, [])
        self.assertEqual(len(decision.retry_entries), 1)
        self.assertEqual(decision.retry_entries[0]["rid"], "rid-chunk")
        self.assertEqual(
            decision.retry_entries[0]["reason"], "missing_before_proxy_recv"
        )
        self.assertEqual(decision.retry_entries[0]["retry_count"], 1)

    def test_authoritative_candidates_keep_materialized_subset(self):
        scheduler = DummyScheduler(is_first_rank=False)
        good_req = make_req("rid-good", extend_batch_idx=0, is_chunked=0)
        bad_req = make_req("rid-bad", extend_batch_idx=1, is_chunked=0)
        scheduler.waiting_queue = [good_req, bad_req]
        scheduler._sgl_pp_upstream_prefill_batch_contract = [
            PPPrefillBatchContract(
                rid="rid-good",
                fill_len=16,
                prefix_len=0,
                extend_len=16,
                extend_batch_idx=1,
                is_chunked=0,
            ),
            PPPrefillBatchContract(
                rid="rid-bad",
                fill_len=16,
                prefix_len=0,
                extend_len=16,
                extend_batch_idx=1,
                is_chunked=0,
            ),
        ]
        scheduler._sgl_pp_scheduled_chunked_rid = None

        selected = scheduler._pp_get_authoritative_prefill_candidates(
            scheduler.waiting_queue
        )

        self.assertEqual([req.rid for req in selected], ["rid-good"])
        self.assertEqual(
            getattr(good_req, "_sgl_pp_authoritative_prefill_contract").rid,
            "rid-good",
        )
        self.assertFalse(hasattr(bad_req, "_sgl_pp_authoritative_prefill_contract"))

    def test_local_ready_views_preview_next_round_host_hits(self):
        scheduler = DummyScheduler(is_first_rank=False)
        scheduler.enable_hicache_storage = True
        scheduler.enable_hierarchical_cache = True
        req = make_req("rid-host-hit")
        scheduler.waiting_queue = [req]

        scheduler.tree_cache = SimpleNamespace(
            check_hicache_events=lambda: None,
            check_prefetch_progress=lambda rid: rid == "rid-host-hit",
            supports_mamba=lambda: False,
            match_prefix=lambda params: SimpleNamespace(
                device_indices=torch.empty((0,), dtype=torch.int64),
                host_hit_length=15360,
            ),
        )

        ready_views = scheduler._pp_get_local_prefill_ready_views()

        self.assertEqual(len(ready_views), 1)
        self.assertEqual(ready_views[0].rid, "rid-host-hit")
        self.assertEqual(ready_views[0].ready_len, 15360)
        self.assertEqual(ready_views[0].next_extend_batch_idx, 1)

    def test_local_ready_views_skip_inflight_req_not_visible_next_round(self):
        scheduler = DummyScheduler(is_first_rank=False)
        scheduler.enable_hicache_storage = True
        scheduler.enable_hierarchical_cache = True
        visible_req = make_req("rid-visible")
        inflight_req = make_req("rid-inflight")
        scheduler.waiting_queue = [visible_req, inflight_req]
        scheduler.running_mbs = [
            SimpleNamespace(
                reqs=[SimpleNamespace(rid="rid-inflight")],
                chunked_req=None,
            )
        ]

        scheduler.tree_cache = SimpleNamespace(
            check_hicache_events=lambda: None,
            check_prefetch_progress=lambda rid: True,
            supports_mamba=lambda: False,
            match_prefix=lambda params: SimpleNamespace(
                device_indices=torch.empty((0,), dtype=torch.int64),
                host_hit_length=15360,
            ),
        )

        ready_views = scheduler._pp_get_local_prefill_ready_views()

        self.assertEqual(
            ready_views, [PPPrefillReadyView("rid-visible", 15360, 1)]
        )

    def test_first_rank_validate_allows_shaped_batch_below_cap(self):
        scheduler = DummyScheduler(is_first_rank=True)
        batch = SimpleNamespace(
            reqs=[
                SimpleNamespace(
                    rid="rid-a",
                    prefix_indices=torch.arange(192, dtype=torch.int64),
                    extend_batch_idx=1,
                )
            ]
        )

        scheduler._pp_validate_authoritative_prefill_ready_views(
            batch, [PPPrefillReadyView("rid-a", 16576, 1)]
        )

    def test_first_rank_validate_ignores_next_round_batch_idx_delta(self):
        scheduler = DummyScheduler(is_first_rank=True)
        batch = SimpleNamespace(
            reqs=[
                SimpleNamespace(
                    rid="rid-a",
                    prefix_indices=torch.arange(192, dtype=torch.int64),
                    extend_batch_idx=2,
                )
            ]
        )

        scheduler._pp_validate_authoritative_prefill_ready_views(
            batch, [PPPrefillReadyView("rid-a", 16576, 1)]
        )

    def test_first_rank_filters_duplicate_authoritative_ready_views(self):
        scheduler = DummyScheduler(is_first_rank=True)
        scheduler.mbs = [
            SimpleNamespace(
                reqs=[SimpleNamespace(rid="rid-a")],
                chunked_req=None,
            )
        ]

        filtered = scheduler._pp_filter_duplicate_authoritative_prefill_ready_views(
            [
                PPPrefillReadyView("rid-a", 16576, 1),
                PPPrefillReadyView("rid-b", 192, 1),
            ]
        )

        self.assertEqual(filtered, [PPPrefillReadyView("rid-b", 192, 1)])

    def test_non_first_rank_validate_requires_exact_batch_contract(self):
        scheduler = DummyScheduler(is_first_rank=False)
        batch = SimpleNamespace(
            reqs=[
                SimpleNamespace(
                    rid="rid-a",
                    fill_ids=[1] * 192,
                    prefix_indices=torch.arange(192, dtype=torch.int64),
                    extend_input_len=0,
                    extend_batch_idx=1,
                    is_chunked=0,
                )
            ]
        )

        with self.assertRaises(RuntimeError):
            scheduler._pp_validate_authoritative_prefill_batch_contract(
                batch,
                [
                    PPPrefillBatchContract(
                        rid="rid-a",
                        fill_len=192,
                        prefix_len=128,
                        extend_len=64,
                        extend_batch_idx=1,
                        is_chunked=0,
                    )
                ],
            )

    def test_send_consensus_bootstrapped_ids_skips_when_downstream_mb_not_ready(self):
        scheduler = DummyScheduler(is_first_rank=True)
        scheduler._pp_send_pyobj_to_next_stage = unittest.mock.Mock(return_value=["work"])

        work, consensus = scheduler._pp_pd_send_consensus_bootstrapped_ids(
            bmbs=[None, [["rid-a"], []]],
            next_first_rank_mb_id=0,
            next_mb_id=0,
            consensus_bootstrapped_rids=[["rid-a"], []],
            bootstrapped_rids=[["rid-a"], []],
        )

        self.assertEqual(work, [])
        self.assertEqual(consensus, [["rid-a"], []])
        scheduler._pp_send_pyobj_to_next_stage.assert_not_called()

    def test_send_consensus_bootstrapped_ids_non_last_sends_bootstrapped_when_consensus_none(
        self,
    ):
        scheduler = DummyScheduler(is_first_rank=True)
        scheduler._pp_send_pyobj_to_next_stage = unittest.mock.Mock(return_value=["work"])
        br = [["rid-a"], []]

        work, consensus = scheduler._pp_pd_send_consensus_bootstrapped_ids(
            bmbs=[br, None],
            next_first_rank_mb_id=0,
            next_mb_id=0,
            consensus_bootstrapped_rids=None,
            bootstrapped_rids=br,
        )

        self.assertEqual(work, ["work"])
        self.assertEqual(consensus, br)
        scheduler._pp_send_pyobj_to_next_stage.assert_called_once_with(br, async_send=True)

    def test_send_consensus_release_ids_skips_when_downstream_mb_not_ready(self):
        scheduler = DummyScheduler(is_first_rank=True)
        scheduler._pp_send_pyobj_to_next_stage = unittest.mock.Mock(return_value=["work"])

        work, release_rids = scheduler._pp_pd_send_consensus_release_ids(
            tmbs=[None, ["rid-a"]],
            next_first_rank_mb_id=0,
            next_mb_id=0,
            release_rids=["rid-a"],
            transferred_rids=["rid-a"],
        )

        self.assertEqual(work, [])
        self.assertEqual(release_rids, ["rid-a"])
        scheduler._pp_send_pyobj_to_next_stage.assert_not_called()

    def test_process_batch_result_disagg_prefill_skips_released_req(self):
        scheduler = DummyPrefillScheduler()
        scheduler.tree_cache = SimpleNamespace(cache_unfinished_req=unittest.mock.Mock())
        scheduler.disagg_prefill_inflight_queue = []
        scheduler.spec_algorithm = SimpleNamespace(is_eagle=lambda: False)
        scheduler.current_scheduler_metrics_enabled = False
        scheduler.send_kv_chunk = unittest.mock.Mock()
        req = make_req("rid-released")
        req.req_pool_idx = None
        req.time_stats = SimpleNamespace(
            set_prefill_finished_time=unittest.mock.Mock(),
        )
        req.grammar = None
        batch = SimpleNamespace(
            reqs=[req],
            return_logprob=False,
            spec_info=None,
        )
        result = SimpleNamespace(
            logits_output=SimpleNamespace(
                next_token_logprobs=None,
                input_token_logprobs=None,
            ),
            next_token_ids=torch.tensor([1], dtype=torch.int64),
            extend_input_len_per_req=None,
            extend_logprob_start_len_per_req=None,
            copy_done=None,
            can_run_cuda_graph=False,
        )

        scheduler.process_batch_result_disagg_prefill(batch, result)

        scheduler.tree_cache.cache_unfinished_req.assert_not_called()
        scheduler.send_kv_chunk.assert_not_called()
        self.assertEqual(scheduler.disagg_prefill_inflight_queue, [])
        self.assertEqual(req.output_ids, [])

    def test_process_batch_result_disagg_prefill_finalizes_aborted_req(self):
        scheduler = DummyPrefillScheduler()
        scheduler.tree_cache = SimpleNamespace(cache_unfinished_req=unittest.mock.Mock())
        scheduler.disagg_prefill_inflight_queue = []
        scheduler.spec_algorithm = SimpleNamespace(is_eagle=lambda: False)
        scheduler.current_scheduler_metrics_enabled = False
        scheduler.send_kv_chunk = unittest.mock.Mock()
        scheduler.stream_output = unittest.mock.Mock()
        scheduler.req_to_metadata_buffer_idx_allocator = SimpleNamespace(
            free=unittest.mock.Mock()
        )
        req = make_req("rid-aborted")
        abort_marker = SimpleNamespace(kind="abort")
        req.req_pool_idx = 0
        req._sgl_abort_requested = True
        req.finished_reason = None
        req.to_finish = abort_marker
        req.metadata_buffer_index = 7
        req.time_stats = SimpleNamespace(
            set_prefill_finished_time=unittest.mock.Mock(),
            set_completion_time=unittest.mock.Mock(),
        )
        req.grammar = None
        req.finished = lambda: req.finished_reason is not None
        req.check_finished = lambda: (
            setattr(req, "finished_reason", req.to_finish),
            setattr(req, "to_finish", None),
        )
        req.disagg_kv_sender = SimpleNamespace(clear=unittest.mock.Mock())
        batch = SimpleNamespace(
            reqs=[req],
            return_logprob=False,
            spec_info=None,
        )
        result = SimpleNamespace(
            logits_output=SimpleNamespace(
                next_token_logprobs=None,
                input_token_logprobs=None,
            ),
            next_token_ids=torch.tensor([1], dtype=torch.int64),
            extend_input_len_per_req=None,
            extend_logprob_start_len_per_req=None,
            copy_done=None,
            can_run_cuda_graph=False,
        )

        with unittest.mock.patch(
            "sglang.srt.disaggregation.prefill.release_kv_cache"
        ) as release_kv_cache:
            scheduler.process_batch_result_disagg_prefill(batch, result)

        scheduler.tree_cache.cache_unfinished_req.assert_not_called()
        scheduler.send_kv_chunk.assert_not_called()
        self.assertEqual(scheduler.disagg_prefill_inflight_queue, [])
        self.assertEqual(req.output_ids, [])
        self.assertIs(req.finished_reason, abort_marker)
        release_kv_cache.assert_called_once_with(req, scheduler.tree_cache)
        req.disagg_kv_sender.clear.assert_called_once()
        scheduler.stream_output.assert_called_once_with([req], False, None)
        scheduler.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(7)

    def test_process_batch_result_disagg_prefill_skips_aborted_chunked_req(self):
        scheduler = DummyPrefillScheduler()
        scheduler.tree_cache = SimpleNamespace(cache_unfinished_req=unittest.mock.Mock())
        scheduler.disagg_prefill_inflight_queue = []
        scheduler.spec_algorithm = SimpleNamespace(is_eagle=lambda: False)
        scheduler.current_scheduler_metrics_enabled = False
        scheduler.enable_overlap = True
        scheduler.send_kv_chunk = unittest.mock.Mock()
        req = make_req("rid-chunk-aborted", is_chunked=1)
        req.req_pool_idx = None
        req._sgl_abort_requested = True
        req.tmp_end_idx = 16
        req.time_stats = SimpleNamespace(
            set_last_chunked_prefill_finish_time=unittest.mock.Mock(),
        )
        batch = SimpleNamespace(
            reqs=[req],
            return_logprob=False,
            spec_info=None,
        )
        result = SimpleNamespace(
            logits_output=SimpleNamespace(
                next_token_logprobs=None,
                input_token_logprobs=None,
            ),
            next_token_ids=torch.tensor([1], dtype=torch.int64),
            extend_input_len_per_req=None,
            extend_logprob_start_len_per_req=None,
            copy_done=None,
            can_run_cuda_graph=False,
        )

        scheduler.process_batch_result_disagg_prefill(batch, result)

        scheduler.tree_cache.cache_unfinished_req.assert_not_called()
        scheduler.send_kv_chunk.assert_not_called()
        self.assertEqual(req.is_chunked, 1)
        req.time_stats.set_last_chunked_prefill_finish_time.assert_not_called()

    def test_abort_request_running_prefill_req_defers_hicache_cleanup(self):
        req = make_req("rid-running")
        req.finished = lambda: False
        req.disagg_kv_sender = SimpleNamespace(abort=unittest.mock.Mock())

        scheduler = SimpleNamespace(
            waiting_queue=[],
            grammar_manager=SimpleNamespace(abort_requests=unittest.mock.Mock()),
            disaggregation_mode=DisaggregationMode.PREFILL,
            disagg_prefill_bootstrap_queue=SimpleNamespace(queue=[]),
            disagg_prefill_inflight_queue=[],
            disagg_decode_prealloc_queue=SimpleNamespace(queue=[], retracted_queue=[]),
            disagg_decode_transfer_queue=SimpleNamespace(queue=[]),
            cur_batch=None,
            running_batch=SimpleNamespace(reqs=[req]),
            chunked_req=None,
            enable_hicache_storage=True,
            tree_cache=SimpleNamespace(release_aborted_request=unittest.mock.Mock()),
            req_to_metadata_buffer_idx_allocator=SimpleNamespace(free=unittest.mock.Mock()),
            send_to_tokenizer=SimpleNamespace(send_output=unittest.mock.Mock()),
        )

        Scheduler.abort_request(scheduler, AbortReq(rid=req.rid))

        self.assertIsNotNone(req.to_finish)
        req.disagg_kv_sender.abort.assert_called_once()
        scheduler.tree_cache.release_aborted_request.assert_not_called()

    def test_abort_request_inactive_chunked_prefill_req_cleans_up_immediately(self):
        req = make_req("rid-chunked")
        req.req_pool_idx = 3
        req.metadata_buffer_index = 9
        req.finished = lambda: False
        req.disagg_kv_sender = SimpleNamespace(abort=unittest.mock.Mock())

        scheduler = SimpleNamespace(
            waiting_queue=[],
            grammar_manager=SimpleNamespace(abort_requests=unittest.mock.Mock()),
            disaggregation_mode=DisaggregationMode.PREFILL,
            disagg_prefill_bootstrap_queue=SimpleNamespace(queue=[]),
            disagg_prefill_inflight_queue=[],
            disagg_decode_prealloc_queue=SimpleNamespace(queue=[], retracted_queue=[]),
            disagg_decode_transfer_queue=SimpleNamespace(queue=[]),
            cur_batch=None,
            running_batch=SimpleNamespace(reqs=[]),
            chunked_req=req,
            enable_hicache_storage=True,
            tree_cache=SimpleNamespace(release_aborted_request=unittest.mock.Mock()),
            req_to_metadata_buffer_idx_allocator=SimpleNamespace(free=unittest.mock.Mock()),
            send_to_tokenizer=SimpleNamespace(send_output=unittest.mock.Mock()),
        )

        with unittest.mock.patch(
            "sglang.srt.managers.scheduler.release_kv_cache"
        ) as release_kv_cache, unittest.mock.patch(
            "sglang.srt.managers.scheduler.release_req_to_metadata_buffer"
        ) as release_req_to_metadata_buffer:
            Scheduler.abort_request(scheduler, AbortReq(rid=req.rid))

        req.disagg_kv_sender.abort.assert_called_once()
        scheduler.tree_cache.release_aborted_request.assert_called_once_with(req.rid)
        release_kv_cache.assert_called_once_with(req, scheduler.tree_cache, is_insert=False)
        release_req_to_metadata_buffer.assert_called_once_with(
            req, scheduler.req_to_metadata_buffer_idx_allocator
        )
        self.assertIsNone(scheduler.chunked_req)

    def test_abort_request_active_chunked_prefill_req_defers_main_release(self):
        req = make_req("rid-active-chunked")
        req.req_pool_idx = 5
        req.finished = lambda: False
        req.disagg_kv_sender = SimpleNamespace(abort=unittest.mock.Mock())

        scheduler = SimpleNamespace(
            waiting_queue=[],
            grammar_manager=SimpleNamespace(abort_requests=unittest.mock.Mock()),
            disaggregation_mode=DisaggregationMode.PREFILL,
            disagg_prefill_bootstrap_queue=SimpleNamespace(queue=[]),
            disagg_prefill_inflight_queue=[],
            disagg_decode_prealloc_queue=SimpleNamespace(queue=[], retracted_queue=[]),
            disagg_decode_transfer_queue=SimpleNamespace(queue=[]),
            cur_batch=None,
            running_batch=SimpleNamespace(reqs=[req]),
            chunked_req=req,
            enable_hicache_storage=True,
            tree_cache=SimpleNamespace(release_aborted_request=unittest.mock.Mock()),
            req_to_metadata_buffer_idx_allocator=SimpleNamespace(free=unittest.mock.Mock()),
            send_to_tokenizer=SimpleNamespace(send_output=unittest.mock.Mock()),
        )

        with unittest.mock.patch(
            "sglang.srt.managers.scheduler.release_kv_cache"
        ) as release_kv_cache, unittest.mock.patch(
            "sglang.srt.managers.scheduler.release_req_to_metadata_buffer"
        ) as release_req_to_metadata_buffer:
            Scheduler.abort_request(scheduler, AbortReq(rid=req.rid))

        req.disagg_kv_sender.abort.assert_called_once()
        scheduler.tree_cache.release_aborted_request.assert_not_called()
        release_kv_cache.assert_not_called()
        release_req_to_metadata_buffer.assert_not_called()
        self.assertIsNotNone(req.to_finish)
        self.assertIsNone(scheduler.chunked_req)

    def test_process_disagg_prefill_inflight_queue_failed_req_releases_hicache_state(self):
        scheduler = DummyPrefillScheduler()
        scheduler.enable_hicache_storage = True
        scheduler.attn_cp_cpu_group = None
        scheduler.attn_tp_cpu_group = None
        scheduler.tp_rank = 0
        scheduler.enable_metrics = False
        scheduler.stream_output = unittest.mock.Mock()
        scheduler.tree_cache = SimpleNamespace(
            release_aborted_request=unittest.mock.Mock(),
        )
        scheduler.req_to_metadata_buffer_idx_allocator = SimpleNamespace(
            free=unittest.mock.Mock()
        )

        req = make_req("rid-failed")
        req.req_pool_idx = 1
        req.metadata_buffer_index = 11
        req.bootstrap_room = None
        req.disagg_kv_sender = SimpleNamespace(
            abort=unittest.mock.Mock(),
            clear=unittest.mock.Mock(),
        )
        req.time_stats = SimpleNamespace(set_completion_time=unittest.mock.Mock())
        req._sgl_pp_transfer_pending_poll = KVPoll.Failed
        scheduler.disagg_prefill_inflight_queue = [req]

        with unittest.mock.patch(
            "sglang.srt.disaggregation.prefill.release_kv_cache"
        ) as release_kv_cache, unittest.mock.patch(
            "sglang.srt.disaggregation.prefill.release_req_to_metadata_buffer"
        ) as release_req_to_metadata_buffer:
            done = scheduler.process_disagg_prefill_inflight_queue(
                use_pending_poll=True,
                authoritative_abort=False,
            )

        self.assertEqual(done, [req])
        scheduler.tree_cache.release_aborted_request.assert_called_once_with(req.rid)
        release_kv_cache.assert_called_once_with(req, scheduler.tree_cache)
        req.disagg_kv_sender.abort.assert_called_once()
        req.disagg_kv_sender.clear.assert_called_once()
        release_req_to_metadata_buffer.assert_called_once_with(
            req, scheduler.req_to_metadata_buffer_idx_allocator
        )
        scheduler.stream_output.assert_called_once_with([req], False, None)

    def test_send_kv_chunk_finalizes_last_chunk_without_new_pages(self):
        scheduler = DummyPrefillScheduler()
        req = make_req("rid-final", is_chunked=0)
        req.req_pool_idx = 0
        req.start_send_idx = 64
        req.fill_ids = [1] * 64
        req.origin_input_ids = [1] * 64
        req.disagg_kv_sender = SimpleNamespace(send=unittest.mock.Mock())
        scheduler.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.zeros((1, 64), dtype=torch.int32),
            req_index_to_mamba_index_mapping=torch.zeros((1,), dtype=torch.int32),
        )
        scheduler.token_to_kv_pool_allocator = SimpleNamespace(
            page_size=64,
            get_kvcache=lambda: object(),
        )
        scheduler.disagg_metadata_buffers = SimpleNamespace(set_buf=unittest.mock.Mock())

        sent = scheduler.send_kv_chunk(req, last_chunk=True)

        self.assertTrue(sent)
        scheduler.disagg_metadata_buffers.set_buf.assert_called_once_with(req)
        req.disagg_kv_sender.send.assert_called_once()
        args, _ = req.disagg_kv_sender.send.call_args
        self.assertEqual(len(args[0]), 0)
        self.assertIsNone(args[1])

    def test_prefill_adder_rejects_second_chunked_candidate(self):
        tree_cache = SimpleNamespace(
            supports_mamba=lambda: False,
            supports_swa=lambda: False,
            is_tree_cache=lambda: False,
            inc_lock_ref=lambda node: None,
            dec_lock_ref=lambda node: None,
            evictable_size=lambda: 0,
        )
        token_allocator = SimpleNamespace(available_size=lambda: 1_000_000)
        req = SimpleNamespace(
            sampling_params=SimpleNamespace(ignore_eos=False, max_new_tokens=16),
            extend_input_len=256,
            output_ids=[],
            host_hit_length=0,
            prefix_indices=torch.empty((0,), dtype=torch.int64),
            last_node=object(),
            fill_ids=list(range(256)),
        )
        req.set_extend_input_len = lambda extend_input_len: setattr(
            req, "extend_input_len", extend_input_len
        )
        adder = PrefillAdder(
            page_size=64,
            tree_cache=tree_cache,
            token_to_kv_pool_allocator=token_allocator,
            running_batch=None,
            new_token_ratio=1.0,
            rem_input_tokens=4096,
            rem_chunk_tokens=64,
        )

        result = adder.add_one_req(
            req, has_chunked_req=True, truncation_align_size=None
        )

        self.assertEqual(result, AddReqResult.OTHER)
        self.assertEqual(adder.can_run_list, [])
        self.assertIsNone(adder.new_chunked_req)

    def test_prefill_adder_exact_chunked_req_does_not_retruncate(self):
        tree_cache = SimpleNamespace(
            supports_mamba=lambda: False,
            supports_swa=lambda: False,
            is_tree_cache=lambda: False,
            inc_lock_ref=lambda node: None,
            dec_lock_ref=lambda node: None,
            evictable_size=lambda: 0,
        )
        token_allocator = SimpleNamespace(available_size=lambda: 1_000_000)
        req = SimpleNamespace(
            sampling_params=SimpleNamespace(ignore_eos=False, max_new_tokens=16),
            extend_input_len=128,
            output_ids=[],
            host_hit_length=0,
            prefix_indices=torch.empty((0,), dtype=torch.int64),
            last_node=object(),
            fill_ids=list(range(128)),
        )
        req.set_extend_input_len = lambda extend_input_len: setattr(
            req, "extend_input_len", extend_input_len
        )
        adder = PrefillAdder(
            page_size=64,
            tree_cache=tree_cache,
            token_to_kv_pool_allocator=token_allocator,
            running_batch=None,
            new_token_ratio=1.0,
            rem_input_tokens=4096,
            rem_chunk_tokens=64,
        )

        req._sgl_pp_force_exact_chunked = True
        req._sgl_pp_force_exact_chunked_truncated = True
        returned_req = adder.add_chunked_req(req)

        self.assertIs(returned_req, req)
        self.assertEqual(req.extend_input_len, 128)
        self.assertEqual(len(req.fill_ids), 128)


if __name__ == "__main__":
    unittest.main(verbosity=3)
