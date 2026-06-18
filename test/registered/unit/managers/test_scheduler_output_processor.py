import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class _FakeMambaPool:
    def __init__(self):
        self.freed_indices = []

    def free(self, indices):
        self.freed_indices.append(indices.clone())


class TestSchedulerOutputProcessorFinishedReq(unittest.TestCase):
    def _new_scheduler(self, tree_cache):
        scheduler = SimpleNamespace()
        scheduler.server_args = SimpleNamespace(
            disaggregation_decode_enable_offload_kvcache=False,
            enable_hisparse=False,
        )
        scheduler.tree_cache = tree_cache
        scheduler.model_worker = SimpleNamespace()
        scheduler._mamba_prefix_cache_update = MagicMock()
        scheduler._maybe_collect_routed_experts = MagicMock()
        scheduler._maybe_collect_indexer_topk = MagicMock()
        scheduler._maybe_collect_customized_info = MagicMock()
        return scheduler

    def _new_finished_req(self, *, req_pool_idx, mamba_pool_idx=None):
        req = SimpleNamespace()
        req.rid = "finished-req"
        req.req_pool_idx = req_pool_idx
        req.mamba_pool_idx = mamba_pool_idx
        req.multimodal_inputs = None
        req.session = None
        req.kv_committed_freed = False
        req.kv_overallocated_freed = False
        req.time_stats = SimpleNamespace(set_completion_time=MagicMock())
        req.finished = MagicMock(return_value=True)
        return req

    def _handle_req(self, scheduler, req):
        with patch(
            "sglang.srt.managers.scheduler_components.batch_result_processor.get_global_server_args",
            return_value=SimpleNamespace(enable_mamba_extra_buffer_lazy=lambda: False),
        ):
            SchedulerBatchResultProcessor._handle_finish_state_updated_req(
                scheduler,
                req,
                batch=SimpleNamespace(),
                result=SimpleNamespace(),
                i=0,
                logits_output=None,
            )

    def test_finished_req_without_req_pool_idx_releases_mamba_slot(self):
        mamba_pool = _FakeMambaPool()
        tree_cache = SimpleNamespace(
            supports_mamba=MagicMock(return_value=True),
            req_to_token_pool=SimpleNamespace(mamba_allocator=mamba_pool),
        )
        scheduler = self._new_scheduler(tree_cache)
        req = self._new_finished_req(
            req_pool_idx=None, mamba_pool_idx=torch.tensor(7, dtype=torch.int64)
        )

        self._handle_req(scheduler, req)

        self.assertIsNone(req.mamba_pool_idx)
        self.assertEqual(len(mamba_pool.freed_indices), 1)
        torch.testing.assert_close(
            mamba_pool.freed_indices[0], torch.tensor([7], dtype=torch.int64)
        )
        tree_cache.supports_mamba.assert_called_once()
        req.time_stats.set_completion_time.assert_called_once()

    def test_finished_req_without_any_pool_idx_keeps_duplicate_kv_release_guard(self):
        tree_cache = SimpleNamespace(supports_mamba=MagicMock(return_value=True))
        scheduler = self._new_scheduler(tree_cache)
        req = self._new_finished_req(req_pool_idx=None, mamba_pool_idx=None)

        with patch(
            "sglang.srt.managers.scheduler_components.batch_result_processor.release_kv_cache"
        ) as release_kv_cache:
            self._handle_req(scheduler, req)

        release_kv_cache.assert_not_called()
        tree_cache.supports_mamba.assert_not_called()
        req.time_stats.set_completion_time.assert_called_once()

    def test_finished_req_with_req_pool_idx_uses_existing_release_path(self):
        tree_cache = MagicMock()
        scheduler = self._new_scheduler(tree_cache)
        scheduler.server_args.enable_hisparse = True
        scheduler.hisparse_coordinator = SimpleNamespace(request_finished=MagicMock())
        req = self._new_finished_req(req_pool_idx=3)

        with patch(
            "sglang.srt.managers.scheduler_components.batch_result_processor.release_kv_cache"
        ) as release_kv_cache:
            self._handle_req(scheduler, req)

        scheduler.hisparse_coordinator.request_finished.assert_called_once_with(req)
        release_kv_cache.assert_called_once_with(req, tree_cache, is_insert=True)
        req.time_stats.set_completion_time.assert_called_once()


if __name__ == "__main__":
    unittest.main()
