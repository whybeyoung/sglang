from types import SimpleNamespace

from sglang.srt.managers.cache_controller import HiCacheAck
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache


class _FakeEvent:
    def __init__(self, ready: bool):
        self._ready = ready
        self.synchronized = False

    def query(self):
        return self._ready

    def synchronize(self):
        self.synchronized = True


def _make_ack(node_ids, ready=True):
    finish_event = _FakeEvent(ready=ready)
    return HiCacheAck(
        start_event=None,
        finish_event=finish_event,
        node_ids=list(node_ids),
    )


def test_loading_check_consumes_only_budgeted_ready_ack_count():
    cache = HiRadixCache.__new__(HiRadixCache)
    first_node = object()
    second_node = object()
    third_node = object()
    ack0 = _make_ack([101], ready=True)
    ack1 = _make_ack([202], ready=True)
    ack2 = _make_ack([303], ready=False)

    cache.cache_controller = SimpleNamespace(
        ack_load_queue=[ack0, ack1, ack2],
        tp_rank=0,
    )
    cache.ongoing_load_back = {
        101: first_node,
        202: second_node,
        303: third_node,
    }
    cache.pp_load_ack_budget = 1
    cache.dec_lock_ref = lambda node: released_nodes.append(node)
    cache.pp_rank = 1
    cache.attn_cp_rank = 0
    released_nodes = []

    cache.loading_check()

    assert released_nodes == [first_node]
    assert cache.pp_load_ack_budget == 0
    assert 101 not in cache.ongoing_load_back
    assert cache.ongoing_load_back[202] is second_node
    assert cache.ongoing_load_back[303] is third_node
    assert cache.cache_controller.ack_load_queue == [ack1, ack2]
    assert ack0.finish_event.synchronized is True
    assert ack1.finish_event.synchronized is False
    assert ack2.finish_event.synchronized is False


def test_loading_check_without_budget_consumes_all_ready_acks():
    cache = HiRadixCache.__new__(HiRadixCache)
    first_node = object()
    second_node = object()
    ack0 = _make_ack([101], ready=True)
    ack1 = _make_ack([202], ready=True)

    cache.cache_controller = SimpleNamespace(
        ack_load_queue=[ack0, ack1],
        tp_rank=0,
    )
    cache.ongoing_load_back = {
        101: first_node,
        202: second_node,
    }
    cache.pp_load_ack_budget = None
    cache.dec_lock_ref = lambda node: released_nodes.append(node)
    cache.pp_rank = 0
    cache.attn_cp_rank = 0
    released_nodes = []

    cache.loading_check()

    assert released_nodes == [first_node, second_node]
    assert cache.pp_load_ack_budget is None
    assert cache.ongoing_load_back == {}
    assert cache.cache_controller.ack_load_queue == []
