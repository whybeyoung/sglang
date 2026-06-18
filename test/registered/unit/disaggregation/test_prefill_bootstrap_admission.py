import threading
import time
import types
import unittest
from collections import defaultdict, deque
from types import SimpleNamespace

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVManager,
    MooncakeKVReceiver,
    MooncakeKVSender,
    TransferKVChunk,
)
from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue
from sglang.srt.disaggregation.utils import ReqToMetadataIdxAllocator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


def _make_req(rid: str):
    return SimpleNamespace(
        rid=rid,
        origin_input_ids=[1, 2, 3],
        sampling_params=SimpleNamespace(max_new_tokens=8),
        finished_reason=None,
        return_logprob=False,
        disagg_kv_sender=None,
    )


def _make_queue(capacity: int):
    queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
    queue.queue = []
    queue.pending_queue = deque()
    queue.req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(capacity)
    queue.max_total_num_tokens = 1024
    queue.scheduler = SimpleNamespace(
        waiting_queue=[],
        disagg_prefill_inflight_queue=[],
        cur_batch=None,
        last_batch=None,
        running_batch=SimpleNamespace(reqs=[]),
        model_config=SimpleNamespace(num_key_value_heads=1),
    )

    def fake_create_kv_sender(self, req, num_kv_heads):
        req.disagg_kv_sender = object()
        self.queue.append(req)

    queue._create_kv_sender = types.MethodType(fake_create_kv_sender, queue)
    return queue


class TestPrefillBootstrapAdmission(unittest.TestCase):
    def test_admission_defers_requests_without_sender(self):
        queue = _make_queue(capacity=2)
        reqs = [_make_req(f"req-{i}") for i in range(3)]

        for req in reqs:
            queue.add(req, num_kv_heads=1)

        self.assertEqual([req.rid for req in queue.queue], ["req-0", "req-1"])
        self.assertEqual([req.rid for req in queue.pending_queue], ["req-2"])
        self.assertIsNotNone(reqs[0].disagg_kv_sender)
        self.assertIsNotNone(reqs[1].disagg_kv_sender)
        self.assertIsNone(reqs[2].disagg_kv_sender)

    def test_pending_request_admits_when_active_sender_finishes(self):
        queue = _make_queue(capacity=2)
        reqs = [_make_req(f"req-{i}") for i in range(3)]
        for req in reqs:
            queue.add(req, num_kv_heads=1)

        queue.queue.pop(0)
        queue._admit_pending(num_kv_heads=1)

        self.assertEqual([req.rid for req in queue.queue], ["req-1", "req-2"])
        self.assertEqual(list(queue.pending_queue), [])
        self.assertIsNotNone(reqs[2].disagg_kv_sender)

    def test_waiting_and_inflight_reqs_count_against_admission(self):
        queue = _make_queue(capacity=2)
        active_waiting = _make_req("waiting")
        active_waiting.disagg_kv_sender = object()
        queue.scheduler.waiting_queue = [active_waiting]

        queue.add(_make_req("req-0"), num_kv_heads=1)
        queue.add(_make_req("req-1"), num_kv_heads=1)

        self.assertEqual([req.rid for req in queue.queue], ["req-0"])
        self.assertEqual([req.rid for req in queue.pending_queue], ["req-1"])

    def test_running_batch_reqs_count_against_admission(self):
        queue = _make_queue(capacity=2)
        active_running = _make_req("running")
        active_running.disagg_kv_sender = object()
        queue.scheduler.running_batch = SimpleNamespace(reqs=[active_running])

        queue.add(_make_req("req-0"), num_kv_heads=1)
        queue.add(_make_req("req-1"), num_kv_heads=1)

        self.assertEqual([req.rid for req in queue.queue], ["req-0"])
        self.assertEqual([req.rid for req in queue.pending_queue], ["req-1"])

    def test_mooncake_sender_bootstrap_timeout_starts_after_metadata_arrives(self):
        class FakeKVManager:
            bootstrap_timeout = 60.0
            transfer_infos = {}

            def check_status(self, bootstrap_room):
                return KVPoll.Bootstrapping

            def record_failure(self, bootstrap_room, reason):
                raise AssertionError("bootstrap should not fail before timeout")

            def update_status(self, bootstrap_room, status):
                raise AssertionError("bootstrap should not update before timeout")

        sender = MooncakeKVSender.__new__(MooncakeKVSender)
        sender.conclude_state = None
        sender.kv_mgr = FakeKVManager()
        sender.bootstrap_room = 7
        sender.init_time = None
        sender.transfer_init_time = None
        sender.trace_ctx = SimpleNamespace(trace_req_finish=lambda: None)

        self.assertEqual(sender.poll(), KVPoll.Bootstrapping)
        self.assertIsNone(sender.init_time)

        sender.kv_mgr.transfer_infos[sender.bootstrap_room] = {"session": object()}
        before_poll = time.time()

        self.assertEqual(sender.poll(), KVPoll.Bootstrapping)
        self.assertIsNotNone(sender.init_time)
        self.assertGreaterEqual(sender.init_time, before_poll)

    def test_mooncake_sender_bootstrap_timeout_fails_after_metadata_timeout(self):
        class FakeKVManager:
            bootstrap_timeout = 60.0

            def __init__(self):
                self.status = KVPoll.Bootstrapping
                self.failures = []
                self.transfer_infos = {7: {"session": object()}}

            def check_status(self, bootstrap_room):
                return self.status

            def record_failure(self, bootstrap_room, reason):
                self.failures.append((bootstrap_room, reason))

            def update_status(self, bootstrap_room, status):
                self.status = status

        sender = MooncakeKVSender.__new__(MooncakeKVSender)
        sender.conclude_state = None
        sender.kv_mgr = FakeKVManager()
        sender.bootstrap_room = 7
        sender.init_time = time.time() - sender.kv_mgr.bootstrap_timeout - 1
        sender.transfer_init_time = None
        sender.trace_ctx = SimpleNamespace(trace_req_finish=lambda: None)

        self.assertEqual(sender.poll(), KVPoll.Failed)
        self.assertEqual(sender.kv_mgr.status, KVPoll.Failed)
        self.assertIn("KVPoll.Bootstrapping", sender.kv_mgr.failures[0][1])

    def test_mooncake_sender_transfer_timeout_fails_inflight_request(self):
        class FakeKVManager:
            waiting_timeout = 60.0

            def __init__(self):
                self.status = KVPoll.Transferring
                self.failures = []

            def check_status(self, bootstrap_room):
                return self.status

            def record_failure(self, bootstrap_room, reason):
                self.failures.append((bootstrap_room, reason))

            def update_status(self, bootstrap_room, status):
                self.status = status

        sender = MooncakeKVSender.__new__(MooncakeKVSender)
        sender.conclude_state = None
        sender.kv_mgr = FakeKVManager()
        sender.bootstrap_room = 17
        sender.init_time = None
        sender.transfer_init_time = None
        sender.trace_ctx = SimpleNamespace(trace_req_finish=lambda: None)

        before_poll = time.time()

        self.assertEqual(sender.poll(), KVPoll.Transferring)
        self.assertIsNotNone(sender.transfer_init_time)
        self.assertGreaterEqual(sender.transfer_init_time, before_poll)
        self.assertEqual(sender.kv_mgr.failures, [])

        sender.transfer_init_time = time.time() - sender.kv_mgr.waiting_timeout - 1

        self.assertEqual(sender.poll(), KVPoll.Failed)
        self.assertEqual(sender.kv_mgr.status, KVPoll.Failed)
        self.assertIn("KVPoll.Transferring", sender.kv_mgr.failures[0][1])

    def test_mooncake_receiver_waiting_timeout_starts_after_transfer_begins(self):
        class FakeKVManager:
            waiting_timeout = 60.0

            def __init__(self):
                self.status = KVPoll.WaitingForInput
                self.failures = []

            def check_status(self, bootstrap_room):
                return self.status

            def record_failure(self, bootstrap_room, reason):
                self.failures.append((bootstrap_room, reason))

            def update_status(self, bootstrap_room, status):
                self.status = status

        receiver = MooncakeKVReceiver.__new__(MooncakeKVReceiver)
        receiver.conclude_state = None
        receiver.kv_mgr = FakeKVManager()
        receiver.bootstrap_room = 11
        receiver.init_time = None
        receiver.abort_notified = False

        self.assertEqual(receiver.poll(), KVPoll.WaitingForInput)
        self.assertIsNone(receiver.init_time)
        self.assertEqual(receiver.kv_mgr.failures, [])

        receiver.kv_mgr.status = KVPoll.Transferring
        before_poll = time.time()

        self.assertEqual(receiver.poll(), KVPoll.Transferring)
        self.assertIsNotNone(receiver.init_time)
        self.assertGreaterEqual(receiver.init_time, before_poll)

        receiver.init_time = time.time() - receiver.kv_mgr.waiting_timeout - 1

        self.assertEqual(receiver.poll(), KVPoll.Failed)
        self.assertIn("KVPoll.WaitingForInput", receiver.kv_mgr.failures[0][1])

    def test_mooncake_transfer_worker_drops_failed_room_without_transfer(self):
        class StopLoop(BaseException):
            pass

        class OneChunkQueue:
            def __init__(self, chunk):
                self.chunk = chunk
                self.calls = 0

            def get(self):
                self.calls += 1
                if self.calls == 1:
                    return self.chunk
                raise StopLoop()

        manager = SimpleNamespace()
        manager.enable_staging = False
        manager.enable_trace = False
        manager.transfer_infos = {3: {"session": object()}}
        manager.req_to_decode_prefix_len = {3: 0}
        manager.request_status = {3: KVPoll.Failed}
        manager.check_status = lambda room: manager.request_status[room]

        chunk = TransferKVChunk(
            room=3,
            prefill_kv_indices=[],
            index_slice=slice(0, 0),
            is_last_chunk=True,
            prefill_aux_index=0,
            state_indices=None,
        )

        with self.assertRaises(StopLoop):
            MooncakeKVManager.transfer_worker(manager, OneChunkQueue(chunk), None)

        self.assertEqual(manager.transfer_infos, {})
        self.assertEqual(manager.req_to_decode_prefix_len, {})

    def test_decode_status_success_ignores_cleared_room(self):
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager.status_lock = threading.RLock()
        manager.request_status = {}
        manager.required_prefill_response_num_table = {7: 1}
        manager.prefill_response_tracker = defaultdict(set)
        manager.enable_staging = False

        MooncakeKVManager._handle_decode_status(manager, 7, KVPoll.Success, 0)

        self.assertNotIn(7, manager.request_status)
        self.assertNotIn(7, manager.prefill_response_tracker)

    def test_decode_status_success_ignores_missing_response_count(self):
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager.status_lock = threading.RLock()
        manager.request_status = {7: KVPoll.Transferring}
        manager.required_prefill_response_num_table = {}
        manager.prefill_response_tracker = defaultdict(set)
        manager.enable_staging = False

        MooncakeKVManager._handle_decode_status(manager, 7, KVPoll.Success, 0)

        self.assertEqual(manager.request_status[7], KVPoll.Transferring)
        self.assertNotIn(7, manager.prefill_response_tracker)

    def test_decode_status_success_completes_after_required_responses(self):
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager.status_lock = threading.RLock()
        manager.request_status = {7: KVPoll.Transferring}
        manager.required_prefill_response_num_table = {7: 2}
        manager.prefill_response_tracker = defaultdict(set)
        manager.enable_staging = False

        MooncakeKVManager._handle_decode_status(manager, 7, KVPoll.Success, 0)
        self.assertEqual(manager.request_status[7], KVPoll.Transferring)

        MooncakeKVManager._handle_decode_status(manager, 7, KVPoll.Success, 1)
        self.assertEqual(manager.request_status[7], KVPoll.Success)

    def test_prefill_decode_status_failed_marks_bootstrap_failed(self):
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager.status_lock = threading.RLock()
        manager.failure_lock = threading.Lock()
        manager.request_status = {7: KVPoll.Bootstrapping}
        manager.failure_records = {}
        manager.transfer_infos = {7: {"session": object()}}
        manager.req_to_decode_prefix_len = {7: 42}

        MooncakeKVManager._handle_prefill_decode_status(
            manager,
            [
                MooncakeKVManager.DECODE_STATUS_HEADER,
                b"7",
                str(KVPoll.Failed).encode("ascii"),
                b"decode aborted before metadata",
            ],
        )

        self.assertEqual(manager.request_status[7], KVPoll.Failed)
        self.assertEqual(manager.failure_records[7], "decode aborted before metadata")
        self.assertNotIn(7, manager.transfer_infos)
        self.assertNotIn(7, manager.req_to_decode_prefix_len)

    def _make_abort_receiver(self, status=KVPoll.Bootstrapping):
        class FakeKVManager:
            def __init__(self, status):
                self.status = status
                self.failures = []
                self.statuses = []

            def check_status(self, bootstrap_room):
                if self.status is None:
                    raise KeyError(bootstrap_room)
                return self.status

            def record_failure(self, bootstrap_room, reason):
                self.failures.append((bootstrap_room, reason))

            def update_status(self, bootstrap_room, status):
                self.statuses.append((bootstrap_room, status))

        class FakeLock:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

        class FakeSocket:
            def __init__(self):
                self.messages = []

            def send_multipart(self, msg):
                self.messages.append(msg)

        fake_socket = FakeSocket()
        receiver = MooncakeKVReceiver.__new__(MooncakeKVReceiver)
        receiver.bootstrap_room = 17
        receiver.bootstrap_infos = [
            {"rank_ip": "127.0.0.1", "rank_port": 12345},
            {"rank_ip": "127.0.0.2", "rank_port": 12346},
        ]
        receiver.kv_mgr = FakeKVManager(status)
        receiver.conclude_state = None
        receiver._connect_to_bootstrap_server = lambda bootstrap_info: (
            fake_socket,
            FakeLock(),
        )
        return receiver, fake_socket

    def test_mooncake_receiver_abort_notifies_prefill_status_before_metadata(self):
        receiver, fake_socket = self._make_abort_receiver(KVPoll.Bootstrapping)
        receiver.abort("decode aborted before metadata")

        self.assertEqual(
            receiver.kv_mgr.failures, [(17, "decode aborted before metadata")]
        )
        self.assertEqual(receiver.kv_mgr.statuses, [(17, KVPoll.Failed)])
        self.assertEqual(receiver.conclude_state, KVPoll.Failed)
        self.assertEqual(len(fake_socket.messages), 2)
        self.assertTrue(
            all(
                msg[0] == MooncakeKVManager.DECODE_STATUS_HEADER
                for msg in fake_socket.messages
            )
        )
        self.assertTrue(all(msg[1] == b"17" for msg in fake_socket.messages))
        self.assertTrue(
            all(
                msg[2] == str(KVPoll.Failed).encode("ascii")
                for msg in fake_socket.messages
            )
        )
        self.assertTrue(
            all(
                msg[3] == b"decode aborted before metadata"
                for msg in fake_socket.messages
            )
        )

    def test_mooncake_receiver_abort_does_not_notify_prefill_after_metadata(self):
        receiver, fake_socket = self._make_abort_receiver(KVPoll.WaitingForInput)
        receiver.abort("stream client disconnected")

        self.assertEqual(receiver.kv_mgr.failures, [(17, "stream client disconnected")])
        self.assertEqual(receiver.kv_mgr.statuses, [(17, KVPoll.Failed)])
        self.assertEqual(receiver.conclude_state, KVPoll.Failed)
        self.assertEqual(fake_socket.messages, [])

    def test_mooncake_receiver_abort_does_not_resurrect_cleared_room(self):
        receiver, fake_socket = self._make_abort_receiver(None)
        receiver.abort("late abort after clear")

        self.assertEqual(receiver.kv_mgr.failures, [])
        self.assertEqual(receiver.kv_mgr.statuses, [])
        self.assertEqual(receiver.conclude_state, KVPoll.Failed)
        self.assertEqual(fake_socket.messages, [])


if __name__ == "__main__":
    unittest.main()
