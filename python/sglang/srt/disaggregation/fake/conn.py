from typing import Dict, List, Optional, Tuple, Union

from sglang.srt.disaggregation.base.conn import (
    BaseKVReceiver,
    BaseKVSender,
    KVArgs,
    KVPoll,
)


# For warmup reqs, we don't kv transfer, we use the fake sender and receiver
class FakeKVSender:
    def __init__(self):
        self.has_sent = False

    def poll(self) -> KVPoll:
        if self.has_sent is False:
            # Assume handshake completed instantly
            return KVPoll.WaitingForInput
        else:
            # Assume transfer completed instantly
            return KVPoll.Success

    def init(
        self,
        kv_indices: list[int],
        aux_index: Optional[int] = None,
        dest_ranks: Optional[list[int]] = None,
    ):
        pass

    def send(self, up_to_index: int):
        self.has_sent = True

    def failure_exception(self):
        raise Exception("Fake KVSender Exception")


class FakeKVReceiver:
    def __init__(self):
        self.has_init = False

    def poll(self) -> KVPoll:
        if self.has_init is False:
            # Assume handshake completed instantly
            return KVPoll.WaitingForInput
        else:
            # Assume transfer completed instantly
            return KVPoll.Success

    def init(self, kv_indices: list[int], aux_index: Optional[int] = None):
        self.has_init = True

    def failure_exception(self):
        raise Exception("Fake KVReceiver Exception")
