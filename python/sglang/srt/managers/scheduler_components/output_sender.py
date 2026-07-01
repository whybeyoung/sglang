from typing import Optional, Union

import zmq

from sglang.srt.managers.io_struct import BaseBatchReq, BaseReq


class SenderWrapper:
    def __init__(self, socket: zmq.Socket):
        self.socket = socket

    def send_output(
        self,
        output: Union[BaseReq, BaseBatchReq],
        recv_obj: Optional[Union[BaseReq, BaseBatchReq]] = None,
    ):
        if self.socket is None:
            return

        # Propagate the originating http worker ipc for the multi-http-worker
        # case. recv_obj may be an io_struct BaseReq OR a scheduler-side
        # schedule_batch.Req (e.g. AbortReq emitted for an aborted req); the
        # latter is not a BaseReq but still carries http_worker_ipc, so use
        # duck typing instead of an isinstance check. Without this, the abort
        # output is dropped with "IPC name is None ... skipping".
        recv_ipc = getattr(recv_obj, "http_worker_ipc", None)
        if recv_ipc is not None and getattr(output, "http_worker_ipc", None) is None:
            output.http_worker_ipc = recv_ipc

        self.socket.send_pyobj(output)
