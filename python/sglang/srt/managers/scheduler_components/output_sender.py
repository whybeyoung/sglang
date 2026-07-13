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

        # handle communicator reqs for multi-http worker case; recv_obj may be
        # an io_struct BaseReq or a scheduler Req (e.g. abort paths pass the
        # scheduler's Req) — both carry http_worker_ipc
        recv_ipc = getattr(recv_obj, "http_worker_ipc", None)
        if recv_ipc is not None and output.http_worker_ipc is None:
            output.http_worker_ipc = recv_ipc

        self.socket.send_pyobj(output)
