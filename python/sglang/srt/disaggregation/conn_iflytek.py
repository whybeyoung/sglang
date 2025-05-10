from __future__ import annotations

import logging
import random
import time
import uuid
from enum import Enum
from typing import Optional, List
import multiprocessing as mp
import numpy as np
import numpy.typing as npt
import threading

import requests

from sglang.srt.bootstrap.app import start_bootstrap_server
from sglang.srt.bootstrap.ucx_server import UcxServer

logger = logging.getLogger(__name__)

global_sessions = {}

class KVArgs:
    engine_rank: int
    kv_data_ptrs: list[int]
    kv_data_lens: list[int]
    kv_item_lens: list[int]
    aux_data_ptrs: list[int]
    aux_data_lens: list[int]
    aux_item_lens: list[int]
    ib_device: str


class KVManager:
    def __init__(self, args: KVArgs):
        self.args = args
        self.engine_rank = args.engine_rank
        self.kv_data_ptrs = args.kv_data_ptrs
        self.kv_data_lens = args.kv_data_lens
        self.kv_item_lens = args.kv_item_lens
        self.aux_data_ptrs = args.aux_data_ptrs
        self.aux_data_lens = args.aux_data_lens
        self.aux_item_lens = args.aux_item_lens
        self.ib_device = args.ib_device
        self.lock = threading.Lock()
        self._validate_args()

        # 存储所有活跃的传输会话
        global global_sessions
        self.active_sessions = global_sessions

        # 用于跟踪内存使用
        self.kv_memory_map = {}
        self.aux_memory_map = {}

        logger.info(f"Initialized KVManager with engine rank {args.engine_rank}")

    def _validate_args(self):
        """验证输入参数的有效性"""
        if not self.args.kv_data_ptrs:
            raise ValueError("KV data pointers cannot be empty")
        if len(self.args.kv_data_ptrs) != len(self.args.kv_data_lens):
            raise ValueError("Mismatched lengths of KV data pointers and lengths")
        if len(self.args.kv_data_lens) != len(self.args.kv_item_lens):
            raise ValueError("Mismatched lengths of KV data and item lengths")

    def register_session(self, session_id: str, num_tokens: int, aux_idx: int) -> bool:
        """注册新的传输会话"""
        with self.lock:
            if session_id in self.active_sessions:
                logger.warning(f"Session {session_id} already exists")
                return False

            self.active_sessions[session_id] = {
                'num_tokens': num_tokens,
                'aux_idx': aux_idx,
                'status': KVPoll.Bootstrapping,
                'transferred_tokens': 0
            }
            return True

    def allocate_memory(self, session_id: str, kv_indices: List[int]) -> bool:
        """为传输分配内存"""
        with self.lock:
            if session_id not in self.active_sessions:
                return False

            session = self.active_sessions[session_id]
            if len(kv_indices) > session['num_tokens']:
                return False

            # 记录内存分配
            self.kv_memory_map[session_id] = kv_indices
            return True


    def start_transfer(self, session_id: str) -> bool:
        """开始传输过程"""
        with self.lock:
            if session_id not in self.active_sessions:
                return False

            threading.Thread(target=self.mock_transfer, args=(self.active_sessions,session_id,)).start()
            return True

    def mock_transfer(self,obj, session_id):
        obj[session_id]['status'] = KVPoll.Transferring
        time.sleep(2)
        obj[session_id]['status'] = KVPoll.Success
        print("传输成功")


    def update_transfer_progress(self, session_id: str, transferred_tokens: int) -> bool:
        """更新传输进度"""
        with self.lock:
            if session_id not in self.active_sessions:
                return False

            session = self.active_sessions[session_id]
            session['transferred_tokens'] = transferred_tokens

            if transferred_tokens >= session['num_tokens']:
                session['status'] = KVPoll.Success

            return True

    def get_transfer_status(self, session_id: str) -> Optional[KVPoll]:
        """获取传输状态"""
        with self.lock:
            if session_id not in self.active_sessions:
                return None
            return self.active_sessions[session_id]['status']

    def get_session_info(self, session_id: str) -> Optional[dict]:
        """获取会话信息"""
        with self.lock:
            return self.active_sessions.get(session_id)

    def cleanup(self):
        """清理所有资源"""
        with self.lock:
            self.active_sessions.clear()
            self.kv_memory_map.clear()
            self.aux_memory_map.clear()

class KVPoll:
    Failed = 0
    Bootstrapping = 1
    WaitingForInput = 2
    Transferring = 3
    Success = 4


class KVSender:
    def __init__(self, mgr: KVManager, bootstrap_addr: str, bootstrap_room: int):
        # 0. * 先跟 receiver handshake
        # 0.1  handshake阶段
          ##  D* Receiver 建立一个uxc server监听循环， 初始化一个端口ucx_port []，基于 ib_device 配置
          ##  D* 该server 结束条件为 transfer DONE
          ##  D-P* 将  {room_id: uxc port}信息通过bootstrap_server发送给 sender， prefill记录 room_id 对应的 uxc_port 握手结束
        # 1. D* 根据请求做预分配显存 ，这块现场地址应该被这个请求的 pd传输独占，这块gpu显存 和  room_id 绑定
        # 2. D->P* 预分配结束后，通知bootstrap server，该请求可以开始prefill
        # 3. While 循环拉取 sender过来的 kv_cache，并检查传输状态，如果结束可以decode

        self.mgr = mgr
        self.bootstrap_addr = bootstrap_addr
        self.bootstrap_room = bootstrap_room
        self.session_id = str(uuid.uuid4())
        self.initialized = False
        self.has_sent = False

    def init(self, num_tokens: int, aux_idx: int) -> bool:
        """初始化发送器"""
        success = self.mgr.register_session(self.session_id, num_tokens, aux_idx)
        if success:
            self.initialized = True
        return success

    def send(self, kv_indices: np.ndarray) -> bool:
        """发送KV缓存"""
        if not self.initialized:
            return False

        # 分配内存并开始传输
        if not self.mgr.allocate_memory(self.session_id, kv_indices.tolist()):
            return False

        return self.mgr.start_transfer(self.session_id)

    def poll(self) -> KVPoll:
        """查询传输状态"""
        status = self.mgr.get_transfer_status(self.session_id)
        return KVPoll.Success if status==KVPoll.Success else KVPoll.WaitingForInput





class KVReceiver:
    def __init__(
        self, mgr: KVManager, bootstrap_addr: str, bootstrap_room: Optional[int] = None
    ):
        # 0. * 先跟bootstrap server handshake
        # 0.1  handshake阶段
          ##  D* Receiver 建立一个uxc server监听循环， 初始化一个端口ucx_port []，基于 ib_device 配置
          ##  D* 该server 结束条件为 transfer DONE
          ##  D-P* 将  {room_id: uxc port}信息通过bootstrap_server发送给 sender， prefill记录 room_id 对应的 uxc_port 握手结束
        # 1. D* 根据请求做预分配显存 ，这块现场地址应该被这个请求的 pd传输独占，这块gpu显存 和  room_id 绑定
        # 2. D->P* 预分配结束后，通知bootstrap server，该请求可以开始prefill
        # 3. While 循环拉取 sender过来的 kv_cache，并检查传输状态，如果结束可以decode

        self.mgr = mgr
        self.bootstrap_addr = bootstrap_addr

        self.bootstrap_room = bootstrap_room
        self.session_id = str(uuid.uuid4())
        self.initialized = self.handshake_uxc()

    def handshake_uxc(self):
        print(self.bootstrap_addr)

        return True


    def init(self, kv_indices: np.ndarray, aux_idx: int) -> bool:
        """初始化接收器"""
        success = self.mgr.register_session(
            self.session_id,
            len(kv_indices),
            aux_idx
        )
        if success:
            self.initialized = True
            # 预分配接收内存
            self.mgr.allocate_memory(self.session_id, kv_indices.tolist())
        return success

    def poll(self) -> KVPoll:
        """查询接收状态"""

        if self.initialized is False:
            # Assume handshake completed instantly
            return KVPoll.WaitingForInput
        else:
            # Assume transfer completed instantly
            return KVPoll.Success
        status = self.mgr.get_transfer_status(self.session_id)
        return status if status is not None else KVPoll.Failed


class KVBootstrapServer:
    def __init__(self, port: int):
        self.bootstrap_server_port = port
        self.ucx_server = self.start_server()
        global global_sessions
        self.global_sessions = global_sessions
    def start_server(self):
        server = start_bootstrap_server("0.0.0.0", self.bootstrap_server_port)
        print(" bootstrap server started")

        #UcxServer(self.bootstrap_server_port, self.global_sessions)

    def poll(self) -> KVPoll:
        print("ppppp bservering")
