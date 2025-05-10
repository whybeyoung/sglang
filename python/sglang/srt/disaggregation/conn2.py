from __future__ import annotations
import logging
import subprocess
import threading
import json
import asyncio
import os
import time

import requests

from sglang.srt.bootstrap.app import start_bootstrap_server

os.environ["RAPIDS_LIBUCX_PREFER_SYSTEM_LIBRARY"] = "true"

import ucp
import uuid
from typing import Dict, Optional
from dataclasses import dataclass
import numpy as np

os.environ['UCPY_LOG_LEVEL'] = "NONE"
os.environ['UCX_LOG_LEVEL'] = "NONE"
os.environ['UCX_TLS'] = "tcp,rc,gdr_copy,cuda_copy,cuda_ipc"
# os.environ['UCX_NET_DEVICES'] = "mlx5_bond_0:1,mlx5_bond_1:1,mlx5_bond_2:1,mlx5_bond_3:1"
# os.environ["UCX_NET_DEVICES"] = 'all'
os.environ['UCX_IB_NUM_PATHS'] = "16"
os.environ['UCX_MAX_RNDV_LANES'] = "16"
os.environ['UCX_IB_GID_INDEX'] = '3'
logger = logging.getLogger(__name__)

from sglang.srt.utils import  get_open_port

UCX_CONFIG = {
    # "RNDV_SCHEME": "put_zcopy"
}


class KVBootstrapServer:
    def __init__(self, port: int):
        self.bootstrap_server_port = port
        self.ucx_server = self.start_server()

    def start_server(self):
        server = start_bootstrap_server("0.0.0.0", self.bootstrap_server_port)
        print(" bootstrap server started")


class KVArgs:
    """Arguments for KV cache management"""
    engine_rank: int
    kv_data_ptrs: list[int]
    kv_data_lens: list[int]
    kv_item_lens: list[int]
    aux_data_ptrs: list[int]
    aux_data_lens: list[int]
    aux_item_lens: list[int]
    ib_device: str = "all"


class KVManager:
    def __init__(self, args: KVArgs, bootstrap_server: KVBootstrapServer = None):
        self.args = args
        self.engine_rank = args.engine_rank
        self.kv_data_ptrs = args.kv_data_ptrs
        self.kv_data_lens = args.kv_data_lens
        self.kv_item_lens = args.kv_item_lens
        self.active_sessions = {}
        self.bootstrap_server = bootstrap_server
        UCX_CONFIG = {}
        UCX_CONFIG.update({
            "NET_DEVICES": "all",
            # "RNDV_SCHEME": "put_zcopy"

        })
        print(UCX_CONFIG)
        ucp.init(options=UCX_CONFIG, blocking_progress_mode=False)

    def set_bootstrap_server(self, bootstrap_server):
        self.bootstrap_server = bootstrap_server


    def calculate_token_kv_address(self,  layer_id: int, token_index: int):
        # 获取基础地址 - 每层的KV数据指针
        base_address = self.args.kv_data_ptrs[layer_id]

        # 每个token的KV数据大小
        token_kv_size =self.args.kv_item_lens[layer_id]
        # 计算偏移量
        offset = token_kv_size * token_index
        # 最终地址 = 基址 + 偏移量
        print("layer_id_base", base_address, offset)

        token_kv_address = base_address + offset
        return token_kv_address, offset

    def calculate_all_token_kv_addresses(self, token_indices: list[int]):
        # 结果存储
        addresses_by_layer = []
        offsets_by_layer = []

        # 对每一层计算
        for layer_id in range(len(self.args.kv_data_ptrs)):
            token_addresses = []
            token_offsets = []

            # 计算每个token的地址和偏移量
            for token_index in token_indices:
                address, offset = self.calculate_token_kv_address( layer_id, token_index)
                token_addresses.append(address)
                token_offsets.append(offset)

            addresses_by_layer.append(token_addresses)
            offsets_by_layer.append(token_offsets)

        return addresses_by_layer, offsets_by_layer
class KVPoll:
    """Status codes for KV operations"""
    Failed = 0
    Bootstrapping = 1
    WaitingForInput = 2
    Transferring = 3
    Success = 4


class KVSender:
    def __init__(self, mgr: KVManager, bootstrap_addr: str, bootstrap_room: int):
        self.mgr = mgr
        self.bootstrap_addr = bootstrap_addr
        self.bootstrap_room = bootstrap_room
        self.session_id = str(uuid.uuid4())
        self.initialized = False
        # self.ucx_server = self.mgr.get_ucx_server()
        self.state = KVPoll.Bootstrapping
        self.transfer_complete = False
        self.metadata_sent = False
        self.data_to_send = None

        # Register with UCX server
        # self.ucx_server.register_sender(self.session_id, self.bootstrap_room)
        logger.info(f"Sender registered with room {self.bootstrap_room}")
        # Initialize transfer metadata
        self.num_tokens = 0
        self.aux_idx = -1

        # target ip
        self.target_ip = None

        # endpoint
        self.ep = None
        # 传输状态
        self.current_indices = None
        self.current_layer = 0
        # 保存初始化信息
       # self.num_layers = mgr.layer_num


    def init(self, num_tokens: int, aux_idx: int) -> bool:
        """Initialize sender with metadata only

        Args:
            num_tokens: Number of tokens to transfer
            aux_idx: Index for auxiliary data

        Returns:
            bool: True if metadata sent successfully
        """
        self.num_tokens = num_tokens
        self.aux_idx = aux_idx
        print("initializing sender req....")
        # Prepare metadata
        self.init_data = {
            "num_tokens": num_tokens,
            "aux_idx": aux_idx,
            "kv_data_ptrs": self.mgr.args.kv_data_ptrs,
            "kv_data_lens": self.mgr.args.kv_data_lens,
            "kv_item_lens": self.mgr.args.kv_item_lens,
            "aux_data_ptrs": self.mgr.args.aux_data_ptrs,
            "aux_data_lens": self.mgr.args.aux_data_lens,
            "aux_item_lens": self.mgr.args.aux_item_lens
        }
        print(self.init_data)
        return True

    def poll(self) -> KVPoll:
        """Poll transfer status"""
        if self.state == KVPoll.Bootstrapping:
            resp = requests.get(f"http://{self.bootstrap_addr}/get_room_info/{self.bootstrap_room}")

            if resp.status_code == 200:
                data = resp.json()
                if not data:
                    self.state = KVPoll.Bootstrapping
                else:
                    print(data)
                    self.target_ip = data.get(str(self.mgr.engine_rank))['ip']
                    self.target_port = data.get(str(self.mgr.engine_rank))['port']

                    self.state = KVPoll.WaitingForInput
        if self.state == KVPoll.Failed:
            return KVPoll.Failed

        if self.transfer_complete:
            return KVPoll.Success

        return self.state

    async def a_send(self, data):
        if self.ep is None:
            host, port = self.target_ip, self.target_port
            self.ep = await ucp.create_endpoint(host, int(port))
            print("sending init,")
            print("data", data)

            await  self.ep.send_obj(json.dumps(self.init_data).encode('utf-8'))
        else:
            print("sending data")
            await  self.ep.send_obj(json.dumps(data))

        # 2. Send connection request
        # todo

    def send(self, kv_indices: np.ndarray[np.int32]):
        """Send actual data synchronously"""

        #self.has_sent = True
        # 收集要传输的数据
        result= self.mgr.calculate_all_token_kv_addresses(kv_indices)

        #transfer_data = self._collect_transfer_data(kv_indices)
        asyncio.run(self.a_send(kv_indices))

    def _collect_transfer_data(self, kv_indices: npt.NDArray[np.int32]):
        """收集要传输的数据

        Args:
            kv_indices: token 的索引数组

        Returns:
            transfer_data: 包含所有要传输数据的字典
        """
        transfer_data = {
            'num_layers': self.num_layers,
            'num_tokens': len(kv_indices),
            'layers': []
        }

        # 对每一层收集数据
        for layer_idx in range(self.num_layers):
            # 获取这一层的 key 和 value buffer
            key_buffer = self.kv_cache.get_key_buffer(layer_idx)
            value_buffer = self.kv_cache.get_value_buffer(layer_idx)

            # 获取数据的形状信息
            _, num_heads, head_dim = key_buffer.shape

            # 收集这一层要传输的数据
            layer_data = {
                'layer_idx': layer_idx,
                'shape': {
                    'num_heads': num_heads,
                    'head_dim': head_dim,
                    'num_tokens': len(kv_indices)
                },
                'data': {
                    # 使用索引选择要传输的数据
                    'key': key_buffer[kv_indices].detach(),  # [num_tokens, num_heads, head_dim]
                    'value': value_buffer[kv_indices].detach()  # [num_tokens, num_heads, head_dim]
                },
                'memory_info': {
                    # 记录内存相关信息，用于接收端重建
                    'key_ptr': key_buffer.data_ptr(),
                    'value_ptr': value_buffer.data_ptr(),
                    'item_size': key_buffer[0].nbytes,  # 单个 token 的 key 大小
                    'stride': key_buffer.stride()  # 内存布局信息
                }
            }

            transfer_data['layers'].append(layer_data)

        return transfer_data

class KVReceiver:
    def __init__(self, mgr: KVManager, bootstrap_addr: str, bootstrap_room: Optional[int] = None):
        self.mgr = mgr
        self.bootstrap_addr = bootstrap_addr
        self.bootstrap_room = bootstrap_room
        self.session_id = str(uuid.uuid4())
        self.initialized = False
        self.ep = None
        self.state = KVPoll.Bootstrapping
        self.transfer_complete = False
        self.num_tokens = 0
        self.aux_idx = -1
        self.kv_indices = None

        self.ucx_port = get_open_port()

        self.start_time = time.time()
        self.handshake()

    def handshake(self):
        post_data = {
            "room_id": self.bootstrap_room,
            "session_id": self.session_id,
            "engine_rank": self.mgr.args.engine_rank,
            "ib_device": self.mgr.args.ib_device,
            "ip_addr": {
                "ip": "10.246.59.104",
                "port": self.ucx_port
            }

        }
        http_start = time.time()
        resp = requests.post(f"http://{self.bootstrap_addr}/handshake", json=post_data)
        http_end = time.time()
        print(f"HD Request time: {http_end - http_start}")
        if resp.status_code != 200:
            self.state = KVPoll.Failed
            print(resp.status_code)
        self.server_thrd = threading.Thread(target=self.run_asyncio_loop)
        self.server_thrd.start()
        # 开始启动loop

    def run_asyncio_loop(self):
        loop = asyncio.new_event_loop()  # 创建新的事件循环
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.create_server())  # 在新事件循环中运行 start()

    async def handle_connection(self, ep):
        """Handle incoming UCX connections"""
        try:
            self.state = KVPoll.WaitingForInput
            self.handshake_time = time.time()
            print("handshake cost time", self.handshake_time - self.start_time)
            print("connected by prefill server....")
            data = await ep.recv_obj()
            self.initialized = KVPoll.WaitingForInput
            print(data)
        except Exception as e:
            print(e)

    async def create_server(self):
        self.listener = ucp.create_listener(self.handle_connection, port=self.ucx_port)
        while not self.listener.closed():
            await asyncio.sleep(0.1)

    def init(self, kv_indices: np.ndarray[np.int32], aux_index: Optional[int] = None):
        """Initialize receiver with KV indices and auxiliary data index

        Args:
            kv_indices: Array of KV indices to receive
            aux_index: Optional index for auxiliary data

        Returns:
            bool: True if initialization successful
        """
        # 收集要传输的数据
        result= self.mgr.calculate_all_token_kv_addresses(kv_indices)

        self.num_tokens = kv_indices
        self.aux_idx = aux_index
        # Prepare metadata
        self.kv_data = {
            "num_tokens": kv_indices,
            "aux_idx": aux_index,
            "kv_data_ptrs": self.mgr.args.kv_data_ptrs,
            "kv_data_lens": self.mgr.args.kv_data_lens,
            "kv_item_lens": self.mgr.args.kv_item_lens,
            "aux_data_ptrs": self.mgr.args.aux_data_ptrs,
            "aux_data_lens": self.mgr.args.aux_data_lens,
            "aux_item_lens": self.mgr.args.aux_item_lens
        }

        print("initializing decode req....")
        print(self.kv_data)

    def poll(self) -> KVPoll:
        """Poll receive status"""
        if not self.initialized:
            return KVPoll.Bootstrapping

        if self.transfer_complete:
            return KVPoll.Success

        if self.state == KVPoll.Failed:
            return KVPoll.Failed

        return self.state

    async def receive(self):
        """Receive data and update state"""
        if not self.initialized:
            logger.error("Cannot receive: not initialized")
            return None

        try:
            self.state = KVPoll.Transferring
            data = await self.ep.recv_obj()
            self.transfer_complete = True
            self.state = KVPoll.Success
            return data
        except Exception as e:
            logger.error(f"Receive failed: {e}")
            self.state = KVPoll.Failed
            return None

    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'loop') and self.loop:
            self.loop.close()
