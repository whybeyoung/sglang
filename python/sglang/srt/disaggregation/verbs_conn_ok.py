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
from sglang.srt.bootstrap.rdma_utils import RdmaQP, RdmaClient
from sglang.srt.disaggregation.group_indics import group_by_continuity_numpy
logger = logging.getLogger(__name__)

from sglang.srt.utils import get_open_port

UCX_CONFIG = {
    # "RNDV_SCHEME": "put_zcopy"
}


class KVBootstrapServer:
    def __init__(self, port: int):
        self.bootstrap_server_port = port
        self.start_server()

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
        self.aux_data_ptrs = args.aux_data_ptrs
        self.aux_data_lens = args.aux_data_lens
        self.aux_item_lens = args.aux_item_lens

        self.active_sessions = {}
        self.bootstrap_server = bootstrap_server

    def set_bootstrap_server(self, bootstrap_server):
        self.bootstrap_server = bootstrap_server

    def calculate_token_kv_address(self, layer_id: int, token_index: int):
        # 获取基础地址 - 每层的KV数据指针
        base_address = self.args.kv_data_ptrs[layer_id]
        # 每个token的KV数据大小
        token_kv_size = self.args.kv_item_lens[layer_id]
        # 计算偏移量
        offset = token_kv_size * token_index
        # 最终地址 = 基址 + 偏移量
        token_kv_address = base_address + offset
        return token_kv_address, offset

    def calculate_all_token_kv_addresses(self, token_indices: list[int]):
        # 结果存储
        addresses_by_layer = []
        offsets_by_layer = []
        addresses_base_and_len = []
        # 对每一层计算
        for layer_id in range(len(self.args.kv_data_ptrs)):
            token_addresses = []
            token_offsets = []

            # 计算每个token的地址和偏移量
            for token_index in token_indices:
                address, offset = self.calculate_token_kv_address(layer_id, token_index)
                token_addresses.append(address)
                token_offsets.append(offset)

            addresses_by_layer.append(token_addresses)
            offsets_by_layer.append(token_offsets)
            addresses_base_and_len.append((token_addresses[0], self.args.kv_item_lens[layer_id] * len(token_indices)))
        return addresses_by_layer, offsets_by_layer, addresses_base_and_len

    def caculate_layer_kv_addresses(self, token_indices: list[int]):
        addresses_base_and_len = []
        for layer_id in range(len(self.args.kv_data_ptrs)):
            # 每个token的KV数据大小
            token_kv_size = self.args.kv_item_lens[layer_id]
            # 计算偏移量
            offset = token_kv_size * token_indices[0]
            token_kv_layer_base_address = self.args.kv_data_ptrs[layer_id] + offset
            addresses_base_and_len.append((token_kv_layer_base_address,
                                           token_kv_size * len(token_indices)))
        return addresses_base_and_len


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
        self.state = KVPoll.Bootstrapping
        self.transfer_complete = False
        self.metadata_sent = False
        self.data_to_send = None

        # self.ucx_server.register_sender(self.session_id, self.bootstrap_room)
        logger.info(f"Sender registered with room {self.bootstrap_room}")
        # Initialize transfer metadata
        self.num_tokens = 0
        self.aux_idx = -1

        # target ip
        self.target_ip = None
        self.ip_address = "10.246.59.104"
        # endpoint
        self.ep = None
        # 传输状态
        self.current_indices = None
        self.current_layer = 0

        self.mrs_to_send = []  # 数据段待发送的内存区域
        self.meta_has_sent = False  # meta 还没有发送

    def handshake(self):
        resp = requests.get(f"http://{self.bootstrap_addr}/get_room_info/{self.bootstrap_room}")

        if resp.status_code == 200:
            data = resp.json()
            return data
        return None

    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):

        """Initialize sender with metadata only

        Args:
            num_tokens: Number of tokens to transfer
            aux_idx: Index for auxiliary data

        Returns:
            bool: True if metadata sent successfully
        """
        self.num_tokens = num_kv_indices
        self.aux_idx = aux_index
        metadata_ptr = self.mgr.aux_data_ptrs[0] + (aux_index * self.mgr.aux_item_lens[0])
        metadata_ptr_length = self.mgr.aux_item_lens[0]

        try:
            self.qp = RdmaClient(host_ip=self.target_ip, socket_port=self.target_port)
            if self.qp.init(metadata_ptr, metadata_ptr_length):
                logger.debug("Transferring...")
                self.state = KVPoll.Transferring
        except Exception as e:
            print(e)
            self.state = KVPoll.Bootstrapping

    def poll(self) -> KVPoll:
        """Poll transfer status"""
        if self.state == KVPoll.Bootstrapping:
            data = self.handshake()
            if not data:
                self.state = KVPoll.Bootstrapping
            else:
                logger.debug(data)
                self.target_ip = data.get(str(self.mgr.engine_rank))['ip']
                self.target_port = data.get(str(self.mgr.engine_rank))['port']

                self.state = KVPoll.WaitingForInput
        if self.state == KVPoll.Failed:
            return KVPoll.Failed

        if self.state == KVPoll.Transferring:
            self.qp.check_send_complete()

            '''
            completed wrs + metadata wrs
            '''
            if self.qp.completed_wrs == len(self.mrs_to_send) + 1 and self.meta_has_sent:
                logger.debug(f"Transfer completed! Bytes: {len(self.mrs_to_send)}")
                # 写入远端 metadata //todo
                self.state = KVPoll.Success
            elif self.qp.completed_wrs == len(self.mrs_to_send) and not self.meta_has_sent:
                self.qp.send_metadata_wrs()
                self.meta_has_sent = True

        return self.state

    def send(self, kv_indices: np.ndarray[np.int32]):
        """Send actual data synchronously"""

        # self.has_sent = True
        # 收集要传输的数据
        address_lengths = self.mgr.caculate_layer_kv_addresses(kv_indices)
        mrs_info = []
        for layer_id, (address, length) in enumerate(address_lengths):
            mr = self.qp.create_mr(address, length)
            self.mrs_to_send.append(mr)
            mrs_info.append({
                "address": address,
                "length": length,
                "rkey": mr.rkey
            })

        self.qp.send_wrs(self.mrs_to_send, mrs_info)
        # 收集要传输的数据


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

        self.rdma_port = get_open_port()

        self.start_time = time.time()
        # todo ip
        self.ip = "10.246.59.104"
        self.qp = RdmaQP(socket_port=self.rdma_port)

        # todo remove http handshake
        self.handshake()
        self.mrs_to_receive = []  # 数据段待接收的内存区域

    def handshake(self):
        post_data = {
            "room_id": self.bootstrap_room,
            "session_id": self.session_id,
            "engine_rank": self.mgr.args.engine_rank,
            "ib_device": self.mgr.args.ib_device,
            "ip_addr": {
                "ip": self.ip,
                "port": self.rdma_port
            }

        }
        http_start = time.time()
        resp = requests.post(f"http://{self.bootstrap_addr}/handshake", json=post_data)
        http_end = time.time()
        print(f"HD Request time: {http_end - http_start}")
        if resp.status_code != 200:
            self.state = KVPoll.Failed
            print(resp.status_code)
        else:
            self.state = KVPoll.WaitingForInput
            self.initialized = True
            print("boostraped success..")

    def init(self, kv_indices: np.ndarray[np.int32], aux_index: Optional[int] = None):
        """Initialize receiver with KV indices and auxiliary data index

        Args:
            kv_indices: Array of KV indices to receive
            aux_index: Optional index for auxiliary data

        Returns:
            bool: True if initialization successful
        """

        metadata_ptr = self.mgr.aux_data_ptrs[0] + (aux_index * self.mgr.aux_item_lens[0])
        metadata_length = self.mgr.aux_item_lens[0]

        # 初始化 rdma server, 注册多个 mr区域并发送给客户端
        # 初始化多个 layer
        # 需要得到  [base_layer_address, layer_sum_offset, layer,length)
        address_lengths = self.mgr.caculate_layer_kv_addresses(kv_indices)
        mrs_info = []

        for layer_id, (address, length) in enumerate(address_lengths):
            mr = self.qp.create_mr(address, length)
            self.mrs_to_receive.append(mr)
            mrs_info.append({
                "address": address,
                "length": length,
                "rkey": mr.rkey
            })

        try:
            self.qp.init(self.mrs_to_receive, mrs_info, metadata_ptr, metadata_length)
            self.state = KVPoll.Transferring
            self.qp.recv_metadata_mr()

        except Exception as e:
            self.state = KVPoll.Bootstrapping
            return

    def poll(self) -> KVPoll:
        """Poll receive status"""
        if not self.initialized:
            return KVPoll.Bootstrapping

        if self.state == KVPoll.Transferring:
            self.qp.check_complete()
            # 轮询
            if self.qp.metadata_mr_complete_num == 1:
                logger.debug("Decode Transferring complete...")
                return KVPoll.Success

        if self.state == KVPoll.Failed:
            return KVPoll.Failed

        return self.state

    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'loop') and self.loop:
            self.loop.close()
