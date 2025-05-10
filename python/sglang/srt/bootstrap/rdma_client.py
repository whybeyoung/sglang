#!/usr/bin/env python
# coding:utf-8
"""
@author: nivic ybyang7
@license: Apache Licence
@file: rdma_client
@time: 2025/03/28
@contact: ybyang7@iflytek.com
@site:
@software: PyCharm

# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃      ☃      ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━┓
                ┃  神兽保佑    ┣┓
                ┃　永无BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
"""

#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.
import time
import torch
import socket
import pickle
from pyverbs.device import Context
from pyverbs.pd import PD
from pyverbs.cq import CQ
from pyverbs.qp import QPInitAttr, QPCap, QPAttr, QP
from pyverbs.mr import MR
from pyverbs.addr import GID, AHAttr
from pyverbs.wr import SGE, SendWR
from pyverbs.enums import *
import logging

logger = logging.getLogger(__name__)


class RDMAClient:
    def __init__(self, device_name: str, server_host: str = '127.0.0.1', server_port: int = 29999):
        """
        初始化 RDMA 客户端

        Args:
            device_name: RDMA 设备名称
            server_host: 服务器主机地址
            server_port: 服务器端口
        """
        self.device_name = device_name
        self.server_host = server_host
        self.server_port = server_port
        self.ctx = None
        self.pd = None
        self.qp = None
        self.cq = None
        self.mr = None
        self.gpu_tensor = None
        self.sock = None
        self.remote_info = None

    def connect(self):
        """连接到服务器"""
        # 创建 TCP 连接
        self.sock = socket.socket()
        self.sock.connect((self.server_host, self.server_port))
        logger.info(f"Connected to server at {self.server_host}:{self.server_port}")

    def _init_rdma(self, ptr, length):
        """初始化 RDMA 资源"""

        # 创建上下文
        self.ctx = Context(name=self.device_name)
        self.pd = PD(self.ctx)

        # 创建完成队列
        self.cq = CQ(self.ctx, 10)
        # 创建 QP
        cap = QPCap(max_send_wr=10, max_recv_wr=10, max_send_sge=1, max_recv_sge=1)
        init_attr = QPInitAttr(qp_type=2, scq=self.cq, rcq=self.cq, cap=cap)
        self.qp = QP(self.pd, init_attr)

        # 创建gpu的内存区域， 相当于注册gpu内存到 rdma
        self.mr = MR(self.pd, address=ptr,
                     length=length, access=0b111)

        # 初始化 QP
        self._init_qp(ptr)

    def _init_qp(self, ptr):
        """初始化 QP 状态"""
        # INIT
        attr = QPAttr()
        attr.qp_state = 2
        attr.pkey_index = 0
        attr.port_num = 1
        attr.qp_access_flags = 0b111
        self.qp.to_init(attr)

        # RTR
        # 交换 RDMA 信息
        self._exchange_info(ptr)

        attr.qp_state = 3
        attr.path_mtu = self.port_attr.active_mtu
        attr.dest_qp_num = self.remote_info['qp_num']
        attr.rq_psn = 0
        attr.max_dest_rd_atomic = 1
        attr.min_rnr_timer = 1
        attr.ah_attr.port_num = 1

        if self.port_attr.lid != 0:
            attr.ah_attr.dlid = self.remote_info['lid']
            attr.ah_attr.is_global = 0
        else:
            ah_attr = AHAttr()

            ah_attr.dlid = 0
            ah_attr.is_global = 1
            ah_attr.dgid = self.remote_info['gid']

            ah_attr.sgid_index = 3
            ah_attr.hop_limit = 1
            attr.ah_attr = ah_attr

        self.qp.to_rtr(attr)

        # RTS
        attr.qp_state = 4
        attr.sq_psn = 0
        attr.timeout = 14
        attr.retry_cnt = 7
        attr.rnr_retry = 7
        attr.max_rd_atomic = 1
        self.qp.to_rts(attr)

    def _exchange_info(self, ptr):
        """交换 RDMA 信息"""
        self.port_attr = self.ctx.query_port(1)
        gid = self.ctx.query_gid(1, 3)

        local_info = {
            'qp_num': self.qp.qp_num,
            'lid': self.port_attr.lid,
            'gid': str(gid),
            'rkey': self.mr.rkey,
            'addr': ptr
        }
        # 发送本地信息
        self.sock.sendall(pickle.dumps(local_info))
        # 接收远程信息
        self.remote_info = pickle.loads(self.sock.recv(4096))
        logger.debug(f"Received remote info: {self.remote_info}")

    def rdma_write(self, ptr, length) -> float:
        """
        执行 RDMA 写入操作

        Args:
            data: 要写入的数据张量

        Returns:
            float: 传输速率 (MB/s)
        """

        # 创建 SGE
        sge = SGE(addr=ptr,
                  length=length,
                  lkey=self.mr.lkey)

        # 创建发送工作请求
        wr = SendWR(wr_id=1, sg=[sge], num_sge=1, opcode=IBV_WR_RDMA_WRITE, send_flags=IBV_SEND_SIGNALED)
        wr.set_wr_rdma(addr=self.remote_info['addr'],
                       rkey=self.remote_info['rkey'])

        # 记录开始时间
        start_time = time.time()

        # 提交发送请求
        self.qp.post_send(wr)
        logger.info("[Client] RDMA Write Posted")

        # 等待完成
        max_retries = 100
        retry_count = 0
        while retry_count < max_retries:
            npolled, wcs = self.cq.poll()
            if npolled > 0:
                for wc in wcs:
                    end_time = time.time()
                    duration = end_time - start_time
                    throughput = length / duration / (1024 * 1024)  # MB/s

                    logger.info(f"[Client] Completion status: {wc.status}, "
                                f"opcode: {wc.opcode}, byte_len: {wc.byte_len}")
                    logger.info(f"[Client] Transfer rate: {throughput:.2f} MB/s")
                    if wc.status == 0:
                        print("completed")
                        break
                    if wc.status > 0:
                        raise RuntimeError(f"RDMA Write failed with status: {wc.status}")
                    return throughput

            time.sleep(0.1)
            retry_count += 1


    def close(self):
        """关闭客户端"""
        if self.mr:
            self.mr.close()
        if self.qp:
            self.qp.close()
        if self.cq:
            self.cq.close()
        if self.pd:
            self.pd.close()
        if self.ctx:
            self.ctx.close()
        if self.sock:
            self.sock.close()
        logger.info("Client closed")


def start_transfer(size=1024):
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 创建并连接客户端
    client = RDMAClient(device_name="mlx5_bond_3", server_host="127.0.0.1", server_port=39999)
    try:
        client.connect()

        # 创建测试数据
        test_data = torch.arange(size, dtype=torch.uint8, device='cuda')

        # 初始化 RDMA, 获取对端server地址
        client._init_rdma(test_data.data_ptr(), test_data.numel())

        # 执行 RDMA 写入
        throughput = client.rdma_write(test_data.data_ptr(), test_data.numel())
        logger.info(f"Final transfer rate: {throughput:.2f} MB/s")

    except KeyboardInterrupt:
        logger.info("Shutting down client...")
    finally:
        client.close()


if __name__ == "__main__":
    # 构造10G
    one1g = 1024 * 1024 * 1024
    start_transfer(one1g * 1)
