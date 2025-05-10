#!/usr/bin/env python
# coding:utf-8
"""
@author: nivic ybyang7
@license: Apache Licence
@file: rdma_server
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
from pyverbs.wr import RecvWR, SGE
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class RDMAServer:
    def __init__(self, device_name: str):
        """
        初始化 RDMA 服务器

        Args:
            device_name: RDMA 设备名称
            port: 监听端口
        """
        self.device_name = device_name
        self.ctx = None
        self.pd = None
        self.qp = None
        self.recv_cq = None
        self.send_cq = None
        self.mr = None
        self.gpu_tensor = None
        self.listen_sock = None
        self.conn = None

    def start(self, size=1024, port=1999):
        """启动服务器"""
        # 创建 TCP 服务器
        self.port = port
        self.listen_sock = socket.socket()
        self.listen_sock.bind(('127.0.0.1', self.port))
        self.listen_sock.listen(1)
        logger.info(f"Rdma Server listening on port {self.port}")

        # 等待客户端连接
        self.conn, _ = self.listen_sock.accept()
        logger.info("Client connected")

        ts = torch.zeros(size, dtype=torch.uint8, device='cuda')
        logger.info(f"now: {ts}")

        # 初始化 RDMA
        self._init_rdma(ts.data_ptr(), ts.numel())

        # 等待 RDMA 写入
        self._wait_for_rdma_write(ts.data_ptr(), ts.numel())
        while True:
             time.sleep(1)
             logger.info(f"writing:::  {ts}")
        #
    def _init_rdma(self, ptr, length):
        """初始化 RDMA 资源"""
        # 创建上下文
        self.ctx = Context(name=self.device_name)
        self.pd = PD(self.ctx)

        # 创建完成队列
        self.recv_cq = CQ(self.ctx, 10)
        self.send_cq = CQ(self.ctx, 10)

        # 创建 QP
        cap = QPCap(max_send_wr=10, max_recv_wr=10, max_send_sge=1, max_recv_sge=1)
        init_attr = QPInitAttr(qp_type=2, scq=self.send_cq, rcq=self.recv_cq, cap=cap)
        self.qp = QP(self.pd, init_attr)

        # 创建 GPU 张量和内存区域
        self.mr = MR(self.pd, address=ptr,
                     length=length, access=0b111)


        # 初始化 QP
        self._init_qp(ptr)



    def _init_qp(self,ptr):
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

    def _exchange_info(self,ptr):
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
        self.conn.sendall(pickle.dumps(local_info))
        # 接收远程信息
        self.remote_info = pickle.loads(self.conn.recv(4096))
        logger.info(f"Received remote info: {self.remote_info}")

    def _wait_for_rdma_write(self, ptr,length):
        """等待 RDMA 写入完成"""
        logger.info("[Server] QP ready, waiting RDMA write...")

        # 预先提交接收工作请求
        # 握手阶段
        recv_sge = SGE(addr=ptr,
                      length=length,
                      lkey=self.mr.lkey)
        recv_wr = RecvWR(wr_id=1, sg=[recv_sge], num_sge=1)
        # self.qp.post_recv(recv_wr)
        # logger.info("[Server] Posted receive request")
        #
        # # 等待接收完成
        # start_time = time.time()
        # while True:
        #     npolled, wc_list = self.recv_cq.poll()
        #
        #     if npolled > 0:
        #         for wc in wc_list:
        #             end_time = time.time()
        #             duration = end_time - start_time
        #             throughput = length / duration / (1024 * 1024)  # MB/s
        #
        #             logger.info(f"[Server] Write Completed: wr_id={wc.wr_id}, "
        #                         f"status={wc.status}, opcode={wc.opcode}")
        #             logger.info(f"[Server] Transfer rate: {throughput:.2f} MB/s")
        #         break
        #     else:
        #         logger.debug("[Server] No completion event received, retrying...")
        #     time.sleep(0.4)

    def close(self):
        """关闭服务器"""
        if self.mr:
            self.mr.close()
        if self.qp:
            self.qp.close()
        if self.send_cq:
            self.send_cq.close()
        if self.recv_cq:
            self.recv_cq.close()
        if self.pd:
            self.pd.close()
        if self.ctx:
            self.ctx.close()
        if self.conn:
            self.conn.close()
        if self.listen_sock:
            self.listen_sock.close()
        logger.info("Server closed")


def main(size, port=39999):
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 创建并启动服务器
    server = RDMAServer(device_name="mlx5_bond_0")
    try:
        server.start(size,port)
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
    finally:
        server.close()


if __name__ == "__main__":
    # 构造10G
    one1g = 1024 *1024*1024
    main(one1g*1)
