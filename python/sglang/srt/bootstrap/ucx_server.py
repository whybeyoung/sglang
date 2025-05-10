#!/usr/bin/env python
# coding:utf-8
"""
@author: nivic ybyang7
@license: Apache Licence
@file: ucx_server
@time: 2025/03/25
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
import asyncio
import os
os.environ["RAPIDS_LIBUCX_PREFER_SYSTEM_LIBRARY"] = "true"


import ucp
import torch
import json
import time

from sglang.utils import logger



#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.


class UcxServer(object):
    def __init__(self, port, sessions, ib_device="all"):
        self.sessions = sessions
        self.port = port
        self.ib_device = ib_device
        self.setup_env()
        asyncio.run(self.start())

    def setup_env(self):
        #os.environ['UCXPY_LOG_LEVEL'] = "DEBUG"
        #os.environ['UCX_LOG_LEVEL'] = "DEBUG"
        os.environ["RAPIDS_LIBUCX_PREFER_SYSTEM_LIBRARY"] = "true"
        os.environ['UCX_TLS'] = "rc,tcp,gdr_copy,cuda_copy,cuda_ipc"
        #os.environ['UCX_NET_DEVICES'] = "mlx5_bond_0:1,mlx5_bond_1:1,mlx5_bond_2:1,mlx5_bond_3:1"
        os.environ["UCX_NET_DEVICES"] = 'all'
        os.environ['UCX_IB_NUM_PATHS'] = "16"
        os.environ['UCX_MAX_RNDV_LANES'] = "16"
        os.environ['UCX_IB_GID_INDEX'] = '3'


    async def handle_connection(self, ep):
        """处理客户端连接 - 接收张量"""
        try:
            shape_bytes = await ep.recv_obj()
            if shape_bytes is None:
                logger.error("接收张量形状失败")
                return

            # 解析形状信息
            shape_str = shape_bytes.decode()
            shape_list = json.loads(shape_str)
            shape = tuple(shape_list)
            logger.info(f"接收到张量形状: {shape}")

            # 接收数据类型
            dtype_str = await ep.recv_obj()
            if isinstance(dtype_str, (bytes, bytearray)):
                dtype_str = dtype_str.decode()

            # 确定 PyTorch 数据类型
            if dtype_str == "float32":
                dtype = torch.float32
            elif dtype_str == "float16":
                dtype = torch.float16
            elif dtype_str == "int64":
                dtype = torch.int64
            else:
                dtype = torch.float32

            logger.info(f"接收到张量数据类型: {dtype_str}")

            # 创建目标张量（直接在GPU显存上分配）
            # self.tensor = torch.zeros(shape, dtype=dtype, device=self.device)

            # 使用 GPUDirect RDMA 接收张量数据到预分配的显存
            start_time = time.time()
            self.tensor = await ep.recv_obj(allocator=self.create_cuda_buffer)
            print(self.tensor)
            end_time = time.time()

            # 计算传输性能
            transfer_time = end_time - start_time
            size_mb = self.tensor.numel() * self.tensor.element_size() / (1024 * 1024)
            bandwidth = size_mb / transfer_time if transfer_time > 0 else 0

            logger.info(f"通过 GPUDirect RDMA 接收张量完成 - 形状: {shape}, 数据类型: {dtype}")
            logger.info(f"数据大小: {size_mb:.2f}MB, 传输时间: {transfer_time:.4f}s, 带宽: {bandwidth:.2f}MB/s")

            # 打印张量摘要信息，验证数据已正确接收
            if self.tensor is not None:
                logger.info(
                    f"接收的张量信息: 均值={self.tensor.mean().item():.4f}, 标准差={self.tensor.std().item():.4f}")

            # 发送确认消息
            await ep.send_obj(b"OK")

        except Exception as e:
            logger.error(f"处理连接时出错: {e}", exc_info=True)
        finally:
            # 关闭连接
            await ep.close()
            logger.debug("客户端连接已关闭")

    def create_cuda_buffer(self, nbytes):
        """创建一个 CUDA 内存缓冲区，尝试与 UCX 兼容"""
        # 1. 创建一个 flat CUDA 张量作为缓冲区
        elem_size = 4  # 使用字节级别的精度
        num_elements = nbytes // elem_size
        buffer = torch.empty(num_elements, dtype=torch.float32, device="cuda")

        # 2. 确保内存布局是连续的
        if not buffer.is_contiguous():
            buffer = buffer.contiguous()

        # 3. 尝试返回 CUDA 内存的字节视图
        # 注意：这可能不被 UCX 支持，需要特殊的 UCX 构建

        # 返回张量，让 UCX 尝试直接使用
        return buffer

    async def start(self):
        """启动服务器"""
        logger.info(f"启动 RDMA Server: 0.0.0.0:{self.port}" +
                    (f" 使用接口 {self.ib_device}" if self.ib_device else ""))

        # 创建监听器
        listener_kwargs = {"port": self.port}

        listener = ucp.create_listener(self.handle_connection, **listener_kwargs)
        return listener
    def close(self):
        pass
