#!/usr/bin/env python
# coding:utf-8
"""
@author: nivic ybyang7
@license: Apache Licence
@file: client
@time: 2025/03/26
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
#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

import os
import threading

os.environ["RAPIDS_LIBUCX_PREFER_SYSTEM_LIBRARY"] = "true"

import ucp
os.environ['UCPY_LOG_LEVEL'] = "NONE"
os.environ['UCX_TLS'] = "tcp,rc,gdr_copy,cuda_copy,cuda_ipc"
#os.environ['UCX_NET_DEVICES'] = "mlx5_bond_0:1,mlx5_bond_1:1,mlx5_bond_2:1,mlx5_bond_3:1"
#os.environ["UCX_NET_DEVICES"] = 'all'
os.environ['UCX_IB_NUM_PATHS'] = "1"
os.environ['UCX_MAX_RNDV_LANES'] = "1"
os.environ['UCX_IB_GID_INDEX'] = '3'
os.environ['UCX_LOG_LEVEL'] = "DEBUG"

async def handle_connection( ep):
    """Handle incoming UCX connections"""
    try:
        print("connecting to")
        data = await ep.recv_obj()
        print(data)
    except Exception as e:
        print(e)




async def main():

    UCX_CONFIG = {}
    UCX_CONFIG.update({
        "NET_DEVICES": "all",
        "RNDV_SCHEME":"put_zcopy",
        "LOG_LEVEL":"INFO"
    })
    print(UCX_CONFIG)
    ucp.init(options=UCX_CONFIG, blocking_progress_mode=False)

    listener = ucp.create_listener(handle_connection, port=9999)
    while not listener.closed():
        await asyncio.sleep(0.1)
    print(f"SharedUcxServer started on port 9988")

def run_asyncio_loop():
    loop = asyncio.new_event_loop()  # 创建新的事件循环
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())  # 在新事件循环中运行 start()

if __name__ == '__main__':
    a  = threading.Thread(target=run_asyncio_loop )
    a.start()
    a.join()
