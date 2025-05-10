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
os.environ["RAPIDS_LIBUCX_PREFER_SYSTEM_LIBRARY"] = "true"

import ucp
os.environ['UCPY_LOG_LEVEL'] = "NONE"
os.environ['UCX_LOG_LEVEL'] = "DEBUG"
os.environ['UCX_TLS'] = "tcp,rc,gdr_copy,cuda_copy,cuda_ipc"
#os.environ['UCX_NET_DEVICES'] = "mlx5_bond_0:1,mlx5_bond_1:1,mlx5_bond_2:1,mlx5_bond_3:1"
#os.environ["UCX_NET_DEVICES"] = 'all'
os.environ['UCX_IB_NUM_PATHS'] = "16"
os.environ['UCX_MAX_RNDV_LANES'] = "16"
os.environ['UCX_IB_GID_INDEX'] = '3'





async def main():
    UCX_CONFIG = {}

    UCX_CONFIG.update({
        "NET_DEVICES": "all",
        # "RNDV_SCHEME": "put_zcopy"

    })
    print(UCX_CONFIG)
    ucp.init(options=UCX_CONFIG, blocking_progress_mode=False)

    host, port = "10.246.59.104", 8998
    ep = await ucp.create_endpoint(host, int(port))

    # 2. Send connection request
    conn_info = {
        "session_id": 1,
        "room_id": 1,
        "role": "receiver"
    }
    print(conn_info)

if __name__ == '__main__':
    asyncio.run(main())
