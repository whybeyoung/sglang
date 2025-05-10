#!/usr/bin/env python
# coding:utf-8
"""
@author: nivic ybyang7
@license: Apache Licence
@file: verbs
@time: 2025/04/08
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
from pyverbs.device import Context
from pyverbs.pd import PD
from pyverbs.qp import QPCap, QPInitAttr, QP
from pyverbs.cq import CQ

ctx = Context(name="mlx5_bond_0")
pd = PD(ctx)

NUM_QP = 10000
start = time.time()

for i in range(NUM_QP):
    cq = CQ(ctx, 128)
    cap = QPCap(max_send_wr=128, max_recv_wr=128, max_send_sge=1, max_recv_sge=1)
    qp_init_attr = QPInitAttr(qp_type=2,  # IBV_QPT_RC
                              scq=cq, rcq=cq,
                              cap=cap)
    qp = QP(pd, qp_init_attr)

end = time.time()
print(f"Created {NUM_QP} QPs in {end - start:.4f} seconds")
print(f"Average per QP: {(end - start) / NUM_QP * 1000:.2f} ms")

