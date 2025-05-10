import threading
import time
from pyverbs.device import Context
from pyverbs.pd import PD
from pyverbs.qp import QPCap, QPInitAttr, QP
from pyverbs.cq import CQ
def create_qps(ctx, pd, num_qps):
    for _ in range(num_qps):
        cq = CQ(ctx, 128)
        cap = QPCap(max_send_wr=128, max_recv_wr=128, max_send_sge=1, max_recv_sge=1)
        qp_init_attr = QPInitAttr(qp_type=2, scq=cq, rcq=cq, cap=cap)
        qp = QP(pd, qp_init_attr)

NUM_THREADS = 4
QPS_PER_THREAD = 2500
threads = []

ctx = Context(name="mlx5_bond_1")
pd = PD(ctx)

start = time.time()
for _ in range(NUM_THREADS):
    t = threading.Thread(target=create_qps, args=(ctx, pd, QPS_PER_THREAD))
    threads.append(t)
    t.start()

for t in threads:
    t.join()
end = time.time()
print(f"Total QP created: {NUM_THREADS * QPS_PER_THREAD}, time: {end - start:.2f}s")
print(f"Average per QP: {(end - start) / NUM_THREADS * 4:.2f} ms")

