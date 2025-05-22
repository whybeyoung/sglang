import datetime

import zmq
import time
from multiprocessing import Process, current_process

PUSH_ENDPOINT = "ipc:///tmp/test_pushpull.ipc"
PUSH_ENDPOINT ="tcp://127.0.0.1:5535"
def worker():
    ctx = zmq.Context()
    socket = ctx.socket(zmq.PULL)
    socket.setsockopt(zmq.RCVHWM, 0)
    socket.setsockopt(zmq.RCVBUF, -1)
    socket.connect(PUSH_ENDPOINT)
    while True:
        msg = socket.recv_pyobj()
        print(f"[{current_process().name}] received: {msg} realtime: {datetime.datetime.now()}")
        time.sleep(0.005)
def producer():
    ctx = zmq.Context()
    socket = ctx.socket(zmq.PUSH)
    socket.setsockopt(zmq.SNDBUF, -1)

    socket.setsockopt(zmq.SNDHWM, 0)
    socket.bind(PUSH_ENDPOINT)

    for i in range(2000):
        msg = f"Message {i} {datetime.datetime.now()}"
        print(f"[Producer] sending: {msg}")
        socket.send_pyobj(msg)

if __name__ == "__main__":
    # 启动多个 worker 进程
    workers = [Process(target=worker, name=f"Worker-{i}") for i in range(16)]
    for p in workers:
        p.start()

    # 启动 producer
    producer()


    for p in workers:
        p.join()
    print(f"All workers Done took {datetime.datetime.now()}")
