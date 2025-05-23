import datetime
import time
import zmq
from multiprocessing import Process, current_process

ROUTER_ENDPOINT = "tcp://127.0.0.1:5559"

def worker():
    ctx = zmq.Context()
    socket = ctx.socket(zmq.DEALER)
    identity = current_process().name.encode()
    socket.setsockopt(zmq.IDENTITY, identity)
    socket.connect(ROUTER_ENDPOINT)

    while True:
        socket.send_multipart([b"", b"READY"])
        parts = socket.recv_multipart()
        if len(parts) >= 2:
            msg = parts[1].decode()
            now = datetime.datetime.now()
            print(f"[{identity.decode()}] received: {msg} at {now}")
            time.sleep(0.1)  # 模拟处理延迟
        else:
            print(f"[{identity.decode()}] received invalid message: {parts}")

def router_producer():
    ctx = zmq.Context()
    socket = ctx.socket(zmq.ROUTER)
    socket.bind(ROUTER_ENDPOINT)

    workers = {}
    task_id = 0
    total_tasks = 2000

    while task_id < total_tasks:
        parts = socket.recv_multipart()
        identity = parts[0]
        if parts[-1] == b"READY":
            msg = f"Message {task_id} {datetime.datetime.now()}".encode()
            socket.send_multipart([identity, b"", msg])
            print(f"[Router] Sent: {msg.decode()}")
            task_id += 1

if __name__ == "__main__":
    # 启动多个 worker (DEALER) 进程
    workers = [Process(target=worker, name=f"Worker-{i}") for i in range(16)]
    for p in workers:
        p.start()

    # 启动 router/producer
    router_producer()

    # 等待所有 worker 完成（不会自动退出）
    for p in workers:
        p.terminate()
