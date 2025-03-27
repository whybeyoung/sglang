#!/usr/bin/env python
# coding:utf-8
"""
@author: nivic ybyang7
@license: Apache Licence
@file: app.py
@time: 2025/03/24
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
from pydantic import BaseModel


import threading
import logging
from typing import Optional, Dict
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


from sglang.srt.utils import global_room_data

#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.

# 定义请求和响应的数据模型
class OpenSessionReqInput(BaseModel):
    session_id: Optional[str] = None
    # 其他字段...

class HandshakeRequest(BaseModel):
    room_id: int
    session_id: str
    engine_rank: int
    ib_device: str
    ip_addr: dict

class PrefillReadyRequest(BaseModel):
    room_id: int
    ready: bool = True

# 用于存储 room_id 到 ucx_port 的映射
room_to_port_mapping = {}
# 用于存储可以开始 prefill 的 room_id
prefill_ready_rooms = set()

def _create_error_response(e):
    return JSONResponse(
        status_code=500,
        content={"error": str(e)},
    )

# Fast API
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.api_route("/handshake", methods=["POST"])
async def handshake(request: HandshakeRequest):
    """
    处理接收方(Receiver)的握手请求
    根据 conn_iflytek.py 的注释，接收方会建立 UCX 服务器并发送 room_id 和 ucx_port 信息
    """
    try:
        room_id = request.room_id
        global room_to_port_mapping
        print(request)
        logging.info(f"Handshake successful for room_id: {room_id}")
        if room_id not in room_to_port_mapping:
            room_to_port_mapping[room_id] = {request.engine_rank: request.ip_addr}
        else:
            room_to_port_mapping[room_id].update({request.engine_rank:request.ip_addr})
        return {
            "status": "success",
            "message": f"Handshake completed for room {room_id}",
            "room_id": room_id,
        }
    except Exception as e:
        logging.error(f"Handshake failed: {str(e)}")
        return _create_error_response(e)

@app.api_route("/get_room_info/{room_id}", methods=["GET"])
async def get_room_info(room_id: int):
    """
    查询指定 room_id 对应的 ucx_port 信息
    发送方(Sender)可以通过此接口获取接收方的 ucx_port
    """
    try:
        if room_id not in room_to_port_mapping:
            return JSONResponse(
                status_code=404,
                content={"error": f"Room {room_id} not found"}
            )

        return room_to_port_mapping.get(room_id, {})
    except Exception as e:
        return _create_error_response(e)

@app.api_route("/prefill_ready", methods=["POST"])
async def prefill_ready(request: PrefillReadyRequest):
    """
    接收方通知 bootstrap server 该请求可以开始 prefill
    根据 conn_iflytek.py 注释的第 2 步: "D->P* 预分配结束后，通知bootstrap server，该请求可以开始prefill"
    """
    try:
        room_id = request.room_id

        if room_id not in room_to_port_mapping:
            return JSONResponse(
                status_code=404,
                content={"error": f"Room {room_id} not found, please handshake first"}
            )

        # 将该 room_id 标记为可以开始 prefill
        prefill_ready_rooms.add(room_id)

        logging.info(f"Room {room_id} is ready for prefill")

        return {
            "status": "success",
            "message": f"Room {room_id} is now ready for prefill"
        }
    except Exception as e:
        return _create_error_response(e)


class UvicornServer:
    def __init__(self, app: FastAPI, host: str, port: int, shared_data: dict):
        self.app = app
        self.host = host
        self.port = port
        # Store shared data that can be accessed from both threads
        self.app.shared_data = shared_data
        self.server = None
        self.thread = None

    def start(self):
        """Start the server in a separate thread"""
        self.thread = threading.Thread(target=self._run,daemon=True)
        self.thread.start()

    def _run(self):
        """Internal method that runs the server"""
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level=logging.INFO,
            timeout_keep_alive=5,
            loop="uvloop"
        )


def start_bootstrap_server(bootstrap_host: str, bootstrap_port: int, server_args: Optional[dict] = None):
    """
    Start the bootstrap server in a separate thread with shared data

    Args:
        bootstrap_host: Host address
        bootstrap_port: Port number
        server_args: Optional server arguments

    Returns:
        tuple: (UvicornServer instance, shared data dictionary)
    """
    global_room_data.update({"server_args":server_args})

    server = UvicornServer(app, bootstrap_host, bootstrap_port, global_room_data)
    server.start()

    return server
