#!/usr/bin/env python
# coding:utf-8
"""
@author: nivic ybyang7
@license: Apache Licence
@file: verbs_engine
@time: 2025/04/04
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

import json
import logging
import os
import uuid
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class XTransferEngineConfig:
    local_hostname: str
    metadata_server: str
    protocol: str
    device_name: str

    @staticmethod
    def from_file(file_path: str) -> "XTransferEngineConfig":
        """Load the config from a JSON file."""
        with open(file_path) as fin:
            config = json.load(fin)
        return XTransferEngineConfig(
            local_hostname=config.get("local_hostname", None),
            metadata_server=config.get("metadata_server"),
            protocol=config.get("protocol", "rdma"),
            device_name=config.get("device_name", ""),
        )

    @staticmethod
    def load_from_env() -> "XTransferEngineConfig":
        """Load config from a file specified in the environment variable."""
        config_file_path = os.getenv("MOONCAKE_CONFIG_PATH")
        if config_file_path is None:
            raise ValueError(
                "The environment variable 'MOONCAKE_CONFIG_PATH' is not set."
            )
        return XTransferEngineConfig.from_file(config_file_path)


class XTransferEngine:

    def __init__(self):
        try:
            import mooncake_sglang_adaptor as msa
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-    ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run SGLang with MooncakeTransferEngine."
            ) from e

        self.engine = msa.TransferEngine()

        try:
            self.config = XTransferEngineConfig.load_from_env()
            logger.info("Mooncake Configuration loaded successfully.")
        except ValueError as e:
            logger.error(e)
            raise
        except Exception as exc:
            logger.error("An error occurred while loading the configuration: %s", exc)
            raise

        self.config = XTransferEngineConfig.load_from_env()

        session_suffix = "_" + str(uuid.uuid4())
        self.session_id = self.config.local_hostname + session_suffix
        self.initialize(
            self.session_id,
            self.config.metadata_server,
            self.config.protocol,
            self.config.device_name,
        )

    def register(self, ptr, length):
        self.engine.expRegisterMemory(ptr, length)

    def deregister(self, ptr):
        self.engine.expUnregisterMemory(ptr)

    def initialize(
        self,
        local_hostname: str,
        metadata_server: str,
        protocol: str,
        device_name: str,
    ) -> None:
        """Initialize the mooncake instance."""
        self.engine.initialize(local_hostname, metadata_server, protocol, device_name)

    def transfer_sync(
        self, session_id: str, buffer: int, peer_buffer_address: int, length: int
    ) -> int:
        """Synchronously transfer data to the specified address."""

        write_op = self.engine.TransferOpcode.WRITE
        ret = self.engine.transferSyncExt(
            session_id, buffer, peer_buffer_address, length, write_op
        )
        if ret < 0:
            logger.error("Transfer Return Error")
            raise Exception("Transfer Return Error")
        return ret

    def get_localhost(self):
        return self.config.local_hostname

    def get_session_id(self):
        return self.session_id
