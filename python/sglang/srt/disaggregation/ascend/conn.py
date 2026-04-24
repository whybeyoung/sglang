import concurrent.futures
import logging
from typing import List, Tuple

import numpy as np
import numpy.typing as npt

from sglang.srt.distributed import is_pipeline_last_stage
from sglang.srt.disaggregation.ascend.transfer_engine import AscendTransferEngine
from sglang.srt.disaggregation.common.utils import group_concurrent_contiguous
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVBootstrapServer,
    MooncakeKVManager,
    MooncakeKVReceiver,
    MooncakeKVSender,
)
from sglang.srt.utils.network import get_local_ip_auto

from sglang.srt.distributed import get_world_rank

logger = logging.getLogger(__name__)


class AscendKVManager(MooncakeKVManager):
    def init_engine(self):
        # TransferEngine initialized on ascend.
        local_ip = get_local_ip_auto()
        self.engine = AscendTransferEngine(
            hostname=local_ip,
            npu_id=self.kv_args.gpu_id,
            disaggregation_mode=self.disaggregation_mode,
        )

    def register_buffer_to_engine(self):
        self.engine.batch_register(self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens)
        # The Ascend backend optimize batch registration for small memory blocks.
        self.engine.batch_register(
            self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens
        )
        # Batch register state/extra pool data buffers
        if self.kv_args.state_data_ptrs and self.kv_args.state_data_lens:
            self.engine.batch_register(
                self.kv_args.state_data_ptrs, self.kv_args.state_data_lens
            )

    def get_mla_kv_ptrs_with_pp(
        self, src_kv_ptrs: List[int], dst_kv_ptrs: List[int]
    ) -> Tuple[List[int], List[int], int]:
        start_layer = self.kv_args.prefill_start_layer

        if self.kv_args.state_type == "nsa":
            ptrs_per_layer = 3
        else:
            ptrs_per_layer = 2

        if self.kv_args.has_draft_pool:
            total_num_layers = len(dst_kv_ptrs) // ptrs_per_layer - 1
        else:
            total_num_layers = len(dst_kv_ptrs) // ptrs_per_layer

        if is_pipeline_last_stage() and self.kv_args.has_draft_pool:
            src_layers = len(src_kv_ptrs) // ptrs_per_layer  - 1
        else:
            src_layers = len(src_kv_ptrs) // ptrs_per_layer

        end_layer = start_layer + src_layers

        from sglang.srt.distributed import get_world_rank
        # print(f'==={get_world_rank}======={src_layers=}=={total_num_layers=}==={start_layer=}=={end_layer=}==={len(dst_kv_ptrs)=}==={len(src_kv_ptrs)=}')
        if src_layers == total_num_layers:
            sliced_dst_kv_ptrs = dst_kv_ptrs
        else:
            k_ptrs = dst_kv_ptrs[start_layer:end_layer]
            v_ptrs = dst_kv_ptrs[total_num_layers + start_layer: total_num_layers + end_layer]
            if self.kv_args.state_type == "nsa":
                index_k_ptrs = dst_kv_ptrs[2 * total_num_layers + start_layer: 2 * total_num_layers + end_layer]
                sliced_dst_kv_ptrs = k_ptrs + v_ptrs + index_k_ptrs
                if is_pipeline_last_stage() and self.kv_args.has_draft_pool:
                    sliced_dst_kv_ptrs += dst_kv_ptrs[-ptrs_per_layer:]
            else:
                sliced_dst_kv_ptrs = k_ptrs + v_ptrs

        layers_current_pp_stage = len(src_kv_ptrs)
        return src_kv_ptrs, sliced_dst_kv_ptrs, layers_current_pp_stage

    def send_kvcache(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        executor: concurrent.futures.ThreadPoolExecutor,
    ):
        # Group by indices
        prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
            prefill_kv_indices, dst_kv_indices
        )

        if self.pp_size > 1:
            if self.is_mla_backend:
                src_kv_ptrs, sliced_dst_kv_ptrs, layers_current_pp_stage = self.get_mla_kv_ptrs_with_pp(self.kv_args.kv_data_ptrs, dst_kv_ptrs)
                layers_params = [
                    (
                        src_kv_ptrs[layer_id],
                        sliced_dst_kv_ptrs[layer_id],
                        self.kv_args.kv_item_lens[layer_id],
                    )
                    for layer_id in range(layers_current_pp_stage)
                ]
                # print(f'{get_world_rank()}========{layers_params=}')
            else:
                src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, layers_current_pp_stage = (
                    self.get_mha_kv_ptrs_with_pp(self.kv_args.kv_data_ptrs, dst_kv_ptrs)
                )

                layers_params = [
                    (
                        src_k_ptrs[layer_id],
                        dst_k_ptrs[layer_id],
                        self.kv_args.kv_item_lens[layer_id],
                    )
                    for layer_id in range(layers_current_pp_stage)
                ] + [
                    (
                        src_v_ptrs[layer_id],
                        dst_v_ptrs[layer_id],
                        self.kv_args.kv_item_lens[layers_current_pp_stage + layer_id],
                    )
                    for layer_id in range(layers_current_pp_stage)
                ]
        else:
            num_layers = len(self.kv_args.kv_data_ptrs)
            layers_params = [
                (
                    self.kv_args.kv_data_ptrs[layer_id],
                    dst_kv_ptrs[layer_id],
                    self.kv_args.kv_item_lens[layer_id],
                )
                for layer_id in range(num_layers)
            ]

        def set_transfer_blocks(
            src_ptr: int, dst_ptr: int, item_len: int
        ) -> List[Tuple[int, int, int]]:
            transfer_blocks = []
            for prefill_index, decode_index in zip(prefill_kv_blocks, dst_kv_blocks):
                src_addr = src_ptr + int(prefill_index[0]) * item_len
                dst_addr = dst_ptr + int(decode_index[0]) * item_len
                length = item_len * len(prefill_index)
                transfer_blocks.append((src_addr, dst_addr, length))
            return transfer_blocks

        # Worker function for processing a single layer
        def process_layer(src_ptr: int, dst_ptr: int, item_len: int) -> int:
            transfer_blocks = set_transfer_blocks(src_ptr, dst_ptr, item_len)
            return self._transfer_data(mooncake_session_id, transfer_blocks)

        # Worker function for processing all layers in a batch
        def process_layers(layers_params: List[Tuple[int, int, int]]) -> int:
            transfer_blocks = []
            for src_ptr, dst_ptr, item_len in layers_params:
                transfer_blocks.extend(set_transfer_blocks(src_ptr, dst_ptr, item_len))
            return self._transfer_data(mooncake_session_id, transfer_blocks)

        if self.enable_custom_mem_pool:
            futures = [
                executor.submit(
                    process_layer,
                    src_ptr,
                    dst_ptr,
                    item_len,
                )
                for (src_ptr, dst_ptr, item_len) in layers_params
            ]
            for future in concurrent.futures.as_completed(futures):
                status = future.result()
                if status != 0:
                    for f in futures:
                        f.cancel()
                    return status
        else:
            # Combining all layers' params in one batch transfer is more efficient
            # compared to using multiple threads
            return process_layers(layers_params)

        return 0


class AscendKVSender(MooncakeKVSender):
    pass


class AscendKVReceiver(MooncakeKVReceiver):
    pass


class AscendKVBootstrapServer(MooncakeKVBootstrapServer):
    pass
