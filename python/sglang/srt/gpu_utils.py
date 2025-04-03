import pynvml
from typing import Optional, Dict, List
import json


class NvidiaTopology:
    def __init__(self):
        """Initialize NVIDIA Management Library"""
        pynvml.nvmlInit()
        self.device_count = pynvml.nvmlDeviceGetCount()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup when done"""
        pynvml.nvmlShutdown()

    def get_ib_device_for_gpu(self, gpu_id: int) -> Optional[Dict]:
        """
        Get IB device information for specific GPU using NVIDIA API

        Args:
            gpu_id: GPU device ID

        Returns:
            Optional[Dict]: Dictionary containing IB device info or None
        """
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            pci_info = pynvml.nvmlDeviceGetPciInfo(handle)

            topology_info = {
                'gpu_pci_info': {
                    'domain': pci_info.domain,
                    'bus': pci_info.bus,
                    'device': pci_info.device,
                    'pci_bus_id': pci_info.busId
                }
            }

            # Try to get NUMA node information
            try:
                numa_node = pynvml.nvmlDeviceGetNumaNode(handle)
                if numa_node >= 0:
                    topology_info['numa_node'] = numa_node
            except pynvml.NVMLError:
                pass

            # Try to get CPU affinity
            try:
                cpu_affinity = pynvml.nvmlDeviceGetCpuAffinity(handle, 0)
                topology_info['cpu_affinity'] = cpu_affinity
            except pynvml.NVMLError:
                pass

            # Try to get P2P capability with other GPUs
            topology_info['p2p_links'] = []
            for other_gpu_id in range(self.device_count):
                if other_gpu_id != gpu_id:
                    try:
                        other_handle = pynvml.nvmlDeviceGetHandleByIndex(other_gpu_id)
                        p2p_status = pynvml.nvmlDeviceGetP2PStatus(
                            handle,
                            other_handle,
                            0  # P2P_FEATURE_NVLINK = 0
                        )
                        if p2p_status == 0:  # P2P_STATUS_OK = 0
                            other_pci = pynvml.nvmlDeviceGetPciInfo(other_handle)
                            topology_info['p2p_links'].append({
                                'gpu_id': other_gpu_id,
                                'pci_bus_id': other_pci.busId.decode('utf-8')
                            })
                    except pynvml.NVMLError:
                        continue

            return topology_info

        except pynvml.NVMLError as e:
            print(f"Error getting GPU topology info: {e}")
            return None

    def get_gpu_info(self, gpu_id: int) -> Dict:
        """
        Get detailed information about a GPU

        Args:
            gpu_id: GPU device ID

        Returns:
            Dict: Dictionary containing GPU information
        """
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)

            # Basic GPU info
            info = {
                'name': pynvml.nvmlDeviceGetName(handle),
                'uuid': pynvml.nvmlDeviceGetUUID(handle),
                'pci_info': {
                    'bus_id': pynvml.nvmlDeviceGetPciInfo(handle).busId,
                    'bus': pynvml.nvmlDeviceGetPciInfo(handle).bus,
                    'device': pynvml.nvmlDeviceGetPciInfo(handle).device,
                    'domain': pynvml.nvmlDeviceGetPciInfo(handle).domain,
                }
            }

            # Add memory info
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                info['memory'] = {
                    'total': mem_info.total,
                    'free': mem_info.free,
                    'used': mem_info.used
                }
            except pynvml.NVMLError:
                pass

            # Add temperature
            try:
                info['temperature'] = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            except pynvml.NVMLError:
                pass

            # Add power usage
            try:
                info['power_usage'] = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
            except pynvml.NVMLError:
                pass

            # Add utilization
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                info['utilization'] = {
                    'gpu': util.gpu,
                    'memory': util.memory
                }
            except pynvml.NVMLError:
                pass

            return info

        except pynvml.NVMLError as e:
            print(f"Error getting GPU info: {e}")
            return {}


def get_gpu_ib_info(gpu_id: Optional[int] = None) -> Dict:
    """
    Get GPU and topology information

    Args:
        gpu_id: Optional specific GPU ID to query

    Returns:
        Dict: Dictionary containing GPU and topology information
    """
    with NvidiaTopology() as topo:
        if gpu_id is not None:
            # Get info for specific GPU
            return {
                'gpu_info': topo.get_gpu_info(gpu_id),
                'topology_info': topo.get_ib_device_for_gpu(gpu_id)
            }
        else:
            # Get info for all GPUs
            result = {}
            for i in range(topo.device_count):
                result[i] = {
                    'gpu_info': topo.get_gpu_info(i),
                    'topology_info': topo.get_ib_device_for_gpu(i)
                }
            return result


if __name__ == "__main__":
    try:
        # 获取特定 GPU 的信息
        gpu_id = 0
        info = get_gpu_ib_info(gpu_id)
        print(f"\nGPU {gpu_id} information:")
        print(json.dumps(info, indent=2))

    except Exception as e:
        print(f"Error: {e}")
