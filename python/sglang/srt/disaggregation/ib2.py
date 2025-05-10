import os
import pynvml
import pyverbs.device as d
from pyverbs.device import Context

def get_gpu_pci_address(gpu_no):
    """ 获取 GPU 设备的 PCI 地址 """
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_no)
    pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
    pynvml.nvmlShutdown()
    return pci_info.busId  # 形如 "0000:19:00.0"

def get_roce_devices():
    """ 获取所有 RoCE 设备及其 PCI 地址（不使用 lspci） """
    roce_devices = {}

    for dev in d.get_device_list():
        dev_name = dev.name.decode()
        ctx = Context(name=dev_name)
        port = 1
        #for port in range(1, ctx.num_ports + 1):
        port_attr = ctx.query_port(port)
        if port_attr.link_layer == 2:  # 2 表示 RoCE (Ethernet)
            # 从 sysfs 获取 PCI 地址
            dev_path = f"/sys/class/infiniband/{dev_name}/device"
            if os.path.exists(dev_path):
                try:
                    pci_addr = os.readlink(dev_path).split("/")[-1]  # 形如 "0000:19:00.0"
                    roce_devices[dev_name] = pci_addr
                except Exception as e:
                    print(f"读取 {dev_name} PCI 地址失败: {e}")
    return roce_devices

def get_net_device_from_rdma(rdma_dev):
    """ 获取 RoCE 设备对应的网卡名（不使用 lspci） """
    net_path = f"/sys/class/infiniband/{rdma_dev}/device/net"
    if os.path.exists(net_path):
        try:
            return os.listdir(net_path)[0]  # 读取网卡名
        except Exception as e:
            print(f"获取 {rdma_dev} 关联网卡失败: {e}")
    return None

def find_best_roce_for_gpu(gpu_no):
    """ 根据 GPU 设备号找到最亲和的 RoCE 网卡 """
    gpu_pci = get_gpu_pci_address(gpu_no)
    roce_devices = get_roce_devices()

    best_rdma_dev = None
    min_distance = float("inf")

    for rdma_dev, rdma_pci in roce_devices.items():
        if rdma_pci and rdma_pci[:7] == gpu_pci[:7]:  # 比较 PCI 地址前缀
            distance = abs(int(rdma_pci.split(":")[1], 16) - int(gpu_pci.split(":")[1], 16))
            if distance < min_distance:
                min_distance = distance
                best_rdma_dev = rdma_dev

    if best_rdma_dev:
        net_dev = get_net_device_from_rdma(best_rdma_dev)
        return best_rdma_dev, net_dev
    return None, None

# 测试 GPU 0 的亲和 RoCE 网卡
gpu_no = 0
rdma_dev, net_dev = find_best_roce_for_gpu(gpu_no)
print(f"GPU {gpu_no} 最亲和的 RDMA 设备: {rdma_dev}, 对应网卡: {net_dev}")
