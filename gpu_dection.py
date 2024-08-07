import torch

# 检查CUDA是否可用
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

# 获取当前设备数量
device_count = torch.cuda.device_count()
print(f"Number of CUDA devices: {device_count}")

# 列出所有CUDA设备的名称
for i in range(device_count):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")

# 获取当前使用的设备
current_device = torch.cuda.current_device()
print(f"Current CUDA device: {current_device}")

# 显示设备能力和其他属性
if cuda_available:
    for i in range(device_count):
        print(f"Device {i} Name: {torch.cuda.get_device_name(i)}")
        print(f"Device {i} Capability: {torch.cuda.get_device_capability(i)}")
        print(f"Device {i} Memory Allocated: {torch.cuda.memory_allocated(i)} bytes")
        print(f"Device {i} Max Memory Cached: {torch.cuda.max_memory_cached(i)} bytes")
