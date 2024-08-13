import torch
import torch.distributed as dist

# 设置分布式环境
def setup(rank, world_size):
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)

# 销毁分布式环境
def cleanup():
    dist.destroy_process_group()