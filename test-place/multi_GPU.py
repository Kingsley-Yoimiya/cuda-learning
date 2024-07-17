import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def all_gather_tensor(tensor, world_size):
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return gathered_tensors

def run(rank, world_size):
    setup(rank, world_size)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    # 模拟数据
    tensor = torch.randn(10, device=device)
    print(f"Rank {rank} tensor before all_gather: {tensor}")

    # 使用 Gloo all_gather
    gathered_tensors = all_gather_tensor(tensor, world_size)

    dist.barrier()  # 同步所有进程

    print(f"Rank {rank} tensor after all_gather: {gathered_tensors}")

    cleanup()

def main():
    world_size = 8
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
