import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, distributed

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=3)
        self.flatten_dim = None
        self._fc = None

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        if self._fc is None:
            self.flatten_dim = x.shape[1]
            self._fc = nn.Linear(self.flatten_dim, 10).to(x.device)
        return self._fc(x)

def setup():
    
    # Step 1. Infer rank, local_rank, and world_size from SLURM environment variables
    # TODO: Replace FIXME with your code.
    rank = FIXME
    local_rank = FIXME
    world_size = FIXME

    # Step 2. Set MASTER_ADDR and MASTER_PORT from SLURM
    # TODO: Replace FIXME with your code.
    if "FIXME" not in os.environ:
        os.environ["FIXME"] = os.environ.get("SLURM_NODELIST", "127.0.0.1")
    if "FIXME" not in os.environ:
        os.environ["FIXME"] = str(12000 + os.getpid() % 10000)

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

    if rank == 0:
        print(f"[Rank 0] MASTER_ADDR={os.environ['MASTER_ADDR']}, MASTER_PORT={os.environ['MASTER_PORT']}")
        print(f"World size: {world_size}, GPUs per node: {torch.cuda.device_count()}")

    return rank, world_size, local_rank, torch.device(f"cuda:{local_rank}")

def cleanup():
    dist.destroy_process_group()

def train():
    rank, world_size, local_rank, device = setup()

    transform = transforms.ToTensor()
    dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
    sampler = distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler)

    model = Net().to(device)
    model = DDP(model, device_ids=[local_rank])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    model.train()
    for epoch in range(100):
        sampler.set_epoch(epoch)
        epoch_start = time.time()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

            if batch_idx % 100 == 0 and rank == 0:
                acc = 100. * correct / total
                print(f"Epoch [{epoch+1}], Batch [{batch_idx}], Loss: {loss.item():.4f}, Acc: {acc:.2f}%")

        if rank == 0:
            acc = 100. * correct / total
            print(f"Epoch {epoch+1} complete | Loss: {total_loss / len(loader):.4f} | Accuracy: {acc:.2f}% | Time: {time.time() - epoch_start:.2f}s")

    if rank == 0:
        print(f"Training finished in {time.time() - start_time:.2f}s")

    cleanup()

if __name__ == "__main__":
    
    # Step 3. Set a direct call to train(), as SLURM + srun handles spawning across nodes
    # TODO: Replace FIXME with your code.    
    FIXME
