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

def setup(rank, world_size):
    # Use IPv4 localhost to avoid IPv6 error
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    transform = transforms.ToTensor()

    dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
    sampler = distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler)

    model = Net().to(device)
    model = DDP(model, device_ids=[rank])
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
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)
