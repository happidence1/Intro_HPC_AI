import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Step 1. Import distributed training modules
# TODO: Replace FIXME with your code.
import torch.FIXME as dist
from torch.nn.parallel import FIXME as DDP
from torch.utils.data import FIXME

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

# Step 2. Adds setup for distributed training.
# TODO: Replace FIXME with your code.
def setup(rank, world_size):
    os.environ['FIXME'] = '127.0.0.1'
    os.environ['FIXME'] = '29500'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# Step 3. Adds cleanup for distributed training.
# TODO: Replace FIXME with your code.
def cleanup():
    dist.FIXME()

def train(rank, world_size):
    setup(rank, world_size)
    
    # Step 4. Bind each process to a GPU based on rank.
    # TODO: Replace FIXME with your code.
    device = torch.device(f'FIXME')
    
    transform = transforms.ToTensor()
    dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
    
    # Step 5. Use DistributedSampler so each process gets a unique shard of data.
    # TODO: Replace FIXME with your code.    
    sampler = distributed.FIXME(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler)

    model = Net().to(device)
    
    # Step 6. Wraps the model in DDP.
    # TODO: Replace FIXME with your code.     
    model = FIXME(model, device_ids=[rank])
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    model.train()
    for epoch in range(100):
    
        # Step 7-1. Shuffle data in each epoch to avoid bias.
        # TODO: Replace FIXME with your code.     
        sampler.FIXME(epoch)
        
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

            # Step 7-2. Only rank 0 (the main process) prints logs.
            # TODO: Replace FIXME with your code. 
            if batch_idx % 100 == 0 and FIXME:
                acc = 100. * correct / total
                print(f"Epoch [{epoch+1}], Batch [{batch_idx}], Loss: {loss.item():.4f}, Acc: {acc:.2f}%")

        # Step 7-3. Only rank 0 (the main process) prints the End-of-epoch stats.
        # TODO: Replace FIXME with your code.
        if FIXME:
            acc = 100. * correct / total
            print(f"Epoch {epoch+1} complete | Loss: {total_loss / len(loader):.4f} | Accuracy: {acc:.2f}% | Time: {time.time() - epoch_start:.2f}s")

    # Step 7-4. Only rank 0 (the main process) prints the training time.
    # TODO: Replace FIXME with your code.
    if FIXME:
        print(f"Training finished in {time.time() - start_time:.2f}s")

    cleanup()

if __name__ == "__main__":
    # Step 8. Use torch.multiprocessing.spawn to launch one process per GPU.
    # TODO: Replace FIXME with your code.
    world_size = torch.cuda.device_count()
    torch.FIXME(train, args=(world_size,), nprocs=world_size, join=True)
