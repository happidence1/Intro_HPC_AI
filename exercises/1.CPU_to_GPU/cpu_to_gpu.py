import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import time

# Step 1. Define the device
# TODO: Replace FIXME with code that selects "cuda" if a GPU is available, otherwise "cpu".
# device = FIXME

# Define the CNN
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=3)  # Output: (batch, 32, 26, 26)
        self.flatten_dim = None
        self._fc = None  # Lazy initialization

    def forward(self, x):
        x = torch.relu(self.conv(x))  # (N, 32, 26, 26)
        x = x.view(x.size(0), -1)     # Flatten
        if self._fc is None:
            self.flatten_dim = x.shape[1]
            
            # Step 5. Handle Lazy Initialization Layer
            # TODO: Replace FIXME with code that nsures the layer is created on the same device as input.
            self._fc = nn.Linear(self.flatten_dim, 10).FIXME
        return self._fc(x)

# Dataset
transform = transforms.ToTensor()
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('.', train=True, download=True, transform=transform),
    batch_size=64, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('.', train=False, download=True, transform=transform),
    batch_size=64, shuffle=False
)

# Initialize model, optimizer, loss

# Step 2. Move the model to GPU
# TODO: Replace FIXME with code that move the model to GPU.
model = Net().FIXME

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Training
num_epochs = 1
total_start = time.time()

for epoch in range(num_epochs):
    model.train()
    epoch_start = time.time()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        batch_start = time.time()

        # Step 3. Move data and labels to GPU during training
        # TODO: Replace FIXME with code that move data and labels to GPU.
        data, target = FIXME

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        # Stats
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == target).sum().item()
        total += target.size(0)

        batch_time = time.time() - batch_start
        if batch_idx % 100 == 0:
            accuracy = 100. * correct / total
            print(f"Epoch [{epoch+1}], Batch [{batch_idx}], "
                  f"Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%, "
                  f"Time: {batch_time:.2f}s")

    epoch_time = time.time() - epoch_start
    epoch_accuracy = 100. * correct / total
    avg_loss = running_loss / len(train_loader)
    print(f"\n Epoch {epoch+1} Summary: Loss = {avg_loss:.4f}, Accuracy = {epoch_accuracy:.2f}%, "
          f"Time = {epoch_time:.2f}s\n")

total_training_time = time.time() - total_start
print(f"Total Training Time: {total_training_time:.2f}s")

# Evaluation
model.eval()
test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for data, target in test_loader:

        # Step 4. Move data and labels to GPU during evaluation
        # TODO: Replace FIXME with code that move data and labels to GPU.
        data, target = FIXME

        outputs = model(data)
        loss = criterion(outputs, target)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

final_test_loss = test_loss / len(test_loader)
final_accuracy = 100. * correct / total

print(f"\n Evaluation Summary:")
print(f"Final Test Loss: {final_test_loss:.4f}")
print(f"Final Test Accuracy: {final_accuracy:.2f}%")
