# Imports
import torch
# All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn as nn
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
# All functions that don't have any parameters (relu, ...)
import torch.nn.functional as F
# Gives easier dataset management and creates mini batches
from torch.utils.data import DataLoader
# Has standard datasets we can import in a nice and easy way
import torchvision.datasets as datasets
# Transformations we can perform on our dataset
import torchvision.transforms as transforms


# Create Fully Connected Network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):  # 28x28 = 784 nodes
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Create Convoluted Neural Network
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):  # if rgb image -> in_channels=3
        # Since we are working with MNIST dataset
        # in_channels=1, num_classes=10 because 0~9
        super(CNN, self).__init__()
        # First CNN channel
        # same convolution : preserves the dimension (doesn't reduce size)
        # n_out = {(n_in + 2*padding - kernel)/stride} + 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        '''class Conv2d(
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: _size_2_t | str = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            device: Any | None = None,
            dtype: Any | None = None
        )
        '''
        # Use max pool
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        '''class MaxPool2d(
            kernel_size: _size_any_t,
            stride: _size_any_t | None = None,
            padding: _size_any_t = 0,
            dilation: _size_any_t = 1,
            return_indices: bool = False,
            ceil_mode: bool = False
        )'''
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(16*7*7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x


# model = CNN()
# x = torch.randn(64, 1, 28, 28)
# print(x.shape) # torch.Size([64, 1, 28, 28])
# print(model(x).shape) # torch.Size([64, 10])
# exit()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

# Load Data
train_dataset = datasets.MNIST(
    root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(
    root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size, shuffle=True)

# Initiate Network
# model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)
model = CNN().to(device)  # already set the parameters by default

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()  # we are using adam here


# Check accuracy on training & test to see how good our model works
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += (predictions.size(0))
        print(
            f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')

    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
