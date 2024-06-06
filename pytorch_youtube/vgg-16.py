# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# VGG16 Architecture summary
VGG16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256,
         'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
# then, flatten and 4096x4096x1000 linear layers


class VGG_16(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(VGG_16, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG16)
        self.fc = nn.Sequential(  # make more compact
            nn.Linear(in_features=512*7*7, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)  # flatten
        x = self.fc(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:  # if integer -> conv layer
                out_channels = x
                # Include batch normalization to increase performance
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(
                    3, 3), stride=(1, 1), padding=(1, 1)), nn.BatchNorm2d(x), nn.ReLU()]
                in_channels = x  # update the in_channel for next conv layer
            elif x == "M":  # not integer  -> max pool
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        # unpacking all layers and create an intire block of these layers
        return nn.Sequential(*layers)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VGG_16(in_channels=3, num_classes=1000).to(device)
# input image is fixed as a 244x244 RGB image
x = torch.randn(1, 3, 244, 244).to(device)
print(x.shape)  # torch.Size([1, 3, 244, 244])
print(model(x).shape)  # torch.Size([1, 1000])
