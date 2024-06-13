import torch
import torch.nn as nn

# 1st place in ILSVRC 2015 classification task
# Problem: Adding more layers to a suitably deep model leads to higher training error
# It chooses waht it wants to learn -> It can learn new things, but it never forgets what it has learned before
# So, in theory, increasing the depth never worsens the performance.


class resnet_layers(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):     # key word arguments
        super(resnet_layers, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=112, kernel_size=(
            7, 7), stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        self.conv2_x = nn.Sequential(
            nn.Conv2d(in_channels=112, out_channels=64,
                      kernel_size=(1, 1), stride=1, padding=1),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=(3, 3), stride=1, padding=1),
            nn.Conv2d(in_channels=64, out_channels=256,
                      kernel_size=(1, 1), stride=1, padding=1),
        )
        self.conv3_x = nn.Sequential(
            nn.Conv2d(in_channels=112, out_channels=56,
                      kernel_size=(1, 1), stride=1, padding=1),
            nn.Conv2d(in_channels=56, out_channels=56,
                      kernel_size=(1, 1), stride=1, padding=1),
            nn.Conv2d(in_channels=56, out_channels=56,
                      kernel_size=(1, 1), stride=1, padding=1),
        )
        self.conv4_x = nn.Sequential(
            nn.Conv2d(in_channels=112, out_channels=56,
                      kernel_size=(1, 1), stride=1, padding=1),
            nn.Conv2d(in_channels=56, out_channels=56,
                      kernel_size=(1, 1), stride=1, padding=1),
            nn.Conv2d(in_channels=56, out_channels=56,
                      kernel_size=(1, 1), stride=1, padding=1),
        )
        self.conv5_x = nn.Sequential(
            nn.Conv2d(in_channels=112, out_channels=56,
                      kernel_size=(1, 1), stride=1, padding=1),
            nn.Conv2d(in_channels=56, out_channels=56,
                      kernel_size=(1, 1), stride=1, padding=1),
            nn.Conv2d(in_channels=56, out_channels=56,
                      kernel_size=(1, 1), stride=1, padding=1),
        )

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))
