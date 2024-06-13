import torch
import torch.nn as nn

# 1st place in ILSVRC 2015 classification task
# Problem: Adding more layers to a suitably deep model leads to higher training error


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):     # key word arguments
        super(conv_block, self).__init__()
        self.conv2_x = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
            nn.Conv2d(in_channels=in_channels, out_channel=50,
                      kernel_size=(1, 1), stride=1, padding=1),
            nn.Conv2d(in_channels=50, out_channel=64,
                      kernel_size=(3, 3), stride=1, padding=1),
            nn.Conv2d(in_channels=64, out_channel=256,
                      kernel_size=(1, 1), stride=1, padding=1),
        )
        self.conv1 = nn.Conv2d()
        # kernel_size = (1,1), (3,3), (5,5)
        self.conv2 = nn.Conv2d(in_channels, out_channels, **kwargs)
        # not in the paper (wasn't invented yet), but increases performance
        self.conv3 = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))
