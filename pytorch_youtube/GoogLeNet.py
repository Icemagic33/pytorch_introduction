import torch
import torch.nn as nn

# Inception architecture are restriced to filter size 1x1, 3x3, 5x5, however this decision was based more on convenience rather than necessity
# One big problem with the above modules, at least in this naive form, is that even a modest number of 5x5 convolutions can be prohibitively expensive on top of a convolutional layer with a large number of filters.
# Inception module with dimension reductions -> have 1x1 conv before 3x3 or 5x5 conv to reduce the dimension that gets passed through larger convolutions.
# --> 1x1 convolutions are used to compute reductinos before the expensive 3x3 and 5x5 convolutions.
# conv(7x7/2) -> max pool(3x3/2) -> conv(3x3/1) -> max pool(3x3/2) -> inception(3a) -> inception(3b) -> max pool(3x3/2) -> inception(4a) -> inception(4b) -> inception(4c) -> inception(4d) -> inception(4e) -> max pool(3x3/2) -> inception(5a) -> inception(5b) -> avg pool(7x7/1) -> dropout(40%) -> linear -> softmax
# Check Table 1 for more details including output size
# Auxiliary classifiers are used only during training


class GoogLenet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(GoogLenet, self).__init__()
        self.conv1 = conv_block(in_channels=in_channels, out_channels=64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3))
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = conv_block(64, 192, kernel_size=3,
                                stride=1, padding=1)  # more compactly
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # In this order: in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool


class Inception_block(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        super(Inception_block, self).__init__()
        self.branch1 = conv_block(in_channels, out_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            conv_block(in_channels, red_3x3, kernel_size=1),
            conv_block(red_3x3, out_3x3, kernel_size=3, stride=1, padding=1)
        )
        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_5x5, kernel_size=1),
            conv_block(red_5x5, out_5x5, kernel_size=5, stride=1, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            conv_block(in_channels, out_1x1, kernel_size=1)
        )

    def forward(self, x):
        # N x filters x 28 x 28
        # Dimension of the concatenation is 1D
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):     # key word arguments
        super(conv_block, self).__init__()
        self.relu = nn.ReLU()
        # kernel_size = (1,1), (3,3), (5,5)
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        # not in the paper (wasn't invented yet), but increases performance
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))
