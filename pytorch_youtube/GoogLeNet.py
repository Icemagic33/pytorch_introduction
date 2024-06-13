import torch
import torch.nn as nn

# Inception architecture are restriced to filter size 1x1, 3x3, 5x5, however this decision was based more on convenience rather than necessity
# One big problem with the above modules, at least in this naive form, is that even a modest number of 5x5 convolutions can be prohibitively expensive on top of a convolutional layer with a large number of filters.
# Inception module with dimension reductions -> have 1x1 conv before 3x3 or 5x5 conv to reduce the dimension that gets passed through larger convolutions.
# --> 1x1 convolutions are used to compute reductinos before the expensive 3x3 and 5x5 convolutions.
# conv(7x7/2) -> max pool(3x3/2) -> conv(3x3/1) -> max pool(3x3/2) -> inception(3a) -> inception(3b) -> max pool(3x3/2) -> inception(4a) -> inception(4b) -> inception(4c) -> inception(4d) -> inception(4e) -> max pool(3x3/2) -> inception(5a) -> inception(5b) -> avg pool(7x7/1) -> dropout(40%) -> linear -> softmax
# Check Table 1 for more details including output size
# Auxiliary classifiers are used only during training


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
