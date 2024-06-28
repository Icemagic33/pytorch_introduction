# Each ouput and label will be relative to the cell
# Each bounding box for each cell will have: [x, y, w, h]
# x, y are between 0 to 1 (boudning boxes have top left corner (0,0) and bottom right corner (1,1))
# w, h can be greater than 1 if the object is wider or taller than the cell
import torch
import torch.nn as nn

architecture_config = [
    # Tuple:  kernel_size, # of filters = output depth, stride, padding
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    # List: tuples, and then last integer represents number of repeats
    [(1, 512, 1, 0), (3, 512, 1, 1), 4],  # last value(=4)
    # is how many times these sequences should be repeated
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],  # last value(=2)
    # is how many times these sequences should be repeated
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        # conv layer -> batch norm (not in paper, but helps performance) -> leaky relu
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs)  # default 3 for RGB image
