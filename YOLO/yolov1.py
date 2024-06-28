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
    # List:
    [(1, 512, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class YOLO(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(YOLO, self).__init__()
        self.layer1 = nn.Sequential{
            nn.Conv2d(in_channels=, out_channels=, kernel_size=(7, 7), stride=, padding=)

        }

    def forward(self, x):
        x = nn.ReLU(x.first)
