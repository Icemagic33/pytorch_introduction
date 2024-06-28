# Each ouput and label will be relative to the cell
# Each bounding box for each cell will have: [x, y, w, h]
# x, y are between 0 to 1 (boudning boxes have top left corner (0,0) and bottom right corner (1,1))
# w, h can be greater than 1 if the object is wider or taller than the cell
import torch
import torch.nn as nn


class YOLO(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(YOLO, self).__init__()
        self.first = nn.Sequential{
            nn.Conv2d(in_channels=, out_channels=, kernel_size=, strid=, padding=)
        }
