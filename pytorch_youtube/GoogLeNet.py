import torch
import torch.nn as nn

# Inception architecture are restriced to filter size 1x1, 3x3, 5x5, however this decision was based more on convenience rather than necessity
# One big problem with the above modules, at least in this naive form, is that even a modest number of 5x5 convolutions can be prohibitively expensive on top of a convolutional layer with a large number of filters.
# Inception module with dimension reductions -> have 1x1 conv before 3x3 or 5x5 conv to reduce the dimension that gets passed through larger convolutions.
# --> 1x1 convolutions are used to compute reductinos before the expensive 3x3 and 5x5 convolutions.
