import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import pydicom
import numpy as np
import os
import glob
from tqdm import tqdm
import warnings

# Access to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# File location
fd = "/content/drive/MyDrive/rsna-2024-lumbar-spine-degenerative-classification"
train = pd.read_csv(f'{fd}/train.csv')

# Check how the training files are organized
print("Total Cases: ", len(train))
print(train.columns)
