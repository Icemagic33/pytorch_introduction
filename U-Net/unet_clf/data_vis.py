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


# Visualize the distribution of different diagnostic categories
figure, axis = plt.subplots(1, 3, figsize=(20, 5))

# three main diagnostic categories
for idx, d in enumerate(['foraminal', 'subarticular', 'canal']):
    # inlucde only the filters that are related to the current diagnostic category
    diagnosis = list(filter(lambda x: x.find(d) > -1, train.columns))

    dff = train[diagnosis]
    with warnings.catch_warnings():
        # calculate the frequency of each diagnostic severity for each condition within the current category
        # and transpose the result for plotting
        warnings.simplefilter(action='ignore', category=FutureWarning)
        value_counts = dff.apply(pd.value_counts).fillna(0).T
    # plot the value counts as a stacked bar chart on the corresponding subplot
    value_counts.plot(kind='bar', stacked=True, ax=axis[idx])
    axis[idx].set_title(f'{d} distribution')

# List out all of the Studies we have on patients.
part_1 = os.listdir(f'{fd}/train_images')
# Filter out any system files that might be present (e.g., '.DS_Store')
part_1 = list(filter(lambda x: x.find('.DS') == -1, part_1))
# Load metadata from the CSV file
df_meta_f = pd.read_csv(f'{fd}/train_series_descriptions.csv')
# Create a list of tuples containing study IDs and their corresponding folder paths
p1 = [(x, f"{fd}/train_images/{x}") for x in part_1]
# Initialize a dictionary to hold metadata for each study
meta_obj = {p[0]: {'folder_path': p[1], 'SeriesInstanceUIDs': []} for p in p1}
