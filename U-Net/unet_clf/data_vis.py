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

# Iterate over each study in the meta_obj dictionary
for m in meta_obj:
    # List all directories (series) within the study folder, filtering out system files
    meta_obj[m]['SeriesInstanceUIDs'] = list(
        filter(lambda x: x.find('.DS') == -1,
               os.listdir(meta_obj[m]['folder_path'])
               )
    )

# Iterate over each study in the meta_obj dictionary using tqdm for progress bar
for k in tqdm(meta_obj):
    for s in meta_obj[k]['SeriesInstanceUIDs']:
        # Initialize the 'SeriesDescriptions' list if not already present
        if 'SeriesDescriptions' not in meta_obj[k]:
            meta_obj[k]['SeriesDescriptions'] = []
        try:
            # Append the series description to the 'SeriesDescriptions' list
            meta_obj[k]['SeriesDescriptions'].append(
                df_meta_f[(df_meta_f['study_id'] == int(k)) &
                          (df_meta_f['series_id'] == int(s))]['series_description'].iloc[0])
        except:
            # Print an error message if the series description cannot be found
            print("Failed on", s, k)

# Access, retrieve, and display the metadata for the second study
print(meta_obj[list(meta_obj.keys())[1]])

# Retrieve data for the second patient in the train DataFrame
patient = train.iloc[1]

# Retrieve the metadata for the study ID associated with the second patient in the train DataFrame from the meta_obj dictionary
ptobj = meta_obj[str(patient['study_id'])]
print(ptobj)

# Get data into the format
"""
im_list_dcm = {
    '{SeriesInstanceUID}': {
        'images': [
            {'SOPInstanceUID': ...,
             'dicom': PyDicom object
            },
            ...,
        ],
        'description': # SeriesDescription
    },
    ...
}
"""

# Initialize the dictionary to hold the DICOM images and series descriptions
im_list_dcm = {}

# Iterate over each SeriesInstanceUID in the patient's study
for idx, i in enumerate(ptobj['SeriesInstanceUIDs']):
    # Initialize the dictionary for each series with an empty list for images and the series description
    im_list_dcm[i] = {'images': [],
                      'description': ptobj['SeriesDescriptions'][idx]}

    # Get the list of all DICOM files in the current series directory
    images = glob.glob(
        f"{ptobj['folder_path']}/{ptobj['SeriesInstanceUIDs'][idx]}/*.dcm")

    # Iterate over the sorted list of DICOM files
    for j in sorted(images, key=lambda x: int(x.split('/')[-1].replace('.dcm', ''))):
        # Append the SOPInstanceUID and the PyDicom object to the list of images for the current series
        im_list_dcm[i]['images'].append({
            'SOPInstanceUID': j.split('/')[-1].replace('.dcm', ''),
            'dicom': pydicom.dcmread(j)})


# Function to display images
def display_images(images, title, max_images_per_row=4):
    # Calculate the number of rows needed
    num_images = len(images)
    num_rows = (num_images + max_images_per_row -
                1) // max_images_per_row  # Ceiling division

    # Handle the case when num_rows is 0
    if num_rows == 0:
        num_rows = 1  # Set at least one row to display images

    # Create a subplot grid
    fig, axes = plt.subplots(
        num_rows, max_images_per_row, figsize=(5, 1.5 * num_rows))

    # Flatten axes array for easier looping if there are multiple rows
    if num_rows > 1:
        axes = axes.flatten()
    else:
        axes = [axes]  # Make it iterable for consistency

    # Plot each image
    for idx, image in enumerate(images):
        ax = axes[idx]
        # Assuming grayscale for simplicity, change cmap as needed
        ax.imshow(image, cmap='gray')
        ax.axis('off')  # Hide axes

    # Turn off unused subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
    fig.suptitle(title, fontsize=16)

    plt.tight_layout()
