import torch
import torch.nn as nn
import torch.nn.functional as F

# Set a larger image size
image_size = (256, 256)  # Example: 256x256
num_images_per_study = 150  # Adjusted based on memory estimation


# Define the multi-head classifier
class MultiHeadClassifier(nn.Module):
    def __init__(self, num_heads=25, num_classes_per_head=3):
        super(MultiHeadClassifier, self).__init__()
        self.num_heads = num_heads
        self.num_classes_per_head = num_classes_per_head

        # Define the shared layers with adjusted input size
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Adjust the size of the linear layer input based on the new image size
        self.shared_fc = nn.Sequential(
            # Adjust according to pooling layers
            nn.Linear(64 * (image_size[0] // 4) * (image_size[1] // 4), 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Define the heads
        self.heads = nn.ModuleList([
            nn.Linear(512, num_classes_per_head) for _ in range(num_heads)
        ])

    def forward(self, x):
        x = self.shared_conv(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.shared_fc(x)

        # Collect outputs from each head
        outputs = [head(x) for head in self.heads]
        return outputs


# Initialize the model
model = MultiHeadClassifier(num_heads=25, num_classes_per_head=3)

# Batch size, image size, num_images_per_study
batch_size = 32
image_size = (256, 256)
num_images_per_study = 150

# Create an example input tensor with the adjusted size
# Shape: (batch_size, num_images_per_study, channels, height, width)
input_tensor = torch.randn(
    batch_size, num_images_per_study, 1, *image_size)  # Batch of 32 studies

# Reshape the input tensor to match the expected input shape for the model
# New shape: (batch_size * num_images_per_study, channels, height, width)
input_tensor = input_tensor.view(-1, 1, *image_size)

# Forward pass through the model
outputs = model(input_tensor)

# Each output corresponds to the predictions for a specific condition-level combination
for i, output in enumerate(outputs):
    print(f"Output for head {i}: {output.shape}")
