import torch
import torch.nn as nn


class MultiHeadClassifier(nn.Module):
    def __init__(self, num_heads=25, num_classes_per_head=3):
        super(MultiHeadClassifier, self).__init__()
        self.num_heads = num_heads
        self.num_classes_per_head = num_classes_per_head

        # Define the shared layers
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

        self.shared_fc = nn.Sequential(
            # Assuming the input image size is (128, 128)
            nn.Linear(64 * 32 * 32, 512),
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


# Example usage
model = MultiHeadClassifier(num_heads=25, num_classes_per_head=3)
input_tensor = torch.randn(4, 1, 128, 128)  # Batch of 4 images
outputs = model(input_tensor)

# Each output corresponds to the predictions for a specific condition-level combination
for i, output in enumerate(outputs):
    print(f"Output for head {i}: {output.shape}")
