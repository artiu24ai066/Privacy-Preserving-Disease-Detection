
import pandas as pd
import torch

from src.config import OUTPUT_DIR, DEVICE
from src.dataset import HAMDataset, train_transform
from src.model import get_model

# Load training data
train_df = pd.read_csv(
    f"{OUTPUT_DIR}/train.csv"
)

# Create dataset
dataset = HAMDataset(
    train_df,
    transform=train_transform
)

print("Dataset size:", len(dataset))

# Load one image
image, label = dataset[0]

print("Image shape:", image.shape)
print("Label:", label)

# Initialize model
model = get_model(num_classes=7)

model = model.to(DEVICE)

# Test forward propagation
image = image.unsqueeze(0).to(DEVICE)

output = model(image)

print("Output shape:", output.shape)

# Check normalization layers
group_norm_count = sum(
    isinstance(m, torch.nn.GroupNorm)
    for m in model.modules()
)

batch_norm_count = sum(
    isinstance(m, torch.nn.BatchNorm2d)
    for m in model.modules()
)

print("GroupNorm layers:", group_norm_count)
print("BatchNorm layers:", batch_norm_count)

assert output.shape == (1, 7)
assert batch_norm_count == 0

print("All tests passed!")
