import sys
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure the correct path to import PatchEmbedding from layers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from layers.Embed import PatchEmbedding

# Define the path to the weather dataset
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dataset_path = os.path.join(base_dir, 'dataset', 'weather/weather.csv')

# Check if the file exists and load Temperature data
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"The file at {dataset_path} was not found.")
    
df = pd.read_csv(dataset_path)


# Extract the first 100 values from the "Temperature (C)" column
temperature_data = df['T (degC)'].values[:100]
temperature_data = torch.tensor(temperature_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 100)

# Instantiate PatchEmbedding with parameters for visualization
d_model = 16  # embedding dimension
patch_len = 8
stride = 8
padding = 0  # No padding for simplicity
dropout = 0.1

patch_embedding = PatchEmbedding(d_model=d_model, patch_len=patch_len, stride=stride, padding=padding, dropout=dropout)

# Forward pass through PatchEmbedding to get patched output
patched_x, n_vars = patch_embedding(temperature_data)

# Visualize the patching with continuity
def visualize_patching_continuous(x, patched_x, patch_len, stride):
    x = x.squeeze().numpy()  # Original sequence
    patches = patched_x.detach().cpu().numpy().reshape(-1, patch_len)
    
    # Plot original data
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(x, marker='o', label='Original Data')
    plt.title("Original Temperature Data (First 100 Values)")
    plt.xlabel("Time Step")
    plt.ylabel("Temperature (C)")
    plt.legend()

    # Plot each patch segment on the original scale
    plt.subplot(2, 1, 2)
    for i, patch in enumerate(patches):
        start_idx = i * stride
        plt.plot(np.arange(start_idx, start_idx + patch_len), patch, marker='o', label=f'Patch {i+1}')
    plt.title("Data Patches with Continuity")
    plt.xlabel("Time Step")
    plt.ylabel("Temperature (C)")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

# Visualize the patching effect
visualize_patching_continuous(temperature_data, patched_x, patch_len, stride)
