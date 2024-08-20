import json
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch
import torch.nn as nn

class ECGImageDataset(Dataset):
    def __init__(self, json_file, transform=None):
        # Load records from JSON file
        with open(json_file, 'r') as f:
            records = json.load(f)
        
        # Convert to DataFrame and reset index
        self.records = pd.DataFrame(records).reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        # Ensure idx is an integer
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        
        # Get the record corresponding to the idx
        record = self.records.iloc[idx]
        
        # Load the image
        image_path = record['path']
        image = Image.open(image_path).convert('RGB')
        
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        
        # Get the label
        label = record['label']
        
        return image, torch.tensor(label, dtype=torch.float)
