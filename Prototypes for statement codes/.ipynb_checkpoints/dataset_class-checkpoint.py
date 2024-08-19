import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import ast

class ECGImageDataset(Dataset):
    def __init__(self, info_df_path, transform=None):
        self.info_df = pd.read_json(info_df_path)
        self.transform = transform

    def __len__(self):
        return len(self.info_df)

    def __getitem__(self, idx):
        img_path = self.info_df.iloc[idx]['path']
        image = Image.open(img_path).convert('RGB')
        
        # Convert diagnostic_superclass_encoded
        diagnostic_encoded = torch.tensor(self.info_df.iloc[idx]['diagnostic_one_hot']).long()

        # Convert scp_labels_encoded
        scp_encoded = torch.tensor(self.info_df.iloc[idx]['scp_one_hot']).long()
 
        if self.transform:
            image = self.transform(image)

        return image, (diagnostic_encoded, scp_encoded)