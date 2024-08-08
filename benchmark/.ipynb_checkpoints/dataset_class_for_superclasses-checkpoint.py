import os
import pandas as pd
import ast
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import torchvision.transforms as transforms

class ECGImageDataset(Dataset):
    def __init__(self, info_df_path, transform=None):
        self.info_df = pd.read_csv(info_df_path)
        self.transform = transform

    def __len__(self):
        return len(self.info_df)

    def __getitem__(self, idx):
        img_path = self.info_df.iloc[idx]['Image Path']
        image = Image.open(img_path).convert('RGB')
        label = self.info_df.iloc[idx]['Label']

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label).long()