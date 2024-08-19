import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class ScpToLabelsDataset(Dataset):
    def __init__(self, info_df_path):
        self.info_df = pd.read_json(info_df_path)

    def __len__(self):
        return len(self.info_df)

    def __getitem__(self, idx):        
        # Convert diagnostic_superclass_encoded
        diagnostic_encoded = torch.tensor(self.info_df.iloc[idx]['diagnostic_one_hot']).long()

        # Convert scp_labels_encoded
        scp_encoded = torch.tensor(self.info_df.iloc[idx]['scp_one_hot']).long()

        return scp_encoded, diagnostic_encoded

