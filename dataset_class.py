import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch

class ECGImageDataset(Dataset):
    def __init__(self, label_df, image_dir, transform=None):
        self.label_df = label_df
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = self._get_image_paths()

    def _get_image_paths(self):
        image_paths = []
        for root, _, files in os.walk(self.image_dir):
            for file in files:
                if file.endswith('.png'):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        img_name = self.label_df.iloc[idx]['ecg_lr_path'] + '-0.png'
        matching_paths = [path for path in self.image_paths if img_name in path]
        
        # Use the first match if it exists
        img_path = matching_paths[0] if matching_paths else None
        
        while img_path is None:
            idx += 1
            img_name = self.label_df.iloc[idx]['ecg_lr_path'] + '-0.png'
            matching_paths = [path for path in self.image_paths if img_name in path]
            # Use the first match if it exists
            img_path = matching_paths[0] if matching_paths else None

        image = Image.open(img_path).convert('RGB')
        label = self.label_df.iloc[idx]['Normal_ECG']
        label = torch.tensor(label)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label.long()