import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import wfdb

class PTBXL_Dataset(Dataset):
    def __init__(self, csv_file, reshape=False):
        """
        Args:
            csv_file (str): Path to the CSV file with ECG paths and labels.
            reshape (bool): Whether to reshape the ECG signal.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.reshape = reshape

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ecg_path = self.data_frame.iloc[idx, 0]  # Get the path to the ECG .dat file
        label = self.data_frame.iloc[idx, 1]     # Get the corresponding label

        # Load the ECG waveform from the .dat file
        ecg_record = wfdb.rdrecord(ecg_path)
        ecg_signal = ecg_record.p_signal  # Get the ECG signal as a NumPy array

        # Normalize the ECG signal (mean = 0, std = 1)
        ecg_signal = (ecg_signal - ecg_signal.mean(axis=0)) / ecg_signal.std(axis=0)

        # Convert to a PyTorch tensor
        ecg_signal = torch.tensor(ecg_signal, dtype=torch.float32)  # This is of shape (12, 1000), if frequency is 100Hz.
        # print(ecg_signal.shape)
        
        if self.reshape:
            # reshaping ecg signal to be in 3 channels, since the image backends work need 3 channels
            ecg_signal = ecg_signal.permute(1, 0).contiguous().view(3, 100, -1)  # Permute if needed

        return ecg_signal, label