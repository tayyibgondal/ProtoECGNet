import os
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import pandas as pd
import ast

class ECGImageDataset(Dataset):
    def __init__(self, scp_statements_df_path, ptb_xl_database_df_path, image_dir, transform=None, test=False):
        self.scp_statements_df_path = scp_statements_df_path
        self.ptb_xl_database_df_path = ptb_xl_database_df_path
        self.image_dir = image_dir
        self.transform = transform
        self.ecg_labels, self.ecg_paths = self.get_ecg_paths_and_labels(test)
        self.image_paths = self._get_image_paths()

    def get_ecg_paths_and_labels(self, test):
        # Load the database file
        ptb_xl_database_df = pd.read_csv(self.ptb_xl_database_df_path, index_col='ecg_id')
        ptb_xl_database_df.scp_codes = ptb_xl_database_df.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load scp_statements.csv for diagnostic aggregation
        agg_df = pd.read_csv(self.scp_statements_df_path, index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]

        def aggregate_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in agg_df.index:
                    tmp.append(agg_df.loc[key].diagnostic_class)
            return list(set(tmp))

        # Apply diagnostic superclass
        ptb_xl_database_df['diagnostic_superclass'] = ptb_xl_database_df.scp_codes.apply(aggregate_diagnostic)
        Y = ptb_xl_database_df

        # Split data into train and test
        test_fold = 10
        y_train = Y[Y.strat_fold != test_fold]
        y_test = Y[Y.strat_fold == test_fold]

        if test:
            y = y_test
        else:
            y = y_train

        y_file_names = y.filename_lr.apply(lambda x: x.split('/')[-1])

        # Filter to get only elements with one class
        y_single_class = y[y.diagnostic_superclass.apply(lambda x: len(x) == 1)]

        # Flatten the list structure
        y_single_class_flat = y_single_class.diagnostic_superclass.apply(lambda x: x[0])

        # Initialize the label encoder
        label_encoder = LabelEncoder()

        # Fit the label encoder and transform the labels to integer encoded labels
        y_encoded = label_encoder.fit_transform(y_single_class_flat)

        return y_encoded, y_file_names.loc[y_single_class.index]

    def _get_image_paths(self):
        image_paths = []
        for root, _, files in os.walk(self.image_dir):
            for file in files:
                if file.endswith('.png'):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def __len__(self):
        return len(self.ecg_labels)

    def __getitem__(self, idx):
        img_name = self.ecg_paths.iloc[idx] + '-0.png'
        matching_paths = [path for path in self.image_paths if img_name in path]

        if not matching_paths:
            raise FileNotFoundError(f"Image {img_name} not found in the dataset.")

        img_path = matching_paths[0]
        image = Image.open(img_path).convert('RGB')
        label = self.ecg_labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label).long()
