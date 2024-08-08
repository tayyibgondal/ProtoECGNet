import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import copy
import torch.nn as nn
import torch.nn.functional as F
from dataset_class_for_superclasses import ECGImageDataset
import numpy as np
from tqdm import tqdm
from vgg_11 import VGG11Classifier  

import wandb
wandb.login()

from settings import img_size, num_train_examples, num_test_examples, num_classes, input_channels, lr, num_epochs, train_batch_size, test_batch_size

# Define the data transformations
data_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),  # Resize image to 224x224
    transforms.ToTensor()                    # Convert image to Tensor
])

# Function to create a subset of the dataset
def create_subset(dataset, num_examples):
    num_examples = min(len(dataset), num_examples)
    indices = np.random.choice(len(dataset), num_examples, replace=False)
    subset = torch.utils.data.Subset(dataset, indices)
    return subset

def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, device, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            all_labels = []
            all_preds = []
            all_scores = []

            for inputs, labels in tqdm(dataloaders[phase], desc=f'{phase} phase', leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_scores.extend(F.softmax(outputs, dim=1).cpu().detach().numpy())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            auroc = roc_auc_score(all_labels, all_scores, multi_class='ovr', average='macro')
            f1 = f1_score(all_labels, all_preds, average='weighted')
            accuracy = accuracy_score(all_labels, all_preds)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} AUROC: {auroc:.4f} F1: {f1:.4f}')

            # Log metrics to wandb
            wandb.log({
                f'{phase}_loss': epoch_loss,
                f'{phase}_acc': epoch_acc,
                f'{phase}_auroc': auroc,
                f'{phase}_f1': f1
            })

    model.load_state_dict(best_model_wts)
    return model

def main():
    parser = argparse.ArgumentParser(description='Train ECG model with VGG11')
    parser.add_argument('-gpuid', type=int, default=0, help='GPU id to use')
    parser.add_argument('-experiment_run', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpuid}" if torch.cuda.is_available() else "cpu")
    
    run = wandb.init(
        project="ExplainableECGModels",
        config={
            "experiment_run": args.experiment_run
        },
    )

    train_dataset = ECGImageDataset(info_df_path='train-100HZ-files-and-labels.csv', transform=data_transform)
    val_dataset = ECGImageDataset(info_df_path='test-100HZ-files-and-labels.csv', transform=data_transform)

    if num_train_examples is not None:
        train_subset = create_subset(train_dataset, num_train_examples)
    else:
        train_subset = train_dataset

    if num_test_examples is not None:
        val_subset = create_subset(val_dataset, num_test_examples)
    else:
        val_subset = val_dataset

    train_loader = DataLoader(train_subset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=test_batch_size, shuffle=False)

    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }

    # Initialize VGG11 model
    model = VGG11Classifier(num_classes=num_classes, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset_sizes = {
        'train': len(train_subset),
        'val': len(val_subset)
    }

    model = train_model(model, criterion, optimizer, dataloaders, dataset_sizes, device, num_epochs=num_epochs)

if __name__ == "__main__":
    main()
