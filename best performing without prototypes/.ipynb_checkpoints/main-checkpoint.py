import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import os
import ast
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import copy
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from tqdm import tqdm
import argparse
from model import VGG_Classifier
from dataset_class_for_superclasses import ECGImageDataset

import wandb
wandb.login()

from settings import img_size, num_train_examples, num_test_examples, num_classes, lr, num_epochs, train_batch_size, test_batch_size, train_df, val_df

# Function to create a subset of the dataset
def create_subset(dataset, num_examples):
    # Ensure num_examples doesn't exceed the dataset length
    num_examples = min(len(dataset), num_examples)
    indices = np.random.choice(len(dataset), num_examples, replace=False)
    subset = torch.utils.data.Subset(dataset, indices)
    return subset

# ===========================================================
# Training Function
# ===========================================================
def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, device, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

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
    parser = argparse.ArgumentParser(description='Train ECG model')
    parser.add_argument('-gpuid', type=int, default=0, help='GPU id to use')
    parser.add_argument('-base', nargs=1, type=str, default='vgg19') 
    parser.add_argument('-experiment_run', type=int, default=0)
    parser.add_argument("-run_name", type=str, default="default_run_name", help="Name of the W&B run")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpuid}" if torch.cuda.is_available() else "cpu")
    
    run = wandb.init(
    # Set the project where this run will be logged
    project="ExplainableECGModels",
    name=args.run_name,
    # Track hyperparameters and run metadata
    config={
        "experiment_run": args.experiment_run
        },
    )

    # ===========================================================
    # Define transformations
    # ===========================================================
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])
    
    # ===========================================================
    # Load datasets, and create dataloaders
    # ===========================================================    
    # Create train and test datasets
    train_dataset = ECGImageDataset(train_df, transform=transform)
    val_dataset = ECGImageDataset(val_df, transform=transform)
    
    if num_train_examples is not None:
        train_subset = create_subset(train_dataset, num_train_examples)
    else: 
        train_subset = train_dataset
    
    if num_test_examples is not None:
        val_subset = create_subset(val_dataset, num_test_examples)
    else:
        val_subset = val_dataset
    
    # Create data loaders for the subsets
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=train_batch_size, shuffle=True, num_workers=4, pin_memory=False)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=test_batch_size, shuffle=True, num_workers=4, pin_memory=False)

    # ===========================================================
    # Call the train function
    # ===========================================================
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }
    
    # Initialize model, criterion, optimizer
    model = VGG_Classifier('vgg11', num_classes=num_classes).to(device) 
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Define dataset sizes
    dataset_sizes = {
        'train': len(train_dataset),
        'val': len(val_dataset)
    }

    # Train the model
    model = train_model(model, criterion, optimizer, dataloaders, dataset_sizes, device, num_epochs=25)
        
if __name__ == "__main__":
    main()
