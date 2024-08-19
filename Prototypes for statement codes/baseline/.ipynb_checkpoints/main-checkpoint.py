import torch
import torch.optim as optim 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataset_class import ScpToLabelsDataset
from settings import train_batch_size, test_batch_size, \
                     input_size, num_classes, lr, num_epochs, \
                     train_df, test_df
from model import NeuralNet
from sklearn.metrics import roc_auc_score
import logging

# Configure logging
logging.basicConfig(filename='training_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

# Create dataset and dataloader
train_dataset = ScpToLabelsDataset(train_df)
train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

test_dataset = ScpToLabelsDataset(test_df)
test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

model = NeuralNet(input_size, num_classes)

# Use Binary Cross Entropy with Logits since it's multi-label classification
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
def train_model(dataloader, model, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for scps, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(scps.float())  # Convert input to float tensor
            loss = criterion(outputs, labels.float())  # Convert labels to float tensor
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(dataloader)
        logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss}')

# Evaluation function
def evaluate_model(dataloader, model):
    model.eval()
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for scps, labels in dataloader:
            outputs = model(scps.float())
            probs = torch.sigmoid(outputs)  # Apply sigmoid to get probabilities
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())

    all_labels = torch.cat(all_labels, dim=0)
    all_probs = torch.cat(all_probs, dim=0)

    # Calculate AUROC
    auroc = roc_auc_score(all_labels.numpy(), all_probs.numpy(), average='macro')

    logging.info(f'AUROC: {auroc}')

# Train the model
train_model(train_dataloader, model, criterion, optimizer, num_epochs=num_epochs)

# Evaluate the model on the test set
evaluate_model(test_dataloader, model)
