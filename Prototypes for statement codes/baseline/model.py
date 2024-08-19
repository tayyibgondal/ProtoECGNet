import torch
import torch.nn as nn

# Define the logistic regression model
class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
        
    def forward(self, x):
        x = self.linear(x)  # Linear layer
        return x