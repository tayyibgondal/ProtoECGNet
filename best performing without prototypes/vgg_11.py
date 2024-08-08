import torch
import torch.nn as nn
from torchvision.models import vgg11

class VGG11Classifier(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(VGG11Classifier, self).__init__()
        # Load the pretrained VGG11 model
        self.vgg11_f = vgg11(pretrained=pretrained)
        
        # Modify the classifier to output the correct number of classes
        self.vgg11_f.classifier[-1] = nn.Linear(self.vgg11_f.classifier[-1].in_features, num_classes)

    def forward(self, x):
        return self.vgg11_f(x)