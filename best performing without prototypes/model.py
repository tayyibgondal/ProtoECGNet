from vgg_features import vgg11_features, vgg13_features, vgg16_features, vgg19_features
import torch.nn as nn
import torch

class VGG_Classifier(nn.Module):
    def __init__(self, model_name, num_classes=5):
        super(VGG_Classifier, self).__init__()
        self.backend_model_bandwidth = 25088  # for vgg family of models
        self.model_loader = self.select_model_loader(model_name)
        self.model = self.model_loader(True)
        self.classifier_head = nn.Linear(self.backend_model_bandwidth, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.model(x)
        x = x.reshape(batch_size, -1)
        x = self.classifier_head(x)
        return x      

    def select_model_loader(self, model_name):
        if model_name == 'vgg11':
            model_loader = vgg11_features
        elif model_name == 'vgg13':
            model_loader = vgg13_features
        elif model_name == 'vgg16':
            model_loader = vgg16_features
        elif model_name == 'vgg19':
            model_loader = vgg19_features

        return model_loader