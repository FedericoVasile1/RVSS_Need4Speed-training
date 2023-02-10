import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from metadata import CLASSES_LIST


class Net(nn.Module):
    def __init__(self, feat_vect_dim, no_global_avg_pool):
        super().__init__()
        num_classes = len(CLASSES_LIST)

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        if no_global_avg_pool:
            self.avgpool = nn.Identity()
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
            feat_vect_dim = 16 * 4 * 4
        self.fc1 = nn.Linear(feat_vect_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_model(model_name, feat_vect_dim=None, no_global_avg_pool=None):
    if model_name == "Net":
        return Net(feat_vect_dim, no_global_avg_pool) 

    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
        for param in model.parameters():
            param.requires_grad = False

        feat_vect_dim = model.classifier[-1].in_features
        model.classifier = nn.Identity()
        
        model = nn.Sequential(
            model,
            nn.Linear(feat_vect_dim, len(CLASSES_LIST))
        )

        return model

    raise NotImplementedError
