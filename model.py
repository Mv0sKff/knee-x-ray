from args import *
from torch import nn
from torch.nn import Module
from torchvision.models import resnet18, resnet34, resnet50


class MyModel(Module):
    def __init__(self, backbone="resnet18", pretrained=True):
        super().__init__()

        # TODO: could be done cleaner
        if backbone == "resnet18":
            self.model = resnet18(pretrained=pretrained)
        elif backbone == "resnet34":
            self.model = resnet34(pretrained=pretrained)
        elif backbone == "resnet50":
            self.model = resnet50(pretrained=pretrained)
        else:
            self.model = resnet18(pretrained=pretrained)

        self.model.fc = nn.Linear(self.model.fc.in_features, 5)

    def forward(self, x):
        return self.model(x)
