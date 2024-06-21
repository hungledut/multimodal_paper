from torchvision.models import resnet18
import torch.nn as nn

class ResNet18Model(nn.Module):
    def __init__(self, num_classes=16):
        super(ResNet18Model, self).__init__()

        self.resnet18 = resnet18(pretrained=False)

        modelOutputFeats = self.resnet18.fc.in_features

        self.resnet18.fc = nn.Sequential(
            nn.Linear(modelOutputFeats,num_classes),
            nn.LogSoftmax(dim=1),
            )


    def forward(self, x):
        x = self.resnet18(x)
        return x


num_classes = 16
vision_model = ResNet18Model(num_classes)