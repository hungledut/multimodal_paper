from torchvision.models import convnext_base
import torch.nn as nn

class ConvNextModel(nn.Module):
    def __init__(self, num_classes=16):
        super(ConvNextModel, self).__init__()

        self.convnext = convnext_base(pretrained=False)

        n_inputs = self.convnext.classifier[2].in_features
        self.convnext.classifier[2] = nn.Sequential(
            nn.Linear(n_inputs, num_classes, bias=True),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.convnext(x)
        return x

num_classes = 16
vision_model = ConvNextModel(num_classes)