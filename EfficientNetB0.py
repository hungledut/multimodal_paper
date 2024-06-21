from torchvision.models import efficientnet_b0
import torch.nn as nn

class EfficientNetB0Model(nn.Module):
    def __init__(self, num_classes=16):
        super(EfficientNetB0Model, self).__init__()

        self.efficientnet_b0 = efficientnet_b0(pretrained=False)

        n_inputs = self.efficientnet_b0.classifier[1].in_features
        self.efficientnet_b0.classifier[1] = nn.Sequential(
            nn.Linear(n_inputs, num_classes, bias=True),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.efficientnet_b0(x)
        return x

num_classes = 16
vision_model = EfficientNetB0Model(num_classes)