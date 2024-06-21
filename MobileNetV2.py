from torchvision.models import mobilenet_v2
import torch.nn as nn

class MobileNetV2Model(nn.Module):
    def __init__(self, num_classes=16):
        super(MobileNetV2Model, self).__init__()

        self.mobilenet_v2 = mobilenet_v2(pretrained=False)

        n_inputs = self.mobilenet_v2.classifier[1].in_features
        self.mobilenet_v2.classifier[1] = nn.Sequential(
            nn.Linear(n_inputs, num_classes, bias=True),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.mobilenet_v2(x)
        return x

num_classes = 16
vision_model = MobileNetV2Model(num_classes)