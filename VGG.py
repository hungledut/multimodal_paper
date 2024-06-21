from torchvision.models import vgg16
import torch.nn as nn

class Vison_model(nn.Module):
    def __init__(self, num_classes=16):
        super(Vison_model, self).__init__()

        self.vgg16 = vgg16(pretrained=False)

        n_inputs = self.vgg16.classifier[6].in_features

        self.vgg16.classifier[6] = nn.Sequential(
            nn.Linear(n_inputs, num_classes, bias=True),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.vgg16(x)
        return x


num_classes = 16
vision_model = Vison_model(num_classes)