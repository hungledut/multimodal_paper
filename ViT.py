import torch.nn as nn
from torchvision.models import vision_transformer

class VisionTransformerModel(nn.Module):
    def __init__(self, num_classes=16):
        super(VisionTransformerModel, self).__init__()

        self.vit = vision_transformer.vit_b_16(weights=None)

        n_inputs = self.vit.heads[0].in_features
        self.vit.heads[0] = nn.Sequential(
            nn.Linear(n_inputs, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.vit(x)
        return x

num_classes = 16
vision_transformer_model = VisionTransformerModel(num_classes)